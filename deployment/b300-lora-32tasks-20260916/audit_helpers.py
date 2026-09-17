"""Record LoRA coverage, packed shapes, gradients and adapter updates."""

import datetime
import hashlib
import json
from pathlib import Path
import time

import torch
import torch.distributed as dist


ROOT = Path(__file__).resolve().parent
_phase = "initial"
_registered = set()
_seen_shapes = set()
_model = None
_nonzero_gradient = False


def _unwrap(module):
    while hasattr(module, "module"):
        module = module.module
    return module


def _record(event, **fields):
    rank = dist.get_rank()
    row = {
        "event": event, "rank": rank,
        "time": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "allocated_bytes": torch.cuda.memory_allocated(),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(), **fields,
    }
    with (ROOT / f"audit-rank-{rank}.jsonl").open("a") as stream:
        stream.write(json.dumps(row) + "\n")


def _gradient(value):
    global _nonzero_gradient
    if not _nonzero_gradient:
        norm = value.detach().float().norm().item()
        if norm > 0:
            assert torch.isfinite(value).all()
            _nonzero_gradient = True
            _record("nonzero_adapter_gradient", norm=norm, shape=list(value.shape))
    return value


def _shape(module, args, kwargs):
    if _phase == "train":
        training_tokens = kwargs.get("input_ids")
        if training_tokens is None and args and isinstance(args[0], torch.Tensor):
            training_tokens = args[0]
        if training_tokens is not None:
            _record("train_microbatch", input_shape=list(training_tokens.shape))
    if _phase in _seen_shapes:
        return
    tokens = kwargs.get("input_ids")
    if tokens is None and args and isinstance(args[0], torch.Tensor):
        tokens = args[0]
    packed = kwargs.get("packed_seq_params")
    if tokens is not None:
        _seen_shapes.add(_phase)
        _record(
            "forward_shape", phase=_phase, input_shape=list(tokens.shape),
            max_seqlen_q=getattr(packed, "max_seqlen_q", None),
            max_seqlen_kv=getattr(packed, "max_seqlen_kv", None),
            cu_seqlens_q=(
                packed.cu_seqlens_q.detach().cpu().tolist()
                if packed is not None and packed.cu_seqlens_q is not None else None
            ),
        )


def _inventory(args, model):
    for chunk in model:
        module = _unwrap(chunk)
        if id(module) in _registered:
            continue
        _registered.add(id(module))
        trainable, frozen = [], 0
        assert args.lora_rank == 32 and module.config.mtp_num_layers == 0
        for name, parameter in module.named_parameters():
            if parameter.requires_grad:
                assert "adapter" in name or "lora" in name, name
                trainable.append({"name": name, "shape": list(parameter.shape), "numel": parameter.numel()})
                parameter.register_hook(_gradient)
            else:
                frozen += parameter.numel()
        assert trainable
        module.register_forward_pre_hook(_shape, with_kwargs=True)
        _record(
            "inventory", trainable=trainable,
            trainable_numel=sum(item["numel"] for item in trainable), frozen_numel=frozen,
            tp=args.tensor_model_parallel_size, cp=args.context_parallel_size,
            micro_batch_size=args.micro_batch_size, rank=dist.get_rank(),
            lora_rank=args.lora_rank, mtp_num_layers=module.config.mtp_num_layers,
            params_dtype=str(module.config.params_dtype), fp8=module.config.fp8,
        )


def _snapshot(model, phase):
    adapters, frozen = {}, {}
    for index, chunk in enumerate(model):
        for name, parameter in _unwrap(chunk).named_parameters():
            key = f"{index}.{name}"
            if parameter.requires_grad:
                adapters[key] = parameter.detach().cpu().clone()
            else:
                sample = parameter.detach().reshape(-1)[:4096].cpu().contiguous()
                frozen[key] = hashlib.sha256(sample.view(torch.uint8).numpy().tobytes()).hexdigest()
    rank = dist.get_rank()
    torch.save(adapters, ROOT / f"adapters-{phase}-rank-{rank}.pt")
    (ROOT / f"frozen-prefix-{phase}-rank-{rank}.json").write_text(json.dumps(frozen) + "\n")
    _record("snapshot", phase=phase, adapter_tensors=len(adapters), frozen_prefixes=len(frozen))


def before_log_prob(args, model, store_prefix):
    global _phase
    _phase = str(store_prefix)
    _inventory(args, model)
    _record("before_log_prob", phase=_phase)


def before_train_step(args, rollout_id, step_id, model, optimizer, scheduler):
    global _phase, _model
    _phase, _model = "train", model
    _inventory(args, model)
    _snapshot(model, "before")
    original = optimizer.step

    def step(*args, **kwargs):
        torch.cuda.synchronize()
        start = time.monotonic()
        result = original(*args, **kwargs)
        torch.cuda.synchronize()
        _record("optimizer_step", elapsed_seconds=time.monotonic() - start, result=str(result))
        return result

    optimizer.step = step
    _record("before_train", rollout_id=rollout_id, step_id=step_id)


def after_save(args, rollout_id, checkpoint_dir, hf_checkpoint_dir):
    assert _model is not None
    _snapshot(_model, "after")
    _record("after_save", rollout_id=rollout_id, checkpoint_dir=str(checkpoint_dir))
