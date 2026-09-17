"""Audit real updates without changing rewards, samples, or the optimizer."""

import importlib.util
from pathlib import Path
import time

import torch


ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    "variable_task_audit_helpers", "/deployment/b300-lora-32tasks-20260916/audit_helpers.py",
)
BASE = importlib.util.module_from_spec(spec)
spec.loader.exec_module(BASE)
BASE.ROOT = ROOT
_current = {}
_original_record = BASE._record


def _record_with_step(event, **fields):
    if event in {"train_microbatch", "nonzero_adapter_gradient"} or (
        event == "forward_shape" and fields.get("phase") == "train"
    ):
        fields = {**_current, **fields}
    _original_record(event, **fields)


BASE._record = _record_with_step


def before_log_prob(args, model, store_prefix):
    BASE._seen_shapes.discard(str(store_prefix))
    BASE.before_log_prob(args, model, store_prefix)


def before_train_step(args, rollout_id, step_id, model, optimizer, scheduler):
    assert args.micro_batch_size in (1, 2) and not args.use_dynamic_batch_size
    assert args.global_batch_size == 256 and args.seq_length == 131072
    assert args.use_dynamic_global_batch_size and args.rollout_batch_size == 32
    assert not args.sglang_speculative_algorithm and not args.enable_mtp_training
    assert args.ash_rollout_branching_return_mode == "all"
    assert args.lora_rank == 32 and not args.lora_train_only
    BASE._phase, BASE._model = "train", model
    BASE._nonzero_gradient = False
    BASE._seen_shapes.discard("train")
    BASE._inventory(args, model)
    _current.update(rollout_id=rollout_id, step_id=step_id)
    if rollout_id < 3:
        BASE._snapshot(model, f"before-rollout-{rollout_id}-step-{step_id}")
    if not getattr(optimizer, "_night_step_audited", False):
        original = optimizer.step

        def step(*values, **kwargs):
            torch.cuda.synchronize()
            started = time.monotonic()
            result = original(*values, **kwargs)
            torch.cuda.synchronize()
            BASE._record("optimizer_step", **_current, elapsed_seconds=time.monotonic() - started, result=str(result))
            return result

        optimizer.step = step
        optimizer._night_step_audited = True
    if not getattr(scheduler, "_variable_task_step_audited", False):
        original_scheduler_step = scheduler.step

        def scheduler_step(*values, **kwargs):
            increment = kwargs.get("increment", values[0] if values else None)
            result = original_scheduler_step(*values, **kwargs)
            BASE._record("scheduler_step", **_current, trajectory_increment=int(increment))
            return result

        scheduler.step = scheduler_step
        scheduler._variable_task_step_audited = True
    BASE._record(
        "before_live_train", **_current, micro_batch_size=args.micro_batch_size,
        tasks_per_update=32, global_batch_size="dynamic", nominal_slot_budget=256,
        loss_weighting=args.ash_rollout_loss_weighting, dynamic_microbatch_packing=False,
    )


def after_save(args, rollout_id, checkpoint_dir, hf_checkpoint_dir):
    if rollout_id < 3:
        BASE._snapshot(BASE._model, f"after-rollout-{rollout_id}")
    BASE._record("after_live_save", rollout_id=rollout_id, checkpoint_dir=str(checkpoint_dir))
