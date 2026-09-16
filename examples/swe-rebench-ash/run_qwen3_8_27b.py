"""Run a real colocated Qwen3.8-27B GRPO update with Ash message rollouts.

Requires a converted torch_dist checkpoint. Fresh rollouts also require a running
Ash v3 driver/worker and a reachable advertised model endpoint for Ash.
Saved real rollout dumps can resume training without those services.

Args:
  --model-dir: Parent directory of HF and converted checkpoints.
  --data-dir: Directory containing miles.jsonl.
  --ash-url: Ash v3 driver URL reachable from the Miles actors.
  --model-endpoint: Public model URL reachable from Ash (for example HTTPS 443).
  --router-port: Port for the router started by Miles.
  --num-rollout: Number of real updates (default 1).
  --max-turns: Maximum model invocations per trajectory, regardless of group size.
  --finalization-timeout: Extra time for final snapshots and Ash grading.
  --log-probs-chunk-size: Token chunk size for memory-bounded logprob computation.
  --load-rollout-data: Reuse a saved real rollout dump for training without SGLang.
  --train-memory-margin-bytes: GPU headroom reserved by the training allocator.
  --lora-rank: Use Bridge LoRA on a saved rollout (0 keeps full-parameter training).
  --extra-args: Additional train.py arguments, including optional audit hooks.

Example:
    python examples/swe-rebench-ash/run_qwen3_8_27b.py \
      --ash-url http://host.docker.internal:19021 \
      --model-endpoint https://MODEL_HOST --router-port 18081
"""

import shlex
from dataclasses import dataclass

import typer

import miles.utils.external_utils.command_utils as U

_LORA_LAYERS = "language_model.decoder.layers.*"
_LORA_TARGETS = ",".join(
    f"{_LORA_LAYERS}.{name}"
    for name in (
        "self_attention.linear_qkv",
        "self_attention.linear_proj",
        "self_attention.in_proj",
        "self_attention.out_proj",
        "mlp.linear_fc1",
        "mlp.linear_fc2",
    )
)


@dataclass
class ScriptArgs(U.ExecuteTrainConfig):
    model_dir: str = "/root/models"
    data_dir: str = "/root/datasets"
    megatron_path: str = "/root/Megatron-LM"
    model_name: str = "Qwen3.8-27B"
    served_model_name: str = "Qwen/Qwen3.8-27B"
    num_gpus_per_node: int = 8
    tp: int = 8
    router_port: int = 18081
    ash_url: str = "http://127.0.0.1:11001"
    model_endpoint: str = ""
    num_rollout: int = 1
    rollout_batch_size: int = 1
    n_samples_per_prompt: int = 2
    max_turns: int = 64
    rollout_timeout: float = 3600.0
    finalization_timeout: float = 1800.0
    max_response_len: int = 16384
    max_context_len: int = 262144
    max_tokens_per_gpu: int = 8192
    log_probs_chunk_size: int = 1024
    load_rollout_data: str = ""
    recompute_loss: bool = True
    train_memory_margin_bytes: int = 1024**3
    lora_rank: int = 0
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    lora_target_modules: str = _LORA_TARGETS
    lr: float = 1e-6
    temperature: float = 0.8
    run_id: str = U.create_run_id()
    extra_args: str = ""

    def __post_init__(self):
        if not self.model_endpoint:
            raise ValueError("--model-endpoint must be reachable from Ash")
        if self.tp != self.num_gpus_per_node:
            raise ValueError("This single-node recipe colocates a full-node TP trainer and rollout engine")
        if self.lora_rank < 0:
            raise ValueError("--lora-rank must be nonnegative")
        if self.lora_rank and not self.load_rollout_data:
            raise ValueError("Ash LoRA currently requires --load-rollout-data; live adapter routing needs validation")


def execute(args: ScriptArgs):
    reference_path = f"{args.model_dir}/{args.model_name}"
    if not args.lora_rank:
        reference_path += "_torch_dist"
    checkpoint_args = (
        f"--hf-checkpoint {args.model_dir}/{args.model_name} "
        f"--ref-load {reference_path} "
        f"--load {args.output_dir}/checkpoints "
        f"--save {args.output_dir}/checkpoints --save-interval 1 "
        "--no-save-optim --no-save-rng --make-vocab-size-divisible-by 64 "
    )
    rollout_args = (
        f"--prompt-data {args.data_dir}/miles.jsonl --input-key prompt --metadata-key metadata "
        "--rollout-function-path miles.rollout.ash.message_rollout.AshMessageRolloutFn "
        f"--ash-rollout-base-url {args.ash_url} --ash-rollout-model-endpoint {args.model_endpoint} "
        f"--ash-rollout-max-turns {args.max_turns} "
        f"--ash-rollout-timeout-seconds {args.rollout_timeout} "
        f"--ash-rollout-finalization-timeout-seconds {args.finalization_timeout} "
        f"--num-rollout {args.num_rollout} --rollout-batch-size {args.rollout_batch_size} "
        f"--n-samples-per-prompt {args.n_samples_per_prompt} "
        f"--global-batch-size {args.rollout_batch_size * args.n_samples_per_prompt} "
        f"--rollout-max-response-len {args.max_response_len} "
        f"--rollout-temperature {args.temperature} --loss-mask-type qwen3 "
        f"--log-probs-chunk-size {args.log_probs_chunk_size} "
    )
    if args.load_rollout_data:
        rollout_args += f"--load-debug-rollout-data {args.load_rollout_data} "
    perf_args = (
        f"--tensor-model-parallel-size {args.tp} --pipeline-model-parallel-size 1 "
        "--context-parallel-size 1 --expert-model-parallel-size 1 --expert-tensor-parallel-size 1 "
        "--sequence-parallel --micro-batch-size 1 "
        "--recompute-granularity full --recompute-method uniform --recompute-num-layers 1 "
        f"--use-dynamic-batch-size --max-tokens-per-gpu {args.max_tokens_per_gpu} "
        f"--seq-length {args.max_context_len} "
        f"--train-memory-margin-bytes {args.train_memory_margin_bytes} "
    )
    if args.recompute_loss:
        perf_args += "--recompute-loss-function "
    if args.lora_rank:
        perf_args += (
            "--megatron-to-hf-mode bridge --mtp-num-layers 0 --no-gradient-accumulation-fusion "
            f"--lora-rank {args.lora_rank} --lora-alpha {args.lora_alpha} "
            f"--lora-dropout {args.lora_dropout} --target-modules {shlex.quote(args.lora_target_modules)} "
            "--lora-base-cpu-backup "
        )
    algorithm_args = (
        "--advantage-estimator grpo --use-kl-loss --kl-loss-coef 0.00 --kl-loss-type low_var_kl "
        "--entropy-coef 0.00 --eps-clip 0.2 --eps-clip-high 0.28 "
    )
    optimizer_args = (
        f"--optimizer adam --lr {args.lr} --lr-decay-style constant "
        "--weight-decay 0.1 --adam-beta1 0.9 --adam-beta2 0.98 "
        "--optimizer-cpu-offload --overlap-cpu-optimizer-d2h-h2d --use-precision-aware-optimizer "
    )
    inference_args = (
        f"--rollout-num-gpus-per-engine {args.tp} "
        "--use-miles-router "
        f"--sglang-router-port {args.router_port} "
        f"--sglang-served-model-name {args.served_model_name} "
        f"--sglang-context-length {args.max_context_len} "
        "--sglang-mem-fraction-static 0.65 --sglang-max-running-requests 4 "
        "--sglang-attention-backend fa3 --sglang-kv-cache-dtype bfloat16 "
        "--sglang-reasoning-parser qwen3 --sglang-tool-call-parser qwen3_coder "
        "--sglang-chunked-prefill-size 8192 --sglang-cuda-graph-max-bs 4 "
    )
    misc_args = (
        "--train-backend megatron --colocate "
        f"--actor-num-nodes 1 --actor-num-gpus-per-node {args.num_gpus_per_node} "
        f"--num-gpus-per-node {args.num_gpus_per_node} "
        "--attention-dropout 0.0 --hidden-dropout 0.0 "
        "--accumulate-allreduce-grads-in-fp32 --attention-softmax-in-fp32 --attention-backend flash "
        f"--dump-details {args.output_dir}/dump_details "
    )
    U.execute_train(
        train_args=(
            checkpoint_args
            + rollout_args
            + perf_args
            + algorithm_args
            + optimizer_args
            + inference_args
            + misc_args
            + U.get_default_wandb_args(__file__, run_id=args.run_id)
            + " "
            + args.extra_args
        ),
        num_gpus_per_node=args.num_gpus_per_node,
        megatron_model_type="qwen3.8-27B",
        megatron_path=args.megatron_path,
        config=args,
        extra_env_vars={
            "NCCL_NVLS_ENABLE": "0",
            "no_proxy": "*",
        },
    )


@U.dataclass_cli
def main(args: ScriptArgs):
    execute(args)


if __name__ == "__main__":
    typer.run(main)
