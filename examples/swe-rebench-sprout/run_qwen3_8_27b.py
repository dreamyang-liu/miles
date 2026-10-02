"""Standard GRPO on SWE-rebench with Sprout rollouts (Qwen3.8-27B, one 8-GPU node).

Sprout's RL driver runs and grades every trajectory; this launcher starts
Miles's trainer and SGLang engines, and SproutRolloutFn hands each prompt group
to the driver, with the engines' router as the policy endpoint Sprout's agents
call. Sprout's sandboxes must reach that router (``--model-endpoint``), and
Miles must reach the driver (``--sprout-url``).

A rollout recorded without GPUs (``rollout_only.py --save-debug-rollout-data``)
trains here without SGLang or Sprout: pass it as ``--load-rollout-data``.

Example:
    python examples/swe-rebench-sprout/run_qwen3_8_27b.py \
      --model-dir /root/models --data-dir /root/swe-inputs \
      --sprout-url http://SPROUT_HOST:11001 --model-endpoint http://THIS_HOST:18081
"""

from dataclasses import dataclass

import typer

from miles.utils.external_utils import command_utils


@dataclass
class ScriptArgs(command_utils.ExecuteTrainConfig):
    run_id: str = command_utils.create_run_id()
    megatron_model_type: str = "qwen3.8-27B"
    megatron_path: str = "/root/Megatron-LM"
    num_gpus_per_node: int = 8
    tp: int = 8
    skip_prepare: bool = False

    # Paths
    model_dir: str = "/root/models"
    model_name: str = "Qwen3.8-27B"
    hf_checkpoint: str = "Qwen/Qwen3.8-27B"
    served_model_name: str = "Qwen/Qwen3.8-27B"
    data_dir: str = "/root/swe-inputs"
    load_rollout_data: str = ""

    # Sprout
    sprout_url: str = "http://127.0.0.1:11001"
    model_endpoint: str = ""
    router_port: int = 18081
    max_turns: int = 64
    rollout_timeout: float = 3600.0
    finalization_timeout: float = 1800.0

    # GRPO
    num_rollout: int = 100
    rollout_batch_size: int = 4
    n_samples_per_prompt: int = 8
    temperature: float = 0.8
    max_response_len: int = 16384
    max_context_len: int = 131072
    max_tokens_per_gpu: int = 16384
    lr: float = 1e-6
    kl_coef: float = 0.0
    save_interval: int = 10
    extra_args: str = ""

    def __post_init__(self):
        if self.tp != self.num_gpus_per_node:
            raise ValueError("This single-node recipe colocates a full-node TP trainer and rollout engine")
        if not self.load_rollout_data and not self.model_endpoint:
            raise ValueError("--model-endpoint is how Sprout's sandboxes reach the policy: this host's router")


def prepare(args: ScriptArgs):
    """Convert the HF checkpoint to torch_dist once."""
    args.create_backend().convert_checkpoint(
        model_name=args.model_name,
        megatron_model_type=args.megatron_model_type,
        num_gpus_per_node=args.num_gpus_per_node,
        dir_dst=args.model_dir,
        hf_checkpoint=args.hf_checkpoint,
        megatron_path=args.megatron_path,
    )


def train_args(args: ScriptArgs) -> str:
    checkpoint_args = (
        f"--hf-checkpoint {args.hf_checkpoint} "
        f"--ref-load {args.model_dir}/{args.model_name}_torch_dist "
        f"--load {args.output_dir}/checkpoints --save {args.output_dir}/checkpoints "
        f"--save-interval {args.save_interval} "
    )
    rollout_args = (
        f"--prompt-data {args.data_dir}/miles.jsonl --input-key prompt --metadata-key metadata "
        "--rollout-shuffle "
        "--rollout-function-path miles.rollout.sprout.rollout_fn.SproutRolloutFn "
        f"--sprout-rollout-base-url {args.sprout_url} "
        f"--sprout-rollout-max-turns {args.max_turns} "
        f"--sprout-rollout-timeout-seconds {args.rollout_timeout} "
        f"--sprout-rollout-finalization-timeout-seconds {args.finalization_timeout} "
        f"--num-rollout {args.num_rollout} --rollout-batch-size {args.rollout_batch_size} "
        f"--n-samples-per-prompt {args.n_samples_per_prompt} "
        f"--global-batch-size {args.rollout_batch_size * args.n_samples_per_prompt} "
        f"--rollout-temperature {args.temperature} "
        f"--rollout-max-response-len {args.max_response_len} "
        # Sprout returns messages; Miles renders them with the policy's own
        # chat template, so the Qwen3 mask type marks the assistant turns.
        "--loss-mask-type qwen3 "
    )
    if args.model_endpoint:
        rollout_args += f"--sprout-rollout-model-endpoint {args.model_endpoint} "
    if args.load_rollout_data:
        rollout_args += f"--load-debug-rollout-data {args.load_rollout_data} "
    # Standard GRPO: group-normalized outcome reward, broadcast over the
    # assistant tokens of each trajectory. Log-probs are recomputed by the
    # trainer on the retokenized sequence (no --use-rollout-logprobs, no TIS).
    grpo_args = (
        "--advantage-estimator grpo "
        f"--use-kl-loss --kl-loss-coef {args.kl_coef} --kl-loss-type low_var_kl "
        "--entropy-coef 0.0 --eps-clip 0.2 --eps-clip-high 0.28 "
    )
    optimizer_args = (
        f"--optimizer adam --lr {args.lr} --lr-decay-style constant "
        "--weight-decay 0.1 --adam-beta1 0.9 --adam-beta2 0.98 "
        "--optimizer-cpu-offload --overlap-cpu-optimizer-d2h-h2d --use-precision-aware-optimizer "
    )
    perf_args = (
        f"--tensor-model-parallel-size {args.tp} --sequence-parallel --pipeline-model-parallel-size 1 "
        "--context-parallel-size 1 --expert-model-parallel-size 1 --expert-tensor-parallel-size 1 "
        "--recompute-granularity full --recompute-method uniform --recompute-num-layers 1 "
        f"--use-dynamic-batch-size --max-tokens-per-gpu {args.max_tokens_per_gpu} "
        f"--seq-length {args.max_context_len} --log-probs-chunk-size 1024 "
    )
    sglang_args = (
        f"--rollout-num-gpus-per-engine {args.tp} "
        f"--sglang-router-port {args.router_port} "
        f"--sglang-served-model-name {args.served_model_name} "
        f"--sglang-context-length {args.max_context_len} "
        "--sglang-mem-fraction-static 0.65 "
        # Sprout's agents call Chat Completions with one bash tool; the engine
        # must return tool calls and reasoning as structured fields.
        "--sglang-reasoning-parser qwen3 --sglang-tool-call-parser qwen3_coder "
    )
    misc_args = (
        "--train-backend megatron --colocate "
        f"--actor-num-nodes {args.num_nodes} --actor-num-gpus-per-node {args.num_gpus_per_node} "
        f"--num-gpus-per-node {args.num_gpus_per_node} "
        "--attention-dropout 0.0 --hidden-dropout 0.0 "
        "--accumulate-allreduce-grads-in-fp32 --attention-softmax-in-fp32 --attention-backend flash "
        f"--dump-details {args.output_dir}/dump_details "
    )
    return (
        checkpoint_args
        + rollout_args
        + grpo_args
        + optimizer_args
        + perf_args
        + sglang_args
        + misc_args
        + args.extra_args
    )


def execute(args: ScriptArgs):
    args.create_backend().execute_train(
        train_args=train_args(args),
        num_gpus_per_node=args.num_gpus_per_node,
        megatron_model_type=args.megatron_model_type,
        megatron_path=args.megatron_path,
        extra_env_vars={"no_proxy": "*"},
    )


@command_utils.dataclass_cli
def main(args: ScriptArgs):
    if not args.skip_prepare:
        prepare(args)
    execute(args)


if __name__ == "__main__":
    typer.run(main)
