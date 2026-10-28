"""RAFO on SWE-rebench with Sprout rollouts: reviewer-guided forest RL (Qwen3.8-27B, one 8-GPU node).

Sprout's RL driver runs and grades every trajectory; this launcher starts
Miles's trainer and SGLang engines, and SproutRolloutFn hands each prompt group
to the driver, with the engines' router as the policy endpoint Sprout's agents
call. Sprout's sandboxes must reach that router (``--model-endpoint``), and
Miles must reach the driver (``--sprout-url``).

Each task is one RAFO search (``--sprout-rollout-rafo``):

1. M roots (``--n-samples-per-prompt``, 4) run from scratch and are graded.
2. Only a task none of whose graded roots resolved is reviewed. At one point of
   each failed root, Sprout's reviewer writes the agent's next turn in its own
   voice: what to check and why, and the commands of the check, never the fix.
   K continuations (``rafo_continuations``, 2) restored at the point run after
   the turn: one event.
3. When every continuation of an event failed, the next depth reviews the first
   of them in the same way, up to D (``rafo_depth``, 3) deep and S events
   (``rafo_events``, 4) in all.

The RAFO loss (``miles.backends.training_utils.loss.hub.rafo_loss``) trains each
root and each continuation with PPO on its local RLOO advantage (among the
task's roots, among its event's continuations), and each reviewer turn with CE
on the turn alone, weighted by how its continuations did (a deeper event that
rescued one counts with its own weight). Every coefficient is a constant of the
search: 1/(M C), 1/(S K C) and w/(S C_h), with C and C_h the token normalizers.
All samples of a task are one rollout, and ``--global-batch-size`` counts the
tasks of an optimizer step: two steps for each rollout batch.

Time limits: roots 50 min, then per depth a review of 15 min, continuations of
15 min and a grade of 10 min (Sprout's ``grade_wall_seconds``), about 2.5 h for a
task that goes three deep. A root or continuation out of time is graded as it
stands; tests that run out of time score 0, and a grade that fails or runs out
of time is missing, not a 0.

``--load-rollout-data`` trains on a saved rollout without SGLang or Sprout.

Example:
    python examples/swe-rebench-sprout/run_qwen3_8_27b.py \
      --model-dir /root/models --data-dir /root/swe-inputs \
      --sprout-url http://SPROUT_HOST:11001 --model-endpoint http://THIS_HOST:18081
"""

from dataclasses import dataclass
from typing import Literal

import typer

from miles.utils.external_utils import command_utils

RAFO_LOSS = "miles.backends.training_utils.loss.hub.rafo_loss.rafo_loss_function"


@dataclass
class ScriptArgs(command_utils.ExecuteTrainConfig):
    run_id: str = command_utils.create_run_id()
    megatron_model_type: str = "qwen3.8-27B"
    megatron_path: str = "/root/Megatron-LM"
    num_gpus_per_node: int = 8
    # The trainer: TP x CP covers the node (data parallel 1); each SGLang engine has engine_tp GPUs.
    tp: int = 2
    cp: int = 4
    engine_tp: int = 2
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
    rollout_timeout: float = 3000.0
    # The roots' grades and the branches' grades, 10 min each, after their phases.
    finalization_timeout: float = 1800.0

    # The RAFO search: K continuations after each reviewer turn, at most D depths and S events per task.
    rafo_continuations: int = 2
    rafo_depth: int = 3
    rafo_events: int = 4
    review_seconds: float = 900.0
    branch_seconds: float = 900.0
    # RAFO's fixed token normalizers C and C_h, and how a turn's weight counts a deeper event.
    rafo_c: float = 64.0
    rafo_c_h: float = 32.0
    rafo_turn_credit: Literal["subtree", "own"] = "subtree"

    # Training
    num_rollout: int = 100
    rollout_batch_size: int = 16
    # RAFO's M roots per task, one local RLOO group.
    n_samples_per_prompt: int = 4
    temperature: float = 0.8
    max_response_len: int = 16384
    max_context_len: int = 131072
    # A bin holds max_tokens_per_gpu * cp tokens: one whole max_context_len trajectory.
    max_tokens_per_gpu: int = 32768
    lr: float = 1e-6
    save_interval: int = 10
    extra_args: str = ""

    def __post_init__(self):
        if self.tp * self.cp != self.num_gpus_per_node:
            raise ValueError("This single-node recipe colocates a TP x CP trainer that covers the node")
        if self.max_tokens_per_gpu * self.cp < self.max_context_len:
            raise ValueError("max_tokens_per_gpu * cp must hold one max_context_len trajectory")
        if self.n_samples_per_prompt < 2 or self.rafo_continuations < 2:
            raise ValueError("a task's roots and an event's continuations are local RLOO groups: at least 2 each")
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
        "--sprout-rollout-rafo "
        f"--sprout-rollout-rafo-continuations {args.rafo_continuations} "
        f"--sprout-rollout-rafo-depth {args.rafo_depth} "
        f"--sprout-rollout-rafo-events {args.rafo_events} "
        f"--sprout-rollout-rafo-review-seconds {args.review_seconds} "
        f"--sprout-rollout-rafo-wall-time-seconds {args.branch_seconds} "
        f"--rollout-temperature {args.temperature} "
        # The full softmax the trainer scores: Miles always sends top_k (-1 here)
        # and Sprout forwards it, so the engine does not fall back to the
        # checkpoint's generation_config; sampling-support replay (top-p below 1,
        # a positive top-k) is refused.
        "--rollout-top-p 1.0 --rollout-top-k -1 "
        f"--rollout-max-response-len {args.max_response_len} "
        # Sprout returns messages; Miles renders them with the policy's own
        # chat template, so the Qwen3 mask type marks the assistant turns. It
        # also trains an inserted turn's reasoning_content, which SproutRolloutFn
        # checks before a RAFO search starts (the qwen mask type fails that check).
        "--loss-mask-type qwen3 "
    )
    if args.model_endpoint:
        rollout_args += f"--sprout-rollout-model-endpoint {args.model_endpoint} "
    if args.load_rollout_data:
        rollout_args += f"--load-debug-rollout-data {args.load_rollout_data} "
    loss_args = _rafo_loss_args(args)
    optimizer_args = (
        f"--optimizer adam --lr {args.lr} --lr-decay-style constant "
        "--weight-decay 0.1 --adam-beta1 0.9 --adam-beta2 0.98 "
        "--optimizer-cpu-offload --overlap-cpu-optimizer-d2h-h2d --use-precision-aware-optimizer "
    )
    perf_args = (
        f"--tensor-model-parallel-size {args.tp} --sequence-parallel --pipeline-model-parallel-size 1 "
        f"--context-parallel-size {args.cp} --expert-model-parallel-size 1 --expert-tensor-parallel-size 1 "
        "--recompute-granularity full --recompute-method uniform --recompute-num-layers 1 "
        f"--use-dynamic-batch-size --max-tokens-per-gpu {args.max_tokens_per_gpu} "
        f"--seq-length {args.max_context_len} --log-probs-chunk-size 1024 "
    )
    sglang_args = (
        f"--rollout-num-gpus-per-engine {args.engine_tp} "
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
        + loss_args
        + optimizer_args
        + perf_args
        + sglang_args
        + misc_args
        + args.extra_args
    )


def _rafo_loss_args(args: ScriptArgs) -> str:
    return (
        # One rollout is one task: two optimizer steps for each rollout batch, each divided by its tasks.
        # A group that failed leaves its step short instead of dropping the tasks after it.
        f"--global-batch-size {args.rollout_batch_size // 2} "
        "--allow-partial-train-step "
        f"--sprout-rollout-rafo-c {args.rafo_c} "
        f"--sprout-rollout-rafo-c-h {args.rafo_c_h} "
        f"--sprout-rollout-rafo-turn-credit {args.rafo_turn_credit} "
        f"--loss-type custom_loss --custom-loss-function-path {RAFO_LOSS} "
        # The RAFO loss reads its advantages from the samples; this pass still computes pi_old.
        "--advantage-estimator grpo --disable-rewards-normalization "
        "--entropy-coef 0.0 --eps-clip 0.2 --eps-clip-high 0.2 --eps-clip-c 3.0 "
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
