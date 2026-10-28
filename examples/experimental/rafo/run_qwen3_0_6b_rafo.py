"""RAFO smoke recipe: Qwen3-0.6B on the toy bug-finding task, Megatron backend, one node.

Trains with reviewer-guided forests (see README.md next to this file): 4 roots for each task, review of the tasks
whose roots all fail, 2 continuations for each reviewer turn, nested review in the failed branch up to depth 3, and
the RAFO custom loss. The toy environment plays the rollout service, including its (oracle) reviewer.

What must exist: nothing. `prepare` downloads Qwen3-0.6B, converts it to Megatron torch_dist next to the HF
checkpoint, and writes the toy dataset if it does not exist yet; --skip-prepare reuses them.

Args:
  --num-gpus-per-node: GPUs of this node; actor and rollout are colocated (default: 1).
  --num-rollout: training iterations (default: 200).
  --model-dir / --data-dir / --output-dir: checkpoint, dataset and output directories.
  --skip-prepare: do not download, convert or write the dataset.
  --extra-args: appended to the train.py arguments.

Example:
  python examples/experimental/rafo/run_qwen3_0_6b_rafo.py \\
      --model-dir /models --data-dir /datasets --output-dir /output
"""

from dataclasses import dataclass, field
from pathlib import Path

import typer

from examples.experimental.rafo.toy_env import write_toy_dataset
from miles.utils.external_utils import command_utils

_MODEL_NAME = "Qwen3-0.6B"
_MEGATRON_MODEL_TYPE = "qwen3-0.6B"
_N_TASKS = 4096
_TASKS_PER_ITERATION = 16
_RAFO = "examples.experimental.rafo"


@dataclass
class ScriptArgs(command_utils.ExecuteTrainConfig):
    run_id: str = field(default_factory=command_utils.create_run_id)
    num_gpus_per_node: int = 1
    num_rollout: int = 200
    model_dir: str = "/root/models"
    data_dir: str = "/root/datasets"
    megatron_path: str = "/root/Megatron-LM"
    skip_prepare: bool = False
    extra_args: str = ""


def _dataset_path(args: ScriptArgs) -> Path:
    return Path(args.data_dir) / "rafo_toy" / "train.jsonl"


def _checkpoint_args(args: ScriptArgs) -> str:
    return (
        f"--hf-checkpoint {args.model_dir}/{_MODEL_NAME} "
        f"--load {args.model_dir}/{_MODEL_NAME}_torch_dist "
        f"--save {args.output_dir}/{args.run_id}/checkpoints "
        "--save-interval 50 "
    )


def _rollout_args(args: ScriptArgs) -> str:
    return (
        f"--prompt-data {_dataset_path(args)} "
        "--input-key prompt "
        "--label-key label "
        "--rollout-shuffle "
        f"--num-rollout {args.num_rollout} "
        f"--rollout-batch-size {_TASKS_PER_ITERATION} "
        # one rollout is one task: the forest samples its own roots
        "--n-samples-per-prompt 1 "
        # two optimizer steps for each iteration; miles divides each step's loss by its number of tasks
        f"--global-batch-size {_TASKS_PER_ITERATION // 2} "
        "--rollout-max-response-len 1024 "
        "--rollout-temperature 1 "
        "--disable-rewards-normalization "
    )


def _rafo_args(args: ScriptArgs) -> str:
    return (
        f"--custom-generate-function-path {_RAFO}.generate.RafoGenerate "
        f"--rafo-env-path {_RAFO}.toy_env.make_toy_env "
        f"--rafo-tool-specs-path {_RAFO}.toy_env.TOOL_SPECS "
        "--rafo-tool-call-parser qwen25 "
        "--rafo-max-turns 8 "
        f"--custom-rollout-log-function-path {_RAFO}.metrics.log_rollout_data "
    )


def _algorithm_args() -> str:
    return (
        # the RAFO loss reads its advantages from the samples; this estimator only keeps pi_old computed
        "--advantage-estimator grpo "
        "--loss-type custom_loss "
        f"--custom-loss-function-path {_RAFO}.loss.rafo_loss_function "
        "--eps-clip 0.2 "
        "--eps-clip-high 0.2 "
        "--eps-clip-c 3 "
    )


def _optimizer_args() -> str:
    return (
        "--optimizer adam "
        "--lr 3e-6 "
        "--lr-decay-style constant "
        "--weight-decay 0.0 "
        "--adam-beta1 0.9 "
        "--adam-beta2 0.999 "
        "--clip-grad 1.0 "
    )


def _perf_args() -> str:
    return (
        "--tensor-model-parallel-size 1 "
        "--pipeline-model-parallel-size 1 "
        "--context-parallel-size 1 "
        "--use-dynamic-batch-size "
        "--max-tokens-per-gpu 16384 "
    )


def _sglang_args() -> str:
    return "--rollout-num-gpus-per-engine 1 --sglang-mem-fraction-static 0.5 "


def _misc_args(args: ScriptArgs) -> str:
    return (
        "--attention-dropout 0.0 "
        "--hidden-dropout 0.0 "
        "--accumulate-allreduce-grads-in-fp32 "
        "--attention-softmax-in-fp32 "
        "--attention-backend flash "
        "--colocate "
        f"--actor-num-nodes {args.num_nodes} "
        f"--actor-num-gpus-per-node {args.num_gpus_per_node} "
        f"--num-gpus-per-node {args.num_gpus_per_node} "
    )


def _build_train_args(args: ScriptArgs) -> str:
    return "".join(
        (
            _checkpoint_args(args),
            _rollout_args(args),
            _rafo_args(args),
            _algorithm_args(),
            _optimizer_args(),
            _perf_args(),
            _sglang_args(),
            _misc_args(args),
            command_utils.get_default_wandb_args(__file__, run_name_prefix="rafo", run_id=args.run_id),
            f" {args.extra_args}",
        )
    )


def prepare(args: ScriptArgs) -> None:
    U = args.create_backend()
    U.exec_command_cpu(f"mkdir -p {args.model_dir} {_dataset_path(args).parent}")
    U.exec_command_cpu(f"hf download Qwen/{_MODEL_NAME} --local-dir {args.model_dir}/{_MODEL_NAME}")
    U.convert_checkpoint(
        model_name=_MODEL_NAME,
        megatron_model_type=_MEGATRON_MODEL_TYPE,
        num_gpus_per_node=args.num_gpus_per_node,
        dir_dst=args.model_dir,
        hf_checkpoint=f"{args.model_dir}/{_MODEL_NAME}",
        megatron_path=args.megatron_path,
    )
    if not _dataset_path(args).exists():
        _dataset_path(args).parent.mkdir(parents=True, exist_ok=True)
        write_toy_dataset(str(_dataset_path(args)), n_tasks=_N_TASKS, seed=0)


def execute(args: ScriptArgs) -> None:
    U = args.create_backend()
    U.execute_train(
        train_args=_build_train_args(args),
        num_gpus_per_node=args.num_gpus_per_node,
        megatron_model_type=_MEGATRON_MODEL_TYPE,
        megatron_path=args.megatron_path,
    )


@command_utils.dataclass_cli
def main(args: ScriptArgs) -> None:
    if not args.skip_prepare:
        prepare(args)
    execute(args)


if __name__ == "__main__":
    typer.run(main)
