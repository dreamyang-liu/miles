"""GRPO on SWE-rebench with Sprout's harness rollouts, after Arctic-SWE's recipe (Qwen3.5-4B, one node).

Each task is one plain group of ``--n-samples-per-prompt`` independent rollouts of Sprout's
``sprout-harness`` slot (no search, no review). Sprout's RL driver config runs the harness in its
training mode (``extra.harness.on_invalid: terminate``): a malformed turn, a response cut off by the
token limit, or a loop of repeated calls ends the rollout with reward -1, a rollout that a budget
ended before ``submit`` scores 0, both without a grade, and a submission gets its 0/1 test verdict
(Arctic-SWE, arXiv 2610.24914, Tables 5 and 18).

The defaults are the paper's overfit run (Section 3.1) on one 8-GPU node: 32 tasks of 5 rollouts an
update, 100 turns, 32k tokens a response, and the harness's 131,072-token trajectory budget (leave
the driver's ``max_total_tokens`` unset). The trainer is TP 2 x DP 4 and SGLang runs one engine a GPU.
The loss follows the paper where Miles can: group-centred advantages without the standard deviation
(``--disable-grpo-std-normalization``), no KL, no entropy term, AdamW at a constant 1e-6 with beta2
0.98, epsilon 1e-15, no weight decay and gradient clipping at 1.0, and the mean over every trained
token of the batch. One optimizer step a rollout batch makes the PPO ratio 1, so the step is a plain
policy gradient (CISPO's importance weights need rollout log-probs, which this path recomputes).

Example (8 GPUs):
    python examples/swe-rebench-sprout/run_harness_grpo.py --skip-prepare \
      --model-dir /data/models --hf-checkpoint /data/models/Qwen3.5-4B --data-dir /data/harness/train \
      --sprout-url http://127.0.0.1:11002 --model-endpoint http://THIS_HOST:18081 --output-dir /data/harness/run

The first overfit run fit two 96 GB GPUs with ``--num-gpus-per-node 2 --rollout-batch-size 16
--max-turns 60 --max-response-len 16384 --max-context-len 81920 --max-tokens-per-gpu 81920`` and a
56k-token trajectory budget in the driver.
"""

from dataclasses import dataclass

import typer

from miles.utils.external_utils import command_utils


@dataclass
class ScriptArgs(command_utils.ExecuteTrainConfig):
    run_id: str = command_utils.create_run_id()
    megatron_model_type: str = "qwen3.5-4B"
    megatron_path: str = "/root/Megatron-LM"
    num_gpus_per_node: int = 8
    #: The trainer's tensor and context parallel sizes; the node's other factor is data parallel.
    tp: int = 2
    cp: int = 1
    #: GPUs a rollout engine: one engine a GPU by default.
    engine_tp: int = 1
    skip_prepare: bool = False

    model_dir: str = "/opt/dlami/nvme/models"
    model_name: str = "Qwen3.5-4B"
    hf_checkpoint: str = "/opt/dlami/nvme/models/Qwen3.5-4B"
    served_model_name: str = "qwen3.5-4b"
    data_dir: str = "/opt/dlami/nvme/harness-exp/data/train"

    sprout_url: str = "http://127.0.0.1:11002"
    model_endpoint: str = ""
    router_port: int = 18081
    max_turns: int = 100
    rollout_timeout: float = 7200.0
    finalization_timeout: float = 1800.0

    num_rollout: int = 300
    rollout_batch_size: int = 32
    n_samples_per_prompt: int = 5
    temperature: float = 1.0
    max_response_len: int = 32768
    # A trajectory stops once a request's prompt and completion pass 131,072 tokens (the harness's
    # max_total_tokens), so with its last tool result it ends below 160k.
    max_context_len: int = 163840
    max_tokens_per_gpu: int = 163840
    lr: float = 1e-6
    save_interval: int = 5
    extra_args: str = ""

    def __post_init__(self):
        if self.num_gpus_per_node % (self.tp * self.cp) or self.num_gpus_per_node % self.engine_tp:
            raise ValueError("The node's GPUs must split into TP x CP trainer groups and into engines")
        if self.max_tokens_per_gpu * self.cp < self.max_context_len:
            raise ValueError("max_tokens_per_gpu * cp must hold one max_context_len trajectory")
        if self.n_samples_per_prompt < 2:
            raise ValueError("a task's rollouts are one GRPO group: at least 2")
        if not self.model_endpoint:
            raise ValueError("--model-endpoint is how Sprout's gateway reaches the policy: this host's router")


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
    samples = args.rollout_batch_size * args.n_samples_per_prompt
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
        f"--sprout-rollout-model-endpoint {args.model_endpoint} "
        f"--sprout-rollout-max-turns {args.max_turns} "
        f"--sprout-rollout-timeout-seconds {args.rollout_timeout} "
        f"--sprout-rollout-finalization-timeout-seconds {args.finalization_timeout} "
        f"--num-rollout {args.num_rollout} --rollout-batch-size {args.rollout_batch_size} "
        f"--n-samples-per-prompt {args.n_samples_per_prompt} "
        f"--rollout-temperature {args.temperature} "
        # The full softmax the trainer scores (Miles always sends top_k).
        "--rollout-top-p 1.0 --rollout-top-k -1 "
        f"--rollout-max-response-len {args.max_response_len} "
        "--loss-mask-type qwen3 "
    )
    loss_args = (
        # A group whose rewards are all equal trains nothing: it is left out, and the one optimizer step of
        # the rollout trains the rest (a static global batch would trim a short step to nothing).
        f"--global-batch-size {samples} --use-dynamic-global-batch-size "
        "--sprout-rollout-drop-zero-advantage-groups "
        "--advantage-estimator grpo --disable-grpo-std-normalization "
        "--entropy-coef 0.0 --eps-clip 0.2 --eps-clip-high 0.28 "
        "--calculate-per-token-loss "
    )
    optimizer_args = (
        f"--optimizer adam --lr {args.lr} --lr-decay-style constant "
        "--weight-decay 0.0 --adam-beta1 0.9 --adam-beta2 0.98 --adam-eps 1e-15 --clip-grad 1.0 "
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
        f"--sglang-context-length {args.max_context_len + args.max_response_len} "
        "--sglang-mem-fraction-static 0.65 "
        # The harness reads the raw text from the log-probs; the parsers still fill the structured fields.
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


def execute(args: ScriptArgs):
    args.create_backend().execute_train(
        train_args=train_args(args),
        num_gpus_per_node=args.num_gpus_per_node,
        megatron_model_type=args.megatron_model_type,
        megatron_path=args.megatron_path,
        extra_env_vars={
            "no_proxy": "*",
            # Variable-length trajectories fragment the allocator pool: the first step of the run asked for
            # an 80k-token micro-batch's 19 GiB of logits with 36 GiB reserved but free. expandable_segments
            # breaks torch_memory_saver under --colocate (see run_glm52_lora_tb2_daytona.py).
            "PYTORCH_CUDA_ALLOC_CONF": "garbage_collection_threshold:0.8,max_split_size_mb:512",
        },
    )


@command_utils.dataclass_cli
def main(args: ScriptArgs):
    if not args.skip_prepare:
        prepare(args)
    execute(args)


if __name__ == "__main__":
    typer.run(main)
