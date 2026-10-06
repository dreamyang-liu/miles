"""Hindsight search on SWE-rebench with Sprout rollouts, trained with GSML credit (Qwen3.8-27B, one 8-GPU node).

Sprout's RL driver runs and grades every trajectory; this launcher starts
Miles's trainer and SGLang engines, and SproutRolloutFn hands each prompt group
to the driver, with the engines' router as the policy endpoint Sprout's agents
call. Sprout's sandboxes must reach that router (``--model-endpoint``), and
Miles must reach the driver (``--sprout-url``).

Each task gets two roots, and how they did decides what Sprout runs next:

    TT  both resolved      nothing
    TF  one resolved       a review picks a point in the failed root; 2 students run from it
    FF  neither resolved   the review picks a point in each root; from each point 1 student,
                           and up to 3 rounds of repair: one turn and one continuation each,
                           a round reviewing the last one's failed continuation for the next
                           turn, until the point resolves

A student continues from the point's restored state with nothing added; a
repair continuation first runs the reviewer's turn, inserted as the
assistant's reasoning (``reasoning_content``) and its commands. A root whose
actor failed, or whose grade gave no verdict, is missing, not a 0: with none
resolved and one missing the task branches as TF does (FF_partial); with every
graded root resolved, or none graded, it ends at its roots.

GSML gives each sample one advantage A, carried by every token it trains
(``rewards.post_process_gsml``). A root gets β_root·z, its z-score among the
task's roots, with β_root = (1 - λ)/√2: ±0.30 for a TF pair at λ = 0.4. A
student or repair continuation gets β_br·z within the group of its point (and
candidate); its restored prefix and the inserted turn are context only. Only
an FF task adds λ credit, split evenly among its points where something
resolved: to the students that resolved there (the policy's own success) or,
where none did, to the repairs whose continuations resolved without changing
the task's test files and whose turn names no hidden test the history had not
shown -- to those continuations, and to the repair turn itself as a one-turn
sample (the â sample, weighted ψ). The coefficients are applied when the step
is trained, so a saved rollout can be re-credited under other GSML flags.

Time limits: roots 50 min, then per repair round a review of 15 min, branches
of 15 min and a grade of 10 min (Sprout's ``grade_wall_seconds``), about 3 h
for an FF task that uses all three rounds. A root or branch out of time is graded as it
stands; tests that run out of time score 0, and a grade that fails or runs out
of time is missing. How many samples a step returns varies, so the step is one
optimizer step over all of them (``--use-dynamic-global-batch-size``), and the
loss is a sum over their trainable tokens (``--calculate-per-token-loss``).

``--load-rollout-data`` trains on a saved rollout without SGLang or Sprout. One
saved by ``rollout_only.py`` lacks the ``dynamic_global_batch_size`` metadata
that ``--use-dynamic-global-batch-size`` requires of it (README, "Without a GPU").

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

    # Hindsight search: one point in each failed root
    search_points: int = 1
    # FF (neither root resolved): students per point, then rounds of one repair turn and one continuation.
    search_candidates: int = 1
    search_student_continuations: int = 1
    search_candidate_continuations: int = 1
    search_repair_rounds: int = 3
    # TF (one root resolved, or none and one missing): students per point, and no repair turns.
    search_tf_student_continuations: int = 2
    review_seconds: float = 900.0
    branch_seconds: float = 900.0

    # GSML: λ is the credit an FF task splits among its points where something
    # resolved, and sets a root's weight (1 - λ)/√2; β_br weighs a continuation's
    # z-score and ψ an â sample's share. c_pre (credit for a branch's restored
    # prefix) and κ+ (guided credit when a root resolved) are not implemented:
    # anything but 0 is refused.
    gsml_lambda: float = 0.4
    gsml_beta_branch: float = 1.0
    gsml_psi: float = 1.0
    gsml_c_pre: float = 0.0
    gsml_kappa_plus: float = 0.0

    # GRPO
    num_rollout: int = 100
    rollout_batch_size: int = 16
    # The roots, two per task (GSML): a search group adds its branches to them.
    n_samples_per_prompt: int = 2
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
        if self.n_samples_per_prompt != 2:
            raise ValueError("GSML credits a task's two roots: n_samples_per_prompt must be 2")
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
        # Nominal: the step trains every sample it got, roots and branches alike.
        f"--global-batch-size {args.rollout_batch_size * args.n_samples_per_prompt} "
        "--use-dynamic-global-batch-size "
        f"--sprout-rollout-search-points {args.search_points} "
        f"--sprout-rollout-search-candidates {args.search_candidates} "
        f"--sprout-rollout-search-student-continuations {args.search_student_continuations} "
        f"--sprout-rollout-search-candidate-continuations {args.search_candidate_continuations} "
        f"--sprout-rollout-search-tf-student-continuations {args.search_tf_student_continuations} "
        f"--sprout-rollout-search-repair-rounds {args.search_repair_rounds} "
        f"--sprout-rollout-search-review-seconds {args.review_seconds} "
        f"--sprout-rollout-search-wall-time-seconds {args.branch_seconds} "
        # Each sample's GSML credit, under these coefficients, becomes its advantage.
        "--custom-reward-post-process-path miles.rollout.sprout.rewards.post_process_gsml "
        f"--sprout-rollout-gsml-lambda {args.gsml_lambda} "
        f"--sprout-rollout-gsml-beta-branch {args.gsml_beta_branch} "
        f"--sprout-rollout-gsml-psi {args.gsml_psi} "
        f"--sprout-rollout-gsml-c-pre {args.gsml_c_pre} "
        f"--sprout-rollout-gsml-kappa-plus {args.gsml_kappa_plus} "
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
        # checks before a search starts (the qwen mask type fails that check).
        "--loss-mask-type qwen3 "
    )
    if args.model_endpoint:
        rollout_args += f"--sprout-rollout-model-endpoint {args.model_endpoint} "
    if args.load_rollout_data:
        rollout_args += f"--load-debug-rollout-data {args.load_rollout_data} "
    # Each sample's advantage is its GSML A as post_process_gsml returns it:
    # --advantage-estimator grpo broadcasts it over the sample's trainable
    # tokens, and nothing normalizes it again (a search refuses
    # --normalize-advantages). Log-probs are recomputed by the trainer on the
    # retokenized sequence (no --use-rollout-logprobs, no TIS). No KL loss: at
    # coefficient 0 it would only add the reference model's forward pass.
    #
    # --calculate-per-token-loss: the loss is a sum over the step's trainable
    # tokens, each weighted by its sample's A, so a token weighs the same
    # whatever its sample's length; the sum is divided by the step's token
    # count. That count varies from step to step (samples whose A is 0 add to it
    # as well), so the update is consistent, not exact; an exact one needs a
    # constant divisor, the CP-aware reducer that is deferred. In raw mode (the
    # default --megatron-to-hf-mode) with CP=4, Miles counts each sample's tokens
    # on every CP rank and scales the loss by cp, and Megatron reduces
    # num_tokens over dp x cp, so the two cancel.
    grpo_args = (
        "--advantage-estimator grpo "
        "--calculate-per-token-loss "
        "--entropy-coef 0.0 --eps-clip 0.2 --eps-clip-high 0.28 "
    )
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
