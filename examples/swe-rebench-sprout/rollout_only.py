"""Run SproutRolloutFn once, on CPU, without a trainer: the integration test.

Everything up to the optimizer step needs no GPU: Sprout runs the agents against
whatever Chat Completions endpoint serves the policy, grades, and returns
messages; this script imports them exactly as the trainer would, gives them the
advantages ``train.py`` would -- Miles's GRPO reward normalization, or for a
search (``--sprout-rollout-search-points``) GSML's credit through
``rewards.post_process_gsml`` -- prints one line per sample, and saves the
samples in the format ``--load-debug-rollout-data`` reads. The file carries no
``dynamic_global_batch_size``: Miles records it when it trims a step to the
trainer's data-parallel size, and nothing here trims, so a trainer run with
``--use-dynamic-global-batch-size`` (``run_qwen3_8_27b.py``) refuses it.

    python examples/swe-rebench-sprout/rollout_only.py \
        --prompt-data /tmp/swe-inputs/miles.jsonl --hf-checkpoint Qwen/Qwen3-8B \
        --sprout-rollout-base-url http://127.0.0.1:11001 \
        --sprout-rollout-model-endpoint http://MODEL_HOST:PORT --model-name Qwen/Qwen3-8B \
        --rollout-batch-size 1 --n-samples-per-prompt 2 --sprout-rollout-max-turns 8 \
        --save-debug-rollout-data /tmp/swe-rollouts/rollout_data/{rollout_id}.pt
"""

import argparse
import json
import logging
from argparse import Namespace
from pathlib import Path

from miles.ray.rollout.debug_data import save_debug_rollout_data
from miles.ray.rollout.rollout_data_conversion import postprocess_rollout_data
from miles.ray.rollout.train_data_conversion import _post_process_rewards
from miles.rollout.base_types import RolloutFnConstructorInput, RolloutFnTrainInput
from miles.rollout.data_source import RolloutDataSource
from miles.rollout.inference_rollout.compatibility import call_rollout_function
from miles.rollout.sprout.rollout_fn import GSML_POST_PROCESS, SproutRolloutFn
from miles.utils.function_registry import load_function
from miles.utils.types import Sample

#: What RolloutDataSource, SproutRolloutFn and the reward path read beyond the
#: flags below: train.py's defaults, except where a comment says otherwise.
DEFAULTS = dict(
    rollout_global_dataset=True,
    rollout_shuffle=False,
    rollout_seed=42,
    rollout_max_prompt_len=None,
    input_key="prompt",
    metadata_key="metadata",
    label_key=None,
    tool_key=None,
    multimodal_keys=None,
    apply_chat_template=False,
    apply_chat_template_kwargs=None,
    chat_template_path=None,
    dump_details=None,
    save=None,
    load=None,
    rollout_top_p=1.0,
    rollout_top_k=-1,
    rollout_stop=None,
    rollout_stop_token_ids=None,
    sprout_rollout_token_env="SPROUT_RL_DRIVER_TOKEN",
    sglang_router_ip=None,
    sglang_router_port=None,
    sglang_served_model_name=None,
    use_rollout_logprobs=False,
    skip_actor_forward_only=False,
    use_tis=False,
    compute_advantages_and_returns=True,
    advantage_estimator="grpo",
    rewards_normalization=True,
    grpo_std_normalization=True,
    reward_key=None,
    normalize_advantages=False,
    # As run_qwen3_8_27b.py trains, and as a search requires: GSML's advantage
    # is every trainable token's, and a step returns however many samples it made.
    calculate_per_token_loss=True,
    use_dynamic_global_batch_size=True,
    # No trainer, so no data-parallel size to trim the step to.
    disable_rollout_trim_samples=True,
    global_batch_size=None,
    save_debug_trajectory_data=None,
    load_debug_rollout_data=None,
    ci_inject_rollout_data_path=None,
    custom_reward_post_process_path=None,
)


def parse_args() -> Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prompt-data", required=True, help="miles.jsonl from prepare_data.py")
    parser.add_argument("--hf-checkpoint", required=True, help="tokenizer and chat template of the policy")
    parser.add_argument("--model-name", default=None, help="the model name Sprout's gateway sends to the endpoint")
    parser.add_argument("--rollout-batch-size", type=int, default=1, help="prompt groups per rollout")
    parser.add_argument(
        "--n-samples-per-prompt",
        type=int,
        default=2,
        help="rollouts per task: the GRPO group, or a search's two roots",
    )
    parser.add_argument("--rollout-id", type=int, default=0)
    parser.add_argument("--rollout-temperature", type=float, default=1.0)
    parser.add_argument("--rollout-max-response-len", type=int, default=None, help="max_new_tokens per model call")
    parser.add_argument(
        "--loss-mask-type",
        default="qwen3",
        choices=("qwen", "qwen3", "distill_qwen"),
        help="as the launcher trains; a search refuses a mask type that does not train an inserted turn's reasoning",
    )
    parser.add_argument(
        "--save-debug-rollout-data",
        default=None,
        help="path template with {rollout_id}; written in the format --load-debug-rollout-data reads",
    )
    parser.add_argument(
        "--save-debug-trajectory-data",
        default=None,
        help="path template with {rollout_id}; one JSON line per sample with its messages",
    )
    SproutRolloutFn.add_arguments(parser)
    parsed = parser.parse_args()
    if not parsed.sprout_rollout_base_url:
        parser.error("--sprout-rollout-base-url is required")
    args = Namespace(**{**DEFAULTS, **vars(parsed)})
    if args.sprout_rollout_search_points:
        # What train.py is given as --custom-reward-post-process-path for a search.
        args.custom_reward_post_process_path = GSML_POST_PROCESS
    return args


def print_samples(samples: list[Sample], rewards: list[float], advantages: list[float]) -> None:
    """One line per sample: its reward and advantage A, and for a search sample its GSML credit --
    the case of its task, its z-score in its group and its share of the task's λ (``gsml.Credit``)."""

    def number(value: float | None) -> str:
        return "-" if value is None else f"{value:.3f}"

    print(
        f"{'group':>5} {'index':>5} {'role':>7} {'status':>9} {'reward':>6} {'case':>10} {'z':>7} {'share':>6} "
        f"{'A':>7} {'tokens':>6} {'response':>8} {'trained':>7}  condition"
    )
    for sample, reward, advantage in zip(samples, rewards, advantages, strict=True):
        lineage = (sample.train_metadata or {}).get("sprout_rollout", {})
        credit = lineage.get("credit") or {}
        print(
            f"{sample.group_index:>5} {sample.index:>5} {lineage.get('role', 'root'):>7} {sample.status.value:>9} "
            f"{reward:>6.2f} {credit.get('case', '-'):>10} {number(credit.get('z')):>7} "
            f"{number(credit.get('lambda_share')):>6} {advantage:>7.3f} {len(sample.tokens):>6} "
            f"{sample.response_length:>8} {sum(sample.loss_mask):>7}  {lineage.get('group') or 'root'}"
        )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = parse_args()
    data_source = RolloutDataSource(args)
    fn = SproutRolloutFn(RolloutFnConstructorInput(args=args, data_source=data_source))
    output = call_rollout_function(fn, RolloutFnTrainInput(rollout_id=args.rollout_id, weight_version=None))
    samples, metadata = postprocess_rollout_data(args, output.samples, train_parallel_config=None)
    if args.custom_reward_post_process_path:
        raw, normalized = load_function(args.custom_reward_post_process_path)(args, samples)
    else:
        raw, normalized = _post_process_rewards(args, samples, None)
    print(json.dumps(output.metrics, indent=2))
    print_samples(samples, raw, normalized)
    if args.save_debug_rollout_data:
        save_debug_rollout_data(args, samples, rollout_id=args.rollout_id, evaluation=False, metadata=metadata)
        print("saved", Path(args.save_debug_rollout_data.format(rollout_id=args.rollout_id)))


if __name__ == "__main__":
    main()
