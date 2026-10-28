"""Run SproutRolloutFn once, on CPU, without a trainer: the integration test.

Everything up to the optimizer step needs no GPU: Sprout runs the agents against
whatever Chat Completions endpoint serves the policy, grades, and returns
messages; this script imports them exactly as the trainer would and prints one
line per sample: a plain group's GRPO advantage, or for a RAFO search
(``--sprout-rollout-rafo``) the role, advantage and coefficient that the RAFO
loss trains the sample with. It saves the samples in the format
``--load-debug-rollout-data`` reads.

    python examples/swe-rebench-sprout/rollout_only.py \
        --prompt-data /tmp/swe-inputs/miles.jsonl --hf-checkpoint Qwen/Qwen3-8B \
        --sprout-rollout-base-url http://127.0.0.1:11001 \
        --sprout-rollout-model-endpoint http://MODEL_HOST:PORT --model-name Qwen/Qwen3-8B \
        --rollout-batch-size 1 --n-samples-per-prompt 4 --sprout-rollout-max-turns 8 \
        --sprout-rollout-rafo --save-debug-rollout-data /tmp/swe-rollouts/rollout_data/{rollout_id}.pt
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
from miles.rollout.sprout.rafo_role import METADATA_KEY, RafoRole
from miles.rollout.sprout.rollout_fn import RAFO_LOSS, SproutRolloutFn
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
    use_rollout_routing_replay=False,
    use_rollout_indexer_replay=False,
    use_sampling_support_replay=False,
    compute_advantages_and_returns=True,
    advantage_estimator="grpo",
    rewards_normalization=True,
    grpo_std_normalization=True,
    reward_key=None,
    normalize_advantages=False,
    loss_type="policy_loss",
    custom_loss_function_path=None,
    train_backend="megatron",
    use_dynamic_global_batch_size=False,
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
        help="rollouts per task: the GRPO group, or a RAFO search's roots",
    )
    parser.add_argument("--rollout-id", type=int, default=0)
    parser.add_argument("--rollout-temperature", type=float, default=1.0)
    parser.add_argument("--rollout-max-response-len", type=int, default=None, help="max_new_tokens per model call")
    parser.add_argument(
        "--loss-mask-type",
        default="qwen3",
        choices=("qwen", "qwen3", "distill_qwen"),
        help="as the launcher trains; RAFO refuses a mask type that does not train an inserted turn's reasoning",
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
    if args.sprout_rollout_rafo:
        # What train.py is given for a RAFO search (run_qwen3_8_27b.py).
        args.loss_type, args.custom_loss_function_path, args.rewards_normalization = "custom_loss", RAFO_LOSS, False
    return args


def print_samples(samples: list[Sample], rewards: list[float], advantages: list[float]) -> None:
    """One line per sample: its reward, and the advantage and coefficient it trains with -- a RAFO sample's
    role (``rafo_role.RafoRole``), or a plain group's GRPO advantage."""
    print(
        f"{'group':>5} {'index':>5} {'role':>7} {'status':>9} {'reward':>6} {'A':>7} {'coef':>9} "
        f"{'tokens':>6} {'response':>8} {'trained':>7}  condition"
    )
    for sample, reward, advantage in zip(samples, rewards, advantages, strict=True):
        lineage = (sample.train_metadata or {}).get("sprout_rollout", {})
        if METADATA_KEY in (sample.train_metadata or {}):
            role = RafoRole.of(sample.train_metadata)
            kind, advantage, coef = role.kind, role.adv, f"{role.coef:.3e}"
        else:
            kind, coef = lineage.get("role", "root"), "-"
        print(
            f"{sample.group_index:>5} {sample.index:>5} {kind:>7} {sample.status.value:>9} {reward:>6.2f} "
            f"{advantage:>7.3f} {coef:>9} {len(sample.tokens):>6} {sample.response_length:>8} "
            f"{sum(sample.loss_mask):>7}  {lineage.get('group') or 'root'}"
        )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = parse_args()
    data_source = RolloutDataSource(args)
    fn = SproutRolloutFn(RolloutFnConstructorInput(args=args, data_source=data_source))
    output = call_rollout_function(fn, RolloutFnTrainInput(rollout_id=args.rollout_id, weight_version=None))
    samples, metadata = postprocess_rollout_data(args, output.samples, train_parallel_config=None)
    raw, normalized = _post_process_rewards(args, samples, None)
    print(json.dumps(output.metrics, indent=2))
    print_samples(samples, raw, normalized)
    if args.save_debug_rollout_data:
        save_debug_rollout_data(args, samples, rollout_id=args.rollout_id, evaluation=False, metadata=metadata)
        print("saved", Path(args.save_debug_rollout_data.format(rollout_id=args.rollout_id)))


if __name__ == "__main__":
    main()
