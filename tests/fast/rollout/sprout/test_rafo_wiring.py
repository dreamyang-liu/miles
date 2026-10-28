"""``--sprout-rollout-credit rafo`` through SproutRolloutFn: the shared example to the rows the RAFO loss reads."""

import argparse
import sys
from argparse import Namespace
from collections import Counter
from unittest.mock import patch

import pytest
from tests.fast.rollout.sprout.conftest import args, group
from tests.fast.rollout.sprout.test_gsml_wiring import ExampleDriver
from tests.fast.rollout.sprout.test_search_request import SEARCH, construct, run
from tests.fast.train_parallel_config_utils import make_train_parallel_config

from miles.ray.rollout.rollout_data_conversion import postprocess_rollout_data
from miles.ray.rollout.train_data_conversion import convert_samples_to_train_data
from miles.rollout.sprout.rafo_role import KIND_REFLECT, KIND_REPAIR, RafoRole
from miles.rollout.sprout.rollout_fn import RAFO_LOSS

#: The search of the shared example, without GSML's trainer settings.
SHAPE = {name: value for name, value in SEARCH.items() if name.startswith("sprout_rollout_search_")}
#: What a RAFO search needs of the trainer.
RAFO = {
    "sprout_rollout_credit": "rafo",
    "loss_type": "custom_loss",
    "custom_loss_function_path": RAFO_LOSS,
    "train_backend": "megatron",
    "rewards_normalization": False,
    "custom_reward_post_process_path": None,
    "use_dynamic_global_batch_size": False,
}


def rafo_args(**updates) -> Namespace:
    return args(**{"n_samples_per_prompt": 2, **SHAPE, **RAFO, **updates})


def roles(samples) -> list[RafoRole]:
    return [RafoRole.of(sample.train_metadata) for sample in samples]


class TestRafoSearchStep:
    def test_the_shared_example_trains_as_one_task_with_rafo_roles(self, mask_generator) -> None:
        """P1: a student resolved, so its turns gain nothing over the students. P2: candidate 1's continuations
        both resolved (w = 1), candidate 2's turn names a hidden test (no CE)."""
        output = run(mask_generator, ExampleDriver(), namespace=rafo_args())

        (samples,) = output.samples
        kinds = Counter(role.kind for role in roles(samples))
        assert kinds == {KIND_REPAIR: 6, KIND_REFLECT: 1}
        assert {sample.rollout_id for sample in samples} == {group(2)[0].group_index}
        assert output.metrics["rollout/sprout/rafo/turns"] == 4 and output.metrics["rollout/sprout/rafo/cases/FF"] == 1
        assert "rollout/sprout/gsml/cases/FF" not in output.metrics

    def test_the_task_reaches_the_train_data_as_one_rollout_whose_rows_carry_their_roles(self, mask_generator) -> None:
        namespace = rafo_args()
        output = run(mask_generator, ExampleDriver(), namespace=namespace)
        trainer = Namespace(
            **vars(namespace),
            disable_rollout_trim_samples=False,
            global_batch_size=1,
            enable_sample_ownership_checker=False,
            advantage_estimator="grpo",
            reward_key=None,
            use_output_store=False,
            rollout_top_logprobs_num=0,
            use_rollout_routing_replay=False,
            use_rollout_indexer_replay=False,
        )

        flat, metadata = postprocess_rollout_data(
            trainer, output.samples, train_parallel_config=make_train_parallel_config(dp_size=1)
        )
        data = convert_samples_to_train_data(trainer, flat, metadata, None, None)

        assert len(set(data["rollout_ids"])) == 1 and len(flat) == 7, "a compact rollout: nothing is trimmed"
        assert [RafoRole.of(row) for row in data["metadata"]] == roles(flat)
        assert data["raw_reward"] == [0.0] * 7, "the episode reward is the share of the task's roots that resolved"


class TestRafoConfiguration:
    @pytest.mark.parametrize(
        ("update", "need"),
        [
            ({"loss_type": "policy_loss"}, "--loss-type custom_loss"),
            ({"train_backend": "fsdp"}, "--train-backend megatron"),
            ({"rewards_normalization": True}, "--disable-rewards-normalization"),
            ({"use_dynamic_global_batch_size": True}, "no --use-dynamic-global-batch-size"),
            ({"custom_reward_post_process_path": "x.y"}, "no --custom-reward-post-process-path"),
            ({"n_samples_per_prompt": 1}, "--n-samples-per-prompt of at least 2"),
        ],
    )
    def test_a_rafo_search_refuses_a_trainer_it_cannot_train_with(self, update: dict, need: str) -> None:
        with pytest.raises(ValueError, match=f"rafo needs .*{need}"):
            construct(rafo_args(**update))

    def test_rafo_credits_search_groups_only(self) -> None:
        with pytest.raises(ValueError, match="rafo trains search groups"):
            construct(rafo_args(sprout_rollout_search_points=0))

    def test_rafo_settings_out_of_range_are_refused_at_construction(self) -> None:
        with pytest.raises(ValueError, match="normalizers must be positive"):
            construct(rafo_args(sprout_rollout_rafo_c_h=0.0))

    def test_the_rafo_flags_parse_once_sprout_rollout_fn_is_the_rollout_function(self) -> None:
        from miles.utils.arguments import get_miles_extra_args_provider

        argv = (
            "--rollout-batch-size",
            "1",
            "--rollout-function-path",
            "miles.rollout.sprout.rollout_fn.SproutRolloutFn",
            "--sprout-rollout-base-url",
            "http://driver:11001",
            "--sprout-rollout-credit",
            "rafo",
            "--sprout-rollout-rafo-c",
            "16",
            "--sprout-rollout-rafo-turn-credit",
            "own",
        )
        with patch.object(sys, "argv", ["test", *argv]):
            parser = argparse.ArgumentParser()
            get_miles_extra_args_provider()(parser)
            parsed, unknown = parser.parse_known_args(argv)

        assert unknown == []
        assert (parsed.sprout_rollout_credit, parsed.sprout_rollout_rafo_c) == ("rafo", 16.0)
        assert (parsed.sprout_rollout_rafo_turn_credit, parsed.sprout_rollout_rafo_baseline) == ("own", "students")
