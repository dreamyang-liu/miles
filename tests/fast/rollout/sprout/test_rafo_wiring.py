"""Sprout's shared RAFO example through Miles: the credit, the train data, and what wires them in.

``fixtures/sprout_rafo_search_result.example.json`` is a byte copy of Sprout's
``examples/rl_driver/rafo_search_result.example.json``, the result its driver and adapter return for a
RAFO search (``tests/search/test_rafo_miles_example.py`` there). Both roots failed. Event 1 in root 1
failed twice, and event 2 in root 2 split. Event 3, inside event 1's first continuation, split. Nothing
in it is annotated by hand: RAFO reads what Sprout's export wrote.
"""

import argparse
import json
import sys
import uuid
from argparse import Namespace
from unittest.mock import patch

import pytest
from tests.fast.rollout.sprout.conftest import FIXTURES, SPROUT_EXAMPLES, group, sprout_examples
from tests.fast.rollout.sprout.test_rafo_request import SearchDriver, construct, rafo_args, run
from tests.fast.train_parallel_config_utils import make_train_parallel_config

from miles.ray.rollout.rollout_data_conversion import postprocess_rollout_data
from miles.ray.rollout.train_data_conversion import convert_samples_to_train_data
from miles.rollout.sprout.protocol import RolloutResult
from miles.rollout.sprout.rafo_role import KIND_REFLECT, KIND_REPAIR, RafoRole
from miles.rollout.sprout.rollout_fn import RAFO_LOSS

EXAMPLE = FIXTURES / "sprout_rafo_search_result.example.json"
#: The example's search: two roots, at most three events of two continuations, two deep.
SEARCH = {"sprout_rollout_rafo_continuations": 2, "sprout_rollout_rafo_depth": 2, "sprout_rollout_rafo_events": 3}
#: Its fixed coefficients at C = 64 and C_h = 32: 1/(S K C), and 1/(S C_h) for each unit of a turn's w.
REPAIR, REFLECT = 1 / (3 * 2 * 64), 1 / (3 * 32)


def example() -> dict:
    return json.loads(EXAMPLE.read_text())


def example_args(**updates) -> Namespace:
    return rafo_args(**{**SEARCH, **updates})


class ExampleDriver(SearchDriver):
    """Sprout's RL driver answering a RAFO request with the shared example, in the slots Miles submitted."""

    def __init__(self) -> None:
        super().__init__({})

    def answer(self, submitted: dict) -> dict:
        body = example()
        assert body["failed_samples"] == [] and len(body["trajectories"]) == len(submitted["sample_slots"])
        for trajectory, slot in zip(body["trajectories"], submitted["sample_slots"], strict=True):
            trajectory["sample_slot_id"] = slot["sample_slot_id"]  # the example fills every slot, in order
        body.update(
            rollout_job_id=submitted["rollout_job_id"],
            prompt_group_id=submitted["prompt_group_id"],
            max_samples=submitted["max_samples"],
        )
        return body


def roles(samples) -> list[RafoRole]:
    return [RafoRole.of(sample.train_metadata) for sample in samples]


def by_event(samples) -> dict[str, list[tuple[str, float, float]]]:
    """(kind, adv, coef) of each sample, by the Sprout group of the trajectory it was rendered from."""
    out: dict[str, list] = {}
    for sample, role in zip(samples, roles(samples), strict=True):
        out.setdefault(sample.train_metadata["sprout_rollout"]["group"], []).append((role.kind, role.adv, role.coef))
    return out


def test_the_example_is_a_result_miles_accepts_as_it_is():
    body = example()

    result = RolloutResult.model_validate(body)

    assert result.model_dump(mode="json") == body
    assert result.status == "completed" and result.actual_samples == result.max_samples == 8
    counts = {"planned": 2, "graded": 2, "resolved": 0}
    assert all(t.metadata["search"]["outcome"] == "FF" for t in result.trajectories)
    assert all(t.metadata["search"]["root_counts"] == counts for t in result.trajectories)


def test_the_request_miles_builds_has_the_examples_slots(mask_generator):
    job_hex = example()["rollout_job_id"].rsplit("-", 1)[1]
    with patch("miles.rollout.sprout.rollout_fn.uuid.uuid4", return_value=uuid.UUID(job_hex)):
        request, _ = construct(example_args())._build_request(group=group(2), rollout_id=1)

    assert (request.rollout_job_id, request.max_samples) == (example()["rollout_job_id"], 8)
    assert [slot.sample_slot_id for slot in request.sample_slots] == [
        t["sample_slot_id"] for t in example()["trajectories"]
    ]


def test_the_shared_example_trains_as_one_task_with_its_rafo_roles(mask_generator):
    """Events 2 and 3 split: their continuations train with PPO. Every turn trains with CE: w2 = w3 = 1/2,
    and w1 = (max(0, w3) + 0)/2 = 1/4, since event 3 rescued event 1's first continuation."""
    output = run(mask_generator, ExampleDriver(), namespace=example_args())

    (samples,) = output.samples
    assert by_event(samples) == {
        "event:1": [(KIND_REFLECT, 0.0, 0.25 * REFLECT)],
        "event:2": [(KIND_REPAIR, 1.0, REPAIR), (KIND_REPAIR, -1.0, REPAIR), (KIND_REFLECT, 0.0, 0.5 * REFLECT)],
        "event:3": [(KIND_REPAIR, 1.0, REPAIR), (KIND_REPAIR, -1.0, REPAIR), (KIND_REFLECT, 0.0, 0.5 * REFLECT)],
    }
    assert {sample.rollout_id for sample in samples} == {group(2)[0].group_index}
    metrics = {key.removeprefix("rollout/sprout/rafo/"): v for key, v in output.metrics.items() if "/rafo/" in key}
    assert {key: metrics[key] for key in ("cases/FF", "events", "nested_events", "events_trained")} == {
        "cases/FF": 1,
        "events": 3,
        "nested_events": 1,
        "events_trained": 3,
    }


def test_the_task_reaches_the_train_data_as_one_rollout_whose_rows_carry_their_roles(mask_generator):
    namespace = example_args()
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
    )

    flat, metadata = postprocess_rollout_data(
        trainer, output.samples, train_parallel_config=make_train_parallel_config(dp_size=1)
    )
    data = convert_samples_to_train_data(trainer, flat, metadata, None, None)

    assert len(set(data["rollout_ids"])) == 1 and len(flat) == 7, "a compact rollout: nothing is trimmed"
    assert [RafoRole.of(row) for row in data["metadata"]] == roles(flat)
    assert data["raw_reward"] == [0.0] * 7, "the episode reward is the share of the task's roots that resolved"


@pytest.mark.parametrize(
    ("update", "need"),
    [
        ({"loss_type": "policy_loss"}, "--loss-type custom_loss"),
        ({"custom_loss_function_path": "x.y"}, "--loss-type custom_loss"),
        ({"train_backend": "fsdp"}, "--train-backend megatron"),
        ({"rewards_normalization": True}, "--disable-rewards-normalization"),
        ({"use_dynamic_global_batch_size": True}, "no --use-dynamic-global-batch-size"),
        ({"custom_reward_post_process_path": "x.y"}, "no --custom-reward-post-process-path"),
        ({"n_samples_per_prompt": 1}, "--n-samples-per-prompt of at least 2"),
        ({"sprout_rollout_rafo_continuations": 1}, "--sprout-rollout-rafo-continuations of at least 2"),
    ],
)
def test_a_rafo_search_refuses_a_trainer_it_cannot_train_with(update: dict, need: str) -> None:
    with pytest.raises(ValueError, match=f"--sprout-rollout-rafo needs .*{need}"):
        construct(example_args(**update))


@pytest.mark.parametrize(
    ("update", "message"),
    [
        ({"sprout_rollout_rafo_events": 0}, "--sprout-rollout-rafo-events must be a positive integer"),
        ({"sprout_rollout_rafo_depth": 1.5}, "--sprout-rollout-rafo-depth must be a positive integer"),
        ({"sprout_rollout_rafo_review_seconds": 0}, "--sprout-rollout-rafo-review-seconds must be finite"),
        ({"sprout_rollout_rafo_c_h": 0.0}, "normalizers must be positive"),
        ({"sprout_rollout_rafo_turn_credit": "followed"}, "not RAFO's"),
    ],
)
def test_a_rafo_search_out_of_range_is_refused_at_construction(update: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        construct(example_args(**update))


def test_the_rafo_flags_parse_once_sprout_rollout_fn_is_the_rollout_function() -> None:
    from miles.utils.arguments import get_miles_extra_args_provider

    argv = (
        "--rollout-batch-size",
        "1",
        "--rollout-function-path",
        "miles.rollout.sprout.rollout_fn.SproutRolloutFn",
        "--sprout-rollout-base-url",
        "http://driver:11001",
        "--sprout-rollout-rafo",
        "--sprout-rollout-rafo-events",
        "6",
        "--sprout-rollout-rafo-turn-credit",
        "own",
    )
    with patch.object(sys, "argv", ["test", *argv]):
        parser = argparse.ArgumentParser()
        get_miles_extra_args_provider()(parser)
        parsed, unknown = parser.parse_known_args(argv)

    assert unknown == []
    assert (parsed.sprout_rollout_rafo, parsed.sprout_rollout_rafo_events) == (True, 6)
    assert (parsed.sprout_rollout_rafo_turn_credit, parsed.sprout_rollout_rafo_c) == ("own", 64.0)
    assert RAFO_LOSS.endswith("rafo_loss.rafo_loss_function")


@pytest.mark.parametrize("fixture", sorted(SPROUT_EXAMPLES))
def test_the_fixtures_are_byte_copies_of_the_examples_sprout_ships(fixture):
    directory = sprout_examples()
    if directory is None:
        pytest.skip("no Sprout checkout beside this one: set SPROUT_REPO to compare the fixtures with its examples")
    shipped = directory / SPROUT_EXAMPLES[fixture]
    assert (FIXTURES / fixture).read_bytes() == shipped.read_bytes(), f"copy {shipped} over fixtures/{fixture}"
