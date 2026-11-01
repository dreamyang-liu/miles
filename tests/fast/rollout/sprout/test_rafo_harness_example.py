"""Sprout's shared harness RAFO example through Miles: invalid and unfinished trajectories in RAFO's forest.

``fixtures/sprout_rafo_harness_result.example.json`` is a byte copy of Sprout's
``examples/rl_driver/rafo_harness_result.example.json``, the result its driver returns for a RAFO search over
harness rollouts in their training mode (``tests/search/test_rafo_miles_example.py`` there). Root 1 failed,
root 2 ended invalid (-1) and root 3 ran out of turns (0, ungraded), so Sprout reviewed root 1 alone.
Event 1's first continuation ended invalid and its second failed, so depth 2 reviewed the second: event 2,
which split, one continuation resolving and one running out of tokens (0, ungraded). Nothing in it is
annotated by hand: RAFO reads what Sprout's export wrote.
"""

import json

import pytest
from tests.fast.rollout.sprout.conftest import FIXTURES, group
from tests.fast.rollout.sprout.test_rafo_request import SearchDriver, rafo_args, run

from miles.rollout.sprout.forest import read_forest
from miles.rollout.sprout.protocol import RafoSearchSpec, RolloutResult
from miles.rollout.sprout.rafo import event_weights
from miles.rollout.sprout.rafo_role import KIND_REFLECT, KIND_REPAIR, KIND_ROOT, RafoRole

EXAMPLE = FIXTURES / "sprout_rafo_harness_result.example.json"
#: The example's search: three roots, at most two events of two continuations, two deep.
SPEC = RafoSearchSpec(roots=3, continuations=2, depth=2, events=2, review_seconds=600.0, wall_time_seconds=900.0)
SEARCH = {
    "n_samples_per_prompt": 3,
    "sprout_rollout_rafo_continuations": 2,
    "sprout_rollout_rafo_depth": 2,
    "sprout_rollout_rafo_events": 2,
}
#: Its fixed coefficients at C = 64 and C_h = 32: 1/(M C), 1/(S K C), and 1/(S C_h) for each unit of a turn's w.
ROOT, REPAIR, REFLECT = 1 / (3 * 64), 1 / (2 * 2 * 64), 1 / (2 * 32)
M = "rollout/sprout/rafo/"


def example() -> dict:
    return json.loads(EXAMPLE.read_text())


class HarnessExampleDriver(SearchDriver):
    """Sprout's RL driver answering a RAFO request with the harness example, in the slots Miles submitted."""

    def __init__(self) -> None:
        super().__init__({})

    def answer(self, submitted: dict) -> dict:
        body = example()
        for trajectory, slot in zip(body["trajectories"], submitted["sample_slots"], strict=True):
            trajectory["sample_slot_id"] = slot["sample_slot_id"]  # the example fills every slot, in order
        body.update(
            rollout_job_id=submitted["rollout_job_id"],
            prompt_group_id=submitted["prompt_group_id"],
            max_samples=submitted["max_samples"],
        )
        return body


def by_event(samples) -> dict[str, list[tuple[str, float, float]]]:
    """(kind, adv, coef) of each sample, by the Sprout group of the trajectory it was rendered from."""
    out: dict[str, list] = {}
    for sample in samples:
        role = RafoRole.of(sample.train_metadata)
        out.setdefault(sample.train_metadata["sprout_rollout"]["group"], []).append((role.kind, role.adv, role.coef))
    return out


def test_the_harness_example_is_a_result_miles_accepts_as_it_is():
    body = example()

    result = RolloutResult.model_validate(body)

    assert result.model_dump(mode="json") == body
    assert [(t.reward, t.stop_reason, t.metadata["zero_reason"]) for t in result.trajectories] == [
        (0.0, None, None),
        (-1.0, "format_invalid", None),
        (0.0, "max_turns_reached", None),
        (-1.0, "format_invalid", None),
        (0.0, None, None),
        (1.0, None, None),
        (0.0, "max_total_tokens", None),
    ], "every one has a verdict: the -1 and the unfinished 0 without a grade"
    counts = {"planned": 3, "graded": 3, "resolved": 0}
    assert all(t.metadata["search"]["outcome"] == "FF" for t in result.trajectories)
    assert all(t.metadata["search"]["root_counts"] == counts for t in result.trajectories)


def test_its_forest_counts_the_invalid_and_unfinished_trajectories_as_graded_failures():
    body = example()
    slots = {trajectory["sample_slot_id"]: None for trajectory in body["trajectories"]}

    forest = read_forest(RolloutResult.model_validate(body), slots, SPEC)

    assert forest.case == "FF"
    assert [(member.value, member.resolved) for member in forest.roots] == [(0.0, False), (-1.0, False), (0.0, False)]
    assert [(event.number, event.depth, event.parent) for event in forest.events] == [(1, 1, "job-0"), (2, 2, "job-6")]
    assert [[member.value for member in event.continuations] for event in forest.events] == [[-1.0, 0.0], [1.0, 0.0]]
    assert event_weights(forest.events, credit="subtree") == [0.25, 0.5], "never below 0: a -1 is unresolved"


def test_the_harness_example_trains_as_one_task_with_its_rafo_roles(mask_generator):
    """The roots and each event's continuations train with PPO on their rewards: [0, -1, 0], [-1, 0] and
    [1, 0]. Each turn trains with CE: w2 = 1/2, and w1 = (0 + max(0, w2))/2 = 1/4, since event 2 rescued event
    1's second continuation."""
    output = run(mask_generator, HarnessExampleDriver(), namespace=rafo_args(**SEARCH), groups=[group(3)])

    (samples,) = output.samples
    assert by_event(samples) == {
        "root": [(KIND_ROOT, 0.5, ROOT), (KIND_ROOT, -1.0, ROOT), (KIND_ROOT, 0.5, ROOT)],
        "event:1": [(KIND_REPAIR, -1.0, REPAIR), (KIND_REPAIR, 1.0, REPAIR), (KIND_REFLECT, 0.0, 0.25 * REFLECT)],
        "event:2": [(KIND_REPAIR, 1.0, REPAIR), (KIND_REPAIR, -1.0, REPAIR), (KIND_REFLECT, 0.0, 0.5 * REFLECT)],
    }
    assert all(RafoRole.of(sample.train_metadata).coef >= 0 for sample in samples)
    assert {sample.metadata["raw_reward"] for sample in samples} == {-1 / 3}, "the roots' mean of 0, -1 and 0"
    metrics = output.metrics
    assert {key.removeprefix(M): metrics[key] for key in (f"{M}cases/FF", f"{M}events", f"{M}nested_events")} == {
        "cases/FF": 1,
        "events": 2,
        "nested_events": 1,
    }
    assert (metrics[f"{M}events_trained"], metrics[f"{M}ppo_samples"], metrics[f"{M}ungradable"]) == (2, 7, 0)
    assert (metrics["rollout/sprout/invalid_mean"], metrics["rollout/sprout/resolved_mean"]) == (
        pytest.approx(2 / 7),
        pytest.approx(1 / 7),
    )
