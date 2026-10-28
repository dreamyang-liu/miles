"""RAFO credit for Sprout search groups: local RLOO on the trajectories, CE on the reviewer turns."""

import itertools
from copy import deepcopy

import pytest
from tests.fast.rollout.sprout.conftest import INSERTED_TURN, PREFIX, ff_example, members, search_result

from miles.rollout.sprout.protocol import RolloutResult, SearchSpec
from miles.rollout.sprout.rafo import RafoSettings, assemble_rafo_group, denominators
from miles.rollout.sprout.rafo_role import KIND_REFLECT, KIND_REPAIR, KIND_ROOT, RafoRole

SPEC = SearchSpec(
    roots=2,
    points=1,
    candidates=2,
    student_continuations=2,
    candidate_continuations=2,
    tf_student_continuations=2,
    repair_rounds=1,
    review_seconds=1.0,
    wall_time_seconds=1.0,
)
CHAIN_SPEC = SPEC.model_copy(update={"candidates": 1, "candidate_continuations": 1, "repair_rounds": 2})
SETTINGS = RafoSettings(c=64.0, c_h=32.0, turn_credit="subtree", baseline="students")
M = "rollout/sprout/rafo/"


def credited(mask_generator, body, slots, *, spec=SPEC, settings=SETTINGS):
    samples, metrics = assemble_rafo_group(
        RolloutResult.model_validate(body),
        slots,
        mask_generator,
        spec=spec,
        settings=settings,
        weight_version=7,
        indices=itertools.count(100),
    )
    return samples, metrics


def role(sample) -> RafoRole:
    return RafoRole.of(sample.train_metadata)


def by_group(samples) -> dict[str, list[tuple[str, float, float]]]:
    """(kind, adv, coef) of each sample, by the Sprout condition it was sampled under (CE samples: ``ce:<group>``)."""
    out: dict[str, list] = {}
    for sample in samples:
        lineage = sample.train_metadata["sprout_rollout"]
        r = role(sample)
        key = f"ce:{lineage['group']}" if r.kind == KIND_REFLECT else lineage["group"]
        out.setdefault(key, []).append((r.kind, r.adv, r.coef))
    return out


def trained_text(mask_generator, sample) -> str:
    response = sample.tokens[-sample.response_length :]
    return mask_generator.tokenizer.decode([t for t, keep in zip(response, sample.loss_mask, strict=True) if keep])


def chained(body, parent_slot: int, child_slot: int, *, turn: str = "inspect tests") -> None:
    """Make the repair in ``child_slot`` round 2 of a chain inside the round-1 repair in ``parent_slot``: its
    history holds the round-1 turn, the parent's own turn after it, then its own inserted turn."""
    (parent,) = [t for t in body["trajectories"] if t["sample_slot_id"] == f"job:slot:{parent_slot}"]
    (child,) = [t for t in body["trajectories"] if t["sample_slot_id"] == f"job:slot:{child_slot}"]
    second = deepcopy(INSERTED_TURN)
    second["reasoning_content"] = turn
    second["tool_calls"][0]["id"] = "inserted-2"
    actor = {
        "role": "assistant",
        "content": "inspect",
        "tool_calls": [{"id": "actor-1", "type": "function", "function": {"name": "shell", "arguments": "{}"}}],
    }
    child["messages"] = [
        *PREFIX,
        INSERTED_TURN,
        {"role": "tool", "tool_call_id": "inserted-1", "content": "tests"},
        actor,
        {"role": "tool", "tool_call_id": "actor-1", "content": "/workspace"},
        second,
        {"role": "tool", "tool_call_id": "inserted-2", "content": "tests"},
        child["messages"][-1],
    ]
    child["metadata"]["provenance"] = {"prefix_messages": 7, "inserted_messages": [3, 7]}
    search = child["metadata"]["search"]
    search.update(round=2, chain_parent_job_id=parent["branch_id"], chain_point_id="rp-2", chain_step=4)


class TestDenominators:
    def test_the_coefficients_count_the_most_groups_and_turns_one_task_can_make(self) -> None:
        # each failed root: points x (one student group + candidates x rounds repair groups)
        assert denominators(SPEC) == (2, 6, 4)
        assert denominators(CHAIN_SPEC) == (2, 6, 4)
        no_repairs = SPEC.model_copy(update={"candidates": 0, "candidate_continuations": 0})
        assert denominators(no_repairs) == (2, 2, 1)


class TestPpo:
    def test_split_roots_and_their_students_train_with_local_rloo(self, mask_generator) -> None:
        body, slots = search_result([1.0, 0.0], {"p1": {"students": [1.0, 0.0]}})

        samples, _ = credited(mask_generator, body, slots)

        groups = by_group(samples)
        g, k_s = 6, SPEC.tf_student_continuations
        assert groups["root"] == [(KIND_ROOT, 1.0, 1 / (2 * 64)), (KIND_ROOT, -1.0, 1 / (2 * 64))]
        assert groups["student:p1"] == [
            (KIND_REPAIR, 1.0, 1 / (g * k_s * 64)),
            (KIND_REPAIR, -1.0, 1 / (g * k_s * 64)),
        ]

    def test_a_group_whose_members_agree_trains_nothing(self, mask_generator) -> None:
        body, slots = ff_example()

        samples, _ = credited(mask_generator, body, slots)

        groups = by_group(samples)
        assert "root" not in groups and "student:p2" not in groups and "repair:p2:1" not in groups
        assert [adv for _, adv, _ in groups["repair:p1:1"]] == [1.0, -1.0]

    def test_a_missing_grade_counts_for_nothing(self, mask_generator) -> None:
        body, slots = search_result([1.0, 0.0], {"p1": {"students": [1.0, None, 0.0]}})

        samples, metrics = credited(mask_generator, body, slots)

        assert [adv for _, adv, _ in by_group(samples)["student:p1"]] == [1.0, -1.0]
        assert metrics[f"{M}ungradable"] == 1


class TestReviewerTurns:
    def test_a_turn_is_weighted_by_its_continuations_over_the_students_at_its_point(self, mask_generator) -> None:
        """P1: students [1, 0] -> baseline 0.5. P2: students [0, 0]; candidate 2's turn names a hidden test."""
        body, slots = ff_example()

        samples, metrics = credited(mask_generator, body, slots)

        ce = {key: value for key, value in by_group(samples).items() if key.startswith("ce:")}
        assert ce == {"ce:repair:p2:1": [(KIND_REFLECT, 0.0, 1.0 / (4 * 32))]}
        assert metrics[f"{M}turns"] == 4 and metrics[f"{M}turns_leaked"] == 1 and metrics[f"{M}turns_trained"] == 1

    def test_without_the_student_baseline_the_weight_is_the_continuations_mean(self, mask_generator) -> None:
        body, slots = ff_example()
        settings = RafoSettings(c=64.0, c_h=32.0, turn_credit="subtree", baseline="none")

        samples, _ = credited(mask_generator, body, slots, settings=settings)

        ce = {key: [coef for _, _, coef in value] for key, value in by_group(samples).items() if key.startswith("ce:")}
        assert ce == {"ce:repair:p1:1": [0.5 / (4 * 32)], "ce:repair:p2:1": [1.0 / (4 * 32)]}

    def test_the_ce_sample_trains_the_turn_alone_after_its_history(self, mask_generator) -> None:
        body, slots = ff_example()

        samples, _ = credited(mask_generator, body, slots)

        (ce,) = [s for s in samples if role(s).kind == KIND_REFLECT]
        assert trained_text(mask_generator, ce).startswith("<think> parser broken </think>")
        assert ce.index >= 100 and ce.reward == 1.0

    def test_a_continuation_that_touched_the_tests_counts_as_unresolved(self, mask_generator) -> None:
        body, slots = ff_example()
        for trajectory in members(body, "repair:p2:1")[:1]:
            trajectory["metadata"]["touched_test_paths"] = ["tests/test_parser.py"]

        samples, metrics = credited(mask_generator, body, slots)

        groups = by_group(samples)
        assert [adv for _, adv, _ in groups["repair:p2:1"]] == [-1.0, 1.0]
        assert [coef for _, _, coef in groups["ce:repair:p2:1"]] == [0.5 / (4 * 32)]
        assert metrics[f"{M}continuations_touching_tests"] == 1


class TestRepairChains:
    def _chain(self):
        """Both roots fail. P1: one student fails; round 1's repair fails; round 2, inside it, resolves."""
        body, slots = search_result([0.0, 0.0], {"p1": {"students": [0.0], "candidates": [[0.0], [1.0]]}})
        chained(body, parent_slot=14, child_slot=15)
        return body, slots

    def test_subtree_credit_gives_the_first_turn_the_weight_of_the_round_that_rescued_its_branch(
        self, mask_generator
    ) -> None:
        body, slots = self._chain()

        samples, metrics = credited(mask_generator, body, slots, spec=CHAIN_SPEC)

        ce = {key: [coef for _, _, coef in value] for key, value in by_group(samples).items() if key.startswith("ce:")}
        assert ce == {"ce:repair:p1:1": [1.0 / (4 * 32)], "ce:repair:p1:2": [1.0 / (4 * 32)]}
        assert metrics[f"{M}chain_turns"] == 1

    def test_own_credit_gives_the_first_turn_nothing(self, mask_generator) -> None:
        body, slots = self._chain()
        settings = RafoSettings(c=64.0, c_h=32.0, turn_credit="own", baseline="students")

        samples, _ = credited(mask_generator, body, slots, spec=CHAIN_SPEC, settings=settings)

        assert [key for key in by_group(samples) if key.startswith("ce:")] == ["ce:repair:p1:2"]

    def test_each_turn_of_the_chain_is_trained_once_in_its_own_ce_sample(self, mask_generator) -> None:
        body, slots = self._chain()

        samples, _ = credited(mask_generator, body, slots, spec=CHAIN_SPEC)

        texts = sorted(trained_text(mask_generator, s) for s in samples if role(s).kind == KIND_REFLECT)
        assert [text.split("</think>")[0] for text in texts] == ["<think> inspect tests ", "<think> parser broken "]


class TestTaskIdentity:
    def test_every_sample_of_a_task_shares_one_rollout_id_and_keeps_its_sprout_lineage(self, mask_generator) -> None:
        body, slots = ff_example()

        samples, _ = credited(mask_generator, body, slots)

        assert {sample.rollout_id for sample in samples} == {3} and {s.group_index for s in samples} == {3}
        assert all(s.metadata["raw_reward"] == 0.0 for s in samples)
        assert all(s.train_metadata["sprout_rollout"]["rollout_job_id"] == "job" for s in samples)

    def test_a_task_without_signal_keeps_one_zero_weight_root_so_that_it_counts_in_the_step(
        self, mask_generator
    ) -> None:
        body, slots = search_result([1.0, 1.0])

        samples, metrics = credited(mask_generator, body, slots)

        assert [(role(s).kind, role(s).coef) for s in samples] == [(KIND_ROOT, 0.0)]
        assert metrics[f"{M}placeholders"] == 1 and metrics[f"{M}cases/TT"] == 1

    def test_a_group_whose_every_root_was_lost_gives_no_sample(self, mask_generator) -> None:
        body, slots = search_result(["lost", "lost"])

        samples, metrics = credited(mask_generator, body, slots)

        assert samples == [] and metrics[f"{M}cases/ungraded"] == 1


@pytest.mark.parametrize(
    "settings",
    [
        {"c": 0.0},
        {"c_h": -1.0},
        {"turn_credit": "followed"},
        {"baseline": "roots"},
    ],
)
def test_settings_out_of_range_are_rejected(settings: dict) -> None:
    with pytest.raises(ValueError):
        RafoSettings(**{"c": 64.0, "c_h": 32.0, "turn_credit": "subtree", "baseline": "students", **settings})
