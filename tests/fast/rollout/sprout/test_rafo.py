"""RAFO credit for a Sprout RAFO search: local RLOO on the trajectories, CE on the reviewer turns."""

import itertools

import pytest
from tests.fast.rollout.sprout.conftest import LOST, forest_result

from miles.rollout.sprout.protocol import RafoSearchSpec, RolloutResult
from miles.rollout.sprout.rafo import RafoSettings, assemble_rafo_group, coefficients, rloo
from miles.rollout.sprout.rafo_role import KIND_REFLECT, KIND_REPAIR, KIND_ROOT, RafoRole

SPEC = RafoSearchSpec(roots=2, continuations=2, depth=3, events=4, review_seconds=1.0, wall_time_seconds=1.0)
SETTINGS = RafoSettings(c=64.0, c_h=32.0, turn_credit="subtree")
M = "rollout/sprout/rafo/"
#: The fixed coefficients under SPEC and SETTINGS: 1/(M C), 1/(S K C), and 1/(S C_h) for each unit of w.
ROOT, REPAIR, REFLECT = 1 / (2 * 64), 1 / (4 * 2 * 64), 1 / (4 * 32)


def credited(mask_generator, body, slots, *, spec=SPEC, settings=SETTINGS):
    return assemble_rafo_group(
        RolloutResult.model_validate(body),
        slots,
        mask_generator,
        spec=spec,
        settings=settings,
        weight_version=7,
        indices=itertools.count(100),
    )


def role(sample) -> RafoRole:
    return RafoRole.of(sample.train_metadata)


def by_group(samples) -> dict[str, list[tuple[str, float, float]]]:
    """(kind, adv, coef) of each sample, by the Sprout group it was sampled in (CE samples: ``ce:<group>``)."""
    out: dict[str, list] = {}
    for sample in samples:
        group = sample.train_metadata["sprout_rollout"]["group"]
        r = role(sample)
        out.setdefault(f"ce:{group}" if r.kind == KIND_REFLECT else group, []).append((r.kind, r.adv, r.coef))
    return out


def trained_text(mask_generator, sample) -> str:
    response = sample.tokens[-sample.response_length :]
    return mask_generator.tokenizer.decode([t for t, keep in zip(response, sample.loss_mask, strict=True) if keep])


def test_the_coefficients_are_constants_of_the_search():
    assert coefficients(SPEC, SETTINGS) == (ROOT, REPAIR, REFLECT)


def test_rloo_compares_each_value_with_the_mean_of_the_others():
    assert rloo([1.0, 0.0, 0.0]) == [1.0, -0.5, -0.5]
    with pytest.raises(ValueError, match="at least 2"):
        rloo([1.0])


class TestPpo:
    def test_split_roots_train_with_local_rloo_and_their_task_is_not_reviewed(self, mask_generator):
        body, slots = forest_result([1.0, 0.0])

        samples, metrics = credited(mask_generator, body, slots)

        assert by_group(samples) == {"root": [(KIND_ROOT, 1.0, ROOT), (KIND_ROOT, -1.0, ROOT)]}
        assert metrics[f"{M}cases/TF"] == 1 and metrics[f"{M}events"] == 0

    def test_the_continuations_of_an_event_are_one_local_rloo_group(self, mask_generator):
        body, slots = forest_result([0.0, 0.0], [("root:0", [1.0, 0.0]), ("root:1", [0.0, 0.0])])

        samples, _ = credited(mask_generator, body, slots)

        groups = by_group(samples)
        assert groups["event:1"] == [(KIND_REPAIR, 1.0, REPAIR), (KIND_REPAIR, -1.0, REPAIR)]
        assert "root" not in groups and "event:2" not in groups, "a group whose members agree trains nothing"

    def test_a_missing_continuation_counts_for_nothing(self, mask_generator):
        body, slots = forest_result([0.0, 0.0], [("root:0", [1.0, None, LOST, 0.0])])
        four = SPEC.model_copy(update={"continuations": 4})

        samples, metrics = credited(mask_generator, body, slots, spec=four)

        assert [adv for _, adv, _ in by_group(samples)["event:1"]] == [1.0, -1.0]
        assert [coef for _, _, coef in by_group(samples)["ce:event:1"]] == [0.5 * REFLECT]
        assert metrics[f"{M}ungradable"] == 1


class TestReviewerTurns:
    def test_a_turn_weighs_the_mean_outcome_of_its_continuations(self, mask_generator):
        body, slots = forest_result([0.0, 0.0], [("root:0", [1.0, 0.0]), ("root:1", [1.0, 1.0])])

        samples, metrics = credited(mask_generator, body, slots)

        ce = {key: value for key, value in by_group(samples).items() if key.startswith("ce:")}
        assert ce == {
            "ce:event:1": [(KIND_REFLECT, 0.0, 0.5 * REFLECT)],
            "ce:event:2": [(KIND_REFLECT, 0.0, 1.0 * REFLECT)],
        }
        assert metrics[f"{M}events_trained"] == 2 and metrics[f"{M}events_without_gain"] == 0

    def test_a_turn_whose_continuations_all_failed_trains_nothing(self, mask_generator):
        body, slots = forest_result([0.0, 0.0], [("root:0", [0.0, 0.0])])

        samples, metrics = credited(mask_generator, body, slots)

        assert [role(s).kind for s in samples] == [KIND_ROOT], "only the placeholder that counts the task"
        assert metrics[f"{M}events_without_gain"] == 1 and metrics[f"{M}placeholders"] == 1

    def test_the_ce_sample_trains_the_turn_alone_after_its_history(self, mask_generator):
        body, slots = forest_result([0.0, 0.0], [("root:0", [1.0, 0.0])])

        samples, _ = credited(mask_generator, body, slots)

        (ce,) = [s for s in samples if role(s).kind == KIND_REFLECT]
        assert trained_text(mask_generator, ce).startswith("<think> parser broken </think>")
        assert ce.index == 100 and ce.reward == 0.5, "a fresh index; its reward shows its w"


class TestNestedEvents:
    """Both roots fail. Event 1 in root 1 fails twice. Event 2, inside event 1's first continuation,
    resolves once of twice."""

    EVENTS = [("root:0", [0.0, 0.0]), ("event:1:0", [1.0, 0.0])]

    def test_subtree_credit_counts_a_rescued_continuation_with_the_weight_of_the_event_that_rescued_it(
        self, mask_generator
    ):
        body, slots = forest_result([0.0, 0.0], self.EVENTS)

        samples, metrics = credited(mask_generator, body, slots)

        ce = {key: [coef for _, _, coef in value] for key, value in by_group(samples).items() if key.startswith("ce:")}
        assert ce == {"ce:event:1": [0.25 * REFLECT], "ce:event:2": [0.5 * REFLECT]}, "w1 = (max(0, 0.5) + 0)/2"
        assert metrics[f"{M}nested_events"] == 1

    def test_own_credit_counts_the_events_own_continuations_only(self, mask_generator):
        body, slots = forest_result([0.0, 0.0], self.EVENTS)
        settings = RafoSettings(c=64.0, c_h=32.0, turn_credit="own")

        samples, _ = credited(mask_generator, body, slots, settings=settings)

        assert [key for key in by_group(samples) if key.startswith("ce:")] == ["ce:event:2"]

    def test_each_turn_of_a_nested_history_is_trained_once_in_its_own_events_ce_sample(self, mask_generator):
        body, slots = forest_result([0.0, 0.0], self.EVENTS)

        samples, _ = credited(mask_generator, body, slots)

        texts = sorted(trained_text(mask_generator, s) for s in samples if role(s).kind == KIND_REFLECT)
        assert [text.split("</think>")[0] for text in texts] == ["<think> inspect tests ", "<think> parser broken "]


class TestTask:
    def test_every_sample_of_a_task_shares_one_rollout_id_and_keeps_its_sprout_lineage(self, mask_generator):
        body, slots = forest_result([0.0, 0.0], TestNestedEvents.EVENTS)

        samples, _ = credited(mask_generator, body, slots)

        assert {s.rollout_id for s in samples} == {3} and {s.group_index for s in samples} == {3}
        assert all(s.metadata["raw_reward"] == 0.0 for s in samples), "the share of the task's roots that resolved"
        assert all(s.train_metadata["sprout_rollout"]["rollout_job_id"] == "job" for s in samples)

    def test_a_task_without_signal_keeps_one_zero_weight_root_so_that_it_counts_in_the_step(self, mask_generator):
        body, slots = forest_result([1.0, 1.0])

        samples, metrics = credited(mask_generator, body, slots)

        assert [(role(s).kind, role(s).coef) for s in samples] == [(KIND_ROOT, 0.0)]
        assert metrics[f"{M}placeholders"] == 1 and metrics[f"{M}cases/TT"] == 1

    def test_a_group_whose_every_root_was_lost_gives_no_sample(self, mask_generator):
        body, slots = forest_result([LOST, LOST])

        samples, metrics = credited(mask_generator, body, slots)

        assert samples == [] and metrics[f"{M}cases/ungraded"] == 1

    def test_the_metrics_count_the_events_and_the_turns_that_nothing_checked(self, mask_generator):
        body, slots = forest_result([0.0, 0.0], TestNestedEvents.EVENTS)
        for trajectory in body["trajectories"][2:4]:
            trajectory["metadata"]["search"]["leak_terms"] = None

        _, metrics = credited(mask_generator, body, slots)

        assert (metrics[f"{M}events"], metrics[f"{M}events_unchecked"], metrics[f"{M}ppo_samples"]) == (2, 1, 2)


@pytest.mark.parametrize("settings", [{"c": 0.0}, {"c_h": -1.0}, {"turn_credit": "followed"}])
def test_settings_out_of_range_are_rejected(settings: dict) -> None:
    with pytest.raises(ValueError):
        RafoSettings(**{"c": 64.0, "c_h": 32.0, "turn_credit": "subtree", **settings})
