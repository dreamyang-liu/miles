"""Tree credit (``RafoSettings.credit == "tree"``): V(node) is the share of resolved leaves in its subtree.

A reviewed trajectory splits at the point: its prefix takes V against its siblings' values, its remainder takes
R - V(turn); the reviewer turn is the head of its subtree, trained with PPO on V(turn) - R(remainder). The
fixture's roots have four messages, and an event's continuations restore the first three (the point is P = 3).
"""

from __future__ import annotations

import pytest

from miles.rollout.sprout.rafo import RafoSettings
from miles.rollout.sprout.rafo_role import KIND_REFLECT, KIND_REPAIR, KIND_ROOT, KIND_TURN
from tests.fast.rollout.sprout.conftest import forest_result
from tests.fast.rollout.sprout.test_rafo import M, REFLECT, REPAIR, ROOT, credited, role

TREE = RafoSettings(c=64.0, c_h=32.0, turn_credit="subtree", credit="tree")


def by_kind(samples) -> dict[str, list]:
    out: dict[str, list] = {}
    for sample in samples:
        out.setdefault(role(sample).kind, []).append(sample)
    return out


def by_slot(samples) -> list:
    return sorted(samples, key=lambda s: s.train_metadata["sprout_rollout"]["sample_slot_id"])


def decoded(mask_generator, sample, start: int, stop: int) -> str:
    response = sample.tokens[-sample.response_length :][start:stop]
    mask = sample.loss_mask[start:stop]
    return mask_generator.tokenizer.decode([t for t, keep in zip(response, mask, strict=True) if keep])


def test_the_credit_setting_is_checked():
    with pytest.raises(ValueError, match="credit"):
        RafoSettings(c=64.0, c_h=32.0, turn_credit="subtree", credit="forest")


def test_a_rescued_root_splits_at_the_point_and_its_siblings_pay_for_its_subtree(mask_generator):
    # roots 0, 0; event 1 in root 1 with continuations 1, 0: the leaves of root 1 are {B1 = 0, C = 1, D = 0}
    body, slots = forest_result([0.0, 0.0], [("root:1", [1.0, 0.0])])

    samples, metrics = credited(mask_generator, body, slots, settings=TREE)
    kinds = by_kind(samples)

    root0, root1 = by_slot(kinds[KIND_ROOT])
    r0, r1 = role(root0), role(root1)
    assert (r0.kind, r0.coef, r0.segments) == (KIND_ROOT, ROOT, ()) and r0.adv == pytest.approx(0 - 1 / 3)
    assert (r1.kind, r1.coef) == (KIND_ROOT, ROOT)
    assert r1.adv == pytest.approx(0 - 0.5), "the remainder B1: R_B - V(turn)"
    (start, prefix_adv), (cut, remainder_adv) = r1.segments
    assert (start, prefix_adv) == (0, pytest.approx(1 / 3 - 0)), "the prefix: V(root 1) - V(root 0)"
    assert remainder_adv == pytest.approx(-0.5) and 0 < cut < root1.response_length
    prefix_text = decoded(mask_generator, root1, 0, cut)
    remainder_text = decoded(mask_generator, root1, cut, root1.response_length)
    assert "done" not in prefix_text and "done" in remainder_text, (prefix_text, remainder_text)

    (turn,) = kinds[KIND_TURN]
    assert (role(turn).kind, role(turn).coef, role(turn).segments) == (KIND_TURN, REFLECT, ())
    assert role(turn).adv == pytest.approx(0.5) and turn.reward == 0.5
    assert KIND_REFLECT not in kinds, "the turn trains with PPO, not CE"
    assert sorted((r.adv, r.coef) for r in map(role, kinds[KIND_REPAIR])) == [(-1.0, REPAIR), (1.0, REPAIR)]
    assert (metrics[f"{M}events"], metrics[f"{M}events_trained"], metrics[f"{M}events_without_gain"]) == (1, 1, 0)


def test_the_turn_loss_setting_rides_on_the_turn_role_only(mask_generator):
    body, slots = forest_result([0.0, 0.0], [("root:1", [1.0, 0.0])])
    shaped = RafoSettings(c=64.0, c_h=32.0, turn_credit="subtree", credit="tree", turn_loss="shaped")

    samples, _ = credited(mask_generator, body, slots, settings=shaped)

    losses = {role(s).kind: role(s).loss for s in samples}
    assert losses == {KIND_ROOT: "ppo", KIND_REPAIR: "ppo", KIND_TURN: "shaped"}
    (turn,) = [s for s in samples if role(s).kind == KIND_TURN]
    assert turn.train_metadata["rafo"]["loss"] == "shaped", "the loss reads it from the sample's metadata"
    assert "loss" not in samples[0].train_metadata["rafo"] or role(samples[0]).kind == KIND_TURN
    with pytest.raises(ValueError, match="turn loss"):
        RafoSettings(c=64.0, c_h=32.0, turn_credit="subtree", credit="tree", turn_loss="softmax")


def test_a_turn_whose_continuations_all_failed_trains_nothing(mask_generator):
    body, slots = forest_result([0.0, 0.0], [("root:1", [0.0, 0.0])])

    samples, metrics = credited(mask_generator, body, slots, settings=TREE)

    assert [(role(s).kind, role(s).coef) for s in samples] == [(KIND_ROOT, 0.0)], "a placeholder keeps the task"
    assert (metrics[f"{M}events_without_gain"], metrics[f"{M}placeholders"]) == (1, 1)


def test_a_nested_event_credits_its_host_continuation_as_a_subtree(mask_generator):
    # event 1 in root 1: C, D both fail; event 2 in C: E = 1, F = 0.
    # leaves(C) = {C = 0, E = 1, F = 0} -> V(C) = 1/3; V(turn 1) over {C, D} = 1/4; leaves(root 1) -> 1/5.
    body, slots = forest_result([0.0, 0.0], [("root:1", [0.0, 0.0]), ("event:1:0", [1.0, 0.0])])

    samples, metrics = credited(mask_generator, body, slots, settings=TREE)
    kinds = by_kind(samples)

    root0, root1 = by_slot(kinds[KIND_ROOT])
    assert role(root0).adv == pytest.approx(-0.2) and role(root0).segments == ()
    assert role(root1).segments[0] == (0, pytest.approx(0.2)) and role(root1).adv == pytest.approx(0 - 0.25)

    repairs = {round(role(s).adv, 6): s for s in kinds[KIND_REPAIR]}
    assert sorted(repairs) == sorted(round(x, 6) for x in (-0.5, -1 / 3, 1.0, -1.0))
    host = repairs[round(-0.5, 6)]  # C: the remainder R_C - V(turn 2) = 0 - 1/2
    assert role(host).segments[0] == (0, pytest.approx(1 / 3 - 0)), "C's prefix: V(C) - V(D)"
    assert role(host).segments[1][0] == host.response_length, "the fixture's nested point lies past C's messages"

    assert sorted(round(role(t).adv, 6) for t in kinds[KIND_TURN]) == [0.25, 0.5]
    assert (metrics[f"{M}events"], metrics[f"{M}nested_events"], metrics[f"{M}events_trained"]) == (2, 1, 2)


def test_an_invalid_root_still_compares_by_its_reward_when_nothing_was_reviewed(mask_generator):
    body, slots = forest_result([0.0, -1.0])

    samples, _ = credited(mask_generator, body, slots, settings=TREE)

    assert sorted(round(role(s).adv, 6) for s in samples) == [-1.0, 1.0]
    assert all(role(s).segments == () for s in samples)
