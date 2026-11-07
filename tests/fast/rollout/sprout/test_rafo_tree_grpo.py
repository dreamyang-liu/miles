"""Tree-GRPO credit (``RafoSettings.credit == "tree_grpo"``): a leaf's advantage is (R - its tree's mean) +
(R - the task's mean); a segment's advantage is the sum of the leaf advantages below it.

The fixture's roots have four messages, and an event's continuations restore the first three (the point is P = 3).
"""

from __future__ import annotations

import pytest

from miles.rollout.sprout.rafo import RafoSettings
from miles.rollout.sprout.rafo_role import KIND_REFLECT, KIND_REPAIR, KIND_ROOT, KIND_TURN
from tests.fast.rollout.sprout.conftest import forest_result
from tests.fast.rollout.sprout.test_rafo import M, REFLECT, REPAIR, ROOT, credited, role
from tests.fast.rollout.sprout.test_rafo_tree import by_kind, by_slot, decoded

TREE_GRPO = RafoSettings(c=64.0, c_h=32.0, turn_credit="subtree", credit="tree_grpo")


def test_a_rescued_root_sums_its_leaves_and_the_task_sums_to_zero(mask_generator):
    # roots 0, 0; event 1 in root 1 with continuations 1, 0. Trees: {0} and {B = 0, C = 1, D = 0} (mean 1/3);
    # the task's leaves {0, 0, 1, 0} have mean 1/4.
    body, slots = forest_result([0.0, 0.0], [("root:1", [1.0, 0.0])])

    samples, metrics = credited(mask_generator, body, slots, settings=TREE_GRPO)
    kinds = by_kind(samples)

    root0, root1 = by_slot(kinds[KIND_ROOT])
    r0, r1 = role(root0), role(root1)
    assert (r0.kind, r0.coef, r0.segments) == (KIND_ROOT, ROOT, ()) and r0.adv == pytest.approx((0 - 0) + (0 - 1 / 4))
    leaf_b, leaf_c, leaf_d = (0 - 1 / 3) + (0 - 1 / 4), (1 - 1 / 3) + (1 - 1 / 4), (0 - 1 / 3) + (0 - 1 / 4)
    assert (r1.kind, r1.coef) == (KIND_ROOT, ROOT) and r1.adv == pytest.approx(leaf_b), "the remainder B is a leaf"
    (start, prefix_adv), (cut, remainder_adv) = r1.segments
    assert (start, prefix_adv) == (0, pytest.approx(leaf_b + leaf_c + leaf_d)), "the prefix sums root 1's leaves"
    assert prefix_adv + r0.adv == pytest.approx(0), "the task's advantages sum to zero"
    assert remainder_adv == pytest.approx(leaf_b) and 0 < cut < root1.response_length
    assert "done" not in decoded(mask_generator, root1, 0, cut) and "done" in decoded(mask_generator, root1, cut, root1.response_length)

    (turn,) = kinds[KIND_TURN]
    assert (role(turn).kind, role(turn).coef, role(turn).segments) == (KIND_TURN, REFLECT, ())
    assert role(turn).adv == pytest.approx(leaf_c + leaf_d) and turn.reward == pytest.approx(leaf_c + leaf_d)
    assert KIND_REFLECT not in kinds, "the turn trains with PPO, not CE"
    assert sorted(round(role(s).adv, 6) for s in kinds[KIND_REPAIR]) == sorted(round(x, 6) for x in (leaf_c, leaf_d))
    assert all(role(s).coef == REPAIR for s in kinds[KIND_REPAIR])
    assert (metrics[f"{M}events"], metrics[f"{M}events_trained"], metrics[f"{M}events_without_gain"]) == (1, 1, 0)


def test_a_task_whose_leaves_all_failed_alike_trains_nothing(mask_generator):
    body, slots = forest_result([0.0, 0.0], [("root:1", [0.0, 0.0])])

    samples, metrics = credited(mask_generator, body, slots, settings=TREE_GRPO)

    assert [(role(s).kind, role(s).coef) for s in samples] == [(KIND_ROOT, 0.0)], "a placeholder keeps the task"
    assert (metrics[f"{M}events_without_gain"], metrics[f"{M}placeholders"]) == (1, 1)


def test_a_nested_event_sums_its_host_continuation_as_a_subtree(mask_generator):
    # event 1 in root 1: C, D both fail; event 2 in C: E = 1, F = 0. Tree 1's leaves {B, C, D, E, F} have mean
    # 1/5; the task's six leaves have mean 1/6.
    body, slots = forest_result([0.0, 0.0], [("root:1", [0.0, 0.0]), ("event:1:0", [1.0, 0.0])])

    samples, metrics = credited(mask_generator, body, slots, settings=TREE_GRPO)
    kinds = by_kind(samples)
    fail, win = (0 - 1 / 5) + (0 - 1 / 6), (1 - 1 / 5) + (1 - 1 / 6)

    root0, root1 = by_slot(kinds[KIND_ROOT])
    assert role(root0).adv == pytest.approx(0 - 1 / 6) and role(root0).segments == ()
    assert role(root1).segments[0] == (0, pytest.approx(4 * fail + win)) and role(root1).adv == pytest.approx(fail)
    assert role(root0).adv + role(root1).segments[0][1] == pytest.approx(0)

    assert sorted(round(role(s).adv, 6) for s in kinds[KIND_REPAIR]) == sorted(round(x, 6) for x in (fail, fail, fail, win))
    (host,) = [s for s in kinds[KIND_REPAIR] if role(s).segments]  # C: prefix = C + E + F, remainder = C's own leaf
    assert role(host).segments[0] == (0, pytest.approx(2 * fail + win)) and role(host).adv == pytest.approx(fail)
    assert role(host).segments[1][0] == host.response_length, "the fixture's nested point lies past C's messages"

    assert sorted(round(role(t).adv, 6) for t in kinds[KIND_TURN]) == sorted(round(x, 6) for x in (3 * fail + win, fail + win))
    assert (metrics[f"{M}events"], metrics[f"{M}nested_events"], metrics[f"{M}events_trained"]) == (2, 1, 2)


def test_an_invalid_root_is_penalized_against_the_task_mean(mask_generator):
    body, slots = forest_result([0.0, -1.0])

    samples, _ = credited(mask_generator, body, slots, settings=TREE_GRPO)

    # single-leaf trees: the intra-tree term is 0; the task's mean is -1/2
    assert sorted(round(role(s).adv, 6) for s in samples) == [-0.5, 0.5]
    assert all(role(s).segments == () for s in samples)
