"""The RAFO forest Sprout returns for a task, as ``read_forest`` reads and checks it."""

import pytest
from tests.fast.rollout.sprout.conftest import LOST, forest_result, members

from miles.rollout.sprout.forest import read_forest, search_outcome
from miles.rollout.sprout.protocol import RafoSearchSpec, RolloutResult

SPEC = RafoSearchSpec(roots=2, continuations=3, depth=3, events=3, review_seconds=1.0, wall_time_seconds=1.0)


def read(body, slots, *, spec=SPEC):
    return read_forest(RolloutResult.model_validate(body), slots, spec)


#: Both roots fail. Event 1 in root 1 fails twice; event 2 in root 2 splits; event 3 is inside event 1's first
#: continuation, and splits.
NESTED = [("root:0", [0.0, 0.0]), ("root:1", [1.0, 0.0]), ("event:1:0", [1.0, 0.0])]


def test_a_forest_holds_its_roots_and_each_event_with_where_its_turn_went():
    body, slots = forest_result([0.0, 0.0], NESTED)

    forest = read(body, slots)

    first = members(body, "event:1")[0]["branch_id"]
    assert forest.case == "FF" and [root.trajectory.branch_id for root in forest.roots] == ["job-11", "job-12"]
    assert [(e.number, e.depth, e.parent) for e in forest.events] == [
        (1, 1, "job-11"),
        (2, 1, "job-12"),
        (3, 2, first),
    ]
    assert [[c.value for c in e.continuations] for e in forest.events] == [[0.0, 0.0], [1.0, 0.0], [1.0, 0.0]]
    assert [e.turn for e in forest.events] == [3, 3, 7], "the turn of its own event, after its restored history"


def test_a_missing_trajectory_is_in_the_forest_but_not_gradable():
    body, slots = forest_result([0.0, None], [("root:0", [1.0, None, LOST])])

    forest = read(body, slots)

    assert forest.case == "FF_partial"
    assert [member.gradable for member in forest.roots] == [True, False]
    (event,) = forest.events
    assert [member.gradable for member in event.continuations] == [True, False], "the lost slot returned nothing"


@pytest.mark.parametrize(
    "counts, label",
    [
        ((2, 2, 2), "TT"),
        ((2, 1, 1), "TT_partial"),
        ((2, 2, 1), "TF"),
        ((3, 2, 1), "TF"),
        ((2, 2, 0), "FF"),
        ((2, 1, 0), "FF_partial"),
        ((2, 0, 0), "ungraded"),
    ],
)
def test_search_outcome_labels_the_roots_as_sprout_does(counts, label):
    assert search_outcome(*counts) == label


@pytest.mark.parametrize("counts", [(2, 3, 0), (2, 1, 2), (-1, 0, 0), (2.0, 1, 0)])
def test_search_outcome_refuses_counts_no_roots_can_have(counts):
    with pytest.raises(ValueError):
        search_outcome(*counts)


def test_a_forest_that_sprout_labelled_otherwise_is_refused():
    body, slots = forest_result([0.0, 0.0], NESTED)
    members(body, "event:2")[0]["metadata"]["search"]["outcome"] = "TF"

    with pytest.raises(ValueError, match="Sprout labels"):
        read(body, slots)


def test_roots_counted_against_another_plan_are_refused():
    body, slots = forest_result([0.0, 0.0])

    with pytest.raises(ValueError, match="2 root trajectories for 1 planned roots"):
        read(body, slots, spec=SPEC.model_copy(update={"roots": 1}))


@pytest.mark.parametrize(
    "update, message",
    [
        ({"events": 2}, "3 events, more than the 2 the search makes"),
        ({"depth": 1}, "event 3 of depth 2, deeper than the search's 1"),
        ({"continuations": 1}, "event 1 has 2 continuations, more than the search's 1"),
    ],
)
def test_a_forest_larger_than_its_search_is_refused(update, message):
    body, slots = forest_result([0.0, 0.0], NESTED)

    with pytest.raises(ValueError, match=message):
        read(body, slots, spec=SPEC.model_copy(update=update))


def _relabel_group(body):
    members(body, "event:2")[0]["group"] = "event:1"


def _parent_unknown(body):
    for trajectory in members(body, "event:3"):
        trajectory["metadata"]["search"]["parent_job_id"] = "job-99"


def _parents_disagree(body):
    members(body, "event:3")[0]["metadata"]["search"]["parent_job_id"] = "job-14"


def _parent_a_root(body):
    for trajectory in members(body, "event:3"):
        trajectory["metadata"]["search"]["parent_job_id"] = "job-11"


def _depth_skipped(body):
    for trajectory in members(body, "event:3"):
        trajectory["metadata"]["search"]["depth"] = 3


def _depths_disagree(body):
    members(body, "event:2")[1]["metadata"]["search"]["depth"] = 2


def _not_reasoning(body):
    for trajectory in members(body, "event:2"):
        turn = trajectory["messages"][3]
        turn["content"], turn["reasoning_content"] = "inspect tests\nnow", ""


def _no_own_turn(body):
    for trajectory in members(body, "event:2"):
        trajectory["metadata"]["provenance"]["inserted_messages"] = []


@pytest.mark.parametrize(
    "change, message",
    [
        (_relabel_group, "is in group 'event:1'"),
        (_parent_unknown, "names no returned trajectory as the parent of its turn"),
        (_parents_disagree, "the continuations of event 3 disagree on where its turn went"),
        (_parent_a_root, "event 3 of depth 2 continues job-11"),
        (_depth_skipped, "event 3 of depth 3 continues"),
        (_depths_disagree, "the continuations of event 2 disagree on its depth"),
        (_not_reasoning, "the turn of event 2 is not in the reasoning form"),
        (_no_own_turn, "has no inserted turn after its restored history"),
    ],
    ids=[
        "group",
        "unknown-parent",
        "parents-disagree",
        "parent-a-root",
        "depth-skipped",
        "depths-disagree",
        "not-reasoning",
        "no-turn",
    ],
)
def test_a_forest_its_annotations_do_not_describe_is_refused(change, message):
    body, slots = forest_result([0.0, 0.0], NESTED)
    change(body)

    with pytest.raises(ValueError, match=message):
        read(body, slots)


@pytest.mark.parametrize(
    "change, message",
    [
        (lambda m: m.pop("zero_reason"), "lacks the search annotations"),
        (lambda m: m["search"].update(kind="student"), "lacks the search annotations"),
        (lambda m: m["search"].pop("event"), "does not name its event"),
        (lambda m: m["search"].update(depth=0), "does not name its event"),
        (lambda m: m["search"].update(parent_job_id=None), "does not name its event"),
    ],
    ids=["no-zero-reason", "a-student", "no-event", "no-depth", "no-parent"],
)
def test_missing_annotations_fail_closed(change, message):
    body, slots = forest_result([0.0, 0.0], NESTED)
    change(members(body, "event:2")[0]["metadata"])

    with pytest.raises(ValueError, match=message):
        read(body, slots)


def test_a_graded_trajectory_carries_a_0_1_verdict():
    body, slots = forest_result([0.0, 0.0], NESTED)
    members(body, "event:2")[0]["reward"] = 0.5

    with pytest.raises(ValueError, match="RAFO credits a 0/1 verdict"):
        read(body, slots)


def test_a_trajectory_in_a_slot_miles_did_not_allocate_is_refused():
    body, slots = forest_result([0.0, 0.0], NESTED)
    slots["job:slot:99"] = slots.pop(members(body, "event:3")[1]["sample_slot_id"])

    with pytest.raises(ValueError, match="unknown sample_slot_id"):
        read(body, slots)
