"""The RAFO forest that Sprout returns for one task: its roots and its events, checked.

Sprout runs a RAFO search (``protocol.RafoSearchSpec``) and returns the trajectories it made: the
roots, and the continuations of each event, which run after a reviewer's turn at a point
(``Trajectory.metadata["search"]``: ``event``, ``depth`` and ``parent_job_id``, the trajectory that
the turn went into). ``read_forest`` reads them into ``Member``s and ``Event``s, and refuses a forest
that its annotations do not describe or that is larger than its search. Sprout decides whether to
review by the label of the roots (``search_outcome``), and Miles recomputes the label from the roots
it receives.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from miles.rollout.sprout.importer import SEARCH_KINDS, unsampled_messages
from miles.rollout.sprout.protocol import RafoSearchSpec, RolloutResult, Trajectory
from miles.utils.types import Sample

#: The labels of a task's roots (``search_outcome``). Sprout reviews FF and FF_partial roots only.
OUTCOMES = ("TT", "TT_partial", "TF", "FF", "FF_partial", "ungraded")


def search_outcome(planned: int, graded: int, resolved: int) -> str:
    """A search task's label from its roots: planned, graded (a verdict came back) and resolved.

    Sprout labels the roots with the same function (``sprout.search.outcome.search_outcome``); Miles
    recomputes it from the returned roots and refuses a group Sprout labelled otherwise.
    """
    if any(type(count) is not int for count in (planned, graded, resolved)) or not (
        0 <= resolved <= graded <= planned
    ):
        raise ValueError(f"expected 0 <= resolved <= graded <= planned, got {planned=}, {graded=}, {resolved=}")
    if graded == 0:
        return "ungraded"
    if resolved == graded:
        return "TT" if graded == planned else "TT_partial"
    if resolved >= 1:
        return "TF"
    return "FF" if graded == planned else "FF_partial"


@dataclass(frozen=True)
class Member:
    """A returned trajectory as the credit reads it: a root, or a continuation of event ``event``."""

    trajectory: Trajectory
    event: int | None
    #: Its grade gave a verdict. A missing trajectory counts for nothing, not as a 0.
    gradable: bool
    #: R, the 0/1 verdict.
    value: float

    @property
    def slot(self) -> str:
        return self.trajectory.sample_slot_id


@dataclass(frozen=True)
class Event:
    """A reviewer's turn at a point, and the continuations after it."""

    number: int
    depth: int
    #: The ``branch_id`` of the trajectory that the turn went into: a root at depth 1, a continuation deeper.
    parent: str
    #: In slot order, the missing ones included.
    continuations: tuple[Member, ...]
    #: The turn's message index in its continuations.
    turn: int
    #: Nothing checked a turn of its history for the task's hidden tests (``leak_terms`` None).
    unchecked: bool

    @property
    def gradable(self) -> tuple[Member, ...]:
        return tuple(member for member in self.continuations if member.gradable)


@dataclass(frozen=True)
class Forest:
    case: str
    roots: tuple[Member, ...]
    #: By number, which is the order Sprout made them in: an event comes after the event it is inside.
    events: tuple[Event, ...]


def read_forest(result: RolloutResult, slot_samples: Mapping[str, Sample], spec: RafoSearchSpec) -> Forest:
    """The forest of a RAFO search group's result. Raises ValueError for one that Miles cannot credit."""
    if result.status not in {"completed", "early_stopped"}:
        raise ValueError(f"cannot credit a rollout group with status={result.status!r}: {result.stop_reason}")
    if result.max_samples > len(slot_samples):
        raise ValueError("the driver's max_samples exceeds the slots Miles allocated")
    for trajectory in result.trajectories:
        if trajectory.sample_slot_id not in slot_samples:
            raise ValueError(f"unknown sample_slot_id: {trajectory.sample_slot_id!r}")
    members = [_member(trajectory) for trajectory in result.trajectories]
    roots = tuple(member for member in members if member.event is None)
    case = _checked_outcome(members, roots, spec.roots)
    events = _events([member for member in members if member.event is not None], spec)
    _check_parents(roots, events)
    return Forest(case=case, roots=roots, events=events)


def _member(trajectory: Trajectory) -> Member:
    """Read what the credit needs from Sprout's annotations, refusing a trajectory that lacks them."""
    slot, metadata = trajectory.sample_slot_id, trajectory.metadata
    search = metadata.get("search")
    if "zero_reason" not in metadata or not isinstance(search, dict) or search.get("kind") not in SEARCH_KINDS:
        raise ValueError(f"trajectory {slot!r} lacks the search annotations RAFO credits it by")
    event = None
    if search["kind"] == "repair":
        event, depth, parent = search.get("event"), search.get("depth"), search.get("parent_job_id")
        if not (_positive(event) and _positive(depth) and isinstance(parent, str) and parent):
            raise ValueError(f"continuation {slot!r} does not name its event, its depth and its parent")
    group = "root" if event is None else f"event:{event}"
    if trajectory.group != group:
        raise ValueError(f"trajectory {slot!r} is in group {trajectory.group!r}; its annotation puts it in {group!r}")
    gradable = metadata["zero_reason"] is None
    if gradable and trajectory.reward not in (0, 1):
        raise ValueError(f"trajectory {slot!r} has reward {trajectory.reward!r}; RAFO credits a 0/1 verdict")
    return Member(trajectory=trajectory, event=event, gradable=gradable, value=float(trajectory.reward))


def _positive(value: object) -> bool:
    return type(value) is int and value >= 1


def _checked_outcome(members: list[Member], roots: tuple[Member, ...], planned_roots: int) -> str:
    """Miles's own label for the task, which every trajectory's annotation must repeat."""
    if len(roots) > planned_roots:
        raise ValueError(f"{len(roots)} root trajectories for {planned_roots} planned roots")
    graded = [member for member in roots if member.gradable]
    counts = {"planned": planned_roots, "graded": len(graded), "resolved": int(sum(m.value for m in graded))}
    case = search_outcome(**counts)
    for member in members:
        search = member.trajectory.metadata["search"]
        if search.get("outcome") != case or search.get("root_counts") != counts:
            raise ValueError(
                f"Sprout labels {member.slot!r} {search.get('outcome')!r} with root counts "
                f"{search.get('root_counts')}; the returned roots make the task {case!r} with {counts}"
            )
    return case


def _events(continuations: list[Member], spec: RafoSearchSpec) -> tuple[Event, ...]:
    """The events of the returned continuations, by number, each within the search's limits."""
    by_number: dict[int, list[Member]] = {}
    for member in continuations:
        by_number.setdefault(member.event, []).append(member)
    if len(by_number) > spec.events:
        raise ValueError(f"{len(by_number)} events, more than the {spec.events} the search makes")
    events = []
    for number, members in sorted(by_number.items()):
        searches = [member.trajectory.metadata["search"] for member in members]
        depths, parents = {s["depth"] for s in searches}, {s["parent_job_id"] for s in searches}
        if len(depths) != 1:
            raise ValueError(f"the continuations of event {number} disagree on its depth: {sorted(depths)}")
        if len(parents) != 1:
            raise ValueError(f"the continuations of event {number} disagree on where its turn went: {sorted(parents)}")
        (depth,), (parent,) = depths, parents
        if depth > spec.depth:
            raise ValueError(f"event {number} of depth {depth}, deeper than the search's {spec.depth}")
        if len(members) > spec.continuations:
            raise ValueError(
                f"event {number} has {len(members)} continuations, more than the search's {spec.continuations}"
            )
        turns = {_own_turn(member.trajectory, number) for member in members}
        if len(turns) != 1:
            raise ValueError(f"the continuations of event {number} disagree on where its turn is: {sorted(turns)}")
        unchecked = any(not isinstance(search.get("leak_terms"), list) for search in searches)
        events.append(Event(number, depth, parent, tuple(members), turns.pop(), unchecked))
    return tuple(events)


def _own_turn(trajectory: Trajectory, number: int) -> int:
    """The message index of event ``number``'s turn in one of its continuations: the first turn inserted
    after its restored history, in the reasoning form that the turn's CE sample trains."""
    unsampled_messages(trajectory)  # the provenance is checked before it is read
    provenance = trajectory.metadata["provenance"]
    own = [index for index in provenance["inserted_messages"] if index >= provenance["prefix_messages"]]
    if not own:
        raise ValueError(f"continuation {trajectory.sample_slot_id!r} has no inserted turn after its restored history")
    turn = trajectory.messages[min(own)]
    reasoning, content = turn.get("reasoning_content"), turn.get("content") or ""
    if not (isinstance(reasoning, str) and reasoning.strip() and "\n" not in content):
        raise ValueError(f"the turn of event {number} is not in the reasoning form that its CE sample trains")
    return min(own)


def _check_parents(roots: tuple[Member, ...], events: tuple[Event, ...]) -> None:
    """Each event's turn went into a root (depth 1), or into a continuation of an event one depth up."""
    depth_of = {member.trajectory.branch_id: 0 for member in roots}
    depth_of |= {member.trajectory.branch_id: event.depth for event in events for member in event.continuations}
    for event in events:
        if event.parent not in depth_of:
            raise ValueError(
                f"event {event.number} names no returned trajectory as the parent of its turn: {event.parent!r}"
            )
        if depth_of[event.parent] != event.depth - 1:
            above = "a root" if event.depth == 1 else f"a continuation of an event of depth {event.depth - 1}"
            raise ValueError(
                f"event {event.number} of depth {event.depth} continues {event.parent}, which is not {above}"
            )
