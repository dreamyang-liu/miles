"""RAFO credit for a Sprout RAFO search: what each sample of a task trains, and with which fixed coefficient.

Sprout's RAFO search (``protocol.RafoSearchSpec``) returns RAFO's forest (``forest.read_forest``): the
task's M roots and its events. An event is a reviewer's turn at a point of a failed trajectory, a
check that exposes the problem, and the K continuations after it. A deeper event sits inside the
first failed continuation of an event whose continuations all failed, and a task has at most S
events. RAFO trains:

- each root with PPO, its advantage the local RLOO among the task's roots, with the coefficient 1/(M C);
- each continuation with PPO, its advantage the local RLOO among its event's continuations, with the
  coefficient 1/(S K C);
- each reviewer turn with CE on the turn alone, with the coefficient w/(S C_h). w is the mean over the
  event's continuations of c_j = R_j, or under ``subtree`` credit max(R_j, w_u) for each event u
  inside continuation j.

M, S and K are constants of the search, so no coefficient depends on what was sampled. R is the 0/1
verdict, and a missing trajectory counts for nothing. Every sample of a task shares one rollout id, so
Miles trains them in one step and divides the step by its tasks; a task without any signal keeps one
zero-weight root, so that it still counts.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import NamedTuple

from miles.rollout.sprout.forest import OUTCOMES, Event, Forest, Member, read_forest, search_outcome
from miles.rollout.sprout.importer import import_trajectory, turn_sample
from miles.rollout.sprout.protocol import RafoSearchSpec, RolloutResult
from miles.rollout.sprout.rafo_role import KIND_REFLECT, KIND_REPAIR, KIND_ROOT, METADATA_KEY, RafoRole
from miles.utils.mask_utils import MultiTurnLossMaskGenerator
from miles.utils.types import Sample

TURN_CREDITS = ("subtree", "own")
METRICS = (
    "events",
    "nested_events",
    "events_trained",
    "events_without_gain",
    "events_unchecked",
    "ppo_samples",
    "placeholders",
    "estimation_only",
    "ungradable",
)


@dataclass(frozen=True)
class RafoSettings:
    c: float  # the token normalizer of PPO samples
    c_h: float  # the token normalizer of reviewer turns
    turn_credit: str  # subtree | own

    def __post_init__(self) -> None:
        if not (self.c > 0 and self.c_h > 0):
            raise ValueError(f"RAFO's token normalizers must be positive, got c={self.c}, c_h={self.c_h}")
        if self.turn_credit not in TURN_CREDITS:
            raise ValueError(f"turn credit {self.turn_credit!r} is not RAFO's; expected one of {TURN_CREDITS}")


class Coefficients(NamedTuple):
    root: float  # 1/(M C)
    repair: float  # 1/(S K C)
    reflect: float  # 1/(S C_h), for each unit of a turn's w


def coefficients(spec: RafoSearchSpec, settings: RafoSettings) -> Coefficients:
    """RAFO's fixed coefficients: constants of the search and the settings."""
    return Coefficients(
        root=1.0 / (spec.roots * settings.c),
        repair=1.0 / (spec.events * spec.continuations * settings.c),
        reflect=1.0 / (spec.events * settings.c_h),
    )


@dataclass(frozen=True)
class _Row:
    member: Member
    role: RafoRole
    turn: int | None = None  # a CE row: the message index of the reviewer turn it trains
    weight: float | None = None  # a CE row: the turn's w


def assemble_rafo_group(
    result: RolloutResult,
    slot_samples: Mapping[str, Sample],
    mask_generator: MultiTurnLossMaskGenerator,
    *,
    spec: RafoSearchSpec,
    settings: RafoSettings,
    weight_version: int | None,
    indices: Iterator[int],
) -> tuple[list[Sample], dict[str, float]]:
    """A RAFO search group's trajectories -> the samples the RAFO loss trains, each with its ``RafoRole``."""
    counts = dict.fromkeys(METRICS, 0) | {f"cases/{label}": 0 for label in OUTCOMES}
    if result.status == "failed" and not result.trajectories:
        counts[f"cases/{search_outcome(spec.roots, 0, 0)}"] = 1  # every root's actor failed
        return [], _metric_names(counts)
    forest = read_forest(result, slot_samples, spec)
    coefs = coefficients(spec, settings)
    members = [*forest.roots, *(member for event in forest.events for member in event.continuations)]
    counts.update({f"cases/{forest.case}": 1, "ungradable": sum(not member.gradable for member in members)})
    weights = event_weights(forest.events, credit=settings.turn_credit)
    rows = _ppo_rows(forest, coefs) + _ce_rows(forest.events, weights, coefs, counts)
    task = _Task(result, slot_samples, mask_generator, weight_version, _task_group_index(slot_samples))
    samples = [sample for row in rows if (sample := _render(task, row, indices, counts)) is not None]
    if not samples:
        samples = _placeholder(task, forest.roots, counts)
    root_values = [member.value for member in forest.roots if member.gradable]
    for sample in samples:
        _annotate_task(sample, task, root_success=sum(root_values) / len(root_values) if root_values else 0.0)
    return samples, _metric_names(counts)


def rloo(values: list[float]) -> list[float]:
    """Leave-one-out advantage inside one group; no division by the group's standard deviation."""
    if len(values) < 2:
        raise ValueError(f"local RLOO needs at least 2 samples per group, got {len(values)}")
    total = sum(values)
    return [value - (total - value) / (len(values) - 1) for value in values]


def event_weights(events: tuple[Event, ...], *, credit: str) -> list[float]:
    """w of each event's turn: the mean over its gradable continuations of R, or under ``subtree`` credit
    of max(R, w_u) for each event u inside that continuation. Without a gradable continuation, 0."""
    if credit not in TURN_CREDITS:
        raise ValueError(f"unknown reviewer-turn credit {credit!r}; expected one of {TURN_CREDITS}")
    inside: dict[str, list[int]] = {}  # a trajectory's branch id -> the events whose turn went into it
    for index, event in enumerate(events):
        inside.setdefault(event.parent, []).append(index)
    weights = [0.0] * len(events)
    for index in sorted(range(len(events)), key=lambda i: -events[i].depth):  # an event before the one it is in
        credits = []
        for member in events[index].gradable:
            rescues = inside.get(member.trajectory.branch_id, []) if credit == "subtree" else []
            credits.append(max([member.value, *(weights[u] for u in rescues)]))
        weights[index] = sum(credits) / len(credits) if credits else 0.0
    return weights


def _ppo_rows(forest: Forest, coefs: Coefficients) -> list[_Row]:
    """The task's roots, and each event's continuations: the gradable members with a nonzero local RLOO."""
    groups = [(KIND_ROOT, coefs.root, forest.roots)]
    groups += [(KIND_REPAIR, coefs.repair, event.continuations) for event in forest.events]
    rows = []
    for kind, coef, members in groups:
        gradable = [member for member in members if member.gradable]
        if len(gradable) < 2:
            continue  # nothing to compare with: no PPO signal
        advantages = rloo([member.value for member in gradable])
        rows += [
            _Row(member, RafoRole(kind=kind, adv=adv, coef=coef))
            for member, adv in zip(gradable, advantages, strict=True)
            if adv != 0
        ]
    return rows


def _ce_rows(
    events: tuple[Event, ...], weights: list[float], coefs: Coefficients, counts: dict[str, int]
) -> list[_Row]:
    rows = []
    for event, weight in zip(events, weights, strict=True):
        counts["events"] += 1
        counts["nested_events"] += event.depth > 1
        counts["events_unchecked"] += event.unchecked
        if weight == 0:
            counts["events_without_gain"] += 1
            continue
        role = RafoRole(kind=KIND_REFLECT, adv=0.0, coef=weight * coefs.reflect)
        rows.append(_Row(event.gradable[0], role, turn=event.turn, weight=weight))
    return rows


@dataclass(frozen=True)
class _Task:
    result: RolloutResult
    slot_samples: Mapping[str, Sample]
    mask_generator: MultiTurnLossMaskGenerator
    weight_version: int | None
    group_index: int


def _task_group_index(slot_samples: Mapping[str, Sample]) -> int:
    """The prompt group's index, which the root slots keep (the branch slots have none)."""
    group_indices = {sample.group_index for sample in slot_samples.values() if sample.group_index is not None}
    if len(group_indices) != 1:
        raise ValueError(f"the slots of one task name one prompt group, got {sorted(group_indices)}")
    return group_indices.pop()


def _render(task: _Task, row: _Row, indices: Iterator[int], counts: dict[str, int]) -> Sample | None:
    trajectory = row.member.trajectory
    slot = task.slot_samples[trajectory.sample_slot_id]
    if row.role.kind == KIND_REFLECT:
        sample = turn_sample(
            trajectory,
            slot,
            task.mask_generator,
            index=next(indices),
            group_index=task.group_index,
            weight_version=task.weight_version,
            turn=row.turn,
        )
        if sample is not None:
            counts["events_trained"] += 1
    else:
        sample = import_trajectory(
            task.result, trajectory, slot, task.mask_generator, weight_version=task.weight_version
        )
        if sample is not None:
            counts["ppo_samples"] += 1
    if sample is None:
        counts["estimation_only"] += 1
        return None
    sample.train_metadata[METADATA_KEY] = row.role.to_metadata()
    sample.metadata[METADATA_KEY] = row.role.to_metadata()
    if row.weight is not None:
        sample.reward = row.weight  # a reviewer turn has no outcome of its own; dumps show its w
    return sample


def _placeholder(task: _Task, roots: tuple[Member, ...], counts: dict[str, int]) -> list[Sample]:
    """One zero-weight root, so that the task counts in the step it adds no gradient to."""
    for member in roots:
        if not member.gradable:
            continue
        row = _Row(member, RafoRole(kind=KIND_ROOT, adv=0.0, coef=0.0))
        if (sample := _render(task, row, iter(()), counts)) is not None:
            counts["ppo_samples"] -= 1
            counts["placeholders"] += 1
            return [sample]
    return []


def _annotate_task(sample: Sample, task: _Task, *, root_success: float) -> None:
    """The task's identity on each of its samples: one rollout of one prompt group."""
    sample.rollout_id = task.group_index
    sample.group_index = task.group_index
    # Miles's episode reward: the policy without review, the share of the task's roots that resolved.
    sample.metadata["raw_reward"] = root_success
    for lineage in (sample.train_metadata["sprout_rollout"], sample.metadata["sprout_rollout"]):
        lineage.update(rollout_job_id=task.result.rollout_job_id, prompt_group_id=task.result.prompt_group_id)


def _metric_names(counts: dict[str, int]) -> dict[str, float]:
    return {f"rollout/sprout/rafo/{name}": value for name, value in counts.items()}
