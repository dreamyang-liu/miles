"""RAFO credit for a Sprout search group: what each sample of a task trains, and with which fixed coefficient.

Sprout's search is RAFO's forest. The roots run from scratch. A review of each failed root inserts a reviewer turn
at a point before the mistake, and continuations run after it (repairs) and with nothing added (students). Later
repair rounds review a failed repair again and insert one more turn in its own history: RAFO's nested review in
the failed branch. RAFO trains:

- each root, student and repair with PPO, its advantage the local RLOO within the condition it was sampled under
  (``Trajectory.group``), with the coefficient 1/(M C) for a root and 1/(G K C) for a continuation;
- each reviewer turn with CE on the turn alone, with the coefficient w/(T C_h). w is the mean over the turn's
  continuations of max(R', w_child), less the mean of the students at its point, and at least 0. Under
  ``subtree`` credit, w_child is the weight of a later round's turn inserted in that continuation. A turn that names
  hidden tests (``leak_terms``), that nothing checked or that is not in canonical form gets w = 0.

M is the roots, G the most continuation groups and T the most reviewer turns one task can have, and K the planned
size of a continuation group. All are constants of the search, so no coefficient depends on what was sampled. R' is
GSML's: a repair that touched the task's tests counts as unresolved. Every sample of the task shares one rollout
id, so Miles trains them in one step and divides the step by its tasks; a task without any signal keeps one
zero-weight root, so that it still counts.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import NamedTuple

from miles.rollout.sprout.gsml import (
    OUTCOMES,
    SearchMember,
    check_group,
    checked_outcome,
    search_candidate,
    search_member,
    search_outcome,
)
from miles.rollout.sprout.importer import ahat_sample, import_trajectory
from miles.rollout.sprout.protocol import RolloutResult, SearchSpec
from miles.rollout.sprout.rafo_role import KIND_REFLECT, KIND_REPAIR, KIND_ROOT, METADATA_KEY, RafoRole
from miles.utils.mask_utils import MultiTurnLossMaskGenerator
from miles.utils.types import Sample

TURN_CREDITS = ("subtree", "own")
BASELINES = ("students", "none")
METRICS = (
    "turns",
    "turns_trained",
    "turns_leaked",
    "turns_unverified",
    "turns_noncanonical",
    "turns_without_gain",
    "chain_turns",
    "ppo_samples",
    "placeholders",
    "estimation_only",
    "ungradable",
    "continuations_touching_tests",
)


@dataclass(frozen=True)
class RafoSettings:
    c: float  # the token normalizer of PPO samples
    c_h: float  # the token normalizer of reviewer turns
    turn_credit: str  # subtree | own
    baseline: str  # students | none: what a reviewer turn's continuations are compared with

    def __post_init__(self) -> None:
        if not (self.c > 0 and self.c_h > 0):
            raise ValueError(f"RAFO's token normalizers must be positive, got c={self.c}, c_h={self.c_h}")
        if self.turn_credit not in TURN_CREDITS or self.baseline not in BASELINES:
            raise ValueError(f"turn credit {self.turn_credit!r} or baseline {self.baseline!r} is not RAFO's")


class Denominators(NamedTuple):
    roots: int  # M
    groups: int  # G: continuation groups
    turns: int  # T: reviewer turns


def denominators(spec: SearchSpec) -> Denominators:
    """The most of each that a search can make for one task; every root failed (FF) or all but one (TF)."""
    every_root_failed = spec.roots * spec.points * (1 + spec.candidates * spec.repair_rounds)
    all_but_one = (spec.roots - 1) * spec.points * int(spec.tf_student_continuations > 0)
    turns = spec.roots * spec.points * spec.candidates * spec.repair_rounds
    return Denominators(roots=spec.roots, groups=max(every_root_failed, all_but_one, 1), turns=max(turns, 1))


@dataclass(frozen=True)
class _Row:
    member: SearchMember
    role: RafoRole
    turn: int | None = None  # a CE row: the message index of the reviewer turn it trains
    weight: float | None = None  # a CE row: the turn's w


@dataclass(frozen=True)
class _Turn:
    """One reviewer turn: the gradable continuations after it, and where it was inserted."""

    group: str
    point_id: str
    round: int
    continuations: tuple[SearchMember, ...]
    position: int  # its message index in its continuations
    host: tuple[int, int] | None  # (turn, continuation) of the repair it was inserted in, for a later round
    gate: str | None  # leaked | unverified | noncanonical


def assemble_rafo_group(
    result: RolloutResult,
    slot_samples: Mapping[str, Sample],
    mask_generator: MultiTurnLossMaskGenerator,
    *,
    spec: SearchSpec,
    settings: RafoSettings,
    weight_version: int | None,
    indices: Iterator[int],
) -> tuple[list[Sample], dict[str, float]]:
    """A search group's trajectories -> the samples the RAFO loss trains, each with its ``RafoRole``."""
    counts = dict.fromkeys(METRICS, 0) | {f"cases/{label}": 0 for label in OUTCOMES}
    if result.status == "failed" and not result.trajectories:
        counts[f"cases/{search_outcome(spec.roots, 0, 0)}"] = 1  # every root's actor failed
        return [], _metric_names(counts)
    check_group(result, slot_samples)
    members = [search_member(trajectory) for trajectory in result.trajectories]
    case = checked_outcome(members, spec.roots)
    gradable = [member for member in members if member.gradable]
    counts.update(
        {
            f"cases/{case}": 1,
            "ungradable": len(members) - len(gradable),
            "continuations_touching_tests": sum(member.touching for member in members),
        }
    )
    sizes = denominators(spec)
    turns = _turns(gradable)
    weights = _turn_weights(turns, gradable, settings)
    rows = _ppo_rows(gradable, case, spec, sizes, settings)
    rows += _ce_rows(turns, weights, sizes, settings, counts)
    task = _Task(result, slot_samples, mask_generator, weight_version, _task_group_index(slot_samples))
    samples = [sample for row in rows if (sample := _render(task, row, indices, counts)) is not None]
    if not samples:
        samples = _placeholder(task, gradable, counts)
    root_values = [member.value for member in gradable if member.kind == "root"]
    for sample in samples:
        _annotate_task(sample, task, root_success=sum(root_values) / len(root_values) if root_values else 0.0)
    return samples, _metric_names(counts)


def rloo(values: list[float]) -> list[float]:
    """Leave-one-out advantage inside one group; no division by the group's standard deviation."""
    if len(values) < 2:
        raise ValueError(f"local RLOO needs at least 2 samples per group, got {len(values)}")
    total = sum(values)
    return [value - (total - value) / (len(values) - 1) for value in values]


def _ppo_rows(
    gradable: list[SearchMember], case: str, spec: SearchSpec, sizes: Denominators, settings: RafoSettings
) -> list[_Row]:
    """Each condition's gradable trajectories with a nonzero local RLOO advantage."""
    groups: dict[str, list[SearchMember]] = {}
    for member in gradable:
        groups.setdefault(member.trajectory.group, []).append(member)
    rows = []
    for group_members in groups.values():
        if len(group_members) < 2:
            continue  # nothing to compare with: no PPO signal
        kind, coef = _ppo_coefficient(group_members[0].kind, case, spec, sizes, settings)
        advantages = rloo([member.value for member in group_members])
        rows += [
            _Row(member, RafoRole(kind=kind, adv=adv, coef=coef))
            for member, adv in zip(group_members, advantages, strict=True)
            if adv != 0
        ]
    return rows


def _ppo_coefficient(
    kind: str, case: str, spec: SearchSpec, sizes: Denominators, settings: RafoSettings
) -> tuple[str, float]:
    if kind == "root":
        return KIND_ROOT, 1.0 / (sizes.roots * settings.c)
    if kind == "student":
        planned = spec.student_continuations if case == "FF" else spec.tf_student_continuations
    else:
        planned = spec.candidate_continuations
    return KIND_REPAIR, 1.0 / (sizes.groups * max(planned, 1) * settings.c)


def _turns(gradable: list[SearchMember]) -> list[_Turn]:
    """The reviewer turns, by the repair condition of their continuations, with the repair each later round's
    turn was inserted in (``chain_parent_job_id``, the job of that repair)."""
    groups: dict[str, list[SearchMember]] = {}
    for member in gradable:
        if member.kind == "repair":
            groups.setdefault(member.trajectory.group, []).append(member)
    hosts = {
        member.trajectory.branch_id: (turn, cont)
        for turn, group_members in enumerate(groups.values())
        for cont, member in enumerate(group_members)
    }
    turns = []
    for group, group_members in groups.items():
        search = group_members[0].trajectory.metadata["search"]
        round_no = search.get("round", 1)
        candidate = search_candidate(group_members)
        gate = "leaked" if candidate.leaked else "unverified" if candidate.unverified else None
        gate = gate or ("noncanonical" if candidate.noncanonical else None)
        turns.append(
            _Turn(
                group=group,
                point_id=group_members[0].point_id,
                round=round_no,
                continuations=tuple(group_members),
                position=max(group_members[0].trajectory.metadata["provenance"]["inserted_messages"]),
                host=hosts.get(search.get("chain_parent_job_id")) if round_no > 1 else None,
                gate=gate,
            )
        )
    return turns


def _turn_weights(turns: list[_Turn], gradable: list[SearchMember], settings: RafoSettings) -> list[float]:
    """w of each turn, the later rounds first so that subtree credit can reach back along the chain."""
    students: dict[str, list[float]] = {}
    for member in gradable:
        if member.kind == "student":
            students.setdefault(member.point_id, []).append(member.value)
    rescues: dict[tuple[int, int], list[int]] = {}
    for index, turn in enumerate(turns):
        if turn.host is not None:
            rescues.setdefault(turn.host, []).append(index)
    weights = [0.0] * len(turns)
    for index in sorted(range(len(turns)), key=lambda i: -turns[i].round):
        turn = turns[index]
        credits = []
        for cont, member in enumerate(turn.continuations):
            rescued = [weights[child] for child in rescues.get((index, cont), [])]
            credits.append(max([member.value, *rescued]) if settings.turn_credit == "subtree" else member.value)
        at_point = students.get(turn.point_id, []) if turn.round == 1 and settings.baseline == "students" else []
        baseline = sum(at_point) / len(at_point) if at_point else 0.0
        weights[index] = 0.0 if turn.gate else max(0.0, sum(credits) / len(credits) - baseline)
    return weights


def _ce_rows(
    turns: list[_Turn], weights: list[float], sizes: Denominators, settings: RafoSettings, counts: dict[str, int]
) -> list[_Row]:
    rows = []
    for turn, weight in zip(turns, weights, strict=True):
        counts["turns"] += 1
        counts["chain_turns"] += turn.round > 1
        if turn.gate:
            counts[f"turns_{turn.gate}"] += 1
        elif weight == 0:
            counts["turns_without_gain"] += 1
        else:
            role = RafoRole(kind=KIND_REFLECT, adv=0.0, coef=weight / (sizes.turns * settings.c_h))
            rows.append(_Row(turn.continuations[0], role, turn=turn.position, weight=weight))
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
        sample = ahat_sample(
            trajectory,
            slot,
            task.mask_generator,
            index=next(indices),
            group_index=task.group_index,
            weight_version=task.weight_version,
            turn=row.turn,
        )
        if sample is not None:
            counts["turns_trained"] += 1
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


def _placeholder(task: _Task, gradable: list[SearchMember], counts: dict[str, int]) -> list[Sample]:
    """One zero-weight root, so that the task counts in the step it adds no gradient to."""
    for member in gradable:
        if member.kind != "root":
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
