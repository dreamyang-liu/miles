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
  event's continuations of c_j = 1[R_j = 1], or under ``subtree`` credit max(c_j, w_u) for each event u
  inside continuation j.

M, S and K are constants of the search, so no coefficient depends on what was sampled. R is the reward:
the 0/1 verdict, or -1 for an invalid attempt. The RLOO advantages compare the rewards themselves; a
turn's weight counts resolves only, -1 and 0 alike unresolved, so that w >= 0. A missing trajectory
counts for nothing. Every sample of a task shares one rollout id, so Miles trains them in one step and
divides the step by its tasks; a task without any signal keeps one zero-weight root, so that it still
counts.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import NamedTuple

from miles.rollout.sprout.forest import OUTCOMES, Event, Forest, Member, read_forest, search_outcome
from miles.rollout.sprout.importer import import_trajectory, prepare_messages, turn_sample, unsampled_messages
from miles.rollout.sprout.protocol import RafoSearchSpec, RolloutResult
from miles.rollout.sprout.rafo_role import KIND_REFLECT, KIND_REPAIR, KIND_ROOT, KIND_TURN, METADATA_KEY, RafoRole
from miles.utils.mask_utils import MultiTurnLossMaskGenerator
from miles.utils.types import Sample

TURN_CREDITS = ("subtree", "own")
#: How a task's samples are credited. ``paper``: the RAFO paper's roles (roots and continuations with local
#: RLOO, the reviewer turn with CE weighted by w). ``tree``: tree credit -- every node's value is the share of
#: resolved leaves in its subtree; a reviewed trajectory splits at the point into a prefix (advantage: its
#: subtree value against its siblings') and a remainder (R against the turn's subtree value); the reviewer turn
#: is the head of its subtree, trained with PPO on V(turn) - R(remainder) instead of CE. ``tree_grpo``: Tree-GRPO
#: credit -- a leaf's advantage is (R - its root's tree mean) + (R - the task's mean), without std, and a segment's
#: advantage is the sum of the leaf advantages below it (so a prefix weighs as many leaves as it has); the reviewer
#: turn takes its continuations' sum when positive, with PPO.
CREDITS = ("paper", "tree", "tree_grpo")
METRICS = (
    "tasks_root_resolved",  # a root resolved the task
    "tasks_resolved",  # a root or a continuation resolved it: the tasks the step solves with review
    "roots",  # the task's roots: with roots_resolved, the policy's own resolve rate, as without RAFO
    "roots_resolved",
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
    #: Tokens. When positive, a PPO sample's coefficient is scaled by length_ref / its trained tokens, so that
    #: every trajectory weighs |A| * length_ref / (M C) whatever its length (a sequence mean, as GRPO's), instead
    #: of the paper's token sum, under which the longest failed trajectories dominate the step. 0 keeps the sum.
    length_ref: int = 0
    #: Tokens. With length_ref, a trajectory shorter than this counts as this long, so that a few hundred tokens
    #: of an attempt that ended at once do not weigh 50-100x per token (GRPO's 1/L bias on short answers).
    length_floor: int = 0
    credit: str = "paper"  # paper | tree | tree_grpo

    def __post_init__(self) -> None:
        if self.credit not in CREDITS:
            raise ValueError(f"RAFO credit {self.credit!r} is unknown; expected one of {CREDITS}")
        if not (self.c > 0 and self.c_h > 0):
            raise ValueError(f"RAFO's token normalizers must be positive, got c={self.c}, c_h={self.c_h}")
        for name in ("length_ref", "length_floor"):
            value = getattr(self, name)
            if not (isinstance(value, int) and value >= 0):
                raise ValueError(f"RAFO's {name} is a token count (0 for none), got {value!r}")
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
    turn: int | None = None  # a turn row (CE or PPO): the message index of the reviewer turn it trains
    weight: float | None = None  # a turn row: the turn's w (CE) or its subtree value (PPO)
    split: int | None = None  # tree credit: the message index where the trajectory's remainder begins
    prefix_adv: float | None = None  # tree credit: the advantage of the messages before ``split``


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
    counts["tasks_root_resolved"] = int(any(member.resolved for member in forest.roots))
    counts["tasks_resolved"] = int(any(member.resolved for member in members))
    counts["roots"] = len(forest.roots)
    counts["roots_resolved"] = sum(member.resolved for member in forest.roots)
    if settings.credit == "tree":
        rows = _tree_rows(forest, coefs, counts)
    elif settings.credit == "tree_grpo":
        rows = _tree_grpo_rows(forest, coefs, counts)
    else:
        weights = event_weights(forest.events, credit=settings.turn_credit)
        rows = _ppo_rows(forest, coefs) + _ce_rows(forest.events, weights, coefs, counts)
    task = _Task(result, slot_samples, mask_generator, weight_version, _task_group_index(slot_samples))
    samples = [
        sample
        for row in rows
        if (
            sample := _render(
                task, row, indices, counts, length_ref=settings.length_ref, length_floor=settings.length_floor
            )
        )
        is not None
    ]
    if not samples:
        samples = _placeholder(task, forest.roots, counts)
    root_rewards = [member.value for member in forest.roots if member.gradable]
    for sample in samples:
        _annotate_task(sample, task, root_reward=sum(root_rewards) / len(root_rewards) if root_rewards else 0.0)
    return samples, _metric_names(counts)


def rloo(values: list[float]) -> list[float]:
    """Leave-one-out advantage inside one group; no division by the group's standard deviation."""
    if len(values) < 2:
        raise ValueError(f"local RLOO needs at least 2 samples per group, got {len(values)}")
    total = sum(values)
    return [value - (total - value) / (len(values) - 1) for value in values]


def event_weights(events: tuple[Event, ...], *, credit: str) -> list[float]:
    """w of each event's turn: the mean over its gradable continuations of 1[R = 1], or under ``subtree``
    credit of max(1[R = 1], w_u) for each event u inside that continuation. Without a gradable
    continuation, 0. A -1 counts as unresolved, as a 0 does, so that w >= 0."""
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
            credits.append(max([float(member.resolved), *(weights[u] for u in rescues)]))
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


def _tree_rows(forest: Forest, coefs: Coefficients, counts: dict[str, int]) -> list[_Row]:
    """Tree credit. A leaf is a graded trajectory's own outcome, 1[R = 1]. A node's value V is the share of
    resolved leaves in its subtree. A trajectory compares with its siblings (the task's roots; an event's
    continuations) by local RLOO on its node value: V if an event's turn went into it, else its reward R, -1
    included. A reviewed trajectory splits at the point: the prefix takes that advantage, the remainder takes
    R - V(turn). The turn takes V(turn) - R(remainder) >= 0 with PPO (``KIND_TURN``), 0 trains nothing."""
    events_in: dict[str, list[Event]] = {}
    for event in forest.events:
        events_in.setdefault(event.parent, []).append(event)

    def leaves(member: Member) -> list[float]:
        found = [float(member.resolved)] if member.gradable else []
        for event in events_in.get(member.trajectory.branch_id, []):
            for continuation in event.continuations:
                found += leaves(continuation)
        return found

    def subtree_value(members: tuple[Member, ...] | list[Member]) -> float | None:
        found = [leaf for member in members for leaf in leaves(member)]
        return sum(found) / len(found) if found else None

    def node_value(member: Member) -> float | None:
        if not member.gradable:
            return None
        return subtree_value([member]) if events_in.get(member.trajectory.branch_id) else member.value

    rows: list[_Row] = []

    def credit_group(members: tuple[Member, ...], kind: str, coef: float) -> None:
        valued = [(member, value) for member in members if (value := node_value(member)) is not None]
        advantages = rloo([value for _, value in valued]) if len(valued) >= 2 else [0.0] * len(valued)
        for (member, _), adv in zip(valued, advantages, strict=True):
            events = events_in.get(member.trajectory.branch_id, [])
            if not events:
                if adv != 0:
                    rows.append(_Row(member, RafoRole(kind=kind, adv=adv, coef=coef)))
                continue
            turn_value = subtree_value([c for event in events for c in event.continuations])
            remainder_adv = member.value - (turn_value or 0.0)
            if adv != 0 or remainder_adv != 0:
                role = RafoRole(kind=kind, adv=remainder_adv, coef=coef)
                rows.append(_Row(member, role, split=min(event.turn for event in events), prefix_adv=adv))
            for event in events:
                counts["events"] += 1
                counts["nested_events"] += event.depth > 1
                counts["events_unchecked"] += event.unchecked
                value = subtree_value(event.continuations) or 0.0
                turn_adv = value - max(member.value, 0.0)
                if turn_adv <= 0 or not event.gradable:
                    counts["events_without_gain"] += 1
                else:
                    role = RafoRole(kind=KIND_TURN, adv=turn_adv, coef=coefs.reflect)
                    rows.append(_Row(event.gradable[0], role, turn=event.turn, weight=value))
                credit_group(event.continuations, KIND_REPAIR, coefs.repair)

    credit_group(forest.roots, KIND_ROOT, coefs.root)
    return rows


def _tree_grpo_rows(forest: Forest, coefs: Coefficients, counts: dict[str, int]) -> list[_Row]:
    """Tree-GRPO credit. A leaf is a graded trajectory's own reward R (1, 0, or -1). Each root with its
    continuations is one tree; a leaf's advantage is (R - the mean of its tree's leaves) + (R - the mean of the
    task's leaves), the intra- and inter-tree terms of Tree-GRPO without the division by std. A segment's
    advantage is the sum of the leaf advantages in its subtree: an unreviewed trajectory takes its own leaf's; a
    reviewed one splits at the point into a prefix (its subtree's sum, so it weighs as many leaves as it has)
    and a remainder (its own leaf's); the turn takes its continuations' subtrees' sum when positive, with PPO.
    The advantages of a task sum to zero."""
    events_in: dict[str, list[Event]] = {}
    for event in forest.events:
        events_in.setdefault(event.parent, []).append(event)

    def leaves(member: Member) -> list[float]:
        found = [member.value] if member.gradable else []
        for event in events_in.get(member.trajectory.branch_id, []):
            for continuation in event.continuations:
                found += leaves(continuation)
        return found

    def mean(values: list[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    task_mean = mean([leaf for root in forest.roots for leaf in leaves(root)])
    rows: list[_Row] = []

    def credit(member: Member, kind: str, coef: float, tree_mean: float) -> None:
        def leaf_adv(reward: float) -> float:
            return (reward - tree_mean) + (reward - task_mean)

        own = leaf_adv(member.value)
        events = events_in.get(member.trajectory.branch_id, [])
        if not events:
            if own != 0:
                rows.append(_Row(member, RafoRole(kind=kind, adv=own, coef=coef)))
            return
        prefix = sum(leaf_adv(leaf) for leaf in leaves(member))
        if prefix != 0 or own != 0:
            role = RafoRole(kind=kind, adv=own, coef=coef)
            rows.append(_Row(member, role, split=min(event.turn for event in events), prefix_adv=prefix))
        for event in events:
            counts["events"] += 1
            counts["nested_events"] += event.depth > 1
            counts["events_unchecked"] += event.unchecked
            turn_adv = sum(leaf_adv(leaf) for c in event.continuations for leaf in leaves(c))
            if turn_adv <= 0 or not event.gradable:
                counts["events_without_gain"] += 1
            else:
                role = RafoRole(kind=KIND_TURN, adv=turn_adv, coef=coefs.reflect)
                rows.append(_Row(event.gradable[0], role, turn=event.turn, weight=turn_adv))
            for continuation in event.continuations:
                if continuation.gradable:
                    credit(continuation, KIND_REPAIR, coefs.repair, tree_mean)

    for root in forest.roots:
        if root.gradable:
            credit(root, KIND_ROOT, coefs.root, mean(leaves(root)))
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


def _render(
    task: _Task,
    row: _Row,
    indices: Iterator[int],
    counts: dict[str, int],
    *,
    length_ref: int = 0,
    length_floor: int = 0,
) -> Sample | None:
    trajectory = row.member.trajectory
    slot = task.slot_samples[trajectory.sample_slot_id]
    if row.role.kind in (KIND_REFLECT, KIND_TURN):
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
    role = row.role
    if row.split is not None:
        cut = _prefix_cut(task.mask_generator, trajectory, sample, split=row.split)
        role = RafoRole(kind=role.kind, adv=role.adv, coef=role.coef, segments=((0, row.prefix_adv), (cut, role.adv)))
    if length_ref and role.kind not in (KIND_REFLECT, KIND_TURN) and role.coef:
        trained = sum(sample.loss_mask)
        if trained:
            scaled = role.coef * length_ref / max(trained, length_floor)
            role = RafoRole(kind=role.kind, adv=role.adv, coef=scaled, segments=role.segments)
    sample.train_metadata[METADATA_KEY] = role.to_metadata()
    sample.metadata[METADATA_KEY] = role.to_metadata()
    if row.weight is not None:
        sample.reward = row.weight  # a reviewer turn has no outcome of its own; dumps show its w
    return sample


def _prefix_cut(mask_generator: MultiTurnLossMaskGenerator, trajectory, sample: Sample, *, split: int) -> int:
    """The response position where the trajectory's tokens from message ``split`` on begin.

    The messages before ``split`` are rendered again with the later ones masked; the tokens are the same
    sequence, so the mask says which response positions belong to the prefix. A ``split`` past the last
    message leaves no remainder (the cut is the response length)."""
    messages = prepare_messages(trajectory)
    for index in unsampled_messages(trajectory):
        messages[index]["step_loss_mask"] = 0
    for index in range(min(split, len(messages)), len(messages)):
        messages[index]["step_loss_mask"] = 0
    tokens, prefix_mask = mask_generator.get_loss_mask(messages, tools=trajectory.tools or None)
    if list(tokens) != list(sample.tokens):
        raise ValueError(f"{trajectory.sample_slot_id!r}: the prefix renders to other tokens than the trajectory")
    response = prefix_mask[-sample.response_length :]
    prefix_positions = [position for position, kept in enumerate(response) if kept]
    cut = prefix_positions[-1] + 1 if prefix_positions else 0
    if any(kept and not response[position] and position < cut for position, kept in enumerate(sample.loss_mask)):
        raise ValueError(f"{trajectory.sample_slot_id!r}: the remainder's tokens interleave with the prefix's")
    return cut


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


def _annotate_task(sample: Sample, task: _Task, *, root_reward: float) -> None:
    """The task's identity on each of its samples: one rollout of one prompt group."""
    sample.rollout_id = task.group_index
    sample.group_index = task.group_index
    # Miles's episode reward: the policy without review, the mean reward of the task's roots. A -1 counts as
    # it does in raw_reward without RAFO, so the curve keeps its meaning when RAFO is turned on; the share
    # that resolved is Sprout's rollout/sprout/resolved_mean.
    sample.metadata["raw_reward"] = root_reward
    for lineage in (sample.train_metadata["sprout_rollout"], sample.metadata["sprout_rollout"]):
        lineage.update(rollout_job_id=task.result.rollout_job_id, prompt_group_id=task.result.prompt_group_id)


def _metric_names(counts: dict[str, int]) -> dict[str, float]:
    return {f"rollout/sprout/rafo/{name}": value for name, value in counts.items()}
