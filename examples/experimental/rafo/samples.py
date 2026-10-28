"""Turn one RAFO forest into miles training samples.

Each sample keeps its own token stream and loss mask; ``train_metadata`` gives the loss its role. All samples of
the forest share one rollout id, so miles trains them in the same step and counts the task once.
"""

from dataclasses import dataclass

from examples.experimental.rafo.credit import EventOutcome, FixedCoefficients, event_weights, rloo
from examples.experimental.rafo.forest import Forest
from examples.experimental.rafo.loss import KIND_REFLECT, KIND_REPAIR, KIND_ROOT
from miles.utils.types import Sample


@dataclass(frozen=True)
class _Row:
    sample: Sample
    kind: str
    adv: float
    coef: float
    reward: float
    depth: int
    event: int | None
    own_tokens: int


def forest_samples(task: Sample, forest: Forest, coefficients: FixedCoefficients, *, credit: str) -> list[Sample]:
    rows = _rows(forest, coefficients, credit=credit)
    kept = [row for row in rows if _has_signal(row)]
    if not kept:
        # miles divides each step by its rollout count: keep the task in it with one zero-weight sample
        shortest = min(forest.roots, key=lambda root: len(root.sample.tokens))
        kept = [_Row(shortest.sample, KIND_ROOT, 0.0, 0.0, shortest.reward, 0, None, shortest.own_tokens)]
    stats = _task_stats(forest, rows)
    rollout_id = task.rollout_id if task.rollout_id is not None else task.index
    for row in kept:
        sample = row.sample
        sample.rollout_id = rollout_id
        sample.reward = row.reward
        sample.train_metadata = {"rafo_kind": row.kind, "rafo_adv": row.adv, "rafo_coef": row.coef}
        # the logged episode reward of every sample is the task's root success: the policy without review
        sample.metadata["raw_reward"] = stats["root_success"]
        sample.metadata["rafo"] = {"kind": row.kind, "depth": row.depth, "event": row.event, "task": dict(stats)}
    return [row.sample for row in kept]


def _rows(forest: Forest, coefficients: FixedCoefficients, *, credit: str) -> list[_Row]:
    rows = [
        _Row(root.sample, KIND_ROOT, adv, coefficients.root, root.reward, 0, None, root.own_tokens)
        for root, adv in zip(forest.roots, rloo([root.reward for root in forest.roots]), strict=True)
    ]
    outcomes = [
        EventOutcome(event.parent, event.parent_cont, tuple(cont.reward for cont in event.conts))
        for event in forest.events
    ]
    for index, (event, weight) in enumerate(zip(forest.events, event_weights(outcomes, credit=credit), strict=True)):
        advantages = rloo([cont.reward for cont in event.conts])
        rows += [
            _Row(cont.sample, KIND_REPAIR, adv, coefficients.repair, cont.reward, event.depth, index, cont.own_tokens)
            for cont, adv in zip(event.conts, advantages, strict=True)
        ]
        n_turn = sum(event.ce_sample.loss_mask)
        rows.append(
            _Row(event.ce_sample, KIND_REFLECT, 0.0, coefficients.reflect(weight), weight, event.depth, index, n_turn)
        )
    return rows


def _has_signal(row: _Row) -> bool:
    """A PPO sample with A = 0 or a reviewer turn with w = 0 has exactly zero loss and gradient."""
    return row.own_tokens > 0 and (row.coef > 0 if row.kind == KIND_REFLECT else row.adv != 0)


def _task_stats(forest: Forest, rows: list[_Row]) -> dict:
    conts = [cont for event in forest.events for cont in event.conts]
    return {
        "root_success": sum(root.reward for root in forest.roots) / len(forest.roots),
        "events": len(forest.events),
        "reviewer_reads": forest.reviewer_reads,
        "abstentions": forest.abstentions,
        "cont_success": sum(cont.reward for cont in conts) / len(conts) if conts else None,
        "reflect_samples": sum(1 for row in rows if row.kind == KIND_REFLECT and _has_signal(row)),
    }
