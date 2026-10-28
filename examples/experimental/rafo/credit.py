"""RAFO credit: local RLOO advantages, reviewer-turn weights, and the fixed loss coefficients.

All functions are pure. The coefficients have no 1/B factor: miles divides the loss of each optimizer step by
the number of rollouts in the step, and one rollout is one task.
"""

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class EventOutcome:
    """The structure and outcome of one event, enough to weight its reviewer turn."""

    parent: int | None  # index of the event that this deeper review continues
    parent_cont: int | None  # continuation of the parent that holds this event's checkpoint (branch re-review)
    cont_rewards: tuple[float, ...]


@dataclass(frozen=True)
class FixedCoefficients:
    root: float
    repair: float
    reflect_per_weight: float

    def reflect(self, weight: float) -> float:
        return weight * self.reflect_per_weight


def rloo(rewards: Sequence[float]) -> list[float]:
    """Leave-one-out advantage inside one group; no division by the group's standard deviation."""
    if len(rewards) < 2:
        raise ValueError(f"local RLOO needs at least 2 samples per group, got {len(rewards)}")
    total = sum(rewards)
    return [r - (total - r) / (len(rewards) - 1) for r in rewards]


def event_weights(events: Sequence[EventOutcome], *, credit: str) -> list[float]:
    """w_v = mean_j c_vj, where c_vj = R_vj, or under subtree credit max(R_vj, w_u) for a deeper review u inside
    continuation j. Events are in creation order, so a child always comes after its parent."""
    if credit not in ("own", "subtree"):
        raise ValueError(f"unknown reviewer-turn credit {credit!r}; expected 'own' or 'subtree'")
    children: dict[tuple[int, int], list[int]] = {}
    for index, event in enumerate(events):
        if event.parent is not None and event.parent_cont is not None:
            children.setdefault((event.parent, event.parent_cont), []).append(index)
    weights = [0.0] * len(events)
    for index in reversed(range(len(events))):
        credits = []
        for j, reward in enumerate(events[index].cont_rewards):
            rescues = [weights[child] for child in children.get((index, j), [])] if credit == "subtree" else []
            credits.append(max([reward, *rescues]))
        weights[index] = sum(credits) / len(credits)
    return weights


def fixed_coefficients(*, m: int, k: int, s: int, c: float, c_h: float) -> FixedCoefficients:
    """kappa_root = 1/(M C), kappa_rep = 1/(S K C), kappa_ref(v) = w_v/(S C_h): constants of the configuration."""
    return FixedCoefficients(root=1.0 / (m * c), repair=1.0 / (s * k * c), reflect_per_weight=1.0 / (s * c_h))
