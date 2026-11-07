"""The role of one RAFO training sample: what the rollout writes into ``train_metadata`` and the loss reads.

Dependency-free, so the trainer reads it without Sprout's client code (``rafo.assemble_rafo_group`` writes it,
``miles.backends.training_utils.loss.hub.rafo_loss`` reads it).
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass

#: A rollout from scratch, trained with PPO.
KIND_ROOT = "root"
#: A continuation from a restored state, after a reviewer turn or with nothing added, trained with PPO.
KIND_REPAIR = "repair"
#: A reviewer turn as a one-turn target, trained with CE.
KIND_REFLECT = "reflect"
#: A reviewer turn as the head of its subtree, trained with PPO on the subtree's advantage (tree credit).
KIND_TURN = "turn"
#: How a ``turn`` sample trains in the RAFO loss: PPO like any sample, or ``shaped``: each token weighs
#: 2 pi_theta / (pi_old + 1) instead of the PPO ratio, so that a token the policy finds unlikely contributes
#: about 2 pi_theta of its gradient (a policy-gradient-like scaling of off-policy reviewer text) instead of all of it.
TURN_LOSSES = ("ppo", "shaped")
KINDS = (KIND_ROOT, KIND_REPAIR, KIND_REFLECT, KIND_TURN)
#: Where the role sits in ``Sample.train_metadata``.
METADATA_KEY = "rafo"


@dataclass(frozen=True)
class RafoRole:
    """``coef`` is the fixed coefficient of the sample's token sum, the token normalizer included; ``adv`` is
    the local RLOO advantage of a PPO sample and 0 for a CE sample."""

    kind: str
    adv: float
    coef: float
    #: Tree credit: (start, adv) pairs over the sample's response positions, ascending starts from 0. From each
    #: start on, the tokens take that advantage: a reviewed trajectory's prefix up to the point, then its own
    #: remainder. Empty: ``adv`` everywhere.
    segments: tuple[tuple[int, float], ...] = ()

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f"unknown RAFO kind {self.kind!r}; expected one of {KINDS}")
        if not (math.isfinite(self.adv) and math.isfinite(self.coef) and self.coef >= 0):
            raise ValueError(f"a RAFO role needs a finite adv and a finite coef >= 0, got {self.adv}, {self.coef}")
        if self.kind == KIND_REFLECT and self.adv != 0:
            raise ValueError("a reviewer turn trains with CE: its adv is 0")
        starts = [start for start, _ in self.segments]
        if self.segments and (
            starts[0] != 0
            or starts != sorted(starts)
            or any(type(start) is not int for start in starts)
            or any(not math.isfinite(adv) for _, adv in self.segments)
        ):
            raise ValueError(f"RAFO segments are (start, adv) pairs with ascending starts from 0, got {self.segments}")
        if self.segments and self.kind in (KIND_REFLECT, KIND_TURN):
            raise ValueError("a reviewer turn is one segment")

    def to_metadata(self) -> dict:
        role = asdict(self)
        role["segments"] = [[start, adv] for start, adv in self.segments]
        return role

    @classmethod
    def of(cls, train_metadata: Mapping | None) -> RafoRole:
        role = (train_metadata or {}).get(METADATA_KEY)
        if not isinstance(role, Mapping) or not {"kind", "adv", "coef"} <= set(role) <= {"kind", "adv", "coef", "segments"}:
            raise ValueError(
                f"a RAFO sample carries train_metadata[{METADATA_KEY!r}] = {{kind, adv, coef[, segments]}} (written by "
                "miles.rollout.sprout.rafo), got "
                f"{role!r}; the loss sees it as batch['metadata'] under the Megatron backend"
            )
        segments = tuple((int(start), float(adv)) for start, adv in (role.get("segments") or ()))
        return cls(kind=role["kind"], adv=float(role["adv"]), coef=float(role["coef"]), segments=segments)
