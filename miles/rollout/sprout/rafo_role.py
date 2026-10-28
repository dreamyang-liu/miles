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
KINDS = (KIND_ROOT, KIND_REPAIR, KIND_REFLECT)
#: Where the role sits in ``Sample.train_metadata``.
METADATA_KEY = "rafo"


@dataclass(frozen=True)
class RafoRole:
    """``coef`` is the fixed coefficient of the sample's token sum, the token normalizer included; ``adv`` is
    the local RLOO advantage of a PPO sample and 0 for a CE sample."""

    kind: str
    adv: float
    coef: float

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f"unknown RAFO kind {self.kind!r}; expected one of {KINDS}")
        if not (math.isfinite(self.adv) and math.isfinite(self.coef) and self.coef >= 0):
            raise ValueError(f"a RAFO role needs a finite adv and a finite coef >= 0, got {self.adv}, {self.coef}")
        if self.kind == KIND_REFLECT and self.adv != 0:
            raise ValueError("a reviewer turn trains with CE: its adv is 0")

    def to_metadata(self) -> dict:
        return asdict(self)

    @classmethod
    def of(cls, train_metadata: Mapping | None) -> RafoRole:
        role = (train_metadata or {}).get(METADATA_KEY)
        if not isinstance(role, Mapping) or set(role) != {"kind", "adv", "coef"}:
            raise ValueError(
                f"a RAFO sample carries train_metadata[{METADATA_KEY!r}] = {{kind, adv, coef}} (written by "
                "miles.rollout.sprout.rafo), got "
                f"{role!r}; the loss sees it as batch['metadata'] under the Megatron backend"
            )
        return cls(kind=role["kind"], adv=float(role["adv"]), coef=float(role["coef"]))
