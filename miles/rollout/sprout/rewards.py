"""GSML advantages for Sprout search rollouts (``--custom-reward-post-process-path``).

    --custom-reward-post-process-path miles.rollout.sprout.rewards.post_process_gsml

``gsml.assemble_search_group`` gives every sample of a search group its credit,
``train_metadata["sprout_rollout"]["credit"]``: which coefficient its z-score
within its group takes, the z-score, and its share of its task's λ mass. The
coefficients are applied here, when the step is trained, and nowhere else, so a
saved rollout (``--load-debug-rollout-data``) can be re-credited under new
flags:

    A = β·z + λ·(ψ for an â sample, else 1)·lambda_share

with β = (1 - λ)/√2 for a root (a split pair of roots gets ±(1 - λ)/2),
``--sprout-rollout-gsml-beta-branch`` for a student or repair continuation and
0 for an â sample; λ is ``--sprout-rollout-gsml-lambda`` and ψ
``--sprout-rollout-gsml-psi``. A is every trainable token's advantage as it
stands: nothing normalizes it again (``--normalize-advantages`` is refused),
and under ``--calculate-per-token-loss`` each of those tokens weighs alike.
"""

from __future__ import annotations

import math
from argparse import Namespace

from miles.rollout.sprout.gsml import Credit
from miles.utils.types import Sample


def post_process_gsml(args: Namespace, samples: list[Sample]) -> tuple[list[float], list[float]]:
    """Each sample's reward and its GSML advantage A."""
    lam, betas, psi = coefficients(args)
    raw, advantages = [], []
    for sample in samples:
        credit = credit_of(sample)
        advantage = betas[credit.beta] * credit.z + lam * (psi if credit.kind == "ahat" else 1.0) * credit.lambda_share
        if credit.kind == "ahat":
            assert advantage >= 0, f"an â sample is only ever reinforced, got A={advantage} for sample {sample.index}"
        raw.append(sample.get_reward_value(args))
        advantages.append(advantage)
    return raw, advantages


def coefficients(args: Namespace) -> tuple[float, dict[str | None, float], float]:
    """λ, the β of each credit's ``beta`` and ψ, from the flags.

    Checked here as well as at construction: a re-credited saved rollout
    never builds the rollout function.
    """
    lam = _gsml_flag(args, "lambda", at_most=1.0)
    beta_branch, psi = _gsml_flag(args, "beta_branch"), _gsml_flag(args, "psi")
    for name in ("c_pre", "kappa_plus"):
        if getattr(args, f"sprout_rollout_gsml_{name}", 0.0) != 0:
            raise ValueError(f"--sprout-rollout-gsml-{name.replace('_', '-')} is not implemented: it must be 0")
    return lam, {"root": (1.0 - lam) / math.sqrt(2.0), "branch": beta_branch, None: 0.0}, psi


def credit_of(sample: Sample) -> Credit:
    """The credit ``gsml.assemble_search_group`` gave the sample."""
    lineage = (sample.train_metadata or {}).get("sprout_rollout") or {}
    if "credit" not in lineage:
        raise ValueError(
            f"sample {sample.index} carries no GSML credit: post_process_gsml trains Sprout search groups, "
            "credited by miles.rollout.sprout.gsml.assemble_search_group"
        )
    credit = Credit.from_dict(lineage["credit"])
    if lineage.get("role") != credit.kind:
        raise ValueError(f"sample {sample.index} is a {lineage.get('role')!r} credited as a {credit.kind!r}")
    return credit


def _gsml_flag(args: Namespace, name: str, *, at_most: float = math.inf) -> float:
    value = getattr(args, f"sprout_rollout_gsml_{name}", None)
    if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= at_most:
        bound = f"in [0, {at_most:g}]" if math.isfinite(at_most) else "nonnegative"
        raise ValueError(f"--sprout-rollout-gsml-{name.replace('_', '-')} must be finite and {bound}, got {value!r}")
    return float(value)
