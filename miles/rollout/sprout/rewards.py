"""Reward post-processing for Sprout search rollouts (``--custom-reward-post-process-path``).

A search step mixes three kinds of samples, told apart by
``train_metadata["sprout_rollout"]["role"]``:

``root``
    scratch rollouts: GRPO, normalized within their prompt group as usual;
``branch``
    student and repair continuations, grouped by the condition they were
    sampled under (``Trajectory.group``): GRPO within the group, then scaled
    by ``--sprout-rollout-branch-advantage-scale`` (0 keeps them out of the
    batch altogether, see ``arrange_search_samples``);
``distill``
    a verified repair turn in its one-shot context: its reward is the weight
    the verification gave it and it is the advantage, not something to
    normalize against a group it is the only member of.

    --custom-reward-post-process-path miles.rollout.sprout.rewards.post_process_rewards
"""

from __future__ import annotations

from argparse import Namespace

from miles.ray.rollout.train_data_conversion import _normalize_rewards_by_rollout
from miles.utils.types import Sample


def role_of(sample: Sample) -> str:
    return ((sample.train_metadata or {}).get("sprout_rollout") or {}).get("role", "root")


def post_process_rewards(args: Namespace, samples: list[Sample]) -> tuple[list[float], list[float]]:
    raw = [sample.get_reward_value(args) for sample in samples]
    normalized = list(raw)
    grouped = [index for index, sample in enumerate(samples) if role_of(sample) != "distill"]
    if grouped and getattr(args, "rewards_normalization", True):
        values = _normalize_rewards_by_rollout(args, [samples[i] for i in grouped], [raw[i] for i in grouped], None)
        for index, value in zip(grouped, values, strict=True):
            normalized[index] = value
    scale = getattr(args, "sprout_rollout_branch_advantage_scale", 1.0)
    for index, sample in enumerate(samples):
        if role_of(sample) == "branch":
            normalized[index] *= scale
    return raw, normalized
