"""Imported Sprout samples through Miles's GRPO reward path, on CPU.

Standard GRPO needs nothing from the rollout function beyond rewards and
group identities: the group mean is subtracted, the group std divides, and
the result is broadcast over every response token the loss mask keeps.
"""

from argparse import Namespace

import pytest
import torch
from tests.fast.rollout.sprout.conftest import group, sprout_result

from miles.backends.training_utils.loss.hub.advantages import compute_advantages
from miles.ray.rollout.rollout_data_conversion import postprocess_rollout_data
from miles.ray.rollout.train_data_conversion import _post_process_rewards
from miles.rollout.sprout.importer import import_trajectories
from miles.rollout.sprout.protocol import RolloutResult


def imported_group(mask_generator, *, group_index, first_index, rewards):
    samples = group(len(rewards), group_index=group_index, first_index=first_index)
    body = sprout_result()
    template = body["trajectories"][0]
    body["trajectories"] = [
        {**template, "sample_slot_id": f"slot-{sample.index}", "branch_id": f"job-{sample.index}", "reward": reward}
        for sample, reward in zip(samples, rewards, strict=True)
    ]
    body.update(max_samples=len(rewards), actual_samples=len(rewards))
    return import_trajectories(
        RolloutResult.model_validate(body), {f"slot-{s.index}": s for s in samples}, mask_generator
    )


def grpo_args(**updates):
    return Namespace(
        **{
            "advantage_estimator": "grpo",
            "rewards_normalization": True,
            "grpo_std_normalization": True,
            "reward_key": None,
            "n_samples_per_prompt": 4,
            "rollout_batch_size": 2,
            "disable_rollout_trim_samples": False,
            "use_dynamic_global_batch_size": False,
            "global_batch_size": 8,
            **updates,
        }
    )


def test_rewards_are_normalized_within_each_sprout_group(mask_generator):
    mixed = imported_group(mask_generator, group_index=3, first_index=11, rewards=[1.0, 0.0, 1.0, 0.0])
    hopeless = imported_group(mask_generator, group_index=4, first_index=15, rewards=[0.0, 0.0, 0.0, 0.0])
    flat, metadata = postprocess_rollout_data(grpo_args(), [mixed, hopeless], train_parallel_config=None)
    assert len(flat) == 8 and metadata == {}
    raw, normalized = _post_process_rewards(grpo_args(), flat, None)
    assert raw == [1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    scale = 1 / (torch.tensor([1.0, 0.0, 1.0, 0.0]).std() + 1e-6)
    expected = [0.5 * scale, -0.5 * scale, 0.5 * scale, -0.5 * scale]
    assert normalized[:4] == pytest.approx([float(v) for v in expected])
    assert normalized[4:] == [0.0] * 4, "a group with one outcome carries no signal"


def test_grouping_follows_group_index_not_position(mask_generator):
    first = imported_group(mask_generator, group_index=3, first_index=11, rewards=[1.0, 0.0])
    second = imported_group(mask_generator, group_index=4, first_index=13, rewards=[1.0, 1.0])
    interleaved = [first[0], second[0], first[1], second[1]]
    _, normalized = _post_process_rewards(grpo_args(n_samples_per_prompt=2), interleaved, None)
    assert normalized[0] > 0 > normalized[2] and normalized[1] == normalized[3] == 0.0


def test_advantages_broadcast_the_group_reward_over_response_tokens(mask_generator):
    samples = imported_group(mask_generator, group_index=3, first_index=11, rewards=[1.0, 0.0, 1.0, 0.0])
    _, normalized = _post_process_rewards(grpo_args(n_samples_per_prompt=4, rollout_batch_size=1), samples, None)
    loss_masks = [torch.tensor(sample.loss_mask, dtype=torch.int32) for sample in samples]
    kl = [torch.zeros(len(mask), dtype=torch.float32) for mask in loss_masks]
    advantages, returns = compute_advantages(
        grpo_args(),
        kl=kl,
        rewards=normalized,
        log_probs=None,
        loss_masks=loss_masks,
        total_lengths=[len(sample.tokens) for sample in samples],
        response_lengths=[sample.response_length for sample in samples],
    )
    for sample, advantage, reward in zip(samples, advantages, normalized, strict=True):
        assert advantage.shape == (sample.response_length,)
        assert torch.allclose(advantage, torch.full_like(advantage, reward))
    assert all(torch.equal(a, r) for a, r in zip(advantages, returns, strict=True))
