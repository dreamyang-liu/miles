"""Imported Sprout samples through Miles's GRPO reward path, on CPU.

Standard GRPO needs nothing from the rollout function beyond rewards and
group identities: the group mean is subtracted, the group std divides, and
the result is broadcast over every response token the loss mask keeps. A
search group's advantages are GSML's (``rewards.post_process_gsml``), taken
as they are and summed over the trainable tokens.
"""

from argparse import Namespace
from types import SimpleNamespace

import pytest
import torch
from tests.fast.rollout.sprout.conftest import assemble, ff_example, group, gsml_args, sprout_result

from miles.backends.training_utils.data import context_parallel
from miles.backends.training_utils.data.context_parallel import get_sum_of_sample_mean
from miles.backends.training_utils.loss.hub.advantages import compute_advantages
from miles.ray.rollout.rollout_data_conversion import postprocess_rollout_data
from miles.ray.rollout.train_data_conversion import _post_process_rewards
from miles.rollout.sprout.importer import import_trajectories
from miles.rollout.sprout.protocol import RolloutResult
from miles.rollout.sprout.rewards import post_process_gsml


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


def gsml_step(mask_generator):
    """The shared FF example as the trainer gets it: one dynamic batch, advantages from GSML."""
    samples, _ = assemble(mask_generator, *ff_example())
    namespace = Namespace(**{**vars(grpo_args(use_dynamic_global_batch_size=True)), **vars(gsml_args())})
    flat, _ = postprocess_rollout_data(namespace, [samples], train_parallel_config={"dp_size": 1})
    _, advantages = _post_process_rewards(namespace, flat, post_process_gsml)
    return flat, advantages


def test_advantages_broadcast_each_search_samples_gsml_a_over_its_response(mask_generator):
    samples, rewards = gsml_step(mask_generator)
    loss_masks = [torch.tensor(sample.loss_mask, dtype=torch.int32) for sample in samples]
    advantages, _ = compute_advantages(
        grpo_args(),
        kl=[torch.zeros(len(mask), dtype=torch.float32) for mask in loss_masks],
        rewards=rewards,
        log_probs=None,
        loss_masks=loss_masks,
        total_lengths=[len(sample.tokens) for sample in samples],
        response_lengths=[sample.response_length for sample in samples],
    )
    (ahat,) = [i for i, s in enumerate(samples) if s.train_metadata["sprout_rollout"]["role"] == "ahat"]
    assert advantages[ahat].shape == (samples[ahat].response_length,)
    assert torch.allclose(advantages[ahat], torch.full_like(advantages[ahat], 0.2)), "λ·ψ·ω/|E| on every position"
    for advantage, reward in zip(advantages, rewards, strict=True):
        assert torch.allclose(advantage, torch.full_like(advantage, reward)), "A as it is, nothing renormalized"


def test_a_token_sum_loss_weighs_each_sample_by_its_trainable_tokens(mask_generator, monkeypatch):
    monkeypatch.setattr(context_parallel, "get_parallel_state", lambda: SimpleNamespace(cp=SimpleNamespace(size=1)))
    samples, rewards = gsml_step(mask_generator)
    loss_masks = [torch.tensor(sample.loss_mask, dtype=torch.float32) for sample in samples]
    per_token = torch.cat([torch.full((s.response_length,), a) for s, a in zip(samples, rewards, strict=True)])
    token_sum = get_sum_of_sample_mean(
        [len(sample.tokens) for sample in samples],
        [sample.response_length for sample in samples],
        loss_masks,
        calculate_per_token_loss=True,
    )(per_token)
    expected = sum(a * sum(s.loss_mask) for s, a in zip(samples, rewards, strict=True))
    assert token_sum.item() == pytest.approx(expected, rel=1e-6), "Σ_n A_n·|trainable tokens_n|"
    assert any(0 in s.loss_mask for s in samples), "a token outside the loss mask weighs nothing"
