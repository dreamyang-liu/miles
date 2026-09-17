from types import SimpleNamespace

import pytest
import torch

from miles.backends.training_utils import cp_utils
from miles.backends.training_utils import loss as loss_module
from miles.ray.rollout.rollout_data_conversion import postprocess_rollout_data
from miles.ray.rollout.train_data_conversion import (
    _task_equal_loss_weights,
    convert_samples_to_train_data,
    split_train_data_by_dp_scheduled_raw,
)
from miles.utils.types import Sample
from tests.fast.ray.rollout.conftest import make_args


TRAIN_CONFIG = {"dp_size": 1, "cp_size": 4, "vpp_size": 1, "microbatch_group_size_per_vp_stage": 1}


def task_groups(sizes):
    return [
        [
            Sample(
                index=group * 8 + child, group_index=group, prompt=f"task-{group}",
                tokens=list(range(16 + 8 * child)), response_length=8 + 8 * child,
                loss_mask=[1] * (8 + 8 * child), reward=float(child % 2),
                status=Sample.Status.COMPLETED,
            )
            for child in range(size)
        ]
        for group, size in enumerate(sizes)
    ]


def arguments(weighting="task", task_count=8):
    return make_args(
        ash_rollout_branching=True, ash_rollout_branching_return_mode="all",
        ash_rollout_loss_weighting=weighting, rollout_batch_size=task_count, n_samples_per_prompt=8,
        global_batch_size=task_count * 8, use_dynamic_global_batch_size=True,
        use_dynamic_batch_size=False, micro_batch_size=2, calculate_per_token_loss=False,
        grpo_std_normalization=True, balance_by_flops=False, balance_data=True,
    )


@pytest.mark.parametrize("weighting", ["task", "trajectory"])
@pytest.mark.parametrize("task_count", [8, 32])
def test_tasks_keep_all_odd_trajectory_count_and_one_optimizer_step(weighting, task_count):
    sizes = [1, 2, 3, 4, 1, 5, 2, 5] if task_count == 8 else list(range(1, 9)) * 4
    if task_count == 32:
        sizes[-1] -= 1
    n = sum(sizes)
    microbatches = (n + 1) // 2
    groups = task_groups(sizes)
    args = arguments(weighting, task_count)
    samples, metadata = postprocess_rollout_data(args, groups, TRAIN_CONFIG)
    assert len(samples) == n == (23 if task_count == 8 else 143)
    assert metadata["dynamic_global_batch_size"] == n
    assert metadata["ash_task_count"] == task_count
    data = convert_samples_to_train_data(args, samples, metadata, None, None)
    (shard,) = split_train_data_by_dp_scheduled_raw(args, data, train_parallel_config=TRAIN_CONFIG)
    assert shard["num_rollouts"] == [n]
    assert shard["num_microbatches"] == [microbatches]
    assert sorted(map(len, shard["micro_batch_indices"])) == [1] + [2] * (microbatches - 1)
    assert sorted(shard["sample_indices"]) == sorted(sample.index for sample in samples)
    assert len(set(shard["sample_indices"])) == n
    assert sorted(i for batch in shard["micro_batch_indices"] for i in batch) == list(range(n))
    for group in groups:
        raw = torch.tensor([sample.reward for sample in group])
        expected = raw - raw.mean()
        if len(raw) > 1 and raw.std() > 0:
            expected /= raw.std() + 1e-6
        positions = [shard["sample_indices"].index(sample.index) for sample in group]
        torch.testing.assert_close(torch.tensor([shard["rewards"][i] for i in positions]), expected)
        if weighting == "task":
            assert sum(shard["sample_loss_weights"][i] for i in positions) == pytest.approx(n / task_count)
    if weighting == "trajectory":
        assert "sample_loss_weights" not in shard


@pytest.mark.parametrize("dp_size,vpp_size", [(2, 1), (1, 2)])
def test_variable_tasks_fail_instead_of_trimming_for_unsupported_parallel_layout(dp_size, vpp_size):
    with pytest.raises(ValueError, match="DP1 and VPP1"):
        postprocess_rollout_data(
            arguments(), task_groups([1, 2, 3, 4, 1, 5, 2, 5]),
            TRAIN_CONFIG | {"dp_size": dp_size, "vpp_size": vpp_size},
        )


@pytest.mark.parametrize("cp_size", [1, 4])
def test_task_equal_gradient_is_invariant_to_microbatch_split_and_context_parallelism(monkeypatch, cp_size):
    sizes = [1, 2, 4]
    samples = [sample for group in task_groups(sizes) for sample in group]
    weights = _task_equal_loss_weights(samples)
    masks = [torch.tensor([1.0] + [float(i % 3 != 0) for i in range(1, sample.response_length)])
             for sample in samples]
    masks[0][1:] = 0
    original_masks = [mask.clone() for mask in masks]
    values = [torch.linspace(-1, 2, sample.response_length, requires_grad=True) for sample in samples]
    reference = [value.detach().clone().requires_grad_() for value in values]
    expected = sum(
        sum((reference[i] * masks[i]).sum() / masks[i].sum() for i, sample in enumerate(samples)
            if sample.group_index == group) / size
        for group, size in enumerate(sizes)
    ) / len(sizes)
    expected.backward()
    accumulated = 0.0
    for rank in range(cp_size):
        monkeypatch.setattr(
            cp_utils, "get_parallel_state",
            lambda rank=rank: SimpleNamespace(cp=SimpleNamespace(size=cp_size, rank=rank)),
        )
        for start in range(0, len(samples), 2):
            indices = list(range(start, min(start + 2, len(samples))))
            reducer = cp_utils.get_sum_of_sample_mean(
                [len(samples[i].tokens) for i in indices],
                [samples[i].response_length for i in indices],
                [masks[i] for i in indices],
                denominators=[masks[i].sum() for i in indices],
                sample_weights=[weights[i] for i in indices],
            )
            pieces = [
                values[i] if cp_size == 1 else cp_utils._slice_loss_mask_for_local_cp(
                    len(samples[i].tokens), samples[i].response_length, values[i], "thd", None,
                )
                for i in indices
            ]
            loss = reducer(torch.cat(pieces)) / len(samples)
            accumulated += loss.detach().item()
            loss.backward()
    assert accumulated == pytest.approx(expected.detach().item(), abs=1e-6)
    for actual, target, mask, original_mask in zip(values, reference, masks, original_masks, strict=True):
        torch.testing.assert_close(actual.grad, target.grad)
        torch.testing.assert_close(mask, original_mask)


def test_learner_uses_actual_count_and_task_weights_for_an_odd_batch(monkeypatch):
    samples = [sample for group in task_groups([1, 2, 4]) for sample in group]
    weights = _task_equal_loss_weights(samples)
    state = SimpleNamespace(
        cp=SimpleNamespace(size=1, rank=0), intra_dp=SimpleNamespace(size=1),
        intra_dp_cp=SimpleNamespace(size=1), is_ulysses_cp=False,
    )
    monkeypatch.setattr(cp_utils, "get_parallel_state", lambda: state)
    monkeypatch.setattr(loss_module, "get_parallel_state", lambda: state)

    def measured_loss(args, batch, logits, reducer):
        value = reducer(logits)
        return value, {"loss": value.detach()}

    monkeypatch.setattr(loss_module, "get_loss_function", lambda _: measured_loss)
    args = arguments()
    args.qkv_format = "thd"
    args.recompute_loss_function = False
    args.allgather_cp = False
    args.true_on_policy_mode = False
    values = [torch.tensor([float(i + 1)], requires_grad=True) for i in range(7)]
    total = 0
    for start in range(0, 7, 2):
        indices = list(range(start, min(start + 2, 7)))
        batch = {
            "total_lengths": [2] * len(indices), "response_lengths": [1] * len(indices),
            "loss_masks": [torch.ones(1) for _ in indices],
            "sample_loss_weights": [weights[i] for i in indices],
            "dynamic_global_batch_size": 7,
        }
        scaled, _, _ = loss_module.loss_function(
            args, batch, num_microbatches=4, logits=torch.cat([values[i] for i in indices]),
            apply_megatron_loss_scaling=True, num_rollouts=7,
        )
        loss = scaled / 4  # Megatron's gradient-accumulation divisor.
        total += loss.detach().item()
        loss.backward()
    assert total == pytest.approx((1 + (2 + 3) / 2 + (4 + 5 + 6 + 7) / 4) / 3)
    assert [value.grad.item() for value in values] == pytest.approx([1 / 3, 1 / 6, 1 / 6] + [1 / 12] * 4)
