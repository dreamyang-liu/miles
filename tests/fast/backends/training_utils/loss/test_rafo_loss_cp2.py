"""The RAFO loss summed over 2 context-parallel ranks equals the loss without context parallelism."""

from functools import partial

import torch
import torch.distributed as dist
from tests.ci.ci_register import register_cpu_ci
from tests.fast.backends.training_utils.loss.loss_test_utils import make_args, make_batch, make_inputs
from tests.fast.dist_utils import init_gloo, run_multiprocess

from miles.backends.training_utils.data.context_parallel import slice_log_prob_with_cp
from miles.backends.training_utils.loss.objective import loss_function
from miles.backends.training_utils.parallel import GroupInfo, ParallelState, set_parallel_state
from miles.utils.args.custom_function import CustomFunctionConfig
from miles.utils.args.custom_view import ImmutableNamespace

register_cpu_ci(est_time=15, suite="stage-a-cpu", labels=[])

# totals divide by 2 * cp for the zigzag layout
PROMPT_LENS = [6, 2, 5, 3]
RESPONSE_LENS = [6, 6, 7, 5]
METADATA = [
    {"rafo": {"kind": "root", "adv": 0.5, "coef": 0.25}},
    {"rafo": {"kind": "repair", "adv": -1.0, "coef": 0.125}},
    {"rafo": {"kind": "reflect", "adv": 0.0, "coef": 0.5}},
    {"rafo": {"kind": "root", "adv": -0.5, "coef": 0.25}},
]


def _args(*, allgather_cp: bool, true_on_policy_mode: bool) -> ImmutableNamespace:
    config = CustomFunctionConfig(path="miles.backends.training_utils.loss.hub.rafo_loss.rafo_loss_function")
    return ImmutableNamespace(
        **dict(
            make_args(
                loss_type="custom_loss",
                custom_loss_function_path=config,
                entropy_coef=0.0,
                eps_clip_c=3.0,
                allgather_cp=allgather_cp,
                true_on_policy_mode=true_on_policy_mode,
            )
        )
    )


def _set_parallel_state(*, rank: int, world_size: int, tp_group) -> None:
    trivial = GroupInfo(rank=0, size=1, group=None)
    cp = GroupInfo(rank=rank, size=world_size, group=dist.group.WORLD if world_size > 1 else None)
    set_parallel_state(
        ParallelState(
            intra_dp=trivial,
            intra_dp_cp=cp,
            cp=cp,
            tp=GroupInfo(rank=0, size=1, group=tp_group),
            pp=trivial,
            ep=trivial,
            etp=trivial,
            indep_dp=trivial,
            is_pp_last_stage=True,
        )
    )


def _batch(inputs: dict, *, local_old_log_probs: bool) -> dict:
    batch = make_batch(inputs, "policy_loss")
    batch["metadata"] = [dict(m) for m in METADATA]
    batch["loss_masks"][2][:2] = 0
    if local_old_log_probs:  # the trainer computes pi_old in the same local CP layout as pi_theta
        batch["log_probs"] = [
            slice_log_prob_with_cp(log_prob, total, response)
            for log_prob, total, response in zip(
                batch["log_probs"], batch["total_lengths"], RESPONSE_LENS, strict=True
            )
        ]
    return batch


def _run(rank: int, world_size: int, port: int) -> None:
    init_gloo(rank, world_size, port=port)
    # the vocab-parallel cross entropy needs a real group per rank; new_group is collective, so build all of them
    tp_group = [dist.new_group([r]) for r in range(world_size)][rank]
    for true_on_policy_mode in (True, False):
        _check_cp_matches_cp1(rank, world_size, tp_group, true_on_policy_mode=true_on_policy_mode)


def _check_cp_matches_cp1(rank: int, world_size: int, tp_group, *, true_on_policy_mode: bool) -> None:
    base_args = _args(allgather_cp=False, true_on_policy_mode=true_on_policy_mode)
    cp_args = _args(allgather_cp=True, true_on_policy_mode=true_on_policy_mode)
    inputs = make_inputs(42, len(PROMPT_LENS), PROMPT_LENS, RESPONSE_LENS, 64, base_args)
    logits = inputs["policy_logits"]
    t_local = logits.size(1) // world_size

    _set_parallel_state(rank=0, world_size=1, tp_group=tp_group)
    base_loss, _, base_log = loss_function(base_args, _batch(inputs, local_old_log_probs=False), 1, logits)

    _set_parallel_state(rank=rank, world_size=world_size, tp_group=tp_group)
    local_logits = logits[:, rank * t_local : (rank + 1) * t_local]
    cp_loss, _, cp_log = loss_function(cp_args, _batch(inputs, local_old_log_probs=True), 1, local_logits)

    summed = torch.stack([cp_loss.detach(), *cp_log["values"][1:]])
    dist.all_reduce(summed)
    expected = torch.stack([base_loss.detach(), *base_log["values"][1:]])
    torch.testing.assert_close(summed, expected, rtol=1e-5, atol=1e-6)


def test_rafo_loss_under_allgather_cp2_matches_cp1() -> None:
    run_multiprocess(partial(_run))
