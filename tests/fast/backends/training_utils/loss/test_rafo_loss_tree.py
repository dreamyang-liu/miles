"""The RAFO loss under tree credit: segment advantages over a sample's response positions, and a reviewer turn
trained with PPO (``turn``) instead of CE (``reflect``)."""

import pytest
import torch
from tests.fast.backends.training_utils.loss.loss_test_utils import make_parallel_state
from tests.fast.backends.training_utils.loss.test_rafo_loss import PROMPT_LENS, RESPONSE_LENS, _args, _batch, _inputs

from miles.backends.training_utils.loss.hub.rafo_loss import token_advantages
from miles.backends.training_utils.loss.objective import loss_function

#: Sample 0: a reviewed root, prefix advantage 0.75 up to position 3, remainder -0.25 from there. Sample 3: a turn.
TREE_METADATA = [
    {"rafo": {"kind": "root", "adv": -0.25, "coef": 0.25, "segments": [[0, 0.75], [3, -0.25]]}},
    {"rafo": {"kind": "root", "adv": -0.25, "coef": 0.25}},
    {"rafo": {"kind": "repair", "adv": -1.0, "coef": 0.125}},
    {"rafo": {"kind": "turn", "adv": 0.5, "coef": 0.5}},
]


def test_token_advantages_are_piecewise_constant_from_each_segment_start():
    adv = token_advantages(0.0, ((0, 0.75), (3, -0.25)), 6, dtype=torch.float32, device="cpu")
    assert adv.tolist() == pytest.approx([0.75, 0.75, 0.75, -0.25, -0.25, -0.25])
    assert token_advantages(0.4, (), 3, dtype=torch.float32, device="cpu").tolist() == pytest.approx([0.4] * 3)
    assert token_advantages(0.0, ((0, 1.0), (9, 2.0)), 4, dtype=torch.float32, device="cpu").tolist() == [1.0] * 4


def _reference(logits: torch.Tensor, inputs: dict, eps: float = 0.2, dual_c: float = 3.0) -> torch.Tensor:
    """The tree-credit objective per token: PPO everywhere, each token with its segment's advantage."""
    total, offset = logits.new_zeros(()), 0
    for i, (p, r) in enumerate(zip(PROMPT_LENS, RESPONSE_LENS, strict=True)):
        tokens = inputs["unconcat_tokens"][i]
        rows = torch.log_softmax(logits[0, offset + p - 1 : offset + p + r - 1].float(), dim=-1)
        new = rows.gather(1, tokens[p : p + r, None]).squeeze(1)
        mask = inputs["loss_masks"][i].bool()
        meta = TREE_METADATA[i]["rafo"]
        adv = token_advantages(meta["adv"], tuple(map(tuple, meta.get("segments", []))), r, dtype=new.dtype, device=new.device)
        ratio = torch.exp(torch.clamp(new - inputs["log_probs"][i], -20, 20))
        pg = torch.maximum(-ratio * adv, -ratio.clamp(1 - eps, 1 + eps) * adv)
        pg = torch.where(adv < 0, torch.minimum(pg, -dual_c * adv), pg)
        total = total + meta["coef"] * pg[mask].sum()
        offset += p + r
    return total


def test_segments_and_a_ppo_turn_match_the_per_token_objective() -> None:
    make_parallel_state()
    args = _args()
    inputs = _inputs(args)
    batch = _batch(inputs)
    batch["metadata"] = [dict(m) for m in TREE_METADATA]
    logits = inputs["policy_logits"].clone().requires_grad_(True)
    reference_logits = inputs["policy_logits"].clone().requires_grad_(True)

    loss, _, log = loss_function(args, batch, 1, logits, num_rollouts=2)
    expected = _reference(reference_logits, inputs) / 2
    loss.backward()
    expected.backward()

    torch.testing.assert_close(loss, expected, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(logits.grad, reference_logits.grad, rtol=1e-5, atol=1e-7)
    metrics = dict(zip(log["keys"], log["values"][1:].tolist(), strict=True))
    assert metrics["rafo_ce_tokens"] == 0, "no CE sample under tree credit"
    assert metrics["rafo_turn_tokens"] == RESPONSE_LENS[3] - 3
    assert metrics["rafo_ppo_tokens"] == sum(RESPONSE_LENS) - 2 - 3


def test_segments_need_context_parallel_one() -> None:
    make_parallel_state()
    args = _args(context_parallel_size=2)
    inputs = _inputs(args)
    batch = _batch(inputs)
    batch["metadata"] = [dict(m) for m in TREE_METADATA]
    with pytest.raises(ValueError, match="context parallel"):
        loss_function(args, batch, 1, inputs["policy_logits"].clone(), num_rollouts=1)


def _reference_shaped(logits: torch.Tensor, inputs: dict, eps: float = 0.2, dual_c: float = 3.0) -> torch.Tensor:
    """Like ``_reference``, but a turn's tokens weigh 2 pi_theta / (pi_old + 1) in place of the PPO ratio."""
    total, offset = logits.new_zeros(()), 0
    for i, (p, r) in enumerate(zip(PROMPT_LENS, RESPONSE_LENS, strict=True)):
        tokens = inputs["unconcat_tokens"][i]
        rows = torch.log_softmax(logits[0, offset + p - 1 : offset + p + r - 1].float(), dim=-1)
        new = rows.gather(1, tokens[p : p + r, None]).squeeze(1)
        mask = inputs["loss_masks"][i].bool()
        meta = TREE_METADATA[i]["rafo"]
        adv = token_advantages(meta["adv"], tuple(map(tuple, meta.get("segments", []))), r, dtype=new.dtype, device=new.device)
        old = inputs["log_probs"][i]
        if meta["kind"] == "turn":
            weight = 2 * torch.exp(new) / (torch.exp(old) + 1)
            total = total - meta["coef"] * (adv * weight)[mask].sum()
        else:
            ratio = torch.exp(torch.clamp(new - old, -20, 20))
            pg = torch.maximum(-ratio * adv, -ratio.clamp(1 - eps, 1 + eps) * adv)
            pg = torch.where(adv < 0, torch.minimum(pg, -dual_c * adv), pg)
            total = total + meta["coef"] * pg[mask].sum()
        offset += p + r
    return total


def test_a_shaped_turn_weighs_each_token_by_two_pi_over_pi_old_plus_one() -> None:
    make_parallel_state()
    args = _args()
    inputs = _inputs(args)
    inputs["log_probs"][3][4] = -6.0  # a turn token the policy finds unlikely: its weight is about 2 pi_theta
    batch = _batch(inputs)
    batch["metadata"] = [dict(m) for m in TREE_METADATA]
    batch["metadata"][3] = {"rafo": {**TREE_METADATA[3]["rafo"], "loss": "shaped"}}
    logits = inputs["policy_logits"].clone().requires_grad_(True)
    reference_logits = inputs["policy_logits"].clone().requires_grad_(True)

    loss, _, log = loss_function(args, batch, 1, logits, num_rollouts=2)
    expected = _reference_shaped(reference_logits, inputs) / 2
    loss.backward()
    expected.backward()

    torch.testing.assert_close(loss, expected, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(logits.grad, reference_logits.grad, rtol=1e-5, atol=1e-7)
    metrics = dict(zip(log["keys"], log["values"][1:].tolist(), strict=True))
    turn_tokens = RESPONSE_LENS[3] - 3
    assert metrics["rafo_turn_tokens"] == turn_tokens
    assert metrics["rafo_ppo_tokens"] == sum(RESPONSE_LENS[:3]) - 2, "the turn's tokens left the PPO set"
    assert 0 < metrics["rafo_turn_shape_weight"] < turn_tokens, "each weight 2 pi / (pi_old + 1) lies in (0, 1]"
    assert metrics["rafo_turn_shaped_loss"] < 0, "a positive advantage: the shaped loss is -coef * adv * weight"


def test_the_turn_loss_is_checked_on_the_role() -> None:
    make_parallel_state()
    args = _args()
    inputs = _inputs(args)
    batch = _batch(inputs)
    batch["metadata"] = [dict(m) for m in TREE_METADATA]
    batch["metadata"][3] = {"rafo": {**TREE_METADATA[3]["rafo"], "loss": "softmax"}}
    with pytest.raises(ValueError, match="turn loss"):
        loss_function(args, batch, 1, inputs["policy_logits"].clone(), num_rollouts=1)
    batch["metadata"][3] = dict(TREE_METADATA[3])
    batch["metadata"][0] = {"rafo": {**TREE_METADATA[0]["rafo"], "loss": "shaped"}}
    with pytest.raises(ValueError, match="only a reviewer turn"):
        loss_function(args, batch, 1, inputs["policy_logits"].clone(), num_rollouts=1)
