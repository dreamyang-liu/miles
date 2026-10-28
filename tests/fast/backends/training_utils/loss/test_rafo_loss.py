import pytest
import torch
from tests.fast.backends.training_utils.loss.loss_test_utils import (
    make_args,
    make_batch,
    make_inputs,
    make_parallel_state,
)
from tests.fast.backends.training_utils.loss.test_training_logprob_reuse import process_group

from miles.backends.training_utils.loss.objective import loss_function
from miles.utils.args.custom_function import CustomFunctionConfig
from miles.utils.args.custom_view import ImmutableNamespace

LOSS_PATH = "miles.backends.training_utils.loss.hub.rafo_loss.rafo_loss_function"
VOCAB = 32
PROMPT_LENS = [5, 3, 4, 6]
RESPONSE_LENS = [6, 4, 7, 5]
METADATA = [
    {"rafo": {"kind": "root", "adv": 0.75, "coef": 0.25}},
    {"rafo": {"kind": "root", "adv": -0.25, "coef": 0.25}},
    {"rafo": {"kind": "repair", "adv": -1.0, "coef": 0.125}},
    {"rafo": {"kind": "reflect", "adv": 0.0, "coef": 0.5}},
]


def _args(**overrides) -> ImmutableNamespace:
    values = dict(
        loss_type="custom_loss",
        custom_loss_function_path=CustomFunctionConfig(path=LOSS_PATH),
        entropy_coef=0.0,
        eps_clip=0.2,
        eps_clip_high=0.2,
        eps_clip_c=3.0,
    )
    return ImmutableNamespace(**dict(make_args(**(values | overrides))))


def _inputs(args: ImmutableNamespace) -> dict:
    inputs = make_inputs(
        seed=7, batch_size=4, prompt_lens=PROMPT_LENS, response_lens=RESPONSE_LENS, vocab_size=VOCAB, args=args
    )
    inputs["loss_masks"][0][:2] = 0  # e.g. tool-output tokens inside a root
    inputs["loss_masks"][3][:3] = 0  # the CE sample trains only its reviewer turn
    inputs["log_probs"][2][1] = -30.0  # pi_old far below pi_theta: rho is huge on a negative-advantage token
    return inputs


def _batch(inputs: dict) -> dict:
    batch = make_batch(inputs, "policy_loss")
    batch["metadata"] = [dict(m) for m in METADATA]
    return batch


def _reference_loss(logits: torch.Tensor, inputs: dict, eps: float = 0.2, dual_c: float = 3.0) -> torch.Tensor:
    """The RAFO objective written out per token from the packed logits, independent of the miles helpers."""
    total, offset = logits.new_zeros(()), 0
    for i, (p, r) in enumerate(zip(PROMPT_LENS, RESPONSE_LENS, strict=True)):
        tokens = inputs["unconcat_tokens"][i]
        rows = torch.log_softmax(logits[0, offset + p - 1 : offset + p + r - 1].float(), dim=-1)
        new = rows.gather(1, tokens[p : p + r, None]).squeeze(1)
        mask = inputs["loss_masks"][i].bool()
        meta = METADATA[i]
        if meta["rafo"]["kind"] == "reflect":
            per_token = -new
        else:
            ratio = torch.exp(torch.clamp(new - inputs["log_probs"][i], -20, 20))
            adv = meta["rafo"]["adv"]
            pg = torch.maximum(-ratio * adv, -ratio.clamp(1 - eps, 1 + eps) * adv)
            per_token = torch.minimum(pg, torch.full_like(pg, -dual_c * adv)) if adv < 0 else pg
        total = total + meta["rafo"]["coef"] * per_token[mask].sum()
        offset += p + r
    return total


__all__ = ["process_group"]


class TestRafoLoss:
    @pytest.mark.parametrize("true_on_policy_mode", [True, False], ids=["full_softmax", "vocab_parallel_ce"])
    def test_loss_and_gradient_match_the_per_token_objective(self, process_group, true_on_policy_mode) -> None:
        """Fixed coefficients, PPO with dual clip on agent tokens, CE on reviewer tokens; miles divides by B.
        Megatron training computes log-probs with the vocab-parallel cross entropy (true_on_policy_mode off)."""
        make_parallel_state()
        args = _args(true_on_policy_mode=true_on_policy_mode)
        inputs = _inputs(args)
        logits = inputs["policy_logits"].clone().requires_grad_(True)
        reference_logits = inputs["policy_logits"].clone().requires_grad_(True)

        loss, _, log = loss_function(args, _batch(inputs), 1, logits, num_rollouts=2)
        expected = _reference_loss(reference_logits, inputs) / 2
        loss.backward()
        expected.backward()

        torch.testing.assert_close(loss, expected, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(logits.grad, reference_logits.grad, rtol=1e-5, atol=1e-7)
        metrics = dict(zip(log["keys"], log["values"][1:].tolist(), strict=True))
        assert metrics["rafo_ce_tokens"] == RESPONSE_LENS[3] - 3
        assert metrics["rafo_ppo_tokens"] == RESPONSE_LENS[0] - 2 + RESPONSE_LENS[1] + RESPONSE_LENS[2]
        assert metrics["loss"] == pytest.approx(expected.item() * 2, rel=1e-5)

    def test_dual_clip_bounds_the_loss_of_a_negative_advantage_token(self) -> None:
        """Token 1 of the repair sample has rho ~ e^27 and A = -1: its loss is c * |A| * coef, not rho * |A|."""
        make_parallel_state()
        args = _args()
        inputs = _inputs(args)
        logits = inputs["policy_logits"].clone().requires_grad_(True)

        loss, _, _ = loss_function(args, _batch(inputs), 1, logits, num_rollouts=1)

        assert torch.isfinite(loss) and loss.item() < 100.0

    def test_a_sample_without_rafo_metadata_is_rejected(self) -> None:
        make_parallel_state()
        args = _args()
        inputs = _inputs(args)
        batch = _batch(inputs)
        del batch["metadata"]

        with pytest.raises(ValueError, match="train_metadata"):
            loss_function(args, batch, 1, inputs["policy_logits"].clone(), num_rollouts=1)

    def test_a_batch_without_trainer_pi_old_is_rejected(self) -> None:
        make_parallel_state()
        args = _args()
        inputs = _inputs(args)
        batch = _batch(inputs)
        batch["log_probs"] = None

        with pytest.raises(ValueError, match="pi_old"):
            loss_function(args, batch, 1, inputs["policy_logits"].clone(), num_rollouts=1)

    @pytest.mark.parametrize(
        "overrides",
        [
            {"use_kl_loss": True},
            {"entropy_coef": 0.01},
            {"use_tis": True},
            {"use_rollout_logprobs": True},
            {"calculate_per_token_loss": True},
        ],
    )
    def test_options_the_loss_would_ignore_are_rejected(self, overrides: dict) -> None:
        make_parallel_state()
        args = _args(**overrides)
        inputs = _inputs(args)

        with pytest.raises(ValueError, match="RAFO"):
            loss_function(args, _batch(inputs), 1, inputs["policy_logits"].clone(), num_rollouts=1)
