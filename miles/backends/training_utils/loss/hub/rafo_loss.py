"""RAFO loss for ``--loss-type custom_loss``.

    --custom-loss-function-path miles.backends.training_utils.loss.hub.rafo_loss.rafo_loss_function

Each sample carries its ``RafoRole`` in ``Sample.train_metadata`` (``batch["metadata"]``): its kind, its local RLOO
advantage and its fixed coefficient, the token normalizer included. Root and repair samples get PPO with dual clip
on their loss-mask tokens; a reflect sample gets CE on the reviewer-turn tokens; a turn sample (tree credit) gets PPO
too, or with ``RafoRole.loss == "shaped"`` (``--sprout-rollout-rafo-turn-loss``) the weight 2 pi_theta / (pi_old + 1) per token in place of
the PPO ratio: its gradient is A * 2 pi_theta / (pi_old + 1) * grad log pi_theta, which fades out the tokens of the
reviewer's text that the policy finds unlikely instead of pulling on each with full force. The loss is a plain
weighted token sum: miles divides each optimizer step by its number of rollouts, one for each task, which supplies the 1/B of the
RAFO objective.
"""

from argparse import Namespace
from collections.abc import Callable

import torch

from miles.backends.training_utils.data.context_parallel import get_local_response_loss_masks
from miles.backends.training_utils.loss.hub.logit_processors import get_log_probs_and_entropy
from miles.backends.training_utils.loss.hub.math_utils import compute_policy_loss
from miles.rollout.sprout.rafo_role import KIND_REFLECT, KIND_TURN, RafoRole
from miles.utils.types import RolloutBatch


def rafo_loss_function(
    args: Namespace,
    batch: RolloutBatch,
    logits: torch.Tensor,
    sum_of_sample_mean: Callable[[torch.Tensor], torch.Tensor],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    del sum_of_sample_mean  # RAFO normalizes by fixed constants, not by per-sample token counts
    _check_supported(args)
    kinds, advantages, coefficients, segments, losses = _sample_columns(batch)
    if any(segments) and getattr(args, "context_parallel_size", 1) > 1:
        raise ValueError("RAFO tree credit's segments index a sample's response positions: use context parallel 1")
    if batch.get("log_probs") is None:
        raise ValueError(
            "the RAFO loss needs pi_old from the trainer (batch['log_probs']); keep the default advantage "
            "computation, which runs the old-policy pass (no --disable-compute-advantages-and-returns)"
        )
    total_lengths, response_lengths = batch["total_lengths"], batch["response_lengths"]
    max_seq_lens = batch.get("max_seq_lens", None)

    log_probs = get_log_probs_and_entropy(
        logits,
        args=args,
        unconcat_tokens=batch["unconcat_tokens"],
        total_lengths=total_lengths,
        response_lengths=response_lengths,
        with_entropy=False,
        max_seq_lens=max_seq_lens,
    )["log_probs"]
    masks = get_local_response_loss_masks(
        total_lengths, response_lengths, batch["loss_masks"], args.qkv_format, max_seq_lens
    )
    new = torch.cat(log_probs, dim=0)
    old = torch.cat([log_prob.detach() for log_prob in batch["log_probs"]], dim=0).to(new.device)
    lengths = torch.tensor([log_prob.numel() for log_prob in log_probs], device=new.device)
    active = torch.cat(masks, dim=0).to(device=new.device).bool()

    is_ce = torch.repeat_interleave(torch.tensor([k == KIND_REFLECT for k in kinds], device=new.device), lengths)
    is_turn = torch.repeat_interleave(torch.tensor([k == KIND_TURN for k in kinds], device=new.device), lengths)
    coef = torch.repeat_interleave(torch.tensor(coefficients, dtype=new.dtype, device=new.device), lengths)
    adv = torch.cat(
        [
            token_advantages(advantage, segs, int(length), dtype=new.dtype, device=new.device)
            for advantage, segs, length in zip(advantages, segments, lengths.tolist(), strict=True)
        ]
    ) if len(lengths) else new.new_zeros(0)
    is_shaped = torch.repeat_interleave(torch.tensor([lo == "shaped" for lo in losses], device=new.device), lengths)
    shaped_tokens = active & is_shaped  # a role allows ``shaped`` on a turn only
    ppo_tokens, ce_tokens = active & ~is_ce & ~shaped_tokens, active & is_ce
    zero = new.new_zeros(())

    ppo_kl = torch.where(ppo_tokens, old - new, zero)
    pg, clipfrac = compute_policy_loss(
        ppo_kl, torch.where(ppo_tokens, adv, zero), args.eps_clip, args.eps_clip_high, args.eps_clip_c
    )
    pg_loss = torch.where(ppo_tokens, coef * pg, zero).sum()
    ce_log_probs = torch.where(ce_tokens, new, zero)
    ce_loss = -(coef * ce_log_probs).sum()
    shape_weight = torch.where(shaped_tokens, 2 * torch.exp(new) / (torch.exp(old) + 1), zero)
    shaped_loss = -(coef * adv * shape_weight).sum()
    loss = pg_loss + ce_loss + shaped_loss
    if new.numel() == 0:
        # keep the graph connected on a rank without response tokens; fp32 sum avoids fp16 inf -> nan
        loss = loss + 0 * logits.sum(dtype=torch.float32)

    # Sums over this micro-batch's tokens; miles reports them divided by the step's rollout count, i.e. per task.
    log_ratio = -ppo_kl
    metrics = {
        "loss": loss.detach(),
        "rafo_pg_loss": pg_loss.detach(),
        "rafo_ce_loss": ce_loss.detach(),
        "rafo_ce_nll": -ce_log_probs.detach().sum(),
        "rafo_ppo_tokens": ppo_tokens.sum().to(new.dtype),
        "rafo_ce_tokens": ce_tokens.sum().to(new.dtype),
        "rafo_turn_tokens": (active & is_turn).sum().to(new.dtype),
        "rafo_turn_shaped_loss": shaped_loss.detach(),
        "rafo_turn_shape_weight": shape_weight.detach().sum(),  # / rafo_turn_tokens = the mean 2 pi / (pi_old + 1)
        "rafo_clipped_tokens": torch.where(ppo_tokens, clipfrac, zero).sum().detach(),
        "rafo_ppo_kl": torch.where(ppo_tokens, torch.expm1(log_ratio) - log_ratio, zero).sum().detach(),
    }
    return loss, metrics


def token_advantages(
    adv: float, segments: tuple[tuple[int, float], ...], length: int, *, dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    """The advantage of each of a sample's ``length`` response positions: ``adv`` everywhere, or under tree
    credit the segment's advantage from each segment's start on."""
    if not segments:
        return torch.full((length,), adv, dtype=dtype, device=device)
    out = torch.full((length,), segments[0][1], dtype=dtype, device=device)
    for start, segment_adv in segments[1:]:
        out[min(start, length) :] = segment_adv
    return out


def _sample_columns(
    batch: RolloutBatch,
) -> tuple[list[str], list[float], list[float], list[tuple[tuple[int, float], ...]], list[str]]:
    metadata = batch.get("metadata")
    if metadata is None:
        raise ValueError(
            "the RAFO loss reads each sample's train_metadata, which reaches it as batch['metadata'] only under "
            "the Megatron backend"
        )
    roles = [RafoRole.of(m) for m in metadata]
    return (
        [r.kind for r in roles],
        [r.adv for r in roles],
        [r.coef for r in roles],
        [r.segments for r in roles],
        [r.loss for r in roles],
    )


def _check_supported(args: Namespace) -> None:
    """Options that the policy loss would apply and this loss would silently drop."""
    unsupported = {
        "--use-kl-loss": args.use_kl_loss,
        "--entropy-coef != 0": args.entropy_coef != 0,
        "--use-tis": args.use_tis,
        "--get-mismatch-metrics": args.get_mismatch_metrics,
        "--use-opsm": args.use_opsm,
        "--custom-pg-loss-reducer-function-path": args.custom_pg_loss_reducer_function_path is not None,
        "--use-rollout-logprobs (RAFO takes pi_old from the trainer)": args.use_rollout_logprobs,
        "--skip-actor-forward-only (RAFO needs pi_old from the trainer)": args.skip_actor_forward_only,
        "--calculate-per-token-loss (RAFO uses fixed normalizers)": args.backend.calculate_per_token_loss,
    }
    if enabled := [name for name, on in unsupported.items() if on]:
        raise ValueError(f"the RAFO loss does not support {', '.join(enabled)}")
