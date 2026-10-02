"""Training samples from Sprout's hint-free messages.

Sprout returns the conversation the agent actually had, minus the branch hints
it marked, and says nothing about tokens: Miles renders the messages with the
policy's own chat template, so the tokens, the loss mask and later the
log-probs all describe one sequence the trainer can score. Nothing from the
sampling process survives on purpose -- a hint-conditioned rollout's token
log-probs would not describe this sequence.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from copy import deepcopy

from miles.rollout.sprout.protocol import HINT_END, HINT_START, RolloutResult, Trajectory
from miles.utils.mask_utils import MultiTurnLossMaskGenerator
from miles.utils.types import Sample

SAMPLE_STATUS = {"completed": Sample.Status.COMPLETED, "truncated": Sample.Status.TRUNCATED}
ROLES = {"system", "user", "assistant", "tool"}


def import_trajectories(
    result: RolloutResult,
    slot_samples: Mapping[str, Sample],
    mask_generator: MultiTurnLossMaskGenerator,
    *,
    weight_version: int | None = None,
) -> list[Sample]:
    """One Miles sample per returned trajectory, in the slots Miles allocated."""
    if result.status not in {"completed", "early_stopped"}:
        raise ValueError(f"cannot import a rollout group with status={result.status!r}: {result.stop_reason}")
    if result.max_samples > len(slot_samples):
        raise ValueError("the driver's max_samples exceeds the slots Miles allocated")
    samples = []
    for trajectory in result.trajectories:
        if trajectory.sample_slot_id not in slot_samples:
            raise ValueError(f"unknown sample_slot_id: {trajectory.sample_slot_id!r}")
        samples.append(
            _import_trajectory(
                result, trajectory, slot_samples[trajectory.sample_slot_id], mask_generator, weight_version
            )
        )
    return samples


def prepare_messages(trajectory: Trajectory) -> list[dict]:
    """The trajectory's messages as the chat template wants them, checked."""
    messages = deepcopy(trajectory.messages)
    pending: set[str] = set()
    seen: set[str] = set()
    for message in messages:
        role = message.get("role")
        if role not in ROLES:
            raise ValueError(f"unsupported message role {role!r}")
        if role in {"user", "system"} and any(
            marker in (message.get("content") or "") for marker in (HINT_START, HINT_END)
        ):
            raise ValueError("Sprout returned an unremoved training hint")
        for call in message.get("tool_calls") or []:
            if role != "assistant":
                raise ValueError("only assistant messages carry tool calls")
            identifier = call.get("id")
            if not identifier or identifier in seen:
                raise ValueError("tool calls need distinct ids")
            seen.add(identifier)
            pending.add(identifier)
            function = call.get("function") or {}
            arguments = function.get("arguments", {})
            if isinstance(arguments, str):
                arguments = json.loads(arguments)
            if not isinstance(arguments, dict):
                raise ValueError("tool arguments must decode to a JSON object")
            # Chat Completions carries arguments as a JSON string; HF tool
            # templates render a mapping.
            function["arguments"] = arguments
        if role == "tool":
            if message.get("tool_call_id") not in pending:
                raise ValueError(f"unknown tool_call_id {message.get('tool_call_id')!r}")
            pending.remove(message["tool_call_id"])
    if pending:
        raise ValueError("the conversation ends with unanswered tool calls")
    if not any(message["role"] == "assistant" for message in messages):
        raise ValueError("a trajectory needs at least one assistant message")
    return messages


def _import_trajectory(
    result: RolloutResult,
    trajectory: Trajectory,
    slot_sample: Sample,
    mask_generator: MultiTurnLossMaskGenerator,
    weight_version: int | None,
) -> Sample:
    messages = prepare_messages(trajectory)
    token_ids, full_mask = mask_generator.get_loss_mask(messages, tools=trajectory.tools or None)
    if len(token_ids) != len(full_mask) or not any(full_mask):
        raise ValueError("the rendered trajectory has no trainable assistant tokens")
    response_length = mask_generator.get_response_lengths([full_mask])[0]
    first_assistant = next(i for i, message in enumerate(messages) if message["role"] == "assistant")

    sample = deepcopy(slot_sample)
    sample.tokens = list(token_ids)
    sample.response_length = response_length
    sample.loss_mask = list(full_mask[-response_length:])
    sample.response = mask_generator.tokenizer.decode(token_ids[-response_length:])
    sample.prompt = messages[:first_assistant]
    sample.reward = float(trajectory.reward)
    sample.status = SAMPLE_STATUS[trajectory.status]
    # Every token-position-dependent record of the sampling process describes
    # a different sequence; the trainer recomputes what it needs from these tokens.
    sample.rollout_log_probs = None
    sample.multimodal_inputs = None
    sample.multimodal_train_inputs = None
    sample.rollout_sampling_mask = None
    sample.rollout_routed_experts = None
    sample.rollout_indexer_topk = None
    sample.teacher_log_probs = None
    sample.opd_reverse_kl = None
    sample.weight_versions = []

    lineage = {
        "rollout_job_id": result.rollout_job_id,
        "prompt_group_id": result.prompt_group_id,
        "sample_slot_id": trajectory.sample_slot_id,
        "branch_id": trajectory.branch_id,
        "parent_branch_id": trajectory.parent_branch_id,
        "stop_reason": trajectory.stop_reason,
        "weight_version": weight_version,
    }
    sample.metadata = {
        **sample.metadata,
        # ``messages`` is where Miles's trajectory dumps and dashboard look.
        "messages": messages,
        "tools": trajectory.tools,
        "sprout_rollout": {
            **lineage,
            "hints_removed": True,
            "logprob_context": "hint_free_messages",
            "trajectory_metadata": deepcopy(trajectory.metadata),
        },
    }
    sample.train_metadata = {**(sample.train_metadata or {}), "sprout_rollout": lineage}
    sample.validate()
    return sample
