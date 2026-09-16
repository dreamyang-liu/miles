"""Reconstruct training samples from hint-free messages, without rollout logprobs."""

import json
from collections.abc import Mapping
from copy import deepcopy

from miles.rollout.ash.importer import _SAMPLE_STATUS, _validate_messages
from miles.rollout.ash.message_protocol import AshMessageResult
from miles.utils.mask_utils import MultiTurnLossMaskGenerator
from miles.utils.types import Sample


def import_ash_messages(
    result: AshMessageResult,
    slot_samples: Mapping[str, Sample],
    mask_generator: MultiTurnLossMaskGenerator,
) -> list[Sample]:
    if result.status not in {"completed", "early_stopped"}:
        raise ValueError(f"cannot import Ash messages with status={result.status!r}")
    if result.max_samples > len(slot_samples):
        raise ValueError("Ash max_samples exceeds allocated slots")
    samples = []
    for trajectory in result.trajectories:
        if trajectory.sample_slot_id not in slot_samples:
            raise ValueError(f"unknown sample_slot_id: {trajectory.sample_slot_id!r}")
        messages = deepcopy(trajectory.messages)
        _validate_messages(messages)
        for message in messages:
            for call in message.get("tool_calls") or []:
                arguments = call["function"].get("arguments", {})
                if isinstance(arguments, str):
                    arguments = json.loads(arguments)
                if not isinstance(arguments, dict):
                    raise ValueError("tool arguments must decode to a JSON object")
                # Native APIs use JSON strings; HF tool templates expect mappings.
                call["function"]["arguments"] = arguments
            if message["role"] in {"user", "system"} and any(
                marker in (message.get("content") or "") for marker in ("<ash_training_hint>", "</ash_training_hint>")
            ):
                raise ValueError("Ash returned an unremoved training hint")
        token_ids, full_mask = mask_generator.get_loss_mask(messages, tools=trajectory.tools or None)
        if len(token_ids) != len(full_mask) or not any(full_mask):
            raise ValueError("message trajectory must contain trainable assistant tokens")
        response_length = mask_generator.get_response_lengths([full_mask])[0]
        sample = deepcopy(slot_samples[trajectory.sample_slot_id])
        # All token-position-dependent execution data is invalid after removing hints.
        sample.tokens = list(token_ids)
        sample.response_length = response_length
        sample.loss_mask = list(full_mask[-response_length:])
        sample.rollout_log_probs = None
        sample.multimodal_inputs = None
        sample.multimodal_train_inputs = None
        sample.rollout_sampling_mask = None
        sample.rollout_routed_experts = None
        sample.rollout_indexer_topk = None
        sample.teacher_log_probs = None
        sample.opd_reverse_kl = None
        sample.weight_versions = []
        sample.response = mask_generator.tokenizer.decode(token_ids[-response_length:])
        sample.reward = trajectory.reward
        sample.status = _SAMPLE_STATUS[trajectory.status]
        first_assistant = next(i for i, message in enumerate(messages) if message["role"] == "assistant")
        sample.prompt = messages[:first_assistant]
        lineage = {
            "protocol_version": result.protocol_version,
            "rollout_job_id": result.rollout_job_id,
            "prompt_group_id": result.prompt_group_id,
            "sample_slot_id": trajectory.sample_slot_id,
            "branch_id": trajectory.branch_id,
            "parent_branch_id": trajectory.parent_branch_id,
            "stop_reason": trajectory.stop_reason,
            "messages": messages,
            "hints_removed": True,
            "logprob_context": "hint_free_messages",
            "trajectory_metadata": deepcopy(trajectory.metadata),
        }
        sample.metadata = {**sample.metadata, "tools": trajectory.tools, "ash_rollout": lineage}
        sample.train_metadata = {
            **(sample.train_metadata or {}),
            "ash_rollout": {
                key: lineage[key]
                for key in (
                    "rollout_job_id",
                    "prompt_group_id",
                    "sample_slot_id",
                    "branch_id",
                    "parent_branch_id",
                    "stop_reason",
                )
            },
        }
        sample.validate()
        samples.append(sample)
    return samples
