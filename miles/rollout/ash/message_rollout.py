"""Ash message rollout: Ash owns environments/rewards; Miles owns training tokens."""

import asyncio
import math
import uuid

from miles.rollout.ash.client import AshRolloutClient
from miles.rollout.ash.message_importer import import_ash_messages
from miles.rollout.ash.message_protocol import (
    AshMessageAcknowledgement,
    AshMessageBudget,
    AshMessageRequest,
    AshMessageResult,
)
from miles.rollout.ash.protocol import AshSampleSlot
from miles.rollout.ash.rollout_fn import (
    AshRolloutFn,
    _delete_without_masking_error,
    _validate_common_configuration,
    _validate_group,
)
from miles.utils.mask_utils import MultiTurnLossMaskGenerator
from miles.utils.processing_utils import load_tokenizer


class AshMessageClient(AshRolloutClient):
    async def submit(self, request: AshMessageRequest) -> AshMessageAcknowledgement:
        response = await self._client.post("/rollout-groups", json=request.to_wire())
        response.raise_for_status()
        return AshMessageAcknowledgement.model_validate(response.json())

    async def get_result(self, rollout_job_id: str) -> AshMessageResult:
        response = await self._client.get(f"/rollout-groups/{rollout_job_id}")
        response.raise_for_status()
        return AshMessageResult.model_validate(response.json())

    async def delete(self, rollout_job_id: str) -> AshMessageAcknowledgement:
        response = await self._client.delete(f"/rollout-groups/{rollout_job_id}")
        response.raise_for_status()
        result = AshMessageAcknowledgement.model_validate(response.json())
        if result.rollout_job_id != rollout_job_id or result.status in {"queued", "running"}:
            raise ValueError("invalid Ash message deletion acknowledgement")
        return result


class AshMessageRolloutFn(AshRolloutFn):
    def __init__(self, input, *, client_factory=AshMessageClient):
        super().__init__(input, client_factory=client_factory)
        configured_turns = getattr(self._args, "ash_rollout_max_turns", None)
        self._max_turns = 64 if configured_turns is None else configured_turns
        self._finalization_timeout_seconds = getattr(self._args, "ash_rollout_finalization_timeout_seconds", 1800.0)
        if not getattr(self._args, "compute_advantages_and_returns", True):
            raise ValueError("Ash message training requires the normal RL forward/advantage path")
        for name in (
            "use_rollout_logprobs",
            "skip_actor_forward_only",
            "use_tis",
            "use_rollout_routing_replay",
            "use_rollout_indexer_replay",
            "use_rollout_sampling_mask",
        ):
            if getattr(self._args, name, False):
                raise ValueError(f"--{name.replace('_', '-')} is incompatible with hint-free message training")
        self._mask_generator = None
        if getattr(self._args, "rollout_stop_token_ids", None):
            raise ValueError("Ash native message APIs support --rollout-stop, not token-id stopping")
        if getattr(self._args, "apply_chat_template_kwargs", None):
            raise ValueError("Ash native message APIs do not support chat_template_kwargs overrides")
        # Native APIs serialize their own text/tool responses. v2's detokenization
        # switches describe its exact-token contract, not this messages contract.
        self._sampling_params = {
            "temperature": self._args.rollout_temperature,
            "top_p": self._args.rollout_top_p,
            "top_k": self._args.rollout_top_k,
            "stop": self._args.rollout_stop,
        }
        if self._args.rollout_max_response_len is not None:
            self._sampling_params["max_new_tokens"] = self._args.rollout_max_response_len

    def _validate_configuration(self) -> None:
        _validate_common_configuration(self._args)
        branching = getattr(self._args, "ash_rollout_branching", False)
        if type(branching) is not bool:
            raise ValueError("--ash-rollout-branching must be boolean")
        if branching and self._args.n_samples_per_prompt != 2:
            raise ValueError("--ash-rollout-branching requires --n-samples-per-prompt 2")
        for name in ("ash_rollout_max_model_calls", "ash_rollout_max_tool_calls"):
            explicit = getattr(self._args, f"_{name}_explicit", None)
            if explicit is True or (explicit is None and getattr(self._args, name, None) is not None):
                raise ValueError(f"--{name.replace('_', '-')} is v2-only; v3 uses --ash-rollout-max-turns")
        max_turns = getattr(self._args, "ash_rollout_max_turns", None)
        if max_turns is not None and (type(max_turns) is not int or max_turns <= 0):
            raise ValueError("--ash-rollout-max-turns must be a positive integer")
        finalization = getattr(self._args, "ash_rollout_finalization_timeout_seconds", 1800.0)
        if type(finalization) not in (int, float) or not math.isfinite(finalization) or finalization <= 0:
            raise ValueError("--ash-rollout-finalization-timeout-seconds must be finite and positive")

    def _build_request(self, *, group, rollout_id, weight_version):
        if len(group) != self._args.n_samples_per_prompt:
            raise ValueError("Ash requires n_samples_per_prompt slots per group")
        group_id, prompt = _validate_group(group)
        identities = []
        for sample in group:
            metadata = sample.metadata or {}
            task_id = metadata.get("task_id")
            image = metadata.get("image") or metadata.get("image_name")
            if not isinstance(task_id, str) or not task_id.strip():
                raise ValueError("Ash message rollout requires metadata['task_id']")
            if not isinstance(image, str) or not image.strip():
                raise ValueError("Ash message rollout requires metadata['image']")
            if sample.multimodal_inputs and any(value is not None for value in sample.multimodal_inputs.values()):
                raise ValueError("Ash message rollout supports text messages only")
            identities.append((task_id, image))
        if any(identity != identities[0] for identity in identities):
            raise ValueError("all samples must share task_id and image")
        task_id, image = identities[0]
        job_id = f"miles-{rollout_id}-{group_id}-{uuid.uuid4().hex}"
        slots = [
            AshSampleSlot(sample_slot_id=f"{job_id}:slot:{sample.index}", sample_index=sample.index)
            for sample in group
        ]
        request = AshMessageRequest(
            rollout_job_id=job_id,
            rollout_id=rollout_id,
            prompt_group_id=group_id,
            task_id=task_id,
            image=image,
            prompt=prompt,
            model_endpoint=self._model_endpoint(),
            model=getattr(self._args, "sglang_served_model_name", None) or getattr(self._args, "model", None),
            sample_slots=slots,
            max_samples=len(slots),
            minimum_returned_samples=len(slots),
            max_turns=self._max_turns,
            finalization_timeout_seconds=self._finalization_timeout_seconds,
            sampling_params=dict(self._sampling_params),
            budgets=AshMessageBudget(
                max_wall_time_seconds=self._rollout_timeout_seconds,
            ),
            branching=getattr(self._args, "ash_rollout_branching", False),
        )
        return request, dict(zip((slot.sample_slot_id for slot in slots), group, strict=True))

    async def _run_group(self, *, client, group, rollout_id, weight_version):
        request, slots = self._build_request(group=group, rollout_id=rollout_id, weight_version=weight_version)
        try:
            acknowledgement = await client.submit(request)
            if acknowledgement.rollout_job_id != request.rollout_job_id:
                raise ValueError("Ash acknowledged a different rollout job")
            result = await asyncio.wait_for(
                client.wait_for_result(request.rollout_job_id, poll_interval_seconds=self._poll_interval_seconds),
                timeout=self._rollout_timeout_seconds + self._finalization_timeout_seconds,
            )
            if (
                result.rollout_job_id != request.rollout_job_id
                or result.prompt_group_id != request.prompt_group_id
                or result.max_samples != request.max_samples
            ):
                raise ValueError("Ash message result identity/slot count mismatch")
            if result.status not in {"completed", "early_stopped"}:
                raise RuntimeError(f"Ash message rollout failed: {result.stop_reason}")
            if result.actual_samples != request.max_samples:
                raise ValueError("Ash must return all allocated samples for fixed-size training groups")
            if self._mask_generator is None:
                tokenizer = load_tokenizer(
                    self._args.hf_checkpoint,
                    chat_template_path=self._args.chat_template_path,
                    trust_remote_code=True,
                )
                self._mask_generator = MultiTurnLossMaskGenerator(
                    tokenizer,
                    tokenizer_type=getattr(self._args, "loss_mask_type", "qwen"),
                )
            return import_ash_messages(result, slots, self._mask_generator), result
        finally:
            await _delete_without_masking_error(client, request.rollout_job_id)
