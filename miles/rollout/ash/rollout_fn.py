import asyncio
import logging
import uuid
from collections.abc import Callable
from typing import Any

from miles.rollout.ash.client import AshRolloutClient
from miles.rollout.ash.importer import import_ash_rollout_result
from miles.rollout.ash.protocol import (
    AshEnvironmentRef,
    AshRolloutBudget,
    AshRolloutRequest,
    AshRolloutResult,
    AshSampleSlot,
)
from miles.rollout.base_types import (
    RolloutFnConstructorInput,
    RolloutFnEvalInput,
    RolloutFnInput,
    RolloutFnOutput,
    RolloutFnTrainInput,
    RolloutFnTrainOutput,
)
from miles.rollout.rm_hub import batched_async_rm
from miles.utils.processing_utils import load_tokenizer
from miles.utils.types import Sample

logger = logging.getLogger(__name__)

AshClientFactory = Callable[..., AshRolloutClient]


class AshRolloutFn:
    """Delegate each prompt group to Ash and import the completed trajectories."""

    def __init__(
        self,
        input: RolloutFnConstructorInput,
        *,
        client_factory: AshClientFactory = AshRolloutClient,
    ) -> None:
        self._args = input.args
        self._data_source = input.data_source
        self._client_factory = client_factory
        self._base_url = _required_arg(self._args, "ash_rollout_base_url")
        self._configured_model_endpoint = getattr(self._args, "ash_rollout_model_endpoint", None)
        self._configured_session_endpoint = getattr(self._args, "ash_rollout_session_server_endpoint", None)
        self._poll_interval_seconds = self._args.ash_rollout_poll_interval_seconds
        self._rollout_timeout_seconds = self._args.ash_rollout_timeout_seconds
        self._client_grace_seconds = self._args.ash_rollout_client_grace_seconds
        self._http_timeout_seconds = self._args.ash_rollout_http_timeout_seconds
        self._tokenizer = None
        self._validate_configuration()
        self._sampling_params = _sampling_params(self._args)

    def _validate_configuration(self) -> None:
        _validate_configuration(self._args)

    async def __call__(self, input: RolloutFnInput) -> RolloutFnOutput:
        if isinstance(input, RolloutFnEvalInput):
            raise NotImplementedError(
                "AshRolloutFn does not serve evaluation yet; configure --eval-function-path separately"
            )
        return await self._call_train(input)

    async def _call_train(self, input: RolloutFnTrainInput) -> RolloutFnTrainOutput:
        groups = self._data_source.get_samples(self._args.rollout_batch_size)
        if len(groups) != self._args.rollout_batch_size:
            raise ValueError(
                f"data source returned {len(groups)} prompt groups; expected {self._args.rollout_batch_size}"
            )

        async with self._client_factory(
            self._base_url,
            timeout=self._http_timeout_seconds,
        ) as client:
            tasks = [
                asyncio.create_task(
                    self._run_group(
                        client=client,
                        group=group,
                        rollout_id=input.rollout_id,
                        weight_version=input.weight_version,
                    )
                )
                for group in groups
            ]
            try:
                completed_groups = await asyncio.gather(*tasks)
            except BaseException:
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                raise

        return RolloutFnTrainOutput(
            samples=[samples for samples, _result in completed_groups],
            metrics=_collect_metrics(result for _samples, result in completed_groups),
        )

    async def _run_group(
        self,
        *,
        client: AshRolloutClient,
        group: list[Sample],
        rollout_id: int,
        weight_version: int | None,
    ) -> tuple[list[Sample], AshRolloutResult]:
        request, slot_samples = self._build_request(
            group=group,
            rollout_id=rollout_id,
            weight_version=weight_version,
        )
        submission_attempted = False
        try:
            # The server may have accepted the deterministic job ID even when
            # the POST response is lost.  Cleanup must therefore follow every
            # submission attempt, not only acknowledged submissions.
            submission_attempted = True
            submission = await client.submit(request)
            if submission.rollout_job_id != request.rollout_job_id:
                raise ValueError(
                    f"Ash accepted rollout_job_id={submission.rollout_job_id!r}; "
                    f"expected {request.rollout_job_id!r}"
                )

            result = await asyncio.wait_for(
                client.wait_for_result(
                    request.rollout_job_id,
                    poll_interval_seconds=self._poll_interval_seconds,
                ),
                timeout=self._rollout_timeout_seconds + self._client_grace_seconds,
            )
            _validate_result_contract(request, result)
            samples = import_ash_rollout_result(result, slot_samples)
            samples_need_reward = [sample for sample in samples if sample.reward is None]
            if samples_need_reward:
                await batched_async_rm(self._args, samples_need_reward, inplace_set_reward_field=True)
            return samples, result
        finally:
            if submission_attempted:
                await _delete_without_masking_error(client, request.rollout_job_id)

    def _build_request(
        self,
        *,
        group: list[Sample],
        rollout_id: int,
        weight_version: int | None,
    ) -> tuple[AshRolloutRequest, dict[str, Sample]]:
        if len(group) != self._args.n_samples_per_prompt:
            raise ValueError(
                f"data source returned {len(group)} samples in a prompt group; "
                f"expected n_samples_per_prompt={self._args.n_samples_per_prompt}"
            )
        prompt_group_id, prompt = _validate_group(group)
        task_id, environment_ref = _task_environment(group)
        prompt_token_ids = self._prompt_token_ids(group)
        job_id = f"miles-{rollout_id}-{prompt_group_id}-{uuid.uuid4().hex}"
        slots = [
            AshSampleSlot(
                sample_slot_id=f"{job_id}:slot:{sample.index}",
                sample_index=sample.index,
            )
            for sample in group
        ]
        slot_samples = dict(zip((slot.sample_slot_id for slot in slots), group, strict=True))
        max_samples = len(slots)
        max_model_calls, max_tool_calls = _legacy_call_limits(self._args)
        request = AshRolloutRequest(
            rollout_job_id=job_id,
            rollout_id=rollout_id,
            prompt_group_id=prompt_group_id,
            task_id=task_id,
            environment_ref=environment_ref,
            sample_slots=slots,
            max_samples=max_samples,
            prompt=prompt,
            prompt_token_ids=prompt_token_ids,
            model_endpoint=self._model_endpoint(),
            session_server_endpoint=self._session_server_endpoint(),
            model=getattr(self._args, "model_name", None) or getattr(self._args, "model", None),
            expected_weight_version=None if weight_version is None else str(weight_version),
            return_rollout_logprobs=self._args.use_rollout_logprobs,
            sampling_params=dict(self._sampling_params),
            budgets=AshRolloutBudget(
                max_model_calls=max_model_calls,
                max_tool_calls=max_tool_calls,
                max_wall_time_seconds=self._rollout_timeout_seconds,
            ),
            # Phase one deliberately keeps Miles' existing fixed group-size semantics.
            minimum_returned_samples=max_samples,
        )
        return request, slot_samples

    def _model_endpoint(self) -> str:
        if self._configured_model_endpoint:
            return self._configured_model_endpoint
        host = getattr(self._args, "sglang_router_ip", None)
        port = getattr(self._args, "sglang_router_port", None)
        if not host or port is None:
            raise RuntimeError("Miles rollout router address is not available when submitting to Ash")
        return f"http://{host}:{port}"

    def _session_server_endpoint(self) -> str | None:
        """Return the v2 session-server base URL when Ash should run an agent loop."""
        if self._configured_session_endpoint:
            return self._configured_session_endpoint.rstrip("/")
        if getattr(self._args, "use_session_server", None) not in ("v1", "v2", True):
            return None
        host = getattr(self._args, "session_server_ip", None)
        ports = getattr(self._args, "session_server_ports", None)
        if not host or not ports:
            return None
        # The rollout service receives one stable base URL per job.  Miles'
        # session router is stateless at this layer and can route all sessions
        # through the first configured worker; direct session creation still
        # records its owning instance in the session metadata.
        return f"http://{host}:{ports[0]}"

    def _prompt_token_ids(self, group: list[Sample]) -> list[int]:
        sample = group[0]
        if sample.tokens:
            if any(item.tokens != sample.tokens for item in group[1:]):
                raise ValueError("all samples in an Ash prompt group must have the same prompt token IDs")
            return list(sample.tokens)
        if any(item.tokens for item in group[1:]):
            raise ValueError("all samples in an Ash prompt group must agree on whether prompt token IDs are set")
        if any(_has_multimodal_inputs(item.multimodal_inputs) for item in group):
            raise NotImplementedError("AshRolloutFn does not support multimodal prompt tokenization yet")

        if self._tokenizer is None:
            self._tokenizer = load_tokenizer(
                self._args.hf_checkpoint,
                chat_template_path=self._args.chat_template_path,
                trust_remote_code=True,
            )
        if isinstance(sample.prompt, str):
            return self._tokenizer.encode(sample.prompt, add_special_tokens=False)
        # Message-mode v3 does not need SGLang's exact-token template helpers.
        from miles.utils import chat_template_utils

        tools = sample.metadata.get("tools") if sample.metadata else None
        return chat_template_utils.apply_chat_template(
            sample.prompt,
            tokenizer=self._tokenizer,
            tools=tools,
            tokenize=True,
            add_generation_prompt=True,
            **(getattr(self._args, "apply_chat_template_kwargs", None) or {}),
        )


def _has_multimodal_inputs(multimodal_inputs: dict[str, Any] | None) -> bool:
    """Match Miles' convention that an all-None media mapping is text-only."""
    return bool(multimodal_inputs) and any(value is not None for value in multimodal_inputs.values())


def _required_arg(args: Any, name: str) -> Any:
    value = getattr(args, name, None)
    if value is None or value == "":
        option = name.replace("_", "-")
        raise ValueError(f"--{option} is required by AshRolloutFn")
    return value


def _task_environment(group: list[Sample]) -> tuple[str, AshEnvironmentRef]:
    """Read one immutable environment identity shared by the prompt group."""
    resolved: list[tuple[str, AshEnvironmentRef]] = []
    for sample in group:
        metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
        task_id = metadata.get("task_id")
        if not isinstance(task_id, str) or not task_id.strip():
            raise ValueError("Ash rollout samples require metadata['task_id']")
        raw_ref = metadata.get("environment_ref")
        if not isinstance(raw_ref, dict):
            raise ValueError("Ash rollout samples require metadata['environment_ref']")
        resolved.append((task_id, AshEnvironmentRef.model_validate(raw_ref)))

    first = resolved[0]
    if any(item != first for item in resolved[1:]):
        raise ValueError(
            "all samples in an Ash prompt group must use the same task_id and environment_ref"
        )
    return first


def _sampling_params(args: Any) -> dict[str, Any]:
    params = {
        "temperature": args.rollout_temperature,
        "top_p": args.rollout_top_p,
        "top_k": args.rollout_top_k,
        "max_new_tokens": args.rollout_max_response_len,
        "stop": args.rollout_stop,
        "stop_token_ids": args.rollout_stop_token_ids,
        "skip_special_tokens": args.rollout_skip_special_tokens,
        "no_stop_trim": True,
        "spaces_between_special_tokens": False,
    }
    if chat_template_kwargs := getattr(args, "apply_chat_template_kwargs", None):
        params["chat_template_kwargs"] = dict(chat_template_kwargs)
    return params


def _validate_common_configuration(args: Any) -> None:
    positive_values = {
        "ash_rollout_poll_interval_seconds": args.ash_rollout_poll_interval_seconds,
        "ash_rollout_timeout_seconds": args.ash_rollout_timeout_seconds,
        "ash_rollout_http_timeout_seconds": args.ash_rollout_http_timeout_seconds,
    }
    for name, value in positive_values.items():
        if value <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be greater than zero")


def _legacy_call_limits(args: Any) -> tuple[int | None, int | None]:
    # Missing programmatic values use the v2 defaults; explicit None stays
    # unbounded, as in the published v2 contract.
    return (
        getattr(args, "ash_rollout_max_model_calls", 100),
        getattr(args, "ash_rollout_max_tool_calls", 100),
    )


def _validate_configuration(args: Any) -> None:
    _validate_common_configuration(args)
    if getattr(args, "ash_rollout_branching", False):
        raise ValueError("--ash-rollout-branching requires AshMessageRolloutFn (v3)")
    if (getattr(args, "ash_rollout_max_sequence_tokens", None) is not None
            or getattr(args, "ash_rollout_truncated_reward_scale", 1.0) != 1.0
            or getattr(args, "ash_rollout_max_unusable_groups", 0) != 0):
        raise ValueError("Ash sequence caps, reward scaling and replacements require AshMessageRolloutFn (v3)")
    if args.ash_rollout_client_grace_seconds <= 0:
        raise ValueError("--ash-rollout-client-grace-seconds must be greater than zero")
    model_calls, tool_calls = _legacy_call_limits(args)
    if model_calls is not None and model_calls <= 0:
        raise ValueError("--ash-rollout-max-model-calls must be greater than zero")
    if tool_calls is not None and tool_calls < 0:
        raise ValueError("--ash-rollout-max-tool-calls must be non-negative")
    if getattr(args, "ash_rollout_max_turns", None) is not None:
        raise ValueError("--ash-rollout-max-turns requires AshMessageRolloutFn (v3)")


def _validate_group(group: list[Sample]) -> tuple[str, str | list[dict[str, Any]]]:
    if not group:
        raise ValueError("Ash rollout cannot submit an empty prompt group")
    group_indices = {sample.group_index for sample in group}
    if None in group_indices or len(group_indices) != 1:
        raise ValueError(f"all samples must have the same non-null group_index, got {group_indices}")
    sample_indices = [sample.index for sample in group]
    if any(index is None for index in sample_indices) or len(sample_indices) != len(set(sample_indices)):
        raise ValueError(f"sample indices must be non-null and unique, got {sample_indices}")
    if any(sample.prompt != group[0].prompt for sample in group[1:]):
        raise ValueError("all samples in an Ash prompt group must have the same prompt")
    return str(group[0].group_index), group[0].prompt


def _validate_result_contract(request: AshRolloutRequest, result: AshRolloutResult) -> None:
    if result.rollout_job_id != request.rollout_job_id:
        raise ValueError(f"Ash returned rollout_job_id={result.rollout_job_id!r}; expected {request.rollout_job_id!r}")
    if result.prompt_group_id != request.prompt_group_id:
        raise ValueError(
            f"Ash returned prompt_group_id={result.prompt_group_id!r}; expected {request.prompt_group_id!r}"
        )
    if result.max_samples != request.max_samples:
        raise ValueError(f"Ash returned max_samples={result.max_samples}; expected {request.max_samples}")
    if result.status not in {"completed", "early_stopped"}:
        detail = f": {result.stop_reason}" if result.stop_reason else ""
        raise RuntimeError(
            f"Ash rollout {request.rollout_job_id!r} ended with status={result.status!r}{detail}"
        )
    if result.actual_samples != request.max_samples:
        raise ValueError(
            f"Ash returned {result.actual_samples} samples for a fixed-size group of {request.max_samples}; "
            "K < N support is not enabled in the Miles training path yet"
        )
    if request.expected_weight_version is not None:
        returned_versions = {
            span.weight_version for trajectory in result.trajectories for span in trajectory.generated_spans
        }
        if returned_versions != {request.expected_weight_version}:
            raise ValueError(
                f"Ash returned weight versions {sorted(returned_versions)}; "
                f"expected only {request.expected_weight_version!r}"
            )
    if request.return_rollout_logprobs:
        missing_response_ids = [
            span.response_id
            for trajectory in result.trajectories
            for span in trajectory.generated_spans
            if span.output_token_log_probs is None
        ]
        if missing_response_ids:
            raise ValueError(
                "Ash omitted output_token_log_probs requested by Miles for responses " f"{missing_response_ids}"
            )


async def _delete_without_masking_error(client: AshRolloutClient, rollout_job_id: str) -> None:
    delete_task = asyncio.create_task(client.delete(rollout_job_id))
    try:
        await asyncio.shield(delete_task)
    except asyncio.CancelledError:
        try:
            await delete_task
        except Exception as error:
            logger.warning("Failed to delete Ash rollout %s during cancellation: %r", rollout_job_id, error)
        raise
    except Exception as error:
        logger.warning("Failed to delete Ash rollout %s during cleanup: %r", rollout_job_id, error)


def _collect_metrics(results) -> dict[str, int | float]:
    results = list(results)
    consumed_model_calls = sum(int(result.consumed_budget.get("model_calls", 0)) for result in results)
    consumed_tool_calls = sum(int(result.consumed_budget.get("tool_calls", 0)) for result in results)
    return {
        "rollout/ash/groups": len(results),
        "rollout/ash/samples": sum(result.actual_samples for result in results),
        "rollout/ash/search_branches": sum(result.search_branches for result in results),
        "rollout/ash/model_calls": consumed_model_calls,
        "rollout/ash/tool_calls": consumed_tool_calls,
    }
