"""``SproutRolloutFn``: Miles's rollout function over Sprout's RL driver.

    --rollout-function-path miles.rollout.sprout.rollout_fn.SproutRolloutFn
    --sprout-rollout-base-url http://SPROUT_DRIVER:11001

Each training step takes ``rollout_batch_size`` prompt groups from the data
source. A group is one task (its ``metadata`` names Sprout's ``task_id`` and the
sandbox ``image``) with ``n_samples_per_prompt`` slots; the whole group is one
``POST /rollout-groups`` and Sprout runs one independent agent rollout per slot
against the policy's inference endpoint, grades every final snapshot and
returns the trajectories as hint-free messages with their 0/1 rewards. The
samples keep their ``group_index``, so Miles's ordinary GRPO reward
normalization (``--advantage-estimator grpo``) applies per group as it would to
a generate_hub rollout.

What Sprout returns is the conversation, not the sampled tokens: Miles renders
it with the policy's chat template and recomputes log-probs. Flags that assume
the rollout engine's own tokens (``--use-rollout-logprobs``, TIS, actor forward
skipping, routing or indexer replay) are refused rather than silently applied
to a different sequence.
"""

from __future__ import annotations

import asyncio
import logging
import math
import os
import uuid
from collections.abc import Callable
from typing import Any

from miles.rollout.base_types import (
    BaseRolloutFn,
    RolloutFnConstructorInput,
    RolloutFnEvalInput,
    RolloutFnInput,
    RolloutFnOutput,
    RolloutFnTrainInput,
    RolloutFnTrainOutput,
)
from miles.rollout.sprout.client import SproutRolloutClient
from miles.rollout.sprout.importer import import_trajectories
from miles.rollout.sprout.protocol import Budget, RolloutRequest, RolloutResult, SampleSlot
from miles.utils.mask_utils import MultiTurnLossMaskGenerator
from miles.utils.processing_utils import load_tokenizer
from miles.utils.types import Sample

logger = logging.getLogger(__name__)

#: Miles features that reuse the rollout engine's tokens or log-probs. The
#: trajectory Sprout returns is retokenized, so none of them can be honoured.
TOKEN_REPLAY_FLAGS = (
    "use_rollout_logprobs",
    "skip_actor_forward_only",
    "use_tis",
    "use_rollout_routing_replay",
    "use_rollout_indexer_replay",
    "use_rollout_sampling_mask",
)


class SproutRolloutFn(BaseRolloutFn):
    def __init__(
        self,
        input: RolloutFnConstructorInput,
        *,
        client_factory: Callable[..., SproutRolloutClient] = SproutRolloutClient,
    ) -> None:
        super().__init__(input)
        self._args = input.args
        self._data_source = input.data_source
        self._client_factory = client_factory
        self._mask_generator: MultiTurnLossMaskGenerator | None = None
        validate_configuration(self._args)
        self._sampling_params = sampling_params(self._args)

    @staticmethod
    def add_arguments(parser):
        """Called by Miles's argument parser once this class is the rollout function."""
        group = parser.add_argument_group("sprout rollout")
        group.add_argument(
            "--sprout-rollout-base-url",
            type=str,
            default=None,
            help="Base URL of Sprout's RL driver (``python -m sprout.rl_driver``, port 11001 by default).",
        )
        group.add_argument(
            "--sprout-rollout-model-endpoint",
            type=str,
            default=None,
            help=(
                "Chat Completions endpoint Sprout's sandboxes reach the policy at. Defaults to Miles's "
                "rollout router, which is right when Sprout runs on a host that can reach it."
            ),
        )
        group.add_argument(
            "--sprout-rollout-token-env",
            type=str,
            default="SPROUT_RL_DRIVER_TOKEN",
            help="Environment variable holding the driver's bearer token; unset means the driver takes no token.",
        )
        group.add_argument(
            "--sprout-rollout-max-turns",
            type=int,
            default=64,
            help="Model calls per trajectory before Sprout stops it (truncated, stop_reason max_turns_reached).",
        )
        group.add_argument(
            "--sprout-rollout-timeout-seconds",
            type=float,
            default=3600.0,
            help="Wall time for a group's execution phase, from submission; the budget Sprout enforces.",
        )
        group.add_argument(
            "--sprout-rollout-finalization-timeout-seconds",
            type=float,
            default=1800.0,
            help="Extra time after the execution deadline for final snapshots, message export and grading.",
        )
        group.add_argument(
            "--sprout-rollout-poll-interval-seconds",
            type=float,
            default=2.0,
            help="Interval between status requests while a group runs.",
        )
        group.add_argument(
            "--sprout-rollout-http-timeout-seconds",
            type=float,
            default=30.0,
            help="Timeout for one HTTP request to the driver.",
        )
        return parser

    async def __call__(self, input: RolloutFnInput) -> RolloutFnOutput:
        if isinstance(input, RolloutFnEvalInput):
            raise NotImplementedError(
                "SproutRolloutFn serves training rollouts only; configure --eval-function-path separately"
            )
        return await self._call_train(input)

    async def _call_train(self, input: RolloutFnTrainInput) -> RolloutFnTrainOutput:
        groups = self._data_source.get_samples(self._args.rollout_batch_size)
        if len(groups) != self._args.rollout_batch_size:
            raise ValueError(
                f"data source returned {len(groups)} prompt groups; expected {self._args.rollout_batch_size}"
            )
        async with self._client_factory(
            self._args.sprout_rollout_base_url,
            timeout=self._args.sprout_rollout_http_timeout_seconds,
            token=os.environ.get(self._args.sprout_rollout_token_env or "") or None,
        ) as client:
            tasks = [
                asyncio.create_task(
                    self._run_group(
                        client=client, group=group, rollout_id=input.rollout_id, weight_version=input.weight_version
                    )
                )
                for group in groups
            ]
            try:
                completed = await asyncio.gather(*tasks)
            except BaseException:
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                raise
        return RolloutFnTrainOutput(
            samples=[samples for samples, _result in completed],
            metrics=collect_metrics([result for _samples, result in completed]),
        )

    async def _run_group(
        self, *, client: SproutRolloutClient, group: list[Sample], rollout_id: int, weight_version: int | None
    ) -> tuple[list[Sample], RolloutResult]:
        request, slots = self._build_request(group=group, rollout_id=rollout_id)
        try:
            acknowledgement = await client.submit(request)
            if acknowledgement.rollout_job_id != request.rollout_job_id:
                raise ValueError(
                    f"driver acknowledged {acknowledgement.rollout_job_id!r}; expected {request.rollout_job_id!r}"
                )
            result = await asyncio.wait_for(
                client.wait_for_result(
                    request.rollout_job_id, poll_interval_seconds=self._args.sprout_rollout_poll_interval_seconds
                ),
                timeout=request.budgets.max_wall_time_seconds + request.finalization_timeout_seconds,
            )
            validate_result(request, result)
            samples = import_trajectories(result, slots, self._get_mask_generator(), weight_version=weight_version)
            return samples, result
        finally:
            # The driver keeps a terminal group until it is released, and a
            # group still running is cancelled; either way nothing leaks.
            await _release_without_masking_error(client, request.rollout_job_id)

    def _build_request(self, *, group: list[Sample], rollout_id: int) -> tuple[RolloutRequest, dict[str, Sample]]:
        if len(group) != self._args.n_samples_per_prompt:
            raise ValueError(
                f"data source returned {len(group)} samples in a prompt group; "
                f"expected n_samples_per_prompt={self._args.n_samples_per_prompt}"
            )
        group_id, prompt = validate_group(group)
        task_id, image = task_identity(group)
        job_id = f"miles-{rollout_id}-{group_id}-{uuid.uuid4().hex}"
        slots = [
            SampleSlot(sample_slot_id=f"{job_id}:slot:{sample.index}", sample_index=sample.index) for sample in group
        ]
        request = RolloutRequest(
            rollout_job_id=job_id,
            rollout_id=rollout_id,
            prompt_group_id=group_id,
            task_id=task_id,
            image=image,
            prompt=prompt,
            model_endpoint=self.model_endpoint(),
            model=self.model_name(),
            sample_slots=slots,
            max_samples=len(slots),
            # Fixed group size: GRPO normalizes within the group, and a group
            # Sprout could not fill is an error, not a smaller group.
            minimum_returned_samples=len(slots),
            max_turns=self._args.sprout_rollout_max_turns,
            finalization_timeout_seconds=self._args.sprout_rollout_finalization_timeout_seconds,
            sampling_params=dict(self._sampling_params),
            budgets=Budget(max_wall_time_seconds=self._args.sprout_rollout_timeout_seconds),
        )
        return request, dict(zip((slot.sample_slot_id for slot in slots), group, strict=True))

    def model_endpoint(self) -> str:
        if configured := getattr(self._args, "sprout_rollout_model_endpoint", None):
            return configured.rstrip("/")
        host = getattr(self._args, "sglang_router_ip", None)
        port = getattr(self._args, "sglang_router_port", None)
        if not host or port is None:
            raise RuntimeError(
                "Miles's rollout router address is not known yet; pass --sprout-rollout-model-endpoint "
                "or submit after the rollout engines start"
            )
        return f"http://{host}:{port}"

    def model_name(self) -> str | None:
        """The ``model`` of the Chat Completions requests Sprout's gateway sends."""
        for name in ("sglang_served_model_name", "model_name"):
            if value := getattr(self._args, name, None):
                return value
        return None

    def _get_mask_generator(self) -> MultiTurnLossMaskGenerator:
        if self._mask_generator is None:
            tokenizer = load_tokenizer(
                self._args.hf_checkpoint,
                chat_template_path=getattr(self._args, "chat_template_path", None),
                trust_remote_code=True,
            )
            self._mask_generator = MultiTurnLossMaskGenerator(
                tokenizer, tokenizer_type=getattr(self._args, "loss_mask_type", "qwen")
            )
        return self._mask_generator


def validate_configuration(args: Any) -> None:
    if not getattr(args, "sprout_rollout_base_url", None):
        raise ValueError("--sprout-rollout-base-url is required by SproutRolloutFn")
    for name in (
        "sprout_rollout_timeout_seconds",
        "sprout_rollout_finalization_timeout_seconds",
        "sprout_rollout_poll_interval_seconds",
        "sprout_rollout_http_timeout_seconds",
    ):
        value = getattr(args, name, None)
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be finite and positive")
    max_turns = getattr(args, "sprout_rollout_max_turns", None)
    if type(max_turns) is not int or max_turns <= 0:
        raise ValueError("--sprout-rollout-max-turns must be a positive integer")
    for name in ("rollout_batch_size", "n_samples_per_prompt"):
        if type(getattr(args, name, None)) is not int or getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be a positive integer")
    if not getattr(args, "compute_advantages_and_returns", True):
        raise ValueError("Sprout rollouts need the ordinary RL forward and advantage path")
    for name in TOKEN_REPLAY_FLAGS:
        if getattr(args, name, False):
            raise ValueError(f"--{name.replace('_', '-')} is incompatible with retokenized Sprout trajectories")
    if getattr(args, "rollout_stop_token_ids", None):
        raise ValueError("Sprout speaks Chat Completions: use --rollout-stop text, not stop token ids")
    if getattr(args, "apply_chat_template_kwargs", None):
        raise ValueError("Sprout's gateway renders the chat template; --apply-chat-template-kwargs cannot reach it")


def sampling_params(args: Any) -> dict[str, Any]:
    """Miles's sampling flags in the vocabulary Sprout's rollout contract takes."""
    params: dict[str, Any] = {"temperature": args.rollout_temperature, "top_p": args.rollout_top_p}
    # Miles's -1 means top-k is off; Sprout forwards top_k verbatim to the
    # provider, so only a value that enables it is sent.
    if getattr(args, "rollout_top_k", -1) > 0:
        params["top_k"] = args.rollout_top_k
    if getattr(args, "rollout_stop", None):
        params["stop"] = list(args.rollout_stop)
    if getattr(args, "rollout_max_response_len", None) is not None:
        params["max_new_tokens"] = args.rollout_max_response_len
    return params


def validate_group(group: list[Sample]) -> tuple[str, str | list[dict[str, Any]]]:
    if not group:
        raise ValueError("cannot submit an empty prompt group")
    group_indices = {sample.group_index for sample in group}
    if None in group_indices or len(group_indices) != 1:
        raise ValueError(f"all samples must share one group_index, got {group_indices}")
    indices = [sample.index for sample in group]
    if any(index is None for index in indices) or len(set(indices)) != len(indices):
        raise ValueError(f"sample indices must be set and unique, got {indices}")
    if any(sample.prompt != group[0].prompt for sample in group[1:]):
        raise ValueError("all samples in a prompt group must share the prompt")
    if any(
        sample.multimodal_inputs and any(v is not None for v in sample.multimodal_inputs.values()) for sample in group
    ):
        raise ValueError("Sprout rollouts take text prompts only")
    return str(group[0].group_index), group[0].prompt


def task_identity(group: list[Sample]) -> tuple[str, str]:
    """The Sprout task and sandbox image a prompt group names in its metadata."""
    identities = []
    for sample in group:
        metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
        task_id = metadata.get("task_id")
        image = metadata.get("image")
        if not isinstance(task_id, str) or not task_id.strip():
            raise ValueError("Sprout rollouts need metadata['task_id'] on every sample")
        if not isinstance(image, str) or not image.strip():
            raise ValueError("Sprout rollouts need metadata['image'] on every sample")
        identities.append((task_id, image))
    if any(identity != identities[0] for identity in identities[1:]):
        raise ValueError("all samples in a prompt group must share task_id and image")
    return identities[0]


def validate_result(request: RolloutRequest, result: RolloutResult) -> None:
    if result.rollout_job_id != request.rollout_job_id or result.prompt_group_id != request.prompt_group_id:
        raise ValueError(
            f"driver result {result.rollout_job_id!r}/{result.prompt_group_id!r} is not the group submitted"
        )
    if result.max_samples != request.max_samples:
        raise ValueError(f"driver reports max_samples={result.max_samples}; expected {request.max_samples}")
    if result.status not in {"completed", "early_stopped"}:
        detail = f": {result.stop_reason}" if result.stop_reason else ""
        raise RuntimeError(f"Sprout rollout {request.rollout_job_id!r} ended with status={result.status!r}{detail}")
    if result.actual_samples != request.max_samples:
        raise ValueError(
            f"Sprout returned {result.actual_samples} of {request.max_samples} samples for a fixed-size group"
            + (f": {result.stop_reason}" if result.stop_reason else "")
        )


async def _release_without_masking_error(client: SproutRolloutClient, rollout_job_id: str) -> None:
    release = asyncio.create_task(client.delete(rollout_job_id))
    try:
        await asyncio.shield(release)
    except asyncio.CancelledError:
        try:
            await release
        except Exception as error:
            logger.warning("Failed to release Sprout rollout %s during cancellation: %r", rollout_job_id, error)
        raise
    except Exception as error:
        logger.warning("Failed to release Sprout rollout %s: %r", rollout_job_id, error)


def collect_metrics(results: list[RolloutResult]) -> dict[str, int | float]:
    trajectories = [trajectory for result in results for trajectory in result.trajectories]
    metrics: dict[str, int | float] = {
        "rollout/sprout/groups": len(results),
        "rollout/sprout/samples": len(trajectories),
        "rollout/sprout/truncated": sum(trajectory.status == "truncated" for trajectory in trajectories),
        "rollout/sprout/search_branches": sum(result.search_branches for result in results),
        "rollout/sprout/model_calls": sum(int(result.consumed_budget.get("model_calls", 0)) for result in results),
        "rollout/sprout/tool_calls": sum(int(result.consumed_budget.get("tool_calls", 0)) for result in results),
    }
    if trajectories:
        metrics["rollout/sprout/reward_mean"] = sum(float(t.reward) for t in trajectories) / len(trajectories)
    return metrics
