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

With ``--sprout-rollout-rafo`` a group is a RAFO search instead
(``protocol.RafoSearchSpec``): the group's roots, then, when none resolved,
the events Sprout runs after a reviewer's turn in each failed root, and deeper
in a failed continuation. Every sample is credited by RAFO
(``rafo.assemble_rafo_group``) and trained by the RAFO loss, which needs the
Megatron backend, a fixed global batch that counts tasks, and no reward
normalization (``_validate_rafo``). The first step probes the chat template
before it submits anything (``importer.check_reasoning_is_trained``).

A group that fails -- rejected, failed at Sprout, or returned in a shape Miles
cannot train -- is logged and left out of the step, counted in
``rollout/sprout/groups_failed``; the step fails only when every group failed
or none returned a sample to train.

What Sprout returns is the conversation, not the sampled tokens: Miles renders
it with the policy's chat template and recomputes log-probs. Flags that assume
the rollout engine's own tokens (``--use-rollout-logprobs``, TIS, actor forward
skipping, routing, indexer or sampling-support replay) are refused rather than
silently applied to a different sequence.
"""

from __future__ import annotations

import asyncio
import copy
import itertools
import logging
import math
import os
import uuid
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
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
from miles.rollout.sprout import rafo
from miles.rollout.sprout.client import SproutRolloutClient
from miles.rollout.sprout.importer import check_reasoning_is_trained, import_trajectories
from miles.rollout.sprout.protocol import (
    INVALID_STOP_REASONS,
    Budget,
    RafoSearchSpec,
    RolloutRequest,
    RolloutResult,
    SampleSlot,
)
from miles.utils.args.custom_function import CustomFunctionConfig
from miles.utils.mask_utils import MultiTurnLossMaskGenerator
from miles.utils.processing_utils import load_tokenizer
from miles.utils.types import Sample

logger = logging.getLogger(__name__)

#: The ``--custom-loss-function-path`` of a RAFO search: PPO and CE on the roles ``rafo.assemble_rafo_group`` gives.
RAFO_LOSS = "miles.backends.training_utils.loss.hub.rafo_loss.rafo_loss_function"


@dataclass(frozen=True)
class _GroupRollout:
    """What one prompt group came back with: its samples for the trainer and Sprout's result."""

    samples: list[Sample]
    result: RolloutResult
    #: ``rollout/sprout/rafo/`` counts for a search group (``rafo.assemble_rafo_group``); empty otherwise.
    measured: dict[str, float]


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
        # A search's roots are the prompt group: n_samples_per_prompt of them (checked per group).
        self._search = self.search_spec(self._args.n_samples_per_prompt)
        self._rafo = rafo_settings(self._args) if self._search is not None else None

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
        group.add_argument(
            "--sprout-rollout-drop-zero-advantage-groups",
            action="store_true",
            help=(
                "Leave out of the step each plain group whose samples all got the same reward: group-centred "
                "advantages give it none, and its tokens would only dilute the batch's token mean "
                "(Arctic-SWE's zero-advantage filter). A step whose every group is so keeps them all."
            ),
        )
        group.add_argument(
            "--sprout-rollout-rafo",
            action="store_true",
            help=(
                "Run each prompt group as a RAFO search: when none of its --n-samples-per-prompt roots resolved, "
                "Sprout's reviewer writes a turn that exposes the problem in each failed root and continuations "
                "run after it, and deeper in a failed continuation. The RAFO loss trains every sample of the "
                f"task: --loss-type custom_loss --custom-loss-function-path {RAFO_LOSS}."
            ),
        )
        group.add_argument(
            "--sprout-rollout-rafo-continuations",
            type=int,
            default=2,
            help="RAFO's K: the continuations after each reviewer turn (an event), one local RLOO group.",
        )
        group.add_argument(
            "--sprout-rollout-rafo-depth",
            type=int,
            default=3,
            help="RAFO's D: the most reviews, each inside the last event's first failed continuation.",
        )
        group.add_argument(
            "--sprout-rollout-rafo-events",
            type=int,
            default=4,
            help="RAFO's S: the most events of a task, and the constant that normalizes their samples.",
        )
        group.add_argument(
            "--sprout-rollout-rafo-review-seconds",
            type=float,
            default=3600.0,
            help="Wall time for each depth's review job.",
        )
        group.add_argument(
            "--sprout-rollout-rafo-wall-time-seconds",
            type=float,
            default=3600.0,
            help="Wall time for each depth's continuations, after that depth's review.",
        )
        group.add_argument(
            "--sprout-rollout-rafo-c",
            type=float,
            default=64.0,
            help="RAFO's token normalizer C of a trajectory's PPO tokens.",
        )
        group.add_argument(
            "--sprout-rollout-rafo-c-h",
            type=float,
            default=32.0,
            help="RAFO's token normalizer C_h of a reviewer turn's CE tokens.",
        )
        group.add_argument(
            "--sprout-rollout-rafo-turn-credit",
            type=str,
            choices=rafo.TURN_CREDITS,
            default="subtree",
            help=(
                "A reviewer turn's weight: a continuation that a deeper event rescued counts with that event's "
                "weight (subtree), or only the turn's own continuations count (own)."
            ),
        )
        return parser

    async def __call__(self, input: RolloutFnInput) -> RolloutFnOutput:
        if isinstance(input, RolloutFnEvalInput):
            raise NotImplementedError(
                "SproutRolloutFn serves training rollouts only; configure --eval-function-path separately"
            )
        return await self._call_train(input)

    async def _call_train(self, input: RolloutFnTrainInput) -> RolloutFnTrainOutput:
        # Loaded, and for a search probed, before anything is submitted.
        mask_generator = self._get_mask_generator()
        groups = self._data_source.get_samples(self._args.rollout_batch_size)
        if len(groups) != self._args.rollout_batch_size:
            raise ValueError(
                f"data source returned {len(groups)} prompt groups; expected {self._args.rollout_batch_size}"
            )
        # Branch and turn samples need identities of their own, after every root's; the groups draw
        # from them as they go.
        indices = _fresh_after(sample.index for group in groups for sample in group)
        async with self._client_factory(
            self._args.sprout_rollout_base_url,
            timeout=self._args.sprout_rollout_http_timeout_seconds,
            token=os.environ.get(self._args.sprout_rollout_token_env or "") or None,
        ) as client:
            tasks = [
                asyncio.create_task(
                    self._run_group_isolated(
                        client=client,
                        group=group,
                        rollout_id=input.rollout_id,
                        weight_version=input.weight_version,
                        mask_generator=mask_generator,
                        indices=indices,
                    )
                )
                for group in groups
            ]
            try:
                outcomes = await asyncio.gather(*tasks)
            except BaseException:
                # Cancelled or interrupted: no group outlives the step.
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                raise
        if getattr(self._args, "sprout_rollout_drop_zero_advantage_groups", False) and self._search is None:
            outcomes = drop_zero_advantage_groups(outcomes)
        return _step_output(outcomes)

    def search_spec(self, roots: int) -> RafoSearchSpec | None:
        args = self._args
        if not args.sprout_rollout_rafo:
            return None
        return RafoSearchSpec(
            roots=roots,
            continuations=args.sprout_rollout_rafo_continuations,
            depth=args.sprout_rollout_rafo_depth,
            events=args.sprout_rollout_rafo_events,
            review_seconds=float(args.sprout_rollout_rafo_review_seconds),
            wall_time_seconds=float(args.sprout_rollout_rafo_wall_time_seconds),
        )

    async def _run_group_isolated(
        self,
        *,
        client: SproutRolloutClient,
        group: list[Sample],
        rollout_id: int,
        weight_version: int | None,
        mask_generator: MultiTurnLossMaskGenerator,
        indices: Iterator[int],
    ) -> _GroupRollout | Exception:
        """One group's rollout, or the error that failed it: no group's failure fails another.

        Only an ``Exception`` is caught; a cancellation or an interrupt still
        stops every group of the step (``_call_train``).
        """
        request = None
        try:
            request, slots = self._build_request(group=group, rollout_id=rollout_id, indices=indices)
            return await self._run_group(
                client=client,
                request=request,
                slots=slots,
                weight_version=weight_version,
                mask_generator=mask_generator,
                indices=indices,
            )
        except Exception as error:
            logger.exception(
                "Sprout rollout %s of prompt group %s failed and is left out of the step: %r",
                "(not submitted)" if request is None else request.rollout_job_id,
                group[0].group_index if group else None,
                error,
            )
            return error

    async def _run_group(
        self,
        *,
        client: SproutRolloutClient,
        request: RolloutRequest,
        slots: dict[str, Sample],
        weight_version: int | None,
        mask_generator: MultiTurnLossMaskGenerator,
        indices: Iterator[int],
    ) -> _GroupRollout:
        try:
            acknowledgement = await client.submit(request)
            if acknowledgement.rollout_job_id != request.rollout_job_id:
                raise ValueError(
                    f"driver acknowledged {acknowledgement.rollout_job_id!r}; expected {request.rollout_job_id!r}"
                )
            timeout = request.budgets.max_wall_time_seconds + request.finalization_timeout_seconds
            if request.search is not None:
                # A grade of a round before the last waits for its round within finalization's allowance.
                timeout += request.search.search_seconds(request.finalization_timeout_seconds)
            result = await asyncio.wait_for(
                client.wait_for_result(
                    request.rollout_job_id, poll_interval_seconds=self._args.sprout_rollout_poll_interval_seconds
                ),
                timeout=timeout,
            )
            validate_result(request, result)
            if request.search is None or self._rafo is None:
                samples = import_trajectories(result, slots, mask_generator, weight_version=weight_version)
                return _GroupRollout(samples=samples, result=result, measured={})
            samples, measured = rafo.assemble_rafo_group(
                result,
                slots,
                mask_generator,
                spec=request.search,
                settings=self._rafo,
                weight_version=weight_version,
                indices=indices,
            )
            return _GroupRollout(samples=samples, result=result, measured=measured)
        finally:
            # The driver keeps a terminal group until it is released, and a
            # group still running is cancelled; either way nothing leaks.
            await _release_without_masking_error(client, request.rollout_job_id)

    def _build_request(
        self, *, group: list[Sample], rollout_id: int, indices: Iterator[int] | None = None
    ) -> tuple[RolloutRequest, dict[str, Sample]]:
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
        slot_samples = list(group)
        search = self._search
        if search is not None:
            if indices is None:
                indices = itertools.count(max(sample.index for sample in group) + 1)
            # The branches' slots: copies of the prompt with identities of their own,
            # grouped later by the condition Sprout reports they were sampled under.
            for _ in range(search.branches()):
                index = next(indices)
                slots.append(SampleSlot(sample_slot_id=f"{job_id}:slot:{index}", sample_index=index))
                slot_samples.append(copy.deepcopy(group[0]))
                slot_samples[-1].index = index
                slot_samples[-1].group_index = None
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
            # Without a search the group size is fixed: GRPO normalizes within
            # it, and a group Sprout could not fill is an error, not a smaller
            # group. A search fills its branch slots as it goes and may lose a
            # root to a failed actor, which it names (``failed_samples``).
            minimum_returned_samples=1 if search is not None else len(group),
            max_turns=self._args.sprout_rollout_max_turns,
            finalization_timeout_seconds=self._args.sprout_rollout_finalization_timeout_seconds,
            sampling_params=dict(self._sampling_params),
            budgets=Budget(max_wall_time_seconds=self._args.sprout_rollout_timeout_seconds),
            search=search,
        )
        return request, dict(zip((slot.sample_slot_id for slot in slots), slot_samples, strict=True))

    def model_endpoint(self) -> str:
        if configured := self._args.sprout_rollout_model_endpoint:
            return configured.rstrip("/")
        host, port = self._args.sglang_router_ip, self._args.sglang_router_port
        if not host or port is None:
            raise RuntimeError(
                "Miles's rollout router address is not known yet; pass --sprout-rollout-model-endpoint "
                "or submit after the rollout engines start"
            )
        return f"http://{host}:{port}"

    def model_name(self) -> str | None:
        """The ``model`` of the Chat Completions requests Sprout's gateway sends."""
        served = getattr(self._args, "sglang_served_model_name", None)  # config-access-exempt: an SGLang server arg
        return served or self._args.model_name or None

    def _get_mask_generator(self) -> MultiTurnLossMaskGenerator:
        """The policy's chat template and loss mask, built once; one set on the instance is used as it is."""
        if self._mask_generator is None:
            tokenizer = load_tokenizer(
                self._args.hf_checkpoint, chat_template_path=self._args.chat_template_path, trust_remote_code=True
            )
            mask_generator = MultiTurnLossMaskGenerator(tokenizer, tokenizer_type=self._args.loss_mask_type)
            if self._search is not None:
                # A search trains the reviewer's inserted turns, their text as reasoning (their CE
                # samples): refuse a template that would train their calls without it.
                check_reasoning_is_trained(mask_generator)
            self._mask_generator = mask_generator
        return self._mask_generator


def validate_configuration(args: Any) -> None:
    if not args.sprout_rollout_base_url:
        raise ValueError("--sprout-rollout-base-url is required by SproutRolloutFn")
    _check_positive(
        {
            "--sprout-rollout-timeout-seconds": args.sprout_rollout_timeout_seconds,
            "--sprout-rollout-finalization-timeout-seconds": args.sprout_rollout_finalization_timeout_seconds,
            "--sprout-rollout-poll-interval-seconds": args.sprout_rollout_poll_interval_seconds,
            "--sprout-rollout-http-timeout-seconds": args.sprout_rollout_http_timeout_seconds,
        },
        integer=False,
    )
    _check_positive(
        {
            "--sprout-rollout-max-turns": args.sprout_rollout_max_turns,
            "--rollout-batch-size": args.rollout_batch_size,
            "--n-samples-per-prompt": args.n_samples_per_prompt,
        },
        integer=True,
    )
    if not args.compute_advantages_and_returns:
        raise ValueError("Sprout rollouts need the ordinary RL forward and advantage path")
    for flag, on in _token_replay(args).items():
        if on:
            raise ValueError(f"{flag} is incompatible with retokenized Sprout trajectories")
    if args.rollout_stop_token_ids:
        raise ValueError("Sprout speaks Chat Completions: use --rollout-stop text, not stop token ids")
    if args.apply_chat_template_kwargs:
        raise ValueError("Sprout's gateway renders the chat template; --apply-chat-template-kwargs cannot reach it")
    if args.sprout_rollout_rafo:
        _validate_rafo(args)
    elif _custom_loss_path(args) == RAFO_LOSS:
        raise ValueError(f"{RAFO_LOSS} trains RAFO search groups: set --sprout-rollout-rafo")


def _validate_rafo(args: Any) -> None:
    """A RAFO search's shape, and what RAFO's credit needs of the trainer (``rafo.assemble_rafo_group`` and
    the RAFO loss)."""
    _check_positive(
        {
            "--sprout-rollout-rafo-continuations": args.sprout_rollout_rafo_continuations,
            "--sprout-rollout-rafo-depth": args.sprout_rollout_rafo_depth,
            "--sprout-rollout-rafo-events": args.sprout_rollout_rafo_events,
        },
        integer=True,
    )
    _check_positive(
        {
            "--sprout-rollout-rafo-review-seconds": args.sprout_rollout_rafo_review_seconds,
            "--sprout-rollout-rafo-wall-time-seconds": args.sprout_rollout_rafo_wall_time_seconds,
        },
        integer=False,
    )
    required = {
        "--n-samples-per-prompt of at least 2: a task's roots are one local RLOO group": (
            args.n_samples_per_prompt >= 2
        ),
        "--sprout-rollout-rafo-continuations of at least 2: an event's continuations are one local RLOO group": (
            args.sprout_rollout_rafo_continuations >= 2
        ),
        f"--loss-type custom_loss --custom-loss-function-path {RAFO_LOSS}": (
            args.loss_type == "custom_loss" and _custom_loss_path(args) == RAFO_LOSS
        ),
        "--train-backend megatron: it trains a task's samples in one step and passes their roles to the loss": (
            args.train_backend == "megatron"
        ),
        "--disable-rewards-normalization: the samples of one task share a rollout id and differ in reward": (
            not args.rewards_normalization
        ),
        "no --custom-reward-post-process-path: the RAFO loss reads its advantages from the samples": (
            args.custom_reward_post_process_path is None
        ),
        "no --use-dynamic-global-batch-size: --global-batch-size counts the tasks of an optimizer step": (
            not args.use_dynamic_global_batch_size
        ),
    }
    if missing := [condition for condition, met in required.items() if not met]:
        raise ValueError(f"--sprout-rollout-rafo needs {'; '.join(missing)}")
    rafo_settings(args)


def _check_positive(values: dict[str, object], *, integer: bool) -> None:
    """Raise ValueError naming the first flag whose value is not a positive integer, or not finite and positive."""
    for flag, value in values.items():
        if integer and (type(value) is not int or value <= 0):
            raise ValueError(f"{flag} must be a positive integer")
        if not integer and (type(value) not in (int, float) or not math.isfinite(value) or value <= 0):
            raise ValueError(f"{flag} must be finite and positive")


def _token_replay(args: Any) -> dict[str, bool]:
    """The Miles features that reuse the rollout engine's tokens or log-probs, by the flag that turns each on.
    The trajectory Sprout returns is retokenized, so none of them can be honoured."""
    return {
        "--use-rollout-logprobs": args.use_rollout_logprobs,
        "--skip-actor-forward-only": args.skip_actor_forward_only,
        "--use-tis": args.use_tis,
        "--use-rollout-routing-replay": args.use_rollout_routing_replay,
        "--use-rollout-indexer-replay": args.use_rollout_indexer_replay,
        # No flag names it: Miles derives it from the sampling flags (miles.utils.arguments).
        "sampling-support replay (--rollout-top-p below 1 or a positive --rollout-top-k)": (
            args.use_sampling_support_replay
        ),
    }


def _custom_loss_path(args: Any) -> str | None:
    loss = args.custom_loss_function_path
    return loss.path if isinstance(loss, CustomFunctionConfig) else loss


def rafo_settings(args: Any) -> rafo.RafoSettings:
    return rafo.RafoSettings(
        c=args.sprout_rollout_rafo_c,
        c_h=args.sprout_rollout_rafo_c_h,
        turn_credit=args.sprout_rollout_rafo_turn_credit,
    )


def sampling_params(args: Any) -> dict[str, Any]:
    """Miles's sampling flags in the vocabulary Sprout's rollout contract takes."""
    params: dict[str, Any] = {
        "temperature": args.rollout_temperature,
        "top_p": args.rollout_top_p,
        # Always sent, Miles's -1 (off) included, and Sprout forwards it verbatim. A request without
        # top_k leaves it to the engine, which takes the checkpoint's generation_config (top_k 20 for
        # Qwen3) and samples from a truncated distribution while the trainer scores the full softmax:
        # Miles's --rollout-top-k wins over the checkpoint's.
        "top_k": args.rollout_top_k,
    }
    if args.rollout_stop:
        params["stop"] = list(args.rollout_stop)
    if args.rollout_max_response_len is not None:
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
    roots = [slot.sample_slot_id for slot in request.sample_slots[: request.search.roots]] if request.search else []
    failed = {sample.sample_slot_id for sample in result.failed_samples}
    if not (
        result.status in {"completed", "early_stopped"}
        or result.status == "failed"
        and not result.trajectories
        and roots
        and set(roots) <= failed
    ):
        detail = f": {result.stop_reason}" if result.stop_reason else ""
        raise RuntimeError(f"Sprout rollout {request.rollout_job_id!r} ended with status={result.status!r}{detail}")
    if request.search is None:
        if result.actual_samples != request.max_samples:
            raise ValueError(
                f"Sprout returned {result.actual_samples} of {request.max_samples} samples for a fixed-size group"
                + (f": {result.stop_reason}" if result.stop_reason else "")
            )
        return
    # A search fills the root slots and as many branch slots as the review led to.
    # A root may be missing only when Sprout says its actor failed.
    returned = {trajectory.sample_slot_id for trajectory in result.trajectories}
    missing = [slot for slot in roots if slot not in returned and slot not in failed]
    if missing:
        raise ValueError(
            f"Sprout did not return root {missing[0]!r} of {request.rollout_job_id!r}"
            + (f": {result.stop_reason}" if result.stop_reason else "")
        )
    for trajectory in result.trajectories:
        if (trajectory.group == "root") != (trajectory.sample_slot_id in roots):
            raise ValueError(
                f"trajectory {trajectory.sample_slot_id!r} is grouped as {trajectory.group!r} in its slot"
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


def _fresh_after(taken: Iterable[int | None]) -> Iterator[int]:
    """Identities after every one already taken; an unset one is the group's to refuse (``validate_group``)."""
    return itertools.count(max((value for value in taken if value is not None), default=-1) + 1)


def drop_zero_advantage_groups(outcomes: list[_GroupRollout | Exception]) -> list[_GroupRollout | Exception]:
    """``outcomes`` with the samples of each group whose rewards are all equal left out, and the count of such
    groups in ``rollout/sprout/groups_zero_advantage``. Unchanged when no group has two rewards."""

    def uniform(outcome: _GroupRollout | Exception) -> bool:
        return isinstance(outcome, _GroupRollout) and len({float(s.reward) for s in outcome.samples}) <= 1

    if all(uniform(outcome) or not isinstance(outcome, _GroupRollout) for outcome in outcomes):
        return outcomes
    kept: list[_GroupRollout | Exception] = []
    dropped = 0
    for outcome in outcomes:
        if uniform(outcome):
            assert isinstance(outcome, _GroupRollout)
            outcome = _GroupRollout(samples=[], result=outcome.result, measured=dict(outcome.measured))
            dropped += 1
        kept.append(outcome)
    for outcome in kept:
        if isinstance(outcome, _GroupRollout):
            outcome.measured["rollout/sprout/groups_zero_advantage"] = dropped
            break
    return kept


def _step_output(outcomes: list[_GroupRollout | Exception]) -> RolloutFnTrainOutput:
    """The step's samples by group, a failed group's as [], and its metrics.

    When every group failed, the first group's error is raised as it is; when
    the groups that came back hold no sample, the step has nothing to train.
    """
    rollouts = [outcome for outcome in outcomes if isinstance(outcome, _GroupRollout)]
    failed = [outcome for outcome in outcomes if not isinstance(outcome, _GroupRollout)]
    if not rollouts:
        raise failed[0]
    samples = [outcome.samples if isinstance(outcome, _GroupRollout) else [] for outcome in outcomes]
    if not any(samples):
        raise RuntimeError(
            f"no trainable sample in this step: {len(failed)} of {len(outcomes)} groups failed and the rest "
            "returned none"
        ) from (failed[0] if failed else None)
    metrics = collect_metrics([rollout.result for rollout in rollouts], groups_failed=len(failed))
    for rollout in rollouts:
        for key, value in rollout.measured.items():
            metrics[key] = metrics.get(key, 0) + value
    return RolloutFnTrainOutput(samples=samples, metrics=metrics)


def collect_metrics(results: list[RolloutResult], *, groups_failed: int = 0) -> dict[str, int | float]:
    """The step's metrics over the groups that came back; ``groups_failed`` more were left out of it."""
    trajectories = [trajectory for result in results for trajectory in result.trajectories]
    # A trajectory whose grade gave no verdict is missing, not a 0 (``Trajectory.metadata``).
    gradable = [trajectory for trajectory in trajectories if trajectory.metadata.get("zero_reason") is None]
    metrics: dict[str, int | float] = {
        "rollout/sprout/groups": len(results),
        "rollout/sprout/groups_failed": groups_failed,
        "rollout/sprout/samples": len(trajectories),
        "rollout/sprout/roots": sum(trajectory.group in (None, "root") for trajectory in trajectories),
        "rollout/sprout/truncated": sum(trajectory.status == "truncated" for trajectory in trajectories),
        "rollout/sprout/ungradable": len(trajectories) - len(gradable),
        "rollout/sprout/failed_samples": sum(len(result.failed_samples) for result in results),
        "rollout/sprout/search_branches": sum(result.search_branches for result in results),
        "rollout/sprout/model_calls": sum(int(result.consumed_budget.get("model_calls", 0)) for result in results),
        "rollout/sprout/tool_calls": sum(int(result.consumed_budget.get("tool_calls", 0)) for result in results),
    }
    if gradable:
        metrics["rollout/sprout/reward_mean"] = sum(float(t.reward) for t in gradable) / len(gradable)
        # The reward mixes resolves (1) with invalid attempts (-1): report each on its own.
        metrics["rollout/sprout/resolved_mean"] = sum(float(t.reward) == 1.0 for t in gradable) / len(gradable)
        metrics["rollout/sprout/invalid_mean"] = sum(t.stop_reason in INVALID_STOP_REASONS for t in gradable) / len(
            gradable
        )
    for trajectory in trajectories:
        if trajectory.stop_reason is not None:
            key = "rollout/sprout/stop/" + trajectory.stop_reason
            metrics[key] = metrics.get(key, 0) + 1
    return metrics
