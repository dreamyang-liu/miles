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

With ``--sprout-rollout-search-points`` a group is a hindsight search instead
(``protocol.SearchSpec``): two roots, then the branches Sprout runs from points
in the failed ones, every sample credited by GSML (``gsml.assemble_search_group``)
and given its advantage by ``rewards.post_process_gsml``. That advantage is
final and each trainable token weighs alike, so construction refuses a search
without ``--n-samples-per-prompt 2``, that post-process,
``--advantage-estimator grpo``, ``--calculate-per-token-loss`` and a dynamic
global batch, or with ``--normalize-advantages``. The first step probes the
chat template before it submits anything (``gsml.check_reasoning_is_trained``).

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
from miles.rollout.sprout import gsml, rewards
from miles.rollout.sprout.client import SproutRolloutClient
from miles.rollout.sprout.importer import import_trajectories
from miles.rollout.sprout.protocol import Budget, RolloutRequest, RolloutResult, SampleSlot, SearchSpec
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
    "use_sampling_support_replay",
)
#: The replay above that no flag names: Miles derives it from the sampling flags (``miles.utils.arguments``).
_DERIVED_REPLAY = {
    "use_sampling_support_replay": "sampling-support replay (--rollout-top-p below 1 or a positive --rollout-top-k)"
}
#: The ``--custom-reward-post-process-path`` a search needs: it applies GSML's credit as the advantage.
GSML_POST_PROCESS = "miles.rollout.sprout.rewards.post_process_gsml"


@dataclass(frozen=True)
class _GroupRollout:
    """What one prompt group came back with: its samples for the trainer and Sprout's result."""

    samples: list[Sample]
    result: RolloutResult
    #: ``rollout/sprout/gsml/`` counts for a search group (``gsml.assemble_search_group``); empty otherwise.
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
            "--sprout-rollout-search-points",
            type=int,
            default=0,
            help=(
                "Hindsight search: after a task's roots are graded, Sprout reviews each failed one on its own and "
                "picks up to this many branch points in it; 0 (the default) runs the roots alone. GSML credits a "
                "search group, which needs --n-samples-per-prompt 2, --custom-reward-post-process-path "
                f"{GSML_POST_PROCESS}, --calculate-per-token-loss and --use-dynamic-global-batch-size."
            ),
        )
        group.add_argument(
            "--sprout-rollout-search-candidates",
            type=int,
            default=4,
            help=(
                "Independent repair turns the review writes per point when every root failed; 0 runs student "
                "continuations only."
            ),
        )
        group.add_argument(
            "--sprout-rollout-search-student-continuations",
            type=int,
            default=2,
            help=(
                "Continuations per point with nothing added when every root failed: the policy's own first step "
                "there."
            ),
        )
        group.add_argument(
            "--sprout-rollout-search-candidate-continuations",
            type=int,
            default=2,
            help="Continuations after each inserted repair turn.",
        )
        group.add_argument(
            "--sprout-rollout-search-tf-student-continuations",
            type=int,
            default=2,
            help=(
                "Continuations per point with nothing added when the roots split, or none resolved and one is "
                "missing: such a task gets no repair turns, and 0 does not branch it at all."
            ),
        )
        group.add_argument(
            "--sprout-rollout-search-repair-rounds",
            type=int,
            default=1,
            help=(
                "Rounds of repair when every root failed: where nothing has resolved, the last round's failed "
                "repair is reviewed again and one more turn inserted in its history. More than one round needs "
                "one candidate and one candidate continuation."
            ),
        )
        group.add_argument(
            "--sprout-rollout-search-review-seconds",
            type=float,
            default=3600.0,
            help="Wall time for each round's review job.",
        )
        group.add_argument(
            "--sprout-rollout-search-wall-time-seconds",
            type=float,
            default=3600.0,
            help="Wall time for each round's branch phase, after the roots and that round's review.",
        )
        # GSML's coefficients, read when the step is trained (rewards.post_process_gsml), not here.
        group.add_argument(
            "--sprout-rollout-gsml-lambda",
            type=float,
            default=0.4,
            help=(
                "GSML's lambda, in [0, 1]: the mass a task whose every root failed spreads over the points where "
                "something resolved; a root's z-score is weighted (1 - lambda)/sqrt(2)."
            ),
        )
        group.add_argument(
            "--sprout-rollout-gsml-beta-branch",
            type=float,
            default=1.0,
            help="GSML's beta_br, nonnegative: the weight of a student or repair continuation's z-score.",
        )
        group.add_argument(
            "--sprout-rollout-gsml-psi",
            type=float,
            default=1.0,
            help=(
                "GSML's psi, nonnegative: the weight of the lambda share of an a-hat sample, a verified repair "
                "turn trained as a one-turn target."
            ),
        )
        group.add_argument(
            "--sprout-rollout-gsml-c-pre",
            type=float,
            default=0.0,
            help="Credit for the history a branch restores, which trains as context only: not implemented, 0.",
        )
        group.add_argument(
            "--sprout-rollout-gsml-kappa-plus",
            type=float,
            default=0.0,
            help="Guided credit for a task one of whose roots resolved: not implemented, 0.",
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
        # Branch and â samples need identities of their own, after every root's, and their
        # conditions groups of their own, after every prompt's; the groups draw from both as they go.
        indices = _fresh_after(sample.index for group in groups for sample in group)
        group_indices = _fresh_after(sample.group_index for group in groups for sample in group)
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
                        group_indices=group_indices,
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
        return _step_output(outcomes)

    def search_spec(self, roots: int) -> SearchSpec | None:
        args = self._args
        if not getattr(args, "sprout_rollout_search_points", 0):
            return None
        return SearchSpec(
            roots=roots,
            points=args.sprout_rollout_search_points,
            candidates=args.sprout_rollout_search_candidates,
            student_continuations=args.sprout_rollout_search_student_continuations,
            candidate_continuations=args.sprout_rollout_search_candidate_continuations,
            tf_student_continuations=args.sprout_rollout_search_tf_student_continuations,
            repair_rounds=getattr(args, "sprout_rollout_search_repair_rounds", 1),
            review_seconds=args.sprout_rollout_search_review_seconds,
            wall_time_seconds=args.sprout_rollout_search_wall_time_seconds,
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
        group_indices: Iterator[int],
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
                group_indices=group_indices,
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
        group_indices: Iterator[int],
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
            if request.search is None:
                samples = import_trajectories(result, slots, mask_generator, weight_version=weight_version)
                return _GroupRollout(samples=samples, result=result, measured={})
            samples, measured = gsml.assemble_search_group(
                result,
                slots,
                mask_generator,
                planned_roots=request.search.roots,
                weight_version=weight_version,
                indices=indices,
                group_indices=group_indices,
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
        """The policy's chat template and loss mask, built once; one set on the instance is used as it is."""
        if self._mask_generator is None:
            tokenizer = load_tokenizer(
                self._args.hf_checkpoint,
                chat_template_path=getattr(self._args, "chat_template_path", None),
                trust_remote_code=True,
            )
            mask_generator = MultiTurnLossMaskGenerator(
                tokenizer, tokenizer_type=getattr(self._args, "loss_mask_type", "qwen")
            )
            if self._search is not None:
                # A search trains the reviewer's inserted turns, their text as reasoning (the â
                # samples): refuse a template that would train their calls without it.
                gsml.check_reasoning_is_trained(mask_generator)
            self._mask_generator = mask_generator
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
            flag = _DERIVED_REPLAY.get(name, f"--{name.replace('_', '-')}")
            raise ValueError(f"{flag} is incompatible with retokenized Sprout trajectories")
    if getattr(args, "rollout_stop_token_ids", None):
        raise ValueError("Sprout speaks Chat Completions: use --rollout-stop text, not stop token ids")
    if getattr(args, "apply_chat_template_kwargs", None):
        raise ValueError("Sprout's gateway renders the chat template; --apply-chat-template-kwargs cannot reach it")
    points = getattr(args, "sprout_rollout_search_points", 0)
    if type(points) is not int or points < 0:
        raise ValueError("--sprout-rollout-search-points must be a nonnegative integer")
    if points:
        _validate_search(args)
    elif getattr(args, "custom_reward_post_process_path", None) == GSML_POST_PROCESS:
        raise ValueError(
            f"--custom-reward-post-process-path {GSML_POST_PROCESS} credits search groups only: set "
            "--sprout-rollout-search-points, or drop it for GRPO's own normalization"
        )


def _validate_search(args: Any) -> None:
    """A search's shape, and what GSML's credit needs of the trainer (``rewards.post_process_gsml``)."""
    for name in (
        "sprout_rollout_search_candidates",
        "sprout_rollout_search_student_continuations",
        "sprout_rollout_search_candidate_continuations",
        "sprout_rollout_search_tf_student_continuations",
    ):
        value = getattr(args, name, None)
        if type(value) is not int or value < 0:
            raise ValueError(f"--{name.replace('_', '-')} must be a nonnegative integer")
    if args.sprout_rollout_search_student_continuations < 1:
        raise ValueError("--sprout-rollout-search-student-continuations must be positive: they are the baseline")
    rounds = getattr(args, "sprout_rollout_search_repair_rounds", 1)
    if type(rounds) is not int or rounds < 1:
        raise ValueError("--sprout-rollout-search-repair-rounds must be a positive integer")
    if rounds > 1 and (args.sprout_rollout_search_candidates, args.sprout_rollout_search_candidate_continuations) != (
        1,
        1,
    ):
        raise ValueError(
            "--sprout-rollout-search-repair-rounds above 1 needs --sprout-rollout-search-candidates 1 and "
            "--sprout-rollout-search-candidate-continuations 1"
        )
    if (args.sprout_rollout_search_candidates == 0) != (args.sprout_rollout_search_candidate_continuations == 0):
        raise ValueError(
            "--sprout-rollout-search-candidates and --sprout-rollout-search-candidate-continuations are both 0 or "
            "both positive"
        )
    for name in ("sprout_rollout_search_review_seconds", "sprout_rollout_search_wall_time_seconds"):
        value = getattr(args, name, None)
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be finite and positive")
    if args.n_samples_per_prompt != 2:
        raise ValueError(
            "a search needs --n-samples-per-prompt 2: GSML's credit is defined for a task's two roots, "
            "a root's z-score weighted (1 - lambda)/sqrt(2)"
        )
    if getattr(args, "advantage_estimator", None) != "grpo":
        raise ValueError(
            "a search needs --advantage-estimator grpo: it broadcasts each sample's GSML advantage over its tokens"
        )
    if getattr(args, "custom_reward_post_process_path", None) != GSML_POST_PROCESS:
        raise ValueError(
            f"a search needs --custom-reward-post-process-path {GSML_POST_PROCESS}: it turns the credit "
            "gsml.assemble_search_group gives each sample into its advantage"
        )
    if getattr(args, "normalize_advantages", False):
        raise ValueError(
            "a search refuses --normalize-advantages: GSML's advantages are final, and whitening them across the "
            "batch would move every zero and rescale the lambda shares"
        )
    if not getattr(args, "calculate_per_token_loss", False):
        raise ValueError(
            "a search needs --calculate-per-token-loss: a GSML advantage is every trainable token's, and a "
            "per-sample mean would weigh a sample's tokens by its length"
        )
    if not getattr(args, "use_dynamic_global_batch_size", False):
        raise ValueError(
            "a search needs --use-dynamic-global-batch-size: how many samples a search step returns varies, and a "
            "fixed --global-batch-size trims the step to a multiple of it, dropping the samples at the end"
        )
    # λ, β_br and ψ in range, c_pre and κ+ 0: what post_process_gsml checks again when it applies them.
    rewards.coefficients(args)


def sampling_params(args: Any) -> dict[str, Any]:
    """Miles's sampling flags in the vocabulary Sprout's rollout contract takes."""
    params: dict[str, Any] = {
        "temperature": args.rollout_temperature,
        "top_p": args.rollout_top_p,
        # Always sent, Miles's -1 (off) included, and Sprout forwards it verbatim. A request without
        # top_k leaves it to the engine, which takes the checkpoint's generation_config (top_k 20 for
        # Qwen3) and samples from a truncated distribution while the trainer scores the full softmax:
        # Miles's --rollout-top-k wins over the checkpoint's.
        "top_k": getattr(args, "rollout_top_k", -1),
    }
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
    roots = [slot.sample_slot_id for slot in request.sample_slots[: request.search.roots]] if request.search else []
    failed = {sample.sample_slot_id for sample in result.failed_samples}
    if not (result.status in {"completed", "early_stopped"}
            or result.status == "failed" and not result.trajectories and roots and set(roots) <= failed):
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
            raise ValueError(f"trajectory {trajectory.sample_slot_id!r} is grouped as {trajectory.group!r} in its slot")


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
    return metrics
