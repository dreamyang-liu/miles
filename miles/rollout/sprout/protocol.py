"""The wire between Miles and Sprout's RL driver (``/rollout-groups``).

Sprout's side is ``sprout.rl_driver.messages.MessageRequest``; the shapes here
mirror it field for field. The contract carries no version tag and Sprout
rejects unknown fields, so every model is strict in both directions: a field
either side adds without the other is an error at the boundary, not a silent
omission in training data.
"""

from __future__ import annotations

import math
from typing import Annotated, Any, Literal

from pydantic import Field, StrictFloat, StrictInt, StringConstraints, model_validator

from miles.utils.pydantic_utils import FrozenStrictBaseModel

NonEmptyStr = Annotated[str, StringConstraints(min_length=1, pattern=r"\S")]
StrictNumber = StrictFloat | StrictInt

GroupStatus = Literal["queued", "running", "completed", "early_stopped", "failed", "cancelled"]
TrajectoryStatus = Literal["completed", "truncated"]
#: ``interrupted``: the attempt failed after closed turns and was cut back to the last of them.
#: ``max_total_tokens``: the trajectory spent its token budget. ``format_invalid``,
#: ``response_truncated`` and ``repetition`` end an invalid attempt (Arctic-SWE, Table 18), whose
#: reward Sprout fixes at -1 without grading it.
StopReason = Literal[
    "max_turns_reached",
    "timeout",
    "interrupted",
    "max_total_tokens",
    "format_invalid",
    "response_truncated",
    "repetition",
]
#: The stop reasons of an invalid attempt.
INVALID_STOP_REASONS = frozenset({"format_invalid", "response_truncated", "repetition"})

#: Group statuses the driver never leaves again.
TERMINAL_GROUP_STATUSES = frozenset({"completed", "early_stopped", "failed", "cancelled"})
#: Sprout wraps every branch continuation prompt in these before the agent sees
#: it and strips the span from user/system messages at export; a trajectory
#: that still carries one was not exported for training.
HINT_START = "<sprout_training_hint>"
HINT_END = "</sprout_training_hint>"


class SampleSlot(FrozenStrictBaseModel):
    sample_slot_id: NonEmptyStr
    sample_index: StrictInt = Field(ge=0)


class Budget(FrozenStrictBaseModel):
    """Wall time for the execution phase, counted from submission (queueing included)."""

    max_wall_time_seconds: float = Field(strict=True, gt=0, allow_inf_nan=False)


class RafoSearchSpec(FrozenStrictBaseModel):
    """RAFO's forest after the roots (Sprout's ``sprout.search.spec.RafoSearchSpec``).

    ``roots`` scratch rollouts take the first slots. Only when no graded root
    resolved does Sprout review: at one point of each failed root, at most
    ``events`` of them, a reviewer writes one turn in the agent's own voice
    that exposes the problem with a check and never fixes it, and
    ``continuations`` branches restored at the point run after the turn: one
    event. When every graded continuation of an event failed, the next depth
    reviews the first of them in the same way, while events are left, up to
    ``depth``. The branches fill the slots after the roots, ``branches()`` of
    them.
    """

    mode: Literal["rafo"] = "rafo"
    roots: StrictInt = Field(gt=0)
    continuations: StrictInt = Field(gt=0)
    depth: StrictInt = Field(gt=0)
    events: StrictInt = Field(gt=0)
    review_seconds: float = Field(strict=True, gt=0, allow_inf_nan=False)
    wall_time_seconds: float = Field(strict=True, gt=0, allow_inf_nan=False)

    def branches(self) -> int:
        """The most branches the search may make: ``continuations`` after each of its ``events`` turns.
        Sprout computes the same number and refuses fewer slots."""
        return self.events * self.continuations

    def search_seconds(self, grade_seconds: float) -> float:
        """How long the search may run after its roots: each depth's review and continuations, and the
        grades of every depth but the last (Sprout's ``RafoSearchSpec.search_seconds``). A depth needs an
        event of its own, so at most ``events`` depths run."""
        depths = min(self.depth, self.events)
        return depths * (self.review_seconds + self.wall_time_seconds) + (depths - 1) * grade_seconds


class RolloutRequest(FrozenStrictBaseModel):
    rollout_job_id: NonEmptyStr
    rollout_id: StrictInt = Field(ge=0)
    prompt_group_id: NonEmptyStr
    task_id: NonEmptyStr
    image: NonEmptyStr
    prompt: str | list[dict[str, Any]]
    model_endpoint: NonEmptyStr
    model: NonEmptyStr | None = None
    sample_slots: list[SampleSlot] = Field(min_length=1)
    max_samples: StrictInt = Field(gt=0)
    minimum_returned_samples: StrictInt = Field(ge=1)
    #: Model calls per trajectory; a response's tool calls do not count.
    max_turns: StrictInt = Field(gt=0)
    #: Extra time after the execution deadline for the final snapshot, the
    #: message export and grading.
    finalization_timeout_seconds: float = Field(default=1800.0, strict=True, gt=0, allow_inf_nan=False)
    sampling_params: dict[str, Any] = Field(default_factory=dict)
    budgets: Budget
    search: RafoSearchSpec | None = None

    @model_validator(mode="after")
    def validate_request(self) -> RolloutRequest:
        if not self.prompt:
            raise ValueError("prompt must be nonempty text or messages")
        if len({slot.sample_slot_id for slot in self.sample_slots}) != len(self.sample_slots) or len(
            {slot.sample_index for slot in self.sample_slots}
        ) != len(self.sample_slots):
            raise ValueError("sample slots must have unique ids and indices")
        if not self.minimum_returned_samples <= self.max_samples <= len(self.sample_slots):
            raise ValueError("expected 1 <= minimum_returned_samples <= max_samples <= len(sample_slots)")
        if self.search is not None:
            if self.minimum_returned_samples > self.search.roots:
                raise ValueError("only the roots are guaranteed: minimum_returned_samples <= search.roots")
            if self.search.roots + self.search.branches() > self.max_samples:
                raise ValueError("max_samples must cover the roots and every branch the search may make")
        return self

    def wire(self) -> dict[str, Any]:
        """The body Sprout takes: ``search`` only when there is one."""
        body = self.model_dump(mode="json")
        if body["search"] is None:
            del body["search"]
        return body


class Acknowledgement(FrozenStrictBaseModel):
    """What ``POST`` and ``DELETE /rollout-groups`` answer."""

    rollout_job_id: NonEmptyStr
    status: GroupStatus


class Trajectory(FrozenStrictBaseModel):
    """One returned rollout: hint-free messages, the 0/1 verdict as ``reward``, and Sprout's ``metadata``.

    ``metadata`` stays a free dict; what Miles reads of it:

    - ``provenance``: ``prefix_messages`` and ``inserted_messages``, the
      messages no policy sampled (a branch's restored history, a repair's
      inserted turn), kept in the context and out of the loss.
    - ``zero_reason``: None when the grade gave a verdict; otherwise why it did
      not (``grade_no_verdict``, ``grade_failed``, ...). Such a trajectory is
      missing: its ``reward`` is 0.0 only because the wire needs a number, and
      a search credits it nothing and counts it in no statistic.
    - ``search``, for a RAFO search: ``kind`` (``root``, or ``repair`` for a
      continuation) and the task's ``outcome`` and ``root_counts``, which must
      match Miles's own; a continuation's ``event`` and ``depth`` and the
      ``parent_job_id`` its event's turn went into (a root's or a
      continuation's ``branch_id``), and ``leak_terms``, the hidden test names
      its history's turns give away (None: there was nothing to check them
      against).

    ``forest.read_forest`` refuses a search trajectory that lacks what it reads.
    """

    sample_slot_id: NonEmptyStr
    branch_id: NonEmptyStr
    parent_branch_id: NonEmptyStr | None = None
    #: Which trajectories were sampled under one condition, for a search
    #: request: ``root`` for the scratch rollouts and ``event:<n>`` for the
    #: continuations of event n; None without a search.
    group: NonEmptyStr | None = None
    messages: list[dict[str, Any]] = Field(min_length=1)
    tools: list[dict[str, Any]] = Field(default_factory=list)
    reward: StrictNumber
    status: TrajectoryStatus
    stop_reason: StopReason | None = None
    hints_removed: Literal[True]
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_trajectory(self) -> Trajectory:
        if self.status == "truncated" and self.stop_reason is None:
            raise ValueError("a truncated trajectory needs a stop_reason")
        if self.status != "truncated" and self.stop_reason is not None:
            raise ValueError("stop_reason belongs to a truncated trajectory")
        if not math.isfinite(self.reward):
            raise ValueError("reward must be finite")
        for message in self.messages:
            content = message.get("content")
            if content is not None and not isinstance(content, str):
                raise ValueError("Sprout trajectories carry text content only")
            if "reasoning_content" in message and not isinstance(message["reasoning_content"], str):
                raise ValueError("reasoning_content must be text")
        return self


class FailedSample(FrozenStrictBaseModel):
    """A slot whose trajectory could not be returned (its actor failed): missing, not unsolved -- it trains
    nothing and counts in no statistic, as a 0 or otherwise."""

    sample_slot_id: NonEmptyStr
    reason: str


class RolloutResult(FrozenStrictBaseModel):
    """``GET /rollout-groups/{id}``: the group's state, with its trajectories once terminal."""

    rollout_job_id: NonEmptyStr
    prompt_group_id: NonEmptyStr
    max_samples: StrictInt = Field(gt=0)
    actual_samples: StrictInt = Field(ge=0)
    trajectories: list[Trajectory] = Field(default_factory=list)
    failed_samples: list[FailedSample] = Field(default_factory=list)
    search_branches: StrictInt = Field(ge=0)
    consumed_budget: dict[str, Any] = Field(default_factory=dict)
    status: GroupStatus
    stop_reason: str | None = None

    @model_validator(mode="after")
    def validate_result(self) -> RolloutResult:
        if self.actual_samples != len(self.trajectories):
            raise ValueError("actual_samples must count the trajectories returned")
        if len({trajectory.sample_slot_id for trajectory in self.trajectories}) != len(self.trajectories):
            raise ValueError("a sample slot is filled at most once")
        if {failed.sample_slot_id for failed in self.failed_samples} & {t.sample_slot_id for t in self.trajectories}:
            raise ValueError("a failed sample slot returned a trajectory")
        return self
