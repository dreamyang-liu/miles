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
StopReason = Literal["max_turns_reached", "timeout"]

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


class SearchSpec(FrozenStrictBaseModel):
    """Hindsight branches after the roots (Sprout's ``rl_driver.messages.SearchSpec``).

    ``roots`` scratch rollouts take the first slots. Once they are graded,
    Sprout reviews the failed ones, picks ``points`` branch points in total and,
    per point, writes ``candidates`` repair turns; it then runs
    ``student_continuations`` branches with nothing added and
    ``candidate_continuations`` after each repair, filling the remaining slots.
    """

    roots: StrictInt = Field(gt=0)
    points: StrictInt = Field(gt=0)
    candidates: StrictInt = Field(ge=0)
    student_continuations: StrictInt = Field(gt=0)
    candidate_continuations: StrictInt = Field(ge=0)
    review_seconds: float = Field(strict=True, gt=0, allow_inf_nan=False)
    wall_time_seconds: float = Field(strict=True, gt=0, allow_inf_nan=False)

    @model_validator(mode="after")
    def validate_search(self) -> SearchSpec:
        if (self.candidates == 0) != (self.candidate_continuations == 0):
            raise ValueError("candidates and candidate_continuations are both zero or both positive")
        return self

    def branches(self) -> int:
        return self.points * (self.student_continuations + self.candidates * self.candidate_continuations)


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
    search: SearchSpec | None = None

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
    sample_slot_id: NonEmptyStr
    branch_id: NonEmptyStr
    parent_branch_id: NonEmptyStr | None = None
    #: Which trajectories were sampled under one condition, for a search
    #: request: ``root`` for the scratch rollouts, ``student:<point>`` and
    #: ``repair:<point>:<candidate>`` for the branches; None without a search.
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


class RolloutResult(FrozenStrictBaseModel):
    """``GET /rollout-groups/{id}``: the group's state, with its trajectories once terminal."""

    rollout_job_id: NonEmptyStr
    prompt_group_id: NonEmptyStr
    max_samples: StrictInt = Field(gt=0)
    actual_samples: StrictInt = Field(ge=0)
    trajectories: list[Trajectory] = Field(default_factory=list)
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
        return self
