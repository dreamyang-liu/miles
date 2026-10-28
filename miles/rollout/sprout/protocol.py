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
StopReason = Literal["max_turns_reached", "timeout", "interrupted"]

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

    ``roots`` scratch rollouts take the first slots. Once they are graded, the
    task's outcome (``gsml.search_outcome``) decides what Sprout samples next.
    Every graded root resolved, or none was graded: nothing. Some resolved
    (TF), or none did but one is missing (FF_partial): a review of each failed
    root picks up to ``points`` branch points in it, and
    ``tf_student_continuations`` branches with nothing added run from each (0:
    none). Every root was graded and none resolved (FF): the review also writes
    ``candidates`` repair turns per point, and ``student_continuations``
    branches with nothing added and ``candidate_continuations`` after each
    repair run from it. Under FF the repairs come in up to ``repair_rounds``
    rounds: where no student and no repair has resolved, the failed repair of
    the last round is reviewed again and one more turn inserted later in its
    own history; more than one round takes one candidate and one continuation
    per round. The branches fill the remaining slots, enough for the outcome
    that makes the most (``branches()``).
    """

    roots: StrictInt = Field(gt=0)
    points: StrictInt = Field(gt=0)
    candidates: StrictInt = Field(ge=0)
    student_continuations: StrictInt = Field(gt=0)
    candidate_continuations: StrictInt = Field(ge=0)
    tf_student_continuations: StrictInt = Field(ge=0)
    repair_rounds: StrictInt = Field(gt=0)
    review_seconds: float = Field(strict=True, gt=0, allow_inf_nan=False)
    wall_time_seconds: float = Field(strict=True, gt=0, allow_inf_nan=False)

    @model_validator(mode="after")
    def validate_search(self) -> SearchSpec:
        if (self.candidates == 0) != (self.candidate_continuations == 0):
            raise ValueError("candidates and candidate_continuations are both zero or both positive")
        if self.repair_rounds > 1 and (self.candidates, self.candidate_continuations) != (1, 1):
            raise ValueError("more than one repair round takes one candidate and one candidate continuation")
        return self

    def branches(self) -> int:
        """The most the search may make, ``points`` in each failed root: all of them failed (FF), or all
        but one (TF, FF_partial). Sprout computes the same number and refuses fewer slots."""
        every_root_failed = self.roots * (
            self.student_continuations + self.candidates * self.candidate_continuations * self.repair_rounds
        )
        all_but_one = (self.roots - 1) * self.tf_student_continuations
        return self.points * max(every_root_failed, all_but_one)

    def search_seconds(self, grade_seconds: float) -> float:
        """How long the search may run after its roots: each round's review and branches, and the grades
        of every round but the last (Sprout's ``SearchSpec.search_seconds``)."""
        rounds = self.repair_rounds if self.candidates else 1
        return rounds * (self.review_seconds + self.wall_time_seconds) + (rounds - 1) * grade_seconds


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
    """One returned rollout: hint-free messages, the 0/1 verdict as ``reward``, and Sprout's ``metadata``.

    ``metadata`` stays a free dict; what Miles reads of it:

    - ``provenance``: ``prefix_messages`` and ``inserted_messages``, the
      messages no policy sampled (a branch's restored history, a repair's
      inserted turn), kept in the context and out of the loss.
    - ``zero_reason``: None when the grade gave a verdict; otherwise why it did
      not (``grade_no_verdict``, ``grade_failed``, ...). Such a trajectory is
      missing: its ``reward`` is 0.0 only because the wire needs a number, and
      a search credits it nothing and counts it in no statistic.
    - ``touched_test_paths``: the task's test files the graded patch changed
      (None when unknown). A repair continuation that touched any, or cannot
      say, counts as unresolved in GSML (R' = 0).
    - ``search``, for a search request: ``kind`` (root, student or repair) and
      the task's ``outcome`` and ``root_counts``, which must match Miles's own;
      a branch's ``point_id``; a repair's ``candidate`` and ``leak_terms``,
      the hidden test names and paths its inserted turn names that its history
      did not show (None: there was nothing to check it against).

    GSML (``gsml.assemble_search_group``) refuses a search trajectory that
    lacks what it reads.
    """

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
