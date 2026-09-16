"""Message trajectories for training after Ash removes branch guidance.

Unlike v2, v3 does not claim to reproduce the tokens or probabilities used
during execution. Miles constructs a new training sequence from these messages.
"""

import math
from typing import Any, Literal

from pydantic import Field, model_validator

from miles.rollout.ash.protocol import (
    AshJobStatus,
    AshRolloutRequest,
    AshRolloutResult,
    AshSampleSlot,
    AshTrajectoryStatus,
    NonEmptyStr,
    StrictNumber,
)
from miles.utils.pydantic_utils import FrozenStrictBaseModel

MESSAGE_PROTOCOL_VERSION = "ash-rollout-v3"


class AshMessageBudget(FrozenStrictBaseModel):
    max_wall_time_seconds: float = Field(strict=True, gt=0, allow_inf_nan=False)


class AshMessageRequest(FrozenStrictBaseModel):
    protocol_version: Literal["ash-rollout-v3"] = MESSAGE_PROTOCOL_VERSION
    rollout_job_id: NonEmptyStr
    rollout_id: int = Field(strict=True, ge=0)
    prompt_group_id: NonEmptyStr
    task_id: NonEmptyStr
    image: NonEmptyStr
    prompt: str | list[dict[str, Any]]
    model_endpoint: NonEmptyStr
    model: NonEmptyStr | None = None
    sample_slots: list[AshSampleSlot] = Field(min_length=1)
    max_samples: int = Field(strict=True, gt=0)
    minimum_returned_samples: int = Field(strict=True, ge=1)
    max_turns: int = Field(strict=True, gt=0)
    finalization_timeout_seconds: float = Field(default=1800.0, strict=True, gt=0, allow_inf_nan=False)
    sampling_params: dict[str, Any] = Field(default_factory=dict)
    budgets: AshMessageBudget
    branching: bool = Field(default=False, strict=True)

    def to_wire(self) -> dict[str, Any]:
        value = self.model_dump(mode="json")
        if not self.branching:
            del value["branching"]
        return value

    @model_validator(mode="after")
    def validate_slots(self) -> "AshMessageRequest":
        AshRolloutRequest.validate_slots(self)
        return self


class AshMessageTrajectory(FrozenStrictBaseModel):
    sample_slot_id: NonEmptyStr
    branch_id: NonEmptyStr
    parent_branch_id: NonEmptyStr | None = None
    messages: list[dict[str, Any]] = Field(min_length=1)
    tools: list[dict[str, Any]] = Field(default_factory=list)
    reward: StrictNumber | dict[str, Any]
    status: AshTrajectoryStatus
    stop_reason: Literal["max_turns_reached", "timeout"] | None = None
    hints_removed: Literal[True]
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_reward(self) -> "AshMessageTrajectory":
        if self.status == "truncated" and self.stop_reason is None:
            raise ValueError("truncated trajectories require a stop_reason")
        if self.status != "truncated" and self.stop_reason is not None:
            raise ValueError("cutoff stop_reason requires a truncated trajectory")
        if isinstance(self.reward, (int, float)) and not math.isfinite(self.reward):
            raise ValueError("reward must be finite")
        for message in self.messages:
            content = message.get("content")
            if content is not None and not isinstance(content, str):
                raise ValueError("Ash message trajectories support text content only")
            if "reasoning_content" in message and not isinstance(message["reasoning_content"], str):
                raise ValueError("reasoning_content must be text")
        return self


class AshMessageResult(AshRolloutResult):
    protocol_version: Literal["ash-rollout-v3"] = MESSAGE_PROTOCOL_VERSION
    trajectories: list[AshMessageTrajectory] = Field(default_factory=list)


class AshMessageAcknowledgement(FrozenStrictBaseModel):
    protocol_version: Literal["ash-rollout-v3"] = MESSAGE_PROTOCOL_VERSION
    rollout_job_id: NonEmptyStr
    status: AshJobStatus
