"""What RAFO needs from the rollout service: a stateful episode environment that also reviews."""

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from miles.utils.types import Sample


@dataclass(frozen=True)
class ToolCall:
    name: str
    arguments: dict[str, Any]
    id: str


@dataclass(frozen=True)
class StepResult:
    outputs: tuple[str, ...]  # one tool output for each call, in call order
    failed_check: bool = False  # the step ran a check that failed, for example a failing test run
    done: bool = False  # the environment ended the episode, for example the agent submitted


@dataclass(frozen=True)
class ReviewRequest:
    messages: tuple[dict[str, Any], ...]  # the full failed episode, including the turns after the checkpoint
    checkpoint: int  # the reviewer turn continues messages[:checkpoint]
    depth: int


@dataclass(frozen=True)
class ReviewerTurn:
    """One assistant turn in the agent's first person: what to check and why, and the tool calls of the check.
    It must not contain the fix or the answer."""

    reasoning: str
    tool_calls: tuple[ToolCall, ...]
    content: str = ""


class RafoEnv(Protocol):
    """One episode environment of the rollout service (a sandbox with a repository, ...)."""

    async def step(self, calls: Sequence[ToolCall], *, check_only: bool) -> StepResult:
        """Run the tool calls of one assistant turn. With check_only (a reviewer turn) the calls must not
        change the code; the environment refuses edits."""
        ...

    async def fork(self) -> "RafoEnv":
        """An independent copy of the current state: the checkpoint snapshot and each continuation's start."""
        ...

    async def review(self, request: ReviewRequest) -> ReviewerTurn | None:
        """Called on a checkpoint snapshot: the rollout service's reviewer reads the failed episode and writes
        the turn that continues it, or abstains (None). miles renders the turn in the agent's chat format and
        runs its tool calls with step(check_only=True) on a fork of the snapshot."""
        ...

    async def reward(self) -> float:
        """The outcome of the episode in the current state, for example 1.0 if the hidden tests pass."""
        ...

    async def close(self) -> None: ...


EnvFactory = Callable[[Sample], Awaitable[RafoEnv]]
