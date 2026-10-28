"""What RAFO needs from a task environment and from a reviewer."""

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


class RafoEnv(Protocol):
    """One stateful episode environment (a sandbox with a repository, ...)."""

    async def step(self, calls: Sequence[ToolCall], *, check_only: bool) -> StepResult:
        """Run the tool calls of one assistant turn. With check_only (a reviewer turn) the calls must not
        change the code; the environment refuses edits."""
        ...

    async def fork(self) -> "RafoEnv":
        """An independent copy of the current state: the checkpoint snapshot and each continuation's start."""
        ...

    async def reward(self) -> float:
        """The outcome of the episode in the current state, for example 1.0 if the hidden tests pass."""
        ...

    async def close(self) -> None: ...


EnvFactory = Callable[[Sample], Awaitable[RafoEnv]]


@dataclass(frozen=True)
class ReviewRequest:
    sample: Sample  # the task: prompt, label, metadata
    messages: tuple[dict[str, Any], ...]  # the full failed episode, including the turns after the checkpoint
    checkpoint: int  # the reviewer turn continues messages[:checkpoint]
    depth: int
    tools: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class ReviewerTurn:
    """One assistant turn in the agent's first person: what to check and why, and the tool calls of the check.
    It must not contain the fix or the answer."""

    reasoning: str
    tool_calls: tuple[ToolCall, ...]
    content: str = ""


Reviewer = Callable[[ReviewRequest], Awaitable[ReviewerTurn | None]]  # None: the reviewer abstains
