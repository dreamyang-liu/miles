"""A small bug-finding environment for tests and smoke runs.

One file of the task holds the bug. ``inspect`` reads a file (a check: it never changes the code); ``patch`` fixes
a file and runs the tests. A patch of the wrong file is a failed check, and its output shows the failing line. The
episode ends when the tests pass or the patch budget is used. Like a rollout service, the environment also reviews:
by default with an oracle that knows the bug file; a real rollout service calls its reviewer model.
"""

import json
import random
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field

from examples.experimental.rafo.protocols import ReviewerTurn, ReviewRequest, StepResult, ToolCall
from miles.utils.types import Sample

_FILE_POOL = ("auth.py", "billing.py", "cache.py", "db.py", "email.py", "jobs.py", "search.py", "users.py")

TOOL_SPECS = [
    {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
        },
    }
    for name, description in (
        ("inspect", "Read one file and report whether it looks suspicious. Does not change the code."),
        ("patch", "Fix one file, then run the test suite."),
    )
]


@dataclass(frozen=True)
class ToyTask:
    files: tuple[str, ...]
    ranges: tuple[tuple[int, int], ...]  # the line range of each file
    bug: str
    bug_line: int
    max_patches: int = 3

    @classmethod
    def from_label(cls, label: str) -> "ToyTask":
        values = json.loads(label)
        return cls(
            files=tuple(values["files"]),
            ranges=tuple(tuple(r) for r in values["ranges"]),
            bug=values["bug"],
            bug_line=values["bug_line"],
            max_patches=values.get("max_patches", 3),
        )

    def to_label(self) -> str:
        return json.dumps(
            {
                "files": list(self.files),
                "ranges": [list(r) for r in self.ranges],
                "bug": self.bug,
                "bug_line": self.bug_line,
                "max_patches": self.max_patches,
            }
        )

    def prompt(self) -> str:
        listing = "\n".join(
            f"- {name}: lines {lo}-{hi}" for name, (lo, hi) in zip(self.files, self.ranges, strict=True)
        )
        return (
            f"The test suite of this repository fails. Find and fix the bug.\nFiles:\n{listing}\n"
            f"You can patch at most {self.max_patches} times."
        )


async def oracle_review(task: ToyTask, request: ReviewRequest) -> ReviewerTurn | None:
    """Exposes the bug: maps the failing line to its file and inspects that file. Never patches."""
    return ReviewerTurn(
        reasoning=(
            f"The traceback ends at line {task.bug_line}, and line {task.bug_line} is in {task.bug}. "
            f"So I should check {task.bug} before I patch again."
        ),
        tool_calls=(ToolCall(name="inspect", arguments={"path": task.bug}, id=f"call_review_{request.depth}"),),
    )


@dataclass(eq=False)
class ToyEnv:
    task: ToyTask
    reviewer: Callable[[ToyTask, ReviewRequest], Awaitable[ReviewerTurn | None]] = oracle_review
    patches: int = 0
    fixed: bool = False
    closed: bool = False
    opened: list["ToyEnv"] = field(default_factory=list)  # every env of the episode tree, shared by forks

    def __post_init__(self) -> None:
        self.opened.append(self)

    async def step(self, calls: Sequence[ToolCall], *, check_only: bool) -> StepResult:
        assert not self.closed, "step on a closed environment"
        outputs, failed = [], False
        for call in calls:
            path = call.arguments.get("path")
            if call.name == "inspect":
                verdict = "found a suspicious line" if path == self.task.bug else "looks fine"
                outputs.append(f"Inspected {path}: {verdict}.")
            elif call.name == "patch" and check_only:
                outputs.append("Refused: a check must not change the code.")
            elif call.name == "patch":
                self.patches += 1
                self.fixed = self.fixed or path == self.task.bug
                failed = failed or not self.fixed
                outputs.append(
                    "All tests passed."
                    if self.fixed
                    else f"Tests failed: the traceback ends at line {self.task.bug_line}."
                )
            else:
                outputs.append(f"Unknown tool {call.name}.")
        done = self.fixed or self.patches >= self.task.max_patches
        return StepResult(outputs=tuple(outputs), failed_check=failed and not self.fixed, done=done)

    async def fork(self) -> "ToyEnv":
        return ToyEnv(
            task=self.task, reviewer=self.reviewer, patches=self.patches, fixed=self.fixed, opened=self.opened
        )

    async def review(self, request: ReviewRequest) -> ReviewerTurn | None:
        assert not self.closed, "review on a closed environment"
        return await self.reviewer(self.task, request)

    async def reward(self) -> float:
        return 1.0 if self.fixed else 0.0

    async def close(self) -> None:
        self.closed = True


async def make_toy_env(sample: Sample) -> ToyEnv:
    return ToyEnv(task=ToyTask.from_label(sample.label))


def sample_toy_task(rng: random.Random, *, n_files: int = 5, max_patches: int = 3) -> ToyTask:
    files = tuple(rng.sample(_FILE_POOL, n_files))
    ranges, start = [], 1
    for _ in files:
        size = rng.randint(10, 40)
        ranges.append((start, start + size - 1))
        start += size
    bug = rng.randrange(n_files)
    return ToyTask(
        files=files,
        ranges=tuple(ranges),
        bug=files[bug],
        bug_line=rng.randint(*ranges[bug]),
        max_patches=max_patches,
    )


def write_toy_dataset(path: str, *, n_tasks: int, seed: int) -> None:
    """A miles prompt dataset: a chat prompt (without --apply-chat-template, RAFO adds the tools) and a label."""
    rng = random.Random(seed)
    with open(path, "w") as f:
        for _ in range(n_tasks):
            task = sample_toy_task(rng)
            row = {"prompt": [{"role": "user", "content": task.prompt()}], "label": task.to_label()}
            f.write(json.dumps(row) + "\n")
