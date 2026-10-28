"""A scripted agent, a scripted reviewer and the toy environment for CPU tests of the RAFO forest."""

import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field

from examples.experimental.rafo.episode import (
    EpisodeContext,
    TurnResult,
    make_tool_call_parser,
    render_reviewer_turn,
    template_renders_reasoning,
)
from examples.experimental.rafo.forest import ForestConfig, ForestContext
from examples.experimental.rafo.protocols import ReviewerTurn, ReviewRequest, ToolCall
from examples.experimental.rafo.toy_env import TOOL_SPECS, ToyEnv, ToyTask
from miles.utils.processing_utils import load_tokenizer
from miles.utils.types import Sample

TOKENIZER = load_tokenizer("Qwen/Qwen3-0.6B", trust_remote_code=True)
TASK = ToyTask(
    files=("auth.py", "billing.py", "cache.py", "db.py"),
    ranges=((1, 20), (21, 55), (56, 70), (71, 99)),
    bug="billing.py",
    bug_line=37,
    max_patches=3,
)
WRONG = "db.py"
SAMPLER_LOG_PROB = -0.25


def tool_call_text(name: str, path: str, thought: str = "Next step.") -> str:
    return f'{thought}\n<tool_call>\n{json.dumps({"name": name, "arguments": {"path": path}})}\n</tool_call>'


def follow_suspicious(text: str) -> str:
    """Patch the file that the latest check found suspicious, otherwise a wrong file."""
    hits = re.findall(r"Inspected (\w+\.py): found a suspicious line", text)
    return tool_call_text("patch", hits[-1] if hits else WRONG)


@dataclass
class ScriptedActor:
    """Appends one scripted assistant turn per call, like update_sample_from_response with a loss mask."""

    policy: Callable[[str], str]
    finish: str = "stop"
    calls: int = 0

    async def __call__(self, sample: Sample) -> TurnResult:
        self.calls += 1
        text = self.policy(TOKENIZER.decode(sample.tokens)) if self.finish != "length" else "I keep think"
        ids = TOKENIZER.encode(text, add_special_tokens=False)
        if self.finish == "stop":
            ids.append(TOKENIZER.eos_token_id)
        sample.tokens = sample.tokens + ids
        sample.response += TOKENIZER.decode(ids)
        sample.response_length += len(ids)
        sample.loss_mask = (sample.loss_mask or []) + [1] * len(ids)
        sample.rollout_log_probs = (sample.rollout_log_probs or []) + [SAMPLER_LOG_PROB] * len(ids)
        return TurnResult(text=TOKENIZER.decode(ids), finish=self.finish)


@dataclass
class ScriptedReviewer:
    """Exposes `targets[depth]` with an inspect; abstains for depths not in `targets`."""

    targets: dict[int, str]
    requests: list[ReviewRequest] = field(default_factory=list)

    async def __call__(self, request: ReviewRequest) -> ReviewerTurn | None:
        self.requests.append(request)
        if (path := self.targets.get(request.depth)) is None:
            return None
        return ReviewerTurn(
            reasoning=f"The traceback ends at line {TASK.bug_line}. I should check {path} before I patch again.",
            tool_calls=(ToolCall(name="inspect", arguments={"path": path}, id=f"call_review_{request.depth}"),),
        )


@dataclass
class EnvLog:
    roots: list[ToyEnv] = field(default_factory=list)

    async def make_env(self, sample: Sample) -> ToyEnv:
        env = ToyEnv(task=ToyTask.from_label(sample.label))
        self.roots.append(env)
        return env

    def all_envs(self) -> list[ToyEnv]:
        return [env for root in self.roots for env in root.opened]


def make_task_sample(task: ToyTask = TASK) -> Sample:
    return Sample(index=3, group_index=3, prompt=[{"role": "user", "content": task.prompt()}], label=task.to_label())


def prompt_ids(sample: Sample) -> list[int]:
    text = TOKENIZER.apply_chat_template(sample.prompt, tokenize=False, add_generation_prompt=True, tools=TOOL_SPECS)
    return TOKENIZER.encode(text, add_special_tokens=False)


def make_context(
    actor: ScriptedActor,
    reviewer: ScriptedReviewer,
    envs: EnvLog,
    *,
    m: int = 2,
    k: int = 2,
    s: int = 4,
    d: int = 3,
    rereview: str = "branch",
    max_turns: int = 8,
) -> ForestContext:
    episode = EpisodeContext(
        generate_turn=actor,
        parse_tool_calls=make_tool_call_parser(TOOL_SPECS, "qwen25"),
        tokenizer=TOKENIZER,
        max_turns=max_turns,
    )
    return ForestContext(
        episode=episode,
        make_env=envs.make_env,
        review=reviewer,
        render_turn=lambda turn: render_reviewer_turn(
            TOKENIZER,
            turn,
            tools=TOOL_SPECS,
            end_of_turn_id=TOKENIZER.eos_token_id,
            reasoning_block=template_renders_reasoning(TOKENIZER),
        ),
        tools=tuple(TOOL_SPECS),
        config=ForestConfig(m=m, k=k, s=s, d=d, rereview=rereview),
    )
