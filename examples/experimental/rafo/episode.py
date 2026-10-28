"""One RAFO episode: a multi-turn agent conversation on one token stream, and its checkpoint.

The episode's ``Sample`` is the token stream of its training sample: the loss mask is 1 on the tokens that this
episode's agent generated, and 0 on the prompt, the tool outputs, reviewer turns and an inherited prefix.
"""

import json
import logging
import uuid
from collections.abc import Awaitable, Callable, Sequence
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from examples.experimental.rafo.protocols import RafoEnv, ReviewerTurn, ToolCall
from miles.rollout.generate_utils.tool_call_utils import create_tool_call_parser, update_sample_with_tool_responses
from miles.utils.types import Sample

logger = logging.getLogger(__name__)

_RENDER_USER = {"role": "user", "content": "x"}


@dataclass(frozen=True)
class TurnResult:
    text: str
    finish: str  # stop | length (unterminated reply) | abort | context (no room for another turn)


# Samples one agent turn for the sample's token stream and appends it, with loss mask 1 and sampler log-probs.
TurnGenerator = Callable[[Sample], Awaitable[TurnResult]]


@dataclass(frozen=True)
class EpisodeContext:
    generate_turn: TurnGenerator
    parse_tool_calls: Callable[[str], list[ToolCall]]
    tokenizer: Any
    max_turns: int  # assistant turns in one sequence, reviewer turns and an inherited prefix included


@dataclass(frozen=True)
class Checkpoint:
    """The state right after the tool output of an episode's first failed check."""

    sample: Sample
    messages: tuple[dict[str, Any], ...]
    turns: int
    env: RafoEnv


@dataclass(eq=False)
class Episode:
    """Mutated only by run_episode, which owns the episode while it runs."""

    sample: Sample
    messages: list[dict[str, Any]]
    env: RafoEnv
    turns: int = 0
    checkpoint: Checkpoint | None = None
    reward: float | None = None
    finish: str | None = None  # done | no_tool_call | max_turns | context | unterminated | abort

    @property
    def own_tokens(self) -> int:
        return sum(self.sample.loss_mask or [])


class EnvTracker:
    """Every environment one forest opens, so that all of them are closed once, also on failure."""

    def __init__(self) -> None:
        self._open: dict[int, RafoEnv] = {}

    def track(self, env: RafoEnv) -> RafoEnv:
        self._open[id(env)] = env
        return env

    async def fork(self, env: RafoEnv) -> RafoEnv:
        return self.track(await env.fork())

    async def close(self, env: RafoEnv) -> None:
        if self._open.pop(id(env), None) is not None:
            await env.close()

    async def close_all(self) -> None:
        envs, self._open = list(self._open.values()), {}
        for env in envs:
            try:
                await env.close()
            except Exception:
                logger.exception("RAFO: failed to close an environment")


async def run_episode(episode: Episode, ctx: EpisodeContext, tracker: EnvTracker) -> None:
    """Generate turns and run their tool calls until the episode ends; set its reward and finish reason."""
    while episode.reward is None:
        if episode.turns >= ctx.max_turns:
            await end_episode(episode, "max_turns", tracker)
            return
        turn = await ctx.generate_turn(episode.sample)
        if turn.finish == "abort":
            _fail(episode, "abort")
            return
        if turn.finish == "context":
            await end_episode(episode, "context", tracker)
            return
        episode.turns += 1
        episode.messages.append({"role": "assistant", "content": turn.text})
        if turn.finish == "length":  # an unterminated reply: its tool calls do not run, and the episode fails
            _fail(episode, "unterminated")
            return
        calls = ctx.parse_tool_calls(turn.text)
        if not calls:
            await end_episode(episode, "no_tool_call", tracker)
            return
        step = await episode.env.step(calls, check_only=False)
        append_tool_outputs(episode, calls, step.outputs, tokenizer=ctx.tokenizer)
        if step.done:
            await end_episode(episode, "done", tracker)
        elif step.failed_check and episode.checkpoint is None:
            episode.checkpoint = Checkpoint(
                sample=deepcopy(episode.sample),
                messages=tuple(episode.messages),
                turns=episode.turns,
                env=await tracker.fork(episode.env),
            )


def append_tool_outputs(episode: Episode, calls: Sequence[ToolCall], outputs: Sequence[str], *, tokenizer) -> None:
    if len(outputs) != len(calls):
        raise ValueError(f"the environment returned {len(outputs)} tool outputs for {len(calls)} tool calls")
    messages = [
        {"role": "tool", "tool_call_id": call.id, "name": call.name, "content": output}
        for call, output in zip(calls, outputs, strict=True)
    ]
    update_sample_with_tool_responses(episode.sample, messages, tokenizer=tokenizer)
    episode.messages.extend(messages)


async def end_episode(episode: Episode, finish: str, tracker: EnvTracker) -> None:
    """Score the episode in its final state and release its environment."""
    episode.reward = float(await episode.env.reward())
    episode.finish = finish
    await tracker.close(episode.env)


def _fail(episode: Episode, finish: str) -> None:
    episode.reward = 0.0
    episode.finish = finish


def make_tool_call_parser(tool_specs: Sequence[dict[str, Any]], parser_name: str) -> Callable[[str], list[ToolCall]]:
    """The agent's tool-call parser (an SGLang function-call parser); calls with invalid JSON arguments are dropped."""
    parser = create_tool_call_parser(list(tool_specs), parser_name)

    def parse(text: str) -> list[ToolCall]:
        try:
            _, items = parser.parse_non_stream(text)
        except Exception:
            logger.debug("RAFO: the tool-call parser rejected a turn", exc_info=True)
            return []
        calls = []
        for item in items:
            try:
                arguments = json.loads(item.parameters) if item.parameters else {}
            except json.JSONDecodeError:
                continue
            calls.append(ToolCall(name=item.name, arguments=arguments, id=f"call_{uuid.uuid4().hex[:24]}"))
        return calls

    return parse


def reviewer_message(turn: ReviewerTurn) -> dict[str, Any]:
    """The reviewer turn as a chat message, in the agent's role."""
    message = {
        "role": "assistant",
        "content": turn.content,
        "reasoning_content": turn.reasoning,
        "tool_calls": [
            {"id": call.id, "type": "function", "function": {"name": call.name, "arguments": call.arguments}}
            for call in turn.tool_calls
        ],
    }
    return message


def render_reviewer_turn(
    tokenizer,
    turn: ReviewerTurn,
    *,
    tools: Sequence[dict[str, Any]],
    end_of_turn_id: int,
    reasoning_block: bool,
) -> list[int]:
    """Token ids of the reviewer turn in the agent's chat format, up to and including its end-of-turn token.

    With reasoning_block (see template_renders_reasoning) the reasoning goes into the template's reasoning block,
    as a thinking model writes it; otherwise it goes in front of the content."""
    message = reviewer_message(turn)
    if not reasoning_block:
        message["content"] = "\n\n".join(part for part in (turn.reasoning, turn.content) if part)
        del message["reasoning_content"]
    tools = list(tools) or None
    with_turn = tokenizer.apply_chat_template(
        [_RENDER_USER, message], tokenize=True, add_generation_prompt=False, tools=tools, return_dict=False
    )
    without = tokenizer.apply_chat_template(
        [_RENDER_USER], tokenize=True, add_generation_prompt=True, tools=tools, return_dict=False
    )
    if with_turn[: len(without)] != without:
        raise ValueError("the chat template does not render an assistant turn as an append to the generation prompt")
    ids = list(with_turn[len(without) :])
    if end_of_turn_id not in ids:
        raise ValueError(f"the rendered reviewer turn has no end-of-turn token {end_of_turn_id}")
    return ids[: ids.index(end_of_turn_id) + 1]


def template_renders_reasoning(tokenizer) -> bool:
    """Whether the chat template renders an assistant message's reasoning_content (a thinking-model template)."""
    probe = "rafo-reasoning-probe"
    text = tokenizer.apply_chat_template(
        [_RENDER_USER, {"role": "assistant", "content": "", "reasoning_content": probe}], tokenize=False
    )
    return probe in text
