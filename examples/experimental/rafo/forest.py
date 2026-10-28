"""The RAFO forest of one task.

M roots run to the end. If all of them fail (gate all_roots_fail), the rollout service's reviewer reads each failed
root and writes one turn at its checkpoint, in the agent's first person, that exposes the problem with a check. The
environment runs the check from the restored snapshot, and K continuations start after its output. If all continuations of an
event fail, the next depth reviews again: inside the first failed continuation after its own first failed check
(branch), or at the same checkpoint (checkpoint). Each event uses one of the task's S slots; depth stops at D.
"""

import asyncio
from collections.abc import Awaitable, Callable, Iterable
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from examples.experimental.rafo.episode import (
    Checkpoint,
    EnvTracker,
    Episode,
    EpisodeContext,
    append_tool_outputs,
    end_episode,
    reviewer_message,
    run_episode,
)
from examples.experimental.rafo.protocols import EnvFactory, ReviewerTurn, ReviewRequest
from miles.rollout.generate_utils.tool_call_utils import append_forced_tokens
from miles.utils.types import Sample


@dataclass(frozen=True)
class ForestConfig:
    m: int  # roots for each task
    k: int  # continuations for each event
    s: int  # slots (events) for each task
    d: int  # maximum review depth
    rereview: str  # branch | checkpoint


@dataclass(frozen=True)
class ForestContext:
    episode: EpisodeContext
    make_env: EnvFactory
    render_turn: Callable[[ReviewerTurn], list[int]]  # reviewer turn -> token ids in the agent's chat format
    config: ForestConfig


@dataclass(frozen=True)
class Event:
    depth: int
    parent: int | None  # index of the event that this deeper review continues
    parent_cont: int | None  # the parent's continuation that holds this checkpoint (branch re-review)
    checkpoint: Checkpoint
    turn: ReviewerTurn
    ce_sample: Sample  # checkpoint prefix + reviewer turn; loss mask 1 only on the reviewer turn
    conts: tuple[Episode, ...]


@dataclass(frozen=True)
class Forest:
    roots: tuple[Episode, ...]
    events: tuple[Event, ...] = ()
    reviewer_reads: int = 0
    abstentions: int = 0

    def episodes(self) -> list[Episode]:
        return [*self.roots, *(cont for event in self.events for cont in event.conts)]

    @property
    def aborted(self) -> bool:
        return any(episode.finish == "abort" for episode in self.episodes())


@dataclass(frozen=True)
class _Candidate:
    checkpoint: Checkpoint
    transcript: tuple[dict[str, Any], ...]  # the full failed episode that the reviewer reads
    parent: int | None
    parent_cont: int | None


async def build_forest(task: Sample, prompt_ids: list[int], ctx: ForestContext) -> Forest:
    tracker = EnvTracker()
    try:
        roots = tuple([await _new_root(task, prompt_ids, ctx, tracker) for _ in range(ctx.config.m)])
        await _run(roots, ctx, tracker)
        forest = Forest(roots=roots)
        if forest.aborted or any(root.reward > 0 for root in roots):
            return forest
        return await _review(forest, ctx, tracker)
    finally:
        await tracker.close_all()


async def run_root(task: Sample, prompt_ids: list[int], ctx: ForestContext) -> Episode:
    """One episode of the agent alone, without review (evaluation)."""
    tracker = EnvTracker()
    try:
        root = await _new_root(task, prompt_ids, ctx, tracker)
        await run_episode(root, ctx.episode, tracker)
        return root
    finally:
        await tracker.close_all()


async def _review(forest: Forest, ctx: ForestContext, tracker: EnvTracker) -> Forest:
    events: list[Event] = []
    reads = abstentions = 0
    slots = ctx.config.s
    frontier = [
        _Candidate(checkpoint=root.checkpoint, transcript=tuple(root.messages), parent=None, parent_cont=None)
        for root in forest.roots
        if root.checkpoint is not None
    ]
    for depth in range(1, ctx.config.d + 1):
        accepted: list[tuple[_Candidate, ReviewerTurn]] = []
        while frontier and len(accepted) < slots:  # in order; one read for each candidate a free slot can take
            batch, frontier = frontier[: slots - len(accepted)], frontier[slots - len(accepted) :]
            turns = await _gather(cand.checkpoint.env.review(_request(cand, depth)) for cand in batch)
            reads += len(batch)
            for cand, turn in zip(batch, turns, strict=True):
                if turn is None or not turn.tool_calls:
                    abstentions += 1
                else:
                    accepted.append((cand, turn))
        if not accepted:
            break
        slots -= len(accepted)
        opened = await _gather(_open_event(cand, turn, depth, ctx, tracker) for cand, turn in accepted)
        await _run([cont for event in opened for cont in event.conts], ctx, tracker)
        first = len(events)
        events.extend(opened)
        frontier = [
            _next_candidate(first + i, event, ctx.config.rereview)
            for i, event in enumerate(opened)
            if not any(cont.reward > 0 for cont in event.conts)
        ]
        if Forest(roots=forest.roots, events=tuple(events)).aborted:
            break
    return Forest(roots=forest.roots, events=tuple(events), reviewer_reads=reads, abstentions=abstentions)


async def _new_root(task: Sample, prompt_ids: list[int], ctx: ForestContext, tracker: EnvTracker) -> Episode:
    sample = deepcopy(task)
    sample.tokens, sample.response, sample.response_length = list(prompt_ids), "", 0
    sample.loss_mask, sample.rollout_log_probs = [], []
    messages = list(task.prompt) if isinstance(task.prompt, list) else [{"role": "user", "content": task.prompt}]
    return Episode(sample=sample, messages=messages, env=tracker.track(await ctx.make_env(task)))


async def _open_event(
    cand: _Candidate, turn: ReviewerTurn, depth: int, ctx: ForestContext, tracker: EnvTracker
) -> Event:
    """Insert the reviewer turn at the checkpoint, run its check on a restored snapshot, start K continuations."""
    tokenizer = ctx.episode.tokenizer
    checkpoint = cand.checkpoint
    turn_ids = ctx.render_turn(turn)
    sample = deepcopy(checkpoint.sample)
    sample.loss_mask = [0] * len(sample.loss_mask)  # the prefix is trained by the episode that generated it
    append_forced_tokens(sample, turn_ids, tokenizer=tokenizer)
    ce_sample = deepcopy(sample)
    ce_sample.loss_mask[len(ce_sample.loss_mask) - len(turn_ids) :] = [1] * len(turn_ids)

    check_env = await tracker.fork(checkpoint.env)
    step = await check_env.step(turn.tool_calls, check_only=True)
    opening = Episode(
        sample=sample,
        messages=[*checkpoint.messages, reviewer_message(turn)],
        env=check_env,
        turns=checkpoint.turns + 1,
    )
    append_tool_outputs(opening, turn.tool_calls, step.outputs, tokenizer=tokenizer)
    conts = tuple([await _continuation(opening, ended=step.done, tracker=tracker) for _ in range(ctx.config.k)])
    await tracker.close(check_env)
    return Event(
        depth=depth,
        parent=cand.parent,
        parent_cont=cand.parent_cont,
        checkpoint=checkpoint,
        turn=turn,
        ce_sample=ce_sample,
        conts=conts,
    )


async def _continuation(opening: Episode, *, ended: bool, tracker: EnvTracker) -> Episode:
    cont = Episode(
        sample=deepcopy(opening.sample),
        messages=list(opening.messages),
        env=await tracker.fork(opening.env),
        turns=opening.turns,
    )
    if ended:  # the check ended the episode: there is nothing left to continue
        await end_episode(cont, "done", tracker)
    return cont


def _next_candidate(index: int, event: Event, rereview: str) -> _Candidate:
    """All continuations of the event failed: where the next depth reviews."""
    if rereview == "branch":
        for j, cont in enumerate(event.conts):
            if cont.checkpoint is not None:
                return _Candidate(
                    checkpoint=cont.checkpoint, transcript=tuple(cont.messages), parent=index, parent_cont=j
                )
    # no failed continuation has a state after its own first failed check: review again at the same checkpoint
    return _Candidate(
        checkpoint=event.checkpoint, transcript=tuple(event.conts[0].messages), parent=index, parent_cont=None
    )


def _request(cand: _Candidate, depth: int) -> ReviewRequest:
    return ReviewRequest(messages=cand.transcript, checkpoint=len(cand.checkpoint.messages), depth=depth)


async def _run(episodes: Iterable[Episode], ctx: ForestContext, tracker: EnvTracker) -> None:
    await _gather(run_episode(episode, ctx.episode, tracker) for episode in episodes if episode.reward is None)


async def _gather(awaitables: Iterable[Awaitable[Any]]) -> list[Any]:
    """asyncio.gather that cancels and awaits the siblings when one fails, so none outlives the forest."""
    tasks = [asyncio.ensure_future(aw) for aw in awaitables]
    try:
        return await asyncio.gather(*tasks)
    except BaseException:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise
