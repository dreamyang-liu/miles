import json

import pytest
from examples.experimental.rafo.episode import make_tool_call_parser, render_reviewer_turn, template_renders_reasoning
from examples.experimental.rafo.forest import build_forest
from examples.experimental.rafo.protocols import ReviewerTurn, ToolCall
from examples.experimental.rafo.toy_env import TOOL_SPECS, ToyEnv
from tests.fast.examples.experimental.rafo.forest_fixtures import (
    TASK,
    TOKENIZER,
    WRONG,
    EnvLog,
    ScriptedActor,
    ScriptedReviewer,
    follow_suspicious,
    make_context,
    make_task_sample,
    prompt_ids,
    tool_call_text,
)


def _always(name: str, path: str):
    return lambda text: tool_call_text(name, path)


async def _forest(actor, reviewer, **config):
    envs = EnvLog(reviewer=reviewer)
    task = make_task_sample()
    return await build_forest(task, prompt_ids(task), make_context(actor, envs, **config)), envs


def _own_text(sample) -> str:
    return TOKENIZER.decode(
        [t for t, m in zip(sample.tokens[-sample.response_length :], sample.loss_mask, strict=True) if m]
    )


class TestRenderReviewerTurn:
    def test_the_reviewer_turn_is_an_agent_turn_whose_tool_calls_the_agent_parser_reads(self) -> None:
        turn = ReviewerTurn(
            reasoning="The traceback ends at line 37. I should check billing.py.",
            tool_calls=(ToolCall(name="inspect", arguments={"path": "billing.py"}, id="c0"),),
        )

        assert template_renders_reasoning(TOKENIZER)  # Qwen3 is a thinking-model template
        ids = render_reviewer_turn(
            TOKENIZER, turn, tools=TOOL_SPECS, end_of_turn_id=TOKENIZER.eos_token_id, reasoning_block=True
        )
        text = TOKENIZER.decode(ids)

        assert ids[-1] == TOKENIZER.eos_token_id and ids.count(TOKENIZER.eos_token_id) == 1
        assert text.startswith("<think>\nThe traceback ends at line 37. I should check billing.py.\n</think>")
        parsed = make_tool_call_parser(TOOL_SPECS, "qwen25")(text)
        assert [(c.name, c.arguments) for c in parsed] == [("inspect", {"path": "billing.py"})]


class TestRoots:
    async def test_a_task_with_a_successful_root_is_not_reviewed(self) -> None:
        answers = iter([tool_call_text("patch", TASK.bug), tool_call_text("patch", WRONG)])
        actor = ScriptedActor(policy=lambda text: next(answers, tool_call_text("patch", WRONG)))
        reviewer = ScriptedReviewer(targets={1: TASK.bug})

        forest, _ = await _forest(actor, reviewer)

        assert [r.reward for r in forest.roots] == [1.0, 0.0]
        assert forest.events == () and reviewer.requests == []

    async def test_the_checkpoint_is_the_state_right_after_the_first_failed_check(self) -> None:
        forest, _ = await _forest(ScriptedActor(policy=_always("patch", WRONG)), ScriptedReviewer(targets={}))

        root = forest.roots[0]
        assert root.reward == 0.0 and root.finish == "done" and root.turns == TASK.max_patches
        checkpoint = root.checkpoint
        assert checkpoint.turns == 1 and checkpoint.env.patches == 1
        assert checkpoint.messages[-1]["role"] == "tool" and "Tests failed" in checkpoint.messages[-1]["content"]
        assert TOKENIZER.decode(checkpoint.sample.tokens).endswith("<|im_start|>assistant\n")
        assert root.sample.tokens[: len(checkpoint.sample.tokens)] == checkpoint.sample.tokens

    async def test_an_unterminated_reply_fails_the_episode_without_running_its_tool_calls(self) -> None:
        forest, envs = await _forest(
            ScriptedActor(policy=_always("patch", WRONG), finish="length"), ScriptedReviewer({})
        )

        assert [(r.reward, r.finish) for r in forest.roots] == [(0.0, "unterminated")] * 2
        assert all(env.patches == 0 for env in envs.roots)


class TestEvents:
    async def test_a_reviewer_turn_starts_k_continuations_from_the_checkpoint_after_its_check(self) -> None:
        reviewer = ScriptedReviewer(targets={1: TASK.bug})
        forest, _ = await _forest(ScriptedActor(policy=follow_suspicious), reviewer, k=2)

        assert len(forest.events) == 2 and forest.reviewer_reads == 2
        event = forest.events[0]
        assert event.depth == 1 and event.parent is None and len(event.conts) == 2
        assert all(c.reward == 1.0 for c in event.conts)
        assert reviewer.requests[0].messages == tuple(forest.roots[0].messages)
        assert reviewer.requests[0].checkpoint == len(forest.roots[0].checkpoint.messages)
        for cont in event.conts:
            assert cont.env.patches == 2  # the reviewer's check did not patch; the continuation patched once
            assert "found a suspicious line" in cont.messages[len(event.checkpoint.messages) + 1]["content"]

    async def test_the_rollout_service_reviews_on_the_checkpoint_snapshot(self, monkeypatch) -> None:
        reviewed_on = []
        review = ToyEnv.review

        async def recording_review(env, request):
            reviewed_on.append(env)
            return await review(env, request)

        monkeypatch.setattr(ToyEnv, "review", recording_review)
        forest, _ = await _forest(ScriptedActor(policy=follow_suspicious), ScriptedReviewer(targets={1: TASK.bug}))

        assert reviewed_on == [root.checkpoint.env for root in forest.roots]
        assert all(env.patches == 1 for env in reviewed_on)  # the state right after the first failed patch

    async def test_the_ce_sample_trains_only_the_reviewer_turn(self) -> None:
        forest, _ = await _forest(ScriptedActor(policy=follow_suspicious), ScriptedReviewer(targets={1: TASK.bug}))

        event = forest.events[0]
        ce = event.ce_sample
        prefix = event.checkpoint.sample
        n_turn = len(ce.tokens) - len(prefix.tokens)
        assert ce.tokens[: len(prefix.tokens)] == prefix.tokens and ce.tokens[-1] == TOKENIZER.eos_token_id
        assert ce.loss_mask == [0] * (ce.response_length - n_turn) + [1] * n_turn
        assert "I should check billing.py" in _own_text(ce)

    async def test_each_agent_token_is_trained_only_by_the_episode_that_generated_it(self) -> None:
        forest, _ = await _forest(ScriptedActor(policy=follow_suspicious), ScriptedReviewer(targets={1: TASK.bug}))

        root, cont = forest.roots[0], forest.events[0].conts[0]
        assert _own_text(root.sample).count("<tool_call>") == root.turns
        assert _own_text(cont.sample).count("<tool_call>") == 1 and "I should check" not in _own_text(cont.sample)
        assert json.dumps({"name": "patch", "arguments": {"path": TASK.bug}}) in _own_text(cont.sample)

    async def test_an_abstaining_reviewer_uses_no_slot(self) -> None:
        reviewer = ScriptedReviewer(targets={})
        forest, _ = await _forest(ScriptedActor(policy=_always("patch", WRONG)), reviewer)

        assert forest.events == () and forest.reviewer_reads == 2 and forest.abstentions == 2

    async def test_events_stop_when_the_task_has_no_free_slot(self) -> None:
        forest, _ = await _forest(
            ScriptedActor(policy=_always("patch", WRONG)),
            ScriptedReviewer(targets={1: "cache.py", 2: "cache.py"}),
            s=3,
        )

        assert len(forest.events) == 3 and forest.reviewer_reads == 3


class TestNestedReview:
    async def test_branch_rereview_starts_inside_the_failed_continuation_with_two_reviewer_turns(self) -> None:
        reviewer = ScriptedReviewer(targets={1: "cache.py", 2: TASK.bug})
        forest, _ = await _forest(ScriptedActor(policy=follow_suspicious), reviewer, m=2, s=4)

        deep = [e for e in forest.events if e.depth == 2]
        assert len(deep) == 2
        for event in deep:
            parent = forest.events[event.parent]
            host = parent.conts[event.parent_cont]
            assert event.parent_cont == 0 and host.reward == 0.0 and host.checkpoint is event.checkpoint
            assert all(c.reward == 1.0 for c in event.conts)
            assert TOKENIZER.decode(event.conts[0].sample.tokens).count("I should check") == 2
            assert _own_text(event.conts[0].sample).count("<tool_call>") == 1

    async def test_checkpoint_rereview_replaces_the_earlier_reviewer_turn(self) -> None:
        reviewer = ScriptedReviewer(targets={1: "cache.py", 2: TASK.bug})
        forest, _ = await _forest(ScriptedActor(policy=follow_suspicious), reviewer, rereview="checkpoint")

        deep = [e for e in forest.events if e.depth == 2]
        assert deep and all(e.parent_cont is None for e in deep)
        for event in deep:
            assert event.checkpoint is forest.events[event.parent].checkpoint
            assert TOKENIZER.decode(event.ce_sample.tokens).count("I should check") == 1

    async def test_review_stops_at_the_maximum_depth(self) -> None:
        reviewer = ScriptedReviewer(targets={1: "cache.py", 2: "cache.py", 3: "cache.py"})
        forest, _ = await _forest(ScriptedActor(policy=follow_suspicious), reviewer, m=2, s=8, d=2)

        assert max(e.depth for e in forest.events) == 2


class TestResources:
    @pytest.mark.parametrize("targets", [{}, {1: TASK.bug}, {1: "cache.py", 2: TASK.bug}])
    async def test_every_environment_of_the_forest_is_closed(self, targets: dict) -> None:
        forest, envs = await _forest(ScriptedActor(policy=follow_suspicious), ScriptedReviewer(targets=targets))

        assert envs.all_envs() and all(env.closed for env in envs.all_envs())

    async def test_environments_are_closed_when_the_forest_fails(self) -> None:
        async def broken_reviewer(task, request):
            raise RuntimeError("reviewer endpoint down")

        envs = EnvLog(reviewer=broken_reviewer)
        task = make_task_sample()
        ctx = make_context(ScriptedActor(policy=_always("patch", WRONG)), envs)
        with pytest.raises(RuntimeError, match="reviewer endpoint down"):
            await build_forest(task, prompt_ids(task), ctx)

        assert envs.all_envs() and all(env.closed for env in envs.all_envs())
