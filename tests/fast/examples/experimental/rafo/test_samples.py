import itertools
import re
from dataclasses import replace

import pytest
from examples.experimental.rafo.credit import fixed_coefficients
from examples.experimental.rafo.forest import build_forest
from examples.experimental.rafo.samples import forest_samples
from tests.fast.examples.experimental.rafo.forest_fixtures import (
    TASK,
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

COEF = fixed_coefficients(m=2, k=2, s=4, c=64.0, c_h=32.0)


async def _samples(policy, targets: dict, *, credit: str = "subtree", task=TASK, **config):
    sample = make_task_sample(task)
    ctx = make_context(ScriptedActor(policy=policy), ScriptedReviewer(targets=targets), EnvLog(), **config)
    forest = await build_forest(sample, prompt_ids(sample), ctx)
    return forest, forest_samples(sample, forest, COEF, credit=credit)


def _meta(samples) -> list[tuple[str, float, float]]:
    return [
        (s.train_metadata["rafo_kind"], s.train_metadata["rafo_adv"], s.train_metadata["rafo_coef"]) for s in samples
    ]


def _follow_every_other():
    """The first turn after a check follows it in every other continuation; a continuation that ignored it fails."""
    coin = itertools.count()

    def policy(text: str) -> str:
        hits = re.findall(r"Inspected (\w+\.py): found a suspicious line", text)
        return tool_call_text("patch", hits[-1] if hits and next(coin) % 2 == 0 else WRONG)

    return policy


class TestForestSamples:
    async def test_roots_of_a_task_without_review_get_local_rloo_advantages(self) -> None:
        answers = iter([tool_call_text("patch", TASK.bug), tool_call_text("patch", WRONG)])
        _, samples = await _samples(lambda text: next(answers, tool_call_text("patch", WRONG)), {})

        assert _meta(samples) == [("root", 1.0, COEF.root), ("root", -1.0, COEF.root)]
        assert [s.reward for s in samples] == [1.0, 0.0]
        assert all(s.rollout_id == 3 and s.metadata["raw_reward"] == 0.5 for s in samples)

    async def test_continuations_with_different_outcomes_train_and_the_reviewer_turn_gets_its_weight(self) -> None:
        forest, samples = await _samples(
            _follow_every_other(), {1: TASK.bug}, task=replace(TASK, max_patches=2), m=2, s=1
        )

        assert [c.reward for c in forest.events[0].conts] == [1.0, 0.0]
        assert _meta(samples) == [
            ("repair", 1.0, COEF.repair),
            ("repair", -1.0, COEF.repair),
            ("reflect", 0.0, COEF.reflect(0.5)),
        ]
        assert samples[2] is forest.events[0].ce_sample and samples[2].reward == 0.5
        assert all(s.metadata["raw_reward"] == 0.0 for s in samples)

    @pytest.mark.parametrize(
        ("credit", "weights"), [("subtree", [0.5, 0.5, 1.0, 1.0]), ("own", [1.0, 1.0])], ids=["subtree", "own"]
    )
    async def test_nested_review_credits_the_first_turn_only_under_subtree_credit(self, credit, weights) -> None:
        """Every group here has one outcome, so only the reviewer turns carry a signal."""
        _, samples = await _samples(follow_suspicious, {1: "cache.py", 2: TASK.bug}, credit=credit)

        assert [kind for kind, _, _ in _meta(samples)] == ["reflect"] * len(weights)
        assert [coef for _, _, coef in _meta(samples)] == pytest.approx([COEF.reflect(w) for w in weights])
        assert [s.metadata["rafo"]["depth"] for s in samples] == ([1, 1, 2, 2] if credit == "subtree" else [2, 2])

    async def test_a_forest_without_signal_keeps_one_zero_weight_sample_so_the_task_counts(self) -> None:
        """miles divides each step by its number of rollouts; a task that disappears would change that count."""
        forest, samples = await _samples(lambda text: tool_call_text("patch", WRONG), {})

        assert _meta(samples) == [("root", 0.0, 0.0)]
        assert samples[0] in [root.sample for root in forest.roots]
        assert samples[0].rollout_id == 3

    async def test_every_sample_records_its_role_for_logging(self) -> None:
        forest, samples = await _samples(_follow_every_other(), {1: TASK.bug}, task=replace(TASK, max_patches=2), s=1)

        stats = samples[0].metadata["rafo"]
        assert stats["kind"] == "repair" and stats["event"] == 0 and stats["depth"] == 1
        assert stats["task"] == {
            "root_success": 0.0,
            "events": 1,
            "reviewer_reads": 1,
            "abstentions": 0,
            "cont_success": 0.5,
            "reflect_samples": 1,
        }
