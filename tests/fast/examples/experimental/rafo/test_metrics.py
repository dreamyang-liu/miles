from dataclasses import replace

import pytest
from examples.experimental.rafo import metrics
from examples.experimental.rafo.forest import build_forest
from examples.experimental.rafo.samples import forest_samples
from tests.fast.examples.experimental.rafo.forest_fixtures import (
    TASK,
    EnvLog,
    ScriptedActor,
    ScriptedReviewer,
    follow_suspicious,
    make_context,
    make_task_sample,
    prompt_ids,
    tool_call_text,
)
from tests.fast.examples.experimental.rafo.test_samples import COEF

from miles.utils.args.custom_view import ImmutableNamespace


async def _task_samples(index: int, policy, targets: dict) -> list:
    task = make_task_sample(replace(TASK))
    task.index = task.group_index = index
    ctx = make_context(ScriptedActor(policy=policy), EnvLog(reviewer=ScriptedReviewer(targets=targets)))
    forest = await build_forest(task, prompt_ids(task), ctx)
    return forest_samples(task, forest, COEF, credit="subtree")


class TestRafoMetrics:
    async def test_task_metrics_count_each_task_once(self) -> None:
        reviewed = await _task_samples(0, follow_suspicious, {1: TASK.bug})
        answers = iter([tool_call_text("patch", TASK.bug)] * 2)
        solved = await _task_samples(1, lambda text: next(answers), {})

        values = metrics.compute_rafo_metrics(reviewed + solved)

        assert values["rafo/root_success"] == pytest.approx(0.5)
        assert values["rafo/reviewed_task_frac"] == pytest.approx(0.5)
        assert values["rafo/events_per_task"] == pytest.approx(1.0)
        assert values["rafo/reviewer_reads_per_task"] == pytest.approx(1.0)
        assert values["rafo/abstention_rate"] == 0.0
        assert values["rafo/cont_success"] == pytest.approx(1.0)
        assert values["rafo/samples_per_task/reflect"] == pytest.approx(1.0)
        assert values["rafo/samples_per_task/root"] == pytest.approx(0.5)  # the zero-weight sample of task 1

    def test_samples_of_other_generate_functions_give_no_metrics(self) -> None:
        assert metrics.compute_rafo_metrics([make_task_sample()]) == {}

    async def test_the_log_hook_adds_rafo_metrics_and_keeps_the_default_rollout_metrics(self, monkeypatch) -> None:
        logged = []
        monkeypatch.setattr(metrics.tracking, "log", lambda args, values, step_key: logged.append((values, step_key)))
        samples = await _task_samples(0, follow_suspicious, {1: TASK.bug})
        args = ImmutableNamespace(wandb_always_use_train_step=False)

        keep_default = not metrics.log_rollout_data(7, args, samples, {}, 1.0)

        assert keep_default
        [(values, step_key)] = logged
        assert step_key == "rollout/step" and values["rollout/step"] == 7
        assert values["rafo/events_per_task"] == 2.0
