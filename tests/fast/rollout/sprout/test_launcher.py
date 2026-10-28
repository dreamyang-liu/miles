"""The SWE-rebench Sprout recipe's train.py arguments: a RAFO search for each task, trained with the RAFO loss."""

import shlex

import pytest
from tests.fast.launch_scripts.py_harness import import_launch_script
from tests.fast.launch_scripts.sh_harness import REPO_ROOT

from miles.rollout.sprout.rollout_fn import RAFO_LOSS

run = import_launch_script(REPO_ROOT / "examples" / "swe-rebench-sprout" / "run_qwen3_8_27b.py")


def _argv(**fields) -> list[str]:
    args = run.ScriptArgs(run_id="261028-sprout", model_endpoint="http://trainer:18081", **fields)
    return shlex.split(run.train_args(args))


def _value(argv: list[str], option: str) -> str:
    return argv[argv.index(option) + 1]


def test_each_task_is_a_rafo_search_trained_with_its_loss_and_counted_in_its_optimizer_step() -> None:
    argv = _argv()

    assert "--sprout-rollout-rafo" in argv
    assert (_value(argv, "--loss-type"), _value(argv, "--custom-loss-function-path")) == ("custom_loss", RAFO_LOSS)
    assert _value(argv, "--n-samples-per-prompt") == "4", "RAFO's M roots"
    assert [_value(argv, f"--sprout-rollout-rafo-{name}") for name in ("continuations", "depth", "events")] == [
        "2",
        "3",
        "4",
    ], "RAFO's K, D and S"
    assert _value(argv, "--global-batch-size") == "8", "two optimizer steps per rollout of 16 tasks"
    assert "--disable-rewards-normalization" in argv and "--allow-partial-train-step" in argv
    assert _value(argv, "--eps-clip-c") == "3.0" and _value(argv, "--eps-clip-high") == "0.2"
    assert _value(argv, "--sprout-rollout-rafo-c-h") == "32.0"
    for option in (
        "--calculate-per-token-loss",
        "--use-dynamic-global-batch-size",
        "--custom-reward-post-process-path",
    ):
        assert option not in argv
    assert not any(option.startswith(("--sprout-rollout-gsml-", "--sprout-rollout-search-")) for option in argv)


def test_the_search_settings_reach_the_rollout_function() -> None:
    argv = _argv(rafo_events=6, rafo_depth=2, rafo_turn_credit="own", review_seconds=300.0, branch_seconds=600.0)

    assert (_value(argv, "--sprout-rollout-rafo-events"), _value(argv, "--sprout-rollout-rafo-depth")) == ("6", "2")
    assert _value(argv, "--sprout-rollout-rafo-turn-credit") == "own"
    assert _value(argv, "--sprout-rollout-rafo-review-seconds") == "300.0"
    assert _value(argv, "--sprout-rollout-rafo-wall-time-seconds") == "600.0"


@pytest.mark.parametrize("field", ["n_samples_per_prompt", "rafo_continuations"])
def test_a_local_rloo_group_of_one_is_refused(field: str) -> None:
    with pytest.raises(ValueError, match="at least 2"):
        run.ScriptArgs(model_endpoint="http://trainer:18081", **{field: 1})
