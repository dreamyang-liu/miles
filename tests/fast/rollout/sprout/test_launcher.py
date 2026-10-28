"""The SWE-rebench Sprout recipe's train.py arguments under each search credit."""

import shlex

import pytest
from tests.fast.launch_scripts.py_harness import import_launch_script
from tests.fast.launch_scripts.sh_harness import REPO_ROOT

from miles.rollout.sprout.rollout_fn import GSML_POST_PROCESS, RAFO_LOSS

run = import_launch_script(REPO_ROOT / "examples" / "swe-rebench-sprout" / "run_qwen3_8_27b.py")


def _argv(**fields) -> list[str]:
    args = run.ScriptArgs(run_id="261028-sprout", model_endpoint="http://trainer:18081", **fields)
    return shlex.split(run.train_args(args))


def _value(argv: list[str], option: str) -> str:
    return argv[argv.index(option) + 1]


def test_gsml_trains_every_sample_of_a_step_with_a_per_token_loss() -> None:
    argv = _argv()

    assert _value(argv, "--custom-reward-post-process-path") == GSML_POST_PROCESS
    assert "--calculate-per-token-loss" in argv and "--use-dynamic-global-batch-size" in argv
    assert "--sprout-rollout-credit" not in argv and "--loss-type" not in argv


def test_rafo_trains_with_its_loss_and_counts_the_tasks_of_each_optimizer_step() -> None:
    argv = _argv(credit="rafo", n_samples_per_prompt=4)

    assert _value(argv, "--sprout-rollout-credit") == "rafo"
    assert (_value(argv, "--loss-type"), _value(argv, "--custom-loss-function-path")) == ("custom_loss", RAFO_LOSS)
    assert _value(argv, "--n-samples-per-prompt") == "4"
    assert _value(argv, "--global-batch-size") == "8", "two optimizer steps per rollout of 16 tasks"
    assert "--disable-rewards-normalization" in argv and "--allow-partial-train-step" in argv
    assert _value(argv, "--eps-clip-c") == "3.0" and _value(argv, "--eps-clip-high") == "0.2"
    for option in (
        "--calculate-per-token-loss",
        "--use-dynamic-global-batch-size",
        "--custom-reward-post-process-path",
    ):
        assert option not in argv
    assert not any(option.startswith("--sprout-rollout-gsml-") for option in argv)
    assert _value(argv, "--sprout-rollout-rafo-c-h") == "32.0"
    assert _value(argv, "--sprout-rollout-search-repair-rounds") == "3", "the search is the same under both credits"


@pytest.mark.parametrize(("credit", "roots"), [("gsml", 4), ("rafo", 1)])
def test_a_root_count_the_credit_is_not_defined_for_is_refused(credit: str, roots: int) -> None:
    with pytest.raises(ValueError, match="n_samples_per_prompt"):
        run.ScriptArgs(model_endpoint="http://trainer:18081", credit=credit, n_samples_per_prompt=roots)
