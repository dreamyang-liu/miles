"""The harness recipe's train.py arguments: plain GRPO groups of Sprout's harness rollouts, after Arctic-SWE."""

import shlex

import pytest
from tests.fast.launch_scripts.py_harness import import_launch_script
from tests.fast.launch_scripts.sh_harness import REPO_ROOT

run = import_launch_script(REPO_ROOT / "examples" / "swe-rebench-sprout" / "run_harness_grpo.py")


def _argv(**fields) -> list[str]:
    args = run.ScriptArgs(run_id="261030-harness", model_endpoint="http://trainer:18081", **fields)
    return shlex.split(run.train_args(args))


def _value(argv: list[str], option: str) -> str:
    return argv[argv.index(option) + 1]


def test_each_task_is_a_plain_grpo_group_with_the_papers_advantages_and_optimizer() -> None:
    argv = _argv()

    assert "--sprout-rollout-rafo" not in argv
    assert _value(argv, "--advantage-estimator") == "grpo" and "--disable-grpo-std-normalization" in argv
    assert "--sprout-rollout-drop-zero-advantage-groups" in argv and "--use-dynamic-global-batch-size" in argv
    assert "--calculate-per-token-loss" in argv and _value(argv, "--entropy-coef") == "0.0"
    assert [
        _value(argv, option) for option in ("--lr", "--adam-beta2", "--adam-eps", "--clip-grad", "--weight-decay")
    ] == [
        "1e-06",
        "0.98",
        "1e-15",
        "1.0",
        "0.0",
    ]


def test_the_defaults_are_the_papers_overfit_run_on_one_8_gpu_node() -> None:
    argv = _argv()

    assert (_value(argv, "--rollout-batch-size"), _value(argv, "--n-samples-per-prompt")) == ("32", "5")
    assert _value(argv, "--sprout-rollout-max-turns") == "100"
    assert _value(argv, "--rollout-max-response-len") == "32768"
    assert (_value(argv, "--tensor-model-parallel-size"), _value(argv, "--actor-num-gpus-per-node")) == ("2", "8")
    assert _value(argv, "--rollout-num-gpus-per-engine") == "1"
    assert int(_value(argv, "--seq-length")) >= 131072 + 16384, "a whole 131k-token trajectory and its ends"


def test_the_first_runs_two_gpu_settings_still_build() -> None:
    argv = _argv(
        num_gpus_per_node=2,
        rollout_batch_size=16,
        max_turns=60,
        max_response_len=16384,
        max_context_len=81920,
        max_tokens_per_gpu=81920,
    )

    assert (_value(argv, "--actor-num-gpus-per-node"), _value(argv, "--seq-length")) == ("2", "81920")


@pytest.mark.parametrize(
    ("fields", "message"),
    [
        ({"num_gpus_per_node": 6, "tp": 4}, "split into TP x CP trainer groups"),
        ({"num_gpus_per_node": 8, "engine_tp": 3}, "split into TP x CP trainer groups and into engines"),
        ({"max_tokens_per_gpu": 65536}, "must hold one max_context_len trajectory"),
        ({"n_samples_per_prompt": 1}, "at least 2"),
    ],
)
def test_a_layout_or_batch_that_cannot_run_is_refused(fields: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        run.ScriptArgs(model_endpoint="http://trainer:18081", **fields)
