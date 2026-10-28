import shlex

import pytest
from tests.fast.launch_scripts.py_harness import import_launch_script
from tests.fast.launch_scripts.sh_harness import REPO_ROOT

from miles.utils.external_utils.command_utils.ray_backend.backend import RayCommandBackend

run = import_launch_script(REPO_ROOT / "examples" / "experimental" / "rafo" / "run_qwen3_0_6b_rafo.py")


def _value(argv: list[str], option: str) -> str:
    return argv[argv.index(option) + 1]


@pytest.fixture
def args(tmp_path):
    return run.ScriptArgs(
        run_id="261028-rafo", model_dir="/models", data_dir=str(tmp_path), output_dir="/output", num_gpus_per_node=2
    )


def test_recipe_wires_the_rafo_generate_function_loss_and_log_hook(args, monkeypatch):
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    argv = shlex.split(run._build_train_args(args))

    assert _value(argv, "--custom-generate-function-path") == "examples.experimental.rafo.generate.RafoGenerate"
    assert _value(argv, "--loss-type") == "custom_loss"
    assert _value(argv, "--custom-loss-function-path") == "examples.experimental.rafo.loss.rafo_loss_function"
    assert _value(argv, "--custom-rollout-log-function-path") == "examples.experimental.rafo.metrics.log_rollout_data"
    assert not any(flag.startswith("--rafo-reviewer") for flag in argv)  # the rollout service reviews
    assert _value(argv, "--eps-clip-c") == "3"
    assert "--disable-rewards-normalization" in argv and "--apply-chat-template" not in argv


def test_one_rollout_is_one_task_and_each_iteration_takes_two_optimizer_steps(args, monkeypatch):
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    argv = shlex.split(run._build_train_args(args))

    assert _value(argv, "--n-samples-per-prompt") == "1"
    assert int(_value(argv, "--rollout-batch-size")) == 2 * int(_value(argv, "--global-batch-size"))
    assert _value(argv, "--num-gpus-per-node") == "2" and _value(argv, "--actor-num-gpus-per-node") == "2"
    assert _value(argv, "--prompt-data") == str(run._dataset_path(args))


def test_execute_submits_a_megatron_job_and_prepare_writes_the_dataset_once(args, monkeypatch):
    calls: list[dict] = []
    commands: list[str] = []
    monkeypatch.setattr(RayCommandBackend, "execute_train", lambda self, **kwargs: calls.append(kwargs))
    monkeypatch.setattr(RayCommandBackend, "exec_command_cpu", lambda self, cmd, **kwargs: commands.append(cmd))
    monkeypatch.setattr(RayCommandBackend, "convert_checkpoint", lambda self, **kwargs: commands.append("convert"))

    run.prepare(args)
    first = run._dataset_path(args).read_text()
    run.prepare(args)
    run.execute(args)

    assert run._dataset_path(args).read_text() == first and first.count("\n") == run._N_TASKS
    assert commands.count("convert") == 2
    [call] = calls
    assert call["megatron_model_type"] == "qwen3-0.6B" and call["num_gpus_per_node"] == 2
