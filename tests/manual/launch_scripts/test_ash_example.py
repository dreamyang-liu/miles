from tests.fast.launch_scripts.py_harness import (
    call_entrypoint,
    format_recording,
    freeze_environment,
    import_launch_script,
    install_command_recorder,
)
from tests.fast.launch_scripts.sh_harness import REPO_ROOT, assert_matches_snapshot


def test_ash_qwen_tp8_launcher_snapshot(monkeypatch, tmp_path):
    freeze_environment(monkeypatch)
    recording = install_command_recorder(monkeypatch)
    rel = "examples/swe-rebench-ash/run_qwen3_8_27b.py"
    module = import_launch_script(REPO_ROOT / rel)
    call_entrypoint(
        module,
        "execute",
        {"model_endpoint": "https://model.example"},
        sandbox=tmp_path,
    )
    assert recording.commands
    assert all("--sglang-router-ip " not in command for command in recording.commands)
    assert any("--ash-rollout-max-turns 64 " in command for command in recording.commands)
    assert all("--ash-rollout-max-model-calls" not in command for command in recording.commands)
    assert all("--ash-rollout-max-tool-calls" not in command for command in recording.commands)
    assert any("--log-probs-chunk-size 1024 " in command for command in recording.commands)
    assert any("--recompute-loss-function " in command for command in recording.commands)
    snapshot = REPO_ROOT / "tests/snapshots/launch_scripts/py" / rel / "execute.txt"
    assert_matches_snapshot(snapshot, format_recording(recording, sandbox=tmp_path), rel)


def test_ash_replay_launcher_uses_complete_recorded_rollout(monkeypatch, tmp_path):
    freeze_environment(monkeypatch)
    recording = install_command_recorder(monkeypatch)
    module = import_launch_script(REPO_ROOT / "examples/swe-rebench-ash/run_qwen3_8_27b.py")
    call_entrypoint(
        module,
        "execute",
        {
            "model_endpoint": "https://model.example",
            "load_rollout_data": "/recorded/0.pt",
            "train_memory_margin_bytes": 0,
        },
        sandbox=tmp_path,
    )
    train = next(command for command in recording.commands if "ray job submit" in command)
    assert "--load-debug-rollout-data /recorded/0.pt " in train
    assert "--load-debug-rollout-data-subsample" not in train
    assert "--debug-disable-optimizer" not in train
    assert "--n-samples-per-prompt 2 " in train
    assert "--train-memory-margin-bytes 0 " in train


def test_ash_lora_replay_uses_bridge_and_explicit_gdn_targets(monkeypatch, tmp_path):
    freeze_environment(monkeypatch)
    recording = install_command_recorder(monkeypatch)
    rel = "examples/swe-rebench-ash/run_qwen3_8_27b.py"
    module = import_launch_script(REPO_ROOT / rel)
    call_entrypoint(
        module,
        "execute",
        {"model_endpoint": "https://model.example", "load_rollout_data": "/recorded/0.pt", "lora_rank": 16},
        sandbox=tmp_path,
    )
    train = next(command for command in recording.commands if "ray job submit" in command)
    assert "--megatron-to-hf-mode bridge " in train
    assert "--lora-rank 16 --lora-alpha 32 --lora-dropout 0.0 " in train
    assert "--ref-load /root/models/Qwen3.8-27B " in train
    assert "--mtp-num-layers 0 " in train
    assert "--no-gradient-accumulation-fusion " in train
    assert ".self_attention.in_proj" in train and ".self_attention.out_proj" in train
    assert "--load-debug-rollout-data /recorded/0.pt " in train
    snapshot = REPO_ROOT / "tests/snapshots/launch_scripts/py" / rel / "execute_lora.txt"
    assert_matches_snapshot(snapshot, format_recording(recording, sandbox=tmp_path), rel)
