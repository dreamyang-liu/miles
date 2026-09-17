"""Train Qwen3.8 LoRA on 32 tasks per step without speculative decoding."""

import argparse
import json
import os
from pathlib import Path
import shlex

import miles.utils.external_utils.command_utils as U


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--length", type=int, default=131072)
    parser.add_argument("--micro-batch-size", type=int, choices=[1, 2], default=2)
    parser.add_argument("--tp", type=int, default=2)
    parser.add_argument("--cp", type=int, default=4)
    parser.add_argument("--inference-tp", type=int, default=1)
    parser.add_argument("--inference-context-length", type=int, default=196608)
    parser.add_argument("--inference-profile", choices=["original", "graph"], default="graph")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--rollout-batch-size", type=int, default=32)
    parser.add_argument("--num-rollout", type=int, default=3)
    parser.add_argument("--loss-weighting", choices=["task", "trajectory"], default="task")
    parser.add_argument("--reward-scale", type=float, default=0.5)
    parser.add_argument("--dataset", default="/data/swe-rebench-filtered-1586-v3.jsonl")
    parser.add_argument("--adapter-path", default="")
    parser.add_argument("--recover-requests", type=Path)
    args = parser.parse_args()
    assert args.rollout_batch_size == 32, "This run collects 32 tasks per update"
    nominal_batch_size = args.rollout_batch_size * 8
    assert args.length == 131072
    # Ray workers put the submitting working directory on sys.path ahead of
    # PYTHONPATH. Keep that directory on this exact source checkout.
    os.chdir(U.repo_base_dir)
    assert 8 % (args.tp * args.cp) == 0 and 8 % args.inference_tp == 0
    assert args.length in {131072, 81920, 65536, 49152}
    assert args.inference_context_length >= args.length
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    if (output / "launch-arguments.json").exists():
        raise FileExistsError("Use a new run directory for a restart")
    recovery_env = {}
    if args.recover_requests:
        assert not args.adapter_path, "Recovery requires the same initial zero-B policy"
        recovery = json.loads(args.recover_requests.read_text())
        assert recovery["optimizer_updates"] == 0 and recovery["speculative_decoding"] is False
        archived = output / "recovery-requests.json"
        archived.write_bytes(args.recover_requests.read_bytes())
        recovery_env["MILES_NIGHT_RECOVER_REQUESTS"] = str(archived)
    (output / "training_audit.py").write_text(
        Path("/deployment/b300-lora-32tasks-20260916/training_audit.py").read_text()
    )
    (output / "night_rollout.py").write_text(
        Path("/deployment/b300-lora-32tasks-20260916/night_rollout.py").read_text()
    )
    model = "/models/Qwen3.8-27B"
    targets = ",".join(
        "language_model.decoder.layers.*." + name
        for name in ("self_attention.linear_qkv", "self_attention.linear_proj", "mlp.linear_fc1", "mlp.linear_fc2")
    )
    flags = [
        "--train-backend", "megatron", "--hf-checkpoint", model, "--load", model, "--ref-load", model,
        "--save", str(output / "checkpoints"), "--save-interval", "1",
        "--megatron-to-hf-mode", "bridge", "--lora-rank", "32", "--lora-alpha", "64",
        "--lora-dropout", "0", "--target-modules", targets, "--no-gradient-accumulation-fusion",
        "--lora-base-cpu-backup", "--mtp-num-layers", "0", "--bf16",
        "--prompt-data", args.dataset, "--input-key", "prompt", "--metadata-key", "metadata",
        "--rollout-shuffle", "--rollout-function-path", "night_rollout.NightAshMessageRolloutFn",
        "--ash-rollout-base-url", "http://host.docker.internal:19051",
        "--ash-rollout-model-endpoint", "http://127.0.0.1:19182",
        "--ash-rollout-branching", "--ash-rollout-max-sequence-tokens", str(args.length),
        "--ash-rollout-branching-return-mode", "all",
        "--ash-rollout-loss-weighting", args.loss_weighting,
        "--ash-rollout-truncated-reward-scale", str(args.reward_scale),
        "--ash-rollout-max-unusable-groups", "8", "--ash-rollout-max-turns", "1000000",
        "--ash-rollout-timeout-seconds", "10800", "--ash-rollout-finalization-timeout-seconds", "1800",
        "--num-rollout", str(args.num_rollout), "--rollout-batch-size", str(args.rollout_batch_size),
        "--n-samples-per-prompt", "8", "--global-batch-size", str(nominal_batch_size),
        "--use-dynamic-global-batch-size", "--check-lora-weight-equal",
        "--micro-batch-size", str(args.micro_batch_size), "--rollout-max-response-len", "32000",
        "--rollout-temperature", "1.0", "--rollout-top-p", "0.95", "--rollout-top-k", "20",
        "--loss-mask-type", "qwen3", "--chat-template-path", model + "/chat_template.jinja",
        "--seq-length", str(args.length), "--max-position-embeddings", str(args.length),
        "--tensor-model-parallel-size", str(args.tp), "--context-parallel-size", str(args.cp),
        "--pipeline-model-parallel-size", "1", "--expert-model-parallel-size", "1",
        "--expert-tensor-parallel-size", "1", "--sequence-parallel",
        "--recompute-granularity", "full", "--recompute-method", "uniform", "--recompute-num-layers", "1",
        "--recompute-loss-function", "--log-probs-chunk-size", "256",
        "--advantage-estimator", "grpo", "--use-kl-loss", "--kl-loss-coef", "0", "--kl-loss-type", "low_var_kl",
        "--entropy-coef", "0.001", "--eps-clip", "0.2", "--eps-clip-high", "0.28",
        "--optimizer", "adam", "--lr", "1e-6", "--lr-decay-style", "constant",
        "--weight-decay", "0.1", "--adam-beta1", "0.9", "--adam-beta2", "0.98",
        "--optimizer-cpu-offload", "--overlap-cpu-optimizer-d2h-h2d", "--use-precision-aware-optimizer",
        "--colocate", "--actor-num-nodes", "1", "--actor-num-gpus-per-node", "8", "--num-gpus-per-node", "8",
        "--attention-dropout", "0", "--hidden-dropout", "0",
        "--accumulate-allreduce-grads-in-fp32", "--attention-softmax-in-fp32", "--attention-backend", "flash",
        "--rollout-num-gpus-per-engine", str(args.inference_tp), "--use-miles-router",
        "--sglang-router-port", "18081", "--sglang-served-model-name", "Qwen3.8-27B",
        "--sglang-context-length", str(args.inference_context_length), "--sglang-mem-fraction-static", "0.55",
        "--sglang-enable-strict-thinking",
        "--sglang-max-running-requests", "4",
        "--sglang-chunked-prefill-size", "4096",
        "--sglang-reasoning-parser", "qwen3",
        "--sglang-tool-call-parser", "qwen3_coder", "--sglang-log-requests", "--sglang-log-requests-level", "1",
        "--custom-megatron-before-log-prob-hook-path", "training_audit.before_log_prob",
        "--custom-megatron-before-train-step-hook-path", "training_audit.before_train_step",
        "--custom-megatron-post-save-hook-path", "training_audit.after_save",
        "--dump-details", str(output / "dump_details"),
        "--use-wandb", "--wandb-mode", "online",
        "--wandb-project", "miles-qwen38-b300-128k",
        "--wandb-group", output.name, "--disable-wandb-random-suffix", "--wandb-dir", str(output / "wandb"),
    ]
    if args.inference_profile == "original":
        flags += ["--sglang-disable-cuda-graph"]
    else:
        flags += [
            "--sglang-attention-backend", "trtllm_mha",
            "--sglang-linear-attn-backend", "triton",
            "--sglang-disable-prefill-cuda-graph",
            "--sglang-cuda-graph-bs-decode", "1", "2", "4",
            "--sglang-max-total-tokens", "600000",
        ]
    if args.adapter_path:
        flags += ["--lora-adapter-path", args.adapter_path]
    assert "--use-dynamic-batch-size" not in flags and "--wandb-key" not in flags
    assert not any(flag.startswith("--sglang-speculative") for flag in flags)
    (output / "launch-arguments.json").write_text(json.dumps({
        "arguments": flags, "model_type": "qwen3.8-27B", "live_dataset": args.dataset,
        "length": args.length, "tp": args.tp, "cp": args.cp, "inference_tp": args.inference_tp,
        "inference_context_length": args.inference_context_length, "strict_thinking": True,
        "inference_profile": args.inference_profile, "dry_run": args.dry_run,
        "train_mtp_layers": 0, "frozen_mtp_draft": False, "speculative_decoding": False,
        "lora_rank": 32, "lora_alpha": 64, "micro_batch_size": args.micro_batch_size,
        "global_batch_size": "dynamic", "nominal_global_batch_size": nominal_batch_size,
        "tasks_per_update": args.rollout_batch_size, "allocated_trajectory_slots_per_task": 8,
        "inference_engines": 8 // args.inference_tp, "requests_per_engine": 4,
        "branching_return_mode": "all", "loss_weighting": args.loss_weighting,
        "capacity_only": False,
        "dynamic_microbatch_packing": False, "truncated_reward_scale": args.reward_scale,
        "branch_limits": [4, 3], "successful_root_limit": 2,
        "num_rollout": args.num_rollout,
        "rollout_batch_size": args.rollout_batch_size,
        "recovered_initial_policy": bool(args.recover_requests),
        "recovery_source_run": recovery["source_run"] if args.recover_requests else None,
    }, indent=2) + "\n")
    if args.dry_run:
        print(json.dumps({
            "status": "prepared", "training_started": False,
            "arguments_file": str(output / "launch-arguments.json"),
        }))
        return
    U.execute_train(
        train_args=shlex.join(flags), num_gpus_per_node=8, megatron_model_type="qwen3.8-27B",
        config=U.ExecuteTrainConfig(output_dir=str(output), extra_env_vars=json.dumps({
            "PYTHONPATH": str(output), "no_proxy": "*", "MILES_SGLANG_DUMMY_LOAD": "0",
            "MILES_NIGHT_SOURCE_ROOT": str(U.repo_base_dir), "MILES_NIGHT_RUN_DIR": str(output),
            **recovery_env,
        })),
    )


if __name__ == "__main__":
    main()
