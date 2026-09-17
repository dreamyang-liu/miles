# B300: 32-task pure LoRA training

This directory preserves the deployment scripts used for the Qwen3.8-27B trial.
The scripts retain the original container paths so they can be compared with
the running deployment. They are a deployment-specific recipe.

The training configuration is 8 B300 GPUs, BF16, LoRA rank 32 / alpha 64,
131072-token sequences, TP2 / CP4 / DP1 / VPP1, and microbatch size 2.
Inference uses eight TP1 engines with four concurrent requests per engine,
CUDA decode graphs, and no MTP or speculative decoding.

Each update collects 32 tasks and all their valid graded root/branch
trajectories. Eight slots per task give a nominal batch size of 256, while
the learner uses the actual trajectory count. Task-equal loss is the default;
the final microbatch may contain one trajectory. Branch widths are capped at
4 then 3, or 2 children for a successful root.

## Required environment

- This Miles checkout and its Megatron/SGLang dependencies must be installed
  in the `miles-b300-dev` container. The source directory passed to
  `run_job.py --source-root` must contain this version of Miles.
- Mount the checkpoint at `/models/Qwen3.8-27B`, including its chat template,
  and the prepared v3 dataset at
  `/data/swe-rebench-filtered-1586-v3.jsonl`.
- Mount this repository's `deployment` directory at `/deployment`.
  The host run root is
  `/opt/dlami/nvme/projects/miles-stack/runs`, mounted at `/runs`.
- Configure the companion Ash source from
  `dreamyang-liu/Ash`, branch `codex/b300-rollout-recovery-20260917`.
  Its dedicated driver must use `miles.branching.return_mode=all`,
  `miles.max_samples=8`, 32 actor workers, and the branch limits above.
- The driver is reached at `host.docker.internal:19051`; the model endpoint
  advertised to Ash is `127.0.0.1:19182`. Set up the corresponding tunnels
  and container host mapping before launching.
- W&B authentication must already be configured in the container.
  Credentials, datasets, checkpoints, trajectory logs, and recovery manifests
  are not included in this source directory.

## Commands

Inside the configured container, prepare and inspect the flags without
starting training:

```bash
python /deployment/b300-lora-32tasks-20260916/launch.py \
  --output /runs/b300-lora-32tasks-20260916/preview \
  --dry-run
```

On the host, start a fresh run:

```bash
python deployment/b300-lora-32tasks-20260916/run_job.py \
  --name live-lora-32tasks-new \
  --micro-batch-size 2 \
  --num-rollout 3 \
  --source-root /runs/b300-lora-32tasks-20260916/source
```

Check progress on the host with `status.py --run live-lora-32tasks-new --compact`.
Inside the container, audit completed updates with:

```bash
python /deployment/b300-lora-32tasks-20260916/audit_run.py \
  --run /runs/b300-lora-32tasks-20260916/live-lora-32tasks-new \
  --steps 3
```

`training_audit.py` and `audit_helpers.py` record per-rank gradients, optimizer
steps, scheduler increments, and adapter snapshots. `night_rollout.py` checks
the loaded source and request configuration, records submitted requests, and
retries transient HTTP/transport failures using the same request IDs for up
to 300 seconds. `test_transport_retry.py` covers that retry behavior.
`ash_progress.py` is a read-only helper for the original Ash deployment path.

The optional recovery flags require externally verified snapshots, matching
request manifests, and a readiness marker. Ordinary fresh runs omit them;
these scripts do not automatically reconstruct or approve a recovery.

## Recorded validation

At 2026-09-17 03:22 UTC, the active deployment had completed and audited
**one of three requested real optimizer updates**. Its first batch contained
32 tasks and 95 trajectories, split into 48 microbatches. All eight ranks
updated their adapters, CP copies matched, a checkpoint was saved, and
subsequent real requests used LoRA version 2. The second batch was sampling.
The deployment's first-update peak was approximately 236.22 GiB per GPU.
This is a record of the checked run, not a claim that all three updates finished.
