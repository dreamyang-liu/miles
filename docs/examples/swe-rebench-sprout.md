---
title: "SWE-rebench with Sprout rollouts"
description: "Standard GRPO on SWE-rebench, with rollouts run, snapshotted and graded by Sprout and trained on as hint-free messages."
# Generated from examples/swe-rebench-sprout/README.md by scripts/tools/sync_example_docs.py. Edit that README, not this file.
---
[Sprout](https://github.com/dreamyang-liu/sprout) runs the agent, snapshots
every step and grades the final state; Miles trains standard GRPO on what comes
back. Each prompt group is one `POST /rollout-groups` to Sprout's RL driver.
Sprout runs `n_samples_per_prompt` independent mini-swe-agent rollouts of the
task against the policy endpoint. It grades each final snapshot with the
official SWE-rebench procedure in a fresh sandbox, and returns every trajectory
as Chat Completions messages with a 0/1 reward.

```text
Miles SproutRolloutFn ── POST /rollout-groups ──> Sprout RL driver (:11001)
        ^                                              │ one rollout job + one grade job per sample
        │                                              v
        │                                     Sprout Run Store (:18110) + workers
        │                                              │ mini-swe-agent, one bash tool,
        │                                              │ podman sandbox (default) or microVM
        │     Chat Completions <───────────────────────┘ through Sprout's gateway
        │     (the Miles router, or any endpoint serving the policy)
        └── GET /rollout-groups/{id}: messages, tools, reward, status ── DELETE when consumed
```

Miles renders the returned messages with the policy's chat template. It masks
everything but the assistant turns and recomputes log-probs on that sequence.
The samples keep their `group_index`, so `--advantage-estimator grpo` normalizes
rewards within each task's group exactly as for any other rollout. Nothing from
the sampling process is imported: Sprout returns no tokens or log-probs. A branch
continuation's hint is wrapped in `<sprout_training_hint>` and removed before
export, so the hint-conditioned probabilities would not describe the trained
sequence anyway. For that reason `--use-rollout-logprobs`,
`--skip-actor-forward-only`, TIS and rollout routing or indexer replay are
refused at construction, as are `--rollout-stop-token-ids` and
`--apply-chat-template-kwargs`, which a Chat Completions endpoint cannot honour.

## Prepare the data

```bash
python examples/swe-rebench-sprout/prepare_data.py \
  --output-dir /root/swe-inputs --sprout-data-dir /home/ubuntu/swe-inputs \
  --model Qwen/Qwen3.8-27B --limit 16
```

The script writes these files:

- `miles.jsonl` is the prompt dataset. Each row's `metadata` names Sprout's `task_id` and the task `image`.
- `tasks.jsonl` and `log_parsers.py` are the frozen grading inputs. They belong in `--sprout-data-dir` on the Sprout host.
- `sprout-driver.json` is the RL driver's configuration. It pins both grading files by SHA-256.
- `manifest.json` records the dataset revision and both hashes.

`--source-parquet` reads a local copy of the dataset, and `--select` keeps
named instances.

## Start Sprout

On the Sprout host, from its repository, following Sprout's
`docs/RUNSTORE.md` and `docs/RL_DRIVER.md`:

```bash
python -m sprout.runstore init   --config src/sprout/runstore/config.example.json
python -m sprout.runstore serve  --config src/sprout/runstore/config.example.json
python -m sprout.runstore worker --config src/sprout/runstore/config.example.json --concurrency 16
python -m sprout.rl_driver --config /home/ubuntu/swe-inputs/sprout-driver.json
```

The example worker profiles run every sandbox as a rootless podman container on
a btrfs store. Name the `mini-microvm` and `grade-microvm` profiles to run on
AgentENV microVMs instead. Pre-pull the task images first, either with
`ops/prepull_images.py` or with `PodmanPool.prepare`. If the policy endpoint
takes a key, `miles.api_key_env` in `sprout-driver.json` names the variable the
worker profile's `worker_env` provides.

## Train

```bash
python examples/swe-rebench-sprout/run_qwen3_8_27b.py \
  --model-dir /root/models --data-dir /root/swe-inputs \
  --sprout-url http://SPROUT_HOST:11001 --model-endpoint http://THIS_HOST:18081
```

The launcher colocates a TP=8 trainer and rollout engine on one 8-GPU node. It
converts the checkpoint once and passes these rollout options to `train.py`:

```text
--rollout-function-path miles.rollout.sprout.rollout_fn.SproutRolloutFn
--sprout-rollout-base-url http://SPROUT_HOST:11001
--sprout-rollout-model-endpoint http://THIS_HOST:18081   # how Sprout's agents reach the router
--sprout-rollout-max-turns 64                            # model calls per trajectory
--advantage-estimator grpo --loss-mask-type qwen3
--sglang-reasoning-parser qwen3 --sglang-tool-call-parser qwen3_coder
```

A group Sprout cannot fill completely fails the step rather than shrinking the
group. A trajectory cut off by `--sprout-rollout-max-turns` or the wall-time
budget is still graded. It arrives with `status="truncated"` and its
`stop_reason`, and trains with its real reward. Each group waits
`--sprout-rollout-timeout-seconds` for execution plus
`--sprout-rollout-finalization-timeout-seconds` for the final snapshots and
grading.

## Without a GPU

Everything up to the optimizer step runs on a CPU host. `rollout_only.py` drives
`SproutRolloutFn` once against any Chat Completions endpoint that serves the
policy. It imports the trajectories as the trainer would and applies Miles's GRPO
reward normalization. It then saves the samples in the format
`--load-debug-rollout-data` reads:

```bash
python examples/swe-rebench-sprout/rollout_only.py \
  --prompt-data /root/swe-inputs/miles.jsonl --hf-checkpoint Qwen/Qwen3.8-27B \
  --model-name Qwen/Qwen3.8-27B --loss-mask-type qwen3 \
  --sprout-rollout-base-url http://SPROUT_HOST:11001 \
  --sprout-rollout-model-endpoint http://MODEL_HOST:PORT \
  --rollout-batch-size 1 --n-samples-per-prompt 2 \
  --save-debug-rollout-data /root/swe-rollouts/rollout_data/{rollout_id}.pt
```

On a GPU host, `run_qwen3_8_27b.py --load-rollout-data /root/swe-rollouts/rollout_data/0.pt`
trains on that rollout without SGLang or Sprout. Only `hf_checkpoint`'s tokenizer
and chat template are needed on the CPU side.

The unit tests (`tests/fast/rollout/sprout/`) run on CPU too. They check the
request Miles builds against the example Sprout ships, the importer against
Sprout's shipped result example, and the rollout function against a fake driver
over `httpx.MockTransport`. They also feed imported samples through Miles's own
GRPO reward normalization and advantage computation.
