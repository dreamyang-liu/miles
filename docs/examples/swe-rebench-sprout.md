---
title: "SWE-rebench with Sprout rollouts"
description: "GRPO with hindsight search on SWE-rebench: Sprout runs, snapshots and grades the rollouts, branches failed roots, and GSML credits the hint-free messages."
# Generated from examples/swe-rebench-sprout/README.md by scripts/tools/sync_example_docs.py. Edit that README, not this file.
---
[Sprout](https://github.com/dreamyang-liu/sprout) runs the agent, snapshots
every step and grades the final state; Miles trains on what comes back. Each
prompt group is one `POST /rollout-groups` to Sprout's RL driver. Sprout runs
`n_samples_per_prompt` independent mini-swe-agent rollouts of the task against
the policy endpoint. It grades each final snapshot with the official
SWE-rebench procedure in a fresh sandbox, and returns every trajectory as Chat
Completions messages with a 0/1 reward. Trained as they come back, those
rollouts are standard GRPO; the launcher below runs each task as a RAFO search
instead (see [RAFO search](#rafo-search)).

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
Without a search the samples keep their `group_index`, so
`--advantage-estimator grpo` normalizes rewards within each task's group
exactly as for any other rollout. Nothing from
the sampling process is imported: Sprout returns no tokens or log-probs, and a
RAFO continuation's history holds a reviewer's turn that the policy did not
sample, so the sampling probabilities would not describe the trained sequence
anyway. For that reason `--use-rollout-logprobs`,
`--skip-actor-forward-only`, TIS and rollout routing, indexer or sampling-support
replay (`--rollout-top-p` below 1 or a positive `--rollout-top-k`) are refused at
construction, as are `--rollout-stop-token-ids` and
`--apply-chat-template-kwargs`, which a Chat Completions endpoint cannot honour.
Every request carries `top_k`, `-1` when `--rollout-top-k` is off: without it the
engine would take the checkpoint's `generation_config` (`top_k` 20 for Qwen3) and
sample from a truncated distribution while the trainer scores the full softmax.

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

Under `miles`, `sprout-driver.json` also holds what a search needs on the
Sprout side:

- `review` (`--review-*`): how a search reviews its failed roots. Without it the
  driver refuses every search request with HTTP 400. `--review-profile`
  (`review`) names the Run Store worker profile the reviews run on, which the
  Run Store config must define; Sprout's example config does.
  `--review-max-output-tokens` (32768), `--review-prompt-chars` (250000) and
  `--review-temperature` (0.8) set the reviewer's model calls.
  `--review-reasoning-effort` and `--review-max-concurrent-requests` are written
  only when given; without them Sprout sends no `reasoning_effort` and runs up
  to 8 of a review's calls at once.
- `grade_wall_seconds` (`--grade-wall-seconds`, 600): how long a grade may run
  before its sample is given up as missing. Sprout keeps 120 s of it for the
  grading sandboxes and the diff, and refuses less than 240.
- `api_key_env` (`--api-key-env`, `SPROUT_MODEL_API_KEY`): the worker-side
  variable holding the policy endpoint's key. A review job without one is
  refused once the roots are graded.

The reviewer is the policy, and its thinking counts against
`max_output_tokens` while Sprout reads only the reply. Qwen3.8's template thinks
at `xhigh` unless the request names a `reasoning_effort` (`low`, `medium` or
`xhigh`), and at Sprout's default of 16384 tokens a reviewer can spend whole
calls thinking and return no plan; a search whose review fails ends at its
roots. A larger budget makes each call longer, and each review makes three
calls in a row for a failed trajectory (the analysis, the point review and the
turn), all within `--sprout-rollout-rafo-review-seconds`. At about three characters per
token, a rough rate for code and logs, 250000 characters are 83k tokens, so a
prompt fits beside 32768 output tokens in the 131072-token context the launcher
gives SGLang, which refuses a request that does not fit.

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
`ops/prepull_images.py` or with `PodmanPool.prepare`. The example `mini` and
`review` profiles pass `SPROUT_MODEL_API_KEY` from the worker's environment to
their jobs (`worker_env`), so start the worker with it set. The rollouts send it
as the policy endpoint's key, and a review refuses to run with it empty, so for
an endpoint that takes no key, such as Miles's router, set it to any value.

## Train

```bash
python examples/swe-rebench-sprout/run_qwen3_8_27b.py \
  --model-dir /root/models --data-dir /root/swe-inputs \
  --sprout-url http://SPROUT_HOST:11001 --model-endpoint http://THIS_HOST:18081
```

The launcher colocates the trainer, tensor-parallel 2 × context-parallel 4 over
the whole node, with SGLang engines of 2 GPUs each on one 8-GPU node, and
checks that `max_tokens_per_gpu * cp` holds one whole `max_context_len`
trajectory. It converts the checkpoint once and passes these rollout options to
`train.py`, with the RAFO options listed under [Flags](#flags):

```text
--rollout-function-path miles.rollout.sprout.rollout_fn.SproutRolloutFn
--sprout-rollout-base-url http://SPROUT_HOST:11001
--sprout-rollout-model-endpoint http://THIS_HOST:18081   # how Sprout's agents reach the router
--sprout-rollout-max-turns 64                            # model calls per trajectory
--n-samples-per-prompt 4                                 # RAFO's M roots of each task
--rollout-top-p 1.0 --rollout-top-k -1                   # the full softmax the trainer scores
--loss-mask-type qwen3
--sglang-reasoning-parser qwen3 --sglang-tool-call-parser qwen3_coder
```

A group that fails is logged and left out of the step, counted in
`rollout/sprout/groups_failed`: refused by the driver, failed at Sprout, or
returned in a shape Miles cannot train, which for a group without a search
includes one Sprout could not fill completely. It is never trained as a smaller
group. The step fails only when every group failed or none returned a sample to
train. A trajectory cut off by `--sprout-rollout-max-turns` or the wall-time
budget is still graded. It arrives with `status="truncated"` and its
`stop_reason`, and trains with its real reward. Each group waits
`--sprout-rollout-timeout-seconds` for execution plus
`--sprout-rollout-finalization-timeout-seconds` for the final snapshots and
grading.

## RAFO search

The model repairs well once it sees what went wrong, and samples worse
first-shot. `--sprout-rollout-rafo` turns each task's rollout into RAFO's forest
(reviewer-guided forest RL): where every attempt failed, a reviewer exposes the
problem with one turn in the agent's own voice, the agent continues on its own,
and the policy learns both its continuations and the turn.

### What Sprout samples

A RAFO search (Sprout's `RafoSearchSpec`, `{"mode": "rafo", ...}` on the wire)
runs in four steps:

1. Roots. The task's `--n-samples-per-prompt` roots (M) run from scratch and
   are graded.
2. Gate. Only a task none of whose graded roots resolved (`FF` or `FF_partial`)
   is reviewed. Any other task ends at its roots.
3. Events. The review reads each failed root, at most S of them, and picks one
   point in it: the state right after the first observation that went wrong.
   There it writes the agent's next turn: what to check and why, and the bash
   commands of the check. The turn never contains the fix, and never edits the
   repository. K continuations restored at the point run after the turn and its
   real output. A turn and its continuations are one event.
4. Depth. When every graded continuation of an event failed, the next review
   reads the first of them and writes one more turn in its history, at most D
   deep and S events in all.

The reviewer is the policy itself, under Sprout's RAFO prompts
(`resources/review/rafo_prompts.yaml`), and it may abstain. Each continuation
comes back with its `group` (`event:<n>`) and with `metadata.search`: its
`event`, `depth` and the `parent_job_id` that the event's turn went into (a
root, or a continuation one depth up). Its restored prefix and every inserted
turn are marked as not the policy's output. Miles recomputes the roots' label
and refuses a group whose annotations say otherwise, or whose forest is larger
than its search (`miles/rollout/sprout/forest.py`).

A trajectory is missing when its actor failed (Sprout names its slot in
`failed_samples`) or its grade gave no verdict (`metadata.zero_reason`, such as
`grade_failed`). A missing trajectory is neither resolved nor failed: it trains
nothing and enters no group below. A patch that does not apply, or tests that
run out of time, is a verdict: 0.

### What each sample trains

`rafo.assemble_rafo_group` gives each sample its role
(`train_metadata["rafo"]`), which the RAFO loss reads:

| Sample | Loss | Advantage | Coefficient |
|---|---|---|---|
| root | PPO with dual clip | local RLOO among the task's gradable roots | 1/(M·C) |
| continuation | PPO with dual clip | local RLOO among its event's gradable continuations | 1/(S·K·C) |
| reviewer turn | CE on the turn alone (its reasoning, its calls, its end of turn), after its history | none | w/(S·C_h) |

M, S and K are constants of the search, so no coefficient depends on what was
sampled; C and C_h are `--sprout-rollout-rafo-c` and `--sprout-rollout-rafo-c-h`.
A turn's weight w is the mean, over its event's gradable continuations, of R,
the 0/1 verdict. Under `--sprout-rollout-rafo-turn-credit subtree`, a
continuation with an event inside it counts max(R, w of that event), so a turn
whose continuation a deeper review rescued gets credit for it; with `own`, only
R counts. A continuation's restored prefix and inserted turns are context only:
each turn trains once, in its own event's CE sample.

These samples train nothing:

- a group whose members all have the same result, or that has only one member;
- a turn with w = 0.

A task with no signal keeps one zero-weight root, so that it still counts in
its step. All samples of a task share its rollout id. Miles trains them in one
step and divides the step's loss by its number of tasks. So
`--global-batch-size` counts tasks: the launcher uses half the rollout batch,
which gives two steps. `--allow-partial-train-step` keeps a short last step when
a group failed.

### Hidden tests

The reviewer reads the grading verdict, so its turn could name what the policy
could not have known at that point. Sprout checks each turn against the task's
hidden tests (`leak_terms`, and the leak judge when one is configured) and makes
no event of a turn that names them. A turn that nothing could check, because the
grade gave no hidden-test facts (Sprout's `harbor` grader, for example), still
makes an event; `rollout/sprout/rafo/events_unchecked` counts them.

### Canonical rendering

Sprout inserts the reviewer's turn as the assistant's reasoning, not as its
reply:
`{"role": "assistant", "content": "", "reasoning_content": "<the reviewer's text>", "tool_calls": [...]}`
(`assistant_turn_rendering: "reasoning"`). The CE sample then trains that turn
as the policy produces one, its thinking and then its calls; Miles refuses a
group whose turn is not in that form. On its first step, before it submits
anything, a RAFO search renders a probe turn of that form and refuses a chat
template and `--loss-mask-type` under which the probe's reasoning is not trained
(`importer.check_reasoning_is_trained`). With Qwen3.8's template, `qwen3` passes
and `qwen` is refused.

### Flags

The launcher passes:

```text
--sprout-rollout-rafo
--n-samples-per-prompt 4                          # M: roots per task
--sprout-rollout-rafo-continuations 2             # K: continuations per event
--sprout-rollout-rafo-depth 3                     # D: reviews, each inside the last
--sprout-rollout-rafo-events 4                    # S: events per task
--sprout-rollout-rafo-review-seconds 900          # each depth's review wall time
--sprout-rollout-rafo-wall-time-seconds 900       # each depth's continuations wall time
--sprout-rollout-rafo-c 64.0 --sprout-rollout-rafo-c-h 32.0
--sprout-rollout-rafo-turn-credit subtree
--loss-type custom_loss
--custom-loss-function-path miles.backends.training_utils.loss.hub.rafo_loss.rafo_loss_function
--advantage-estimator grpo --disable-rewards-normalization
--entropy-coef 0.0 --eps-clip 0.2 --eps-clip-high 0.2 --eps-clip-c 3.0
--global-batch-size 8 --allow-partial-train-step
```

Miles's own defaults for both wall times are 3600 s. A request carries a slot
for every continuation the search may make, `M + S·K` (12 with the launcher's
values); slots left unfilled are not failures. Each depth's review and
continuations, and the grades between depths, are added to what Miles waits for
a group: min(D, S)·(review + wall time) + (min(D, S) − 1)·finalization time. On
the Sprout side the driver config needs the `review` section `prepare_data.py`
writes (see [Prepare the data](#prepare-the-data)), and the driver, the Run
Store and the workers need a Sprout build with the RAFO search.

Construction refuses a RAFO search without each of these:

- `--loss-type custom_loss --custom-loss-function-path miles.backends.training_utils.loss.hub.rafo_loss.rafo_loss_function`;
- `--train-backend megatron`, which passes `train_metadata` to the loss and
  keeps a task's samples in one step;
- `--disable-rewards-normalization`;
- `--n-samples-per-prompt` and `--sprout-rollout-rafo-continuations` of 2 or
  more: each is a local RLOO group.

It also refuses `--custom-reward-post-process-path` and
`--use-dynamic-global-batch-size`, and the RAFO loss without
`--sprout-rollout-rafo`. The loss refuses these:

- `--calculate-per-token-loss`;
- KL and entropy terms;
- TIS and OPSM;
- rollout log-probs as π_old.

### Metrics

Each step sums these values over its tasks, under `rollout/sprout/rafo/`:

- `cases/<outcome>`: how many tasks had each label of their roots;
- `events`, with `nested_events` (deeper than 1), `events_trained` (a CE sample
  made), `events_without_gain` (w = 0) and `events_unchecked`;
- `ppo_samples`, `placeholders`, `estimation_only` (gradable trajectories with
  nothing to train) and `ungradable`.

`rollout/sprout/failed_samples` counts the slots whose actor failed, and
`rollout/sprout/reward_mean` is taken over the gradable trajectories. The loss
logs `rafo_*` sums for each task: the PPO and CE losses, the PPO and CE tokens,
the clipped tokens and the k3 KL.

## Without a GPU

Everything up to the optimizer step runs on a CPU host. `rollout_only.py` drives
`SproutRolloutFn` once against any Chat Completions endpoint that serves the
policy. It imports the trajectories as the trainer would. A plain group gets
Miles's GRPO reward normalization; with `--sprout-rollout-rafo`, each sample
gets its RAFO role, and the script sets the loss flags the launcher passes. It
defaults to the launcher's `--loss-mask-type qwen3`, prints one line per sample
with its role, reward, advantage and coefficient, and then saves the samples in
the format `--load-debug-rollout-data` reads:

```bash
python examples/swe-rebench-sprout/rollout_only.py \
  --prompt-data /root/swe-inputs/miles.jsonl --hf-checkpoint Qwen/Qwen3.8-27B \
  --model-name Qwen/Qwen3.8-27B \
  --sprout-rollout-base-url http://SPROUT_HOST:11001 \
  --sprout-rollout-model-endpoint http://MODEL_HOST:PORT \
  --rollout-batch-size 1 --n-samples-per-prompt 4 --sprout-rollout-rafo \
  --save-debug-rollout-data /root/swe-rollouts/rollout_data/{rollout_id}.pt
```

Only `hf_checkpoint`'s tokenizer, chat template and processor are needed on the
CPU side; a Qwen3.8 checkpoint's processor needs `torchvision` installed.

`run_qwen3_8_27b.py --load-rollout-data <file>` trains on a saved rollout
without SGLang or Sprout. Training on a file that `rollout_only.py` saved has not
been tried.

The unit tests (`tests/fast/rollout/sprout/`) run on CPU too. They check the
request Miles builds against the example Sprout ships, the importer against
Sprout's shipped result example, and the rollout function against a fake driver
over `httpx.MockTransport`. They check RAFO's credit on synthetic forests and on
Sprout's shipped RAFO example, which they also take through Miles's train-data
conversion, and they feed plain groups through Miles's own GRPO reward
normalization and advantage computation. `tests/fast/backends/training_utils/loss/`
checks the RAFO loss against a per-token reference.
