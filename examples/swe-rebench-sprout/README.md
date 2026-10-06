# SWE-rebench with Sprout rollouts

[Sprout](https://github.com/dreamyang-liu/sprout) runs the agent, snapshots
every step and grades the final state; Miles trains on what comes back. Each
prompt group is one `POST /rollout-groups` to Sprout's RL driver. Sprout runs
`n_samples_per_prompt` independent mini-swe-agent rollouts of the task against
the policy endpoint. It grades each final snapshot with the official
SWE-rebench procedure in a fresh sandbox, and returns every trajectory as Chat
Completions messages with a 0/1 reward. Trained as they come back, those
rollouts are standard GRPO; the launcher below runs a hindsight search instead,
credited by GSML (see [Hindsight search](#hindsight-search)).

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
the sampling process is imported: Sprout returns no tokens or log-probs. A branch
continuation's hint is wrapped in `<sprout_training_hint>` and removed before
export, so the hint-conditioned probabilities would not describe the trained
sequence anyway. For that reason `--use-rollout-logprobs`,
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
roots. A larger budget makes each call longer, and an `FF` task's review makes
three calls in a row (the analysis, the point review and a repair turn), all
within `--sprout-rollout-search-review-seconds`. At about three characters per
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
`train.py`, with the search options listed under [Flags](#flags):

```text
--rollout-function-path miles.rollout.sprout.rollout_fn.SproutRolloutFn
--sprout-rollout-base-url http://SPROUT_HOST:11001
--sprout-rollout-model-endpoint http://THIS_HOST:18081   # how Sprout's agents reach the router
--sprout-rollout-max-turns 64                            # model calls per trajectory
--n-samples-per-prompt 2                                 # the two roots of each task
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

## Hindsight search

The model repairs well once it has seen how an attempt failed, and samples
worse first-shot. `--sprout-rollout-search-points N` turns each task's rollout
into a small search whose purpose is to move that repair ability into the
one-shot policy. It samples the policy again from inside its own failed
attempts, on its own and after a reviewer's repair turn, and GSML
(`miles.rollout.sprout.gsml`) decides what each of those samples is credited
with: the repair turn earns credit only where the policy on its own did not
recover.

### What Sprout samples

A search has two roots, the task's `n_samples_per_prompt`; construction refuses
any other number. Once the roots are graded, the task's outcome decides what
Sprout runs next:

| outcome | roots | what Sprout runs next |
|---|---|---|
| `TT` | both graded, both resolved | nothing |
| `TF` | one resolved, one failed | a review of the failed root picks up to `N` points in it; `--sprout-rollout-search-tf-student-continuations` students run from each |
| `FF` | both graded, neither resolved | a review of each root picks up to `N` points in it and writes `--sprout-rollout-search-candidates` repair turns per point; from each point run `--sprout-rollout-search-student-continuations` students and `--sprout-rollout-search-candidate-continuations` continuations after each repair turn, for up to `--sprout-rollout-search-repair-rounds` rounds |
| `FF_partial` | one failed, the other missing | as for `TF` |
| `TT_partial` | one resolved, the other missing | nothing |
| `ungraded` | neither graded | nothing |

With `--sprout-rollout-search-tf-student-continuations 0`, `TF` and `FF_partial`
tasks are not branched either, and get no review. Miles recomputes the outcome
from the roots it gets back and refuses a group whose annotations
(`metadata.search.outcome`) say otherwise.

A student continues from the point's restored state with nothing added: the
policy's own next turn there. A repair continuation first runs the reviewer's
turn, inserted into the history and executed, so its real tool results follow
it. Every branch is limited to the turns its root had left at that point, and
comes back with its `group` (`student:<point>` or `repair:<point>:<candidate>`)
and its restored prefix and inserted turn marked as not the policy's output.
The reviewer is the policy itself under SPROUT's analyst and reviewer prompts:
it reads each failed root with its grading verdict, and identical repair turns
merge, keeping their multiplicity.

A trajectory is missing when its actor failed (Sprout names its slot in
`failed_samples`) or its grade gave no verdict (`metadata.zero_reason`, such as
`grade_failed`). A missing trajectory is neither resolved nor failed: it trains
nothing and enters no count or z-score below. A patch that does not apply, or
tests that run out of time, is a verdict: 0.

### What each sample is credited with

Every sample gets one advantage `A`, carried by each of its trainable tokens.
`gsml.assemble_search_group` annotates the sample with its credit
(`train_metadata["sprout_rollout"]["credit"]`), and
`rewards.post_process_gsml` computes
`A = β·z + λ·(ψ for an â sample, else 1)·share` when the step is trained, so a
saved rollout can be re-credited under other GSML flags.

| sample | trained tokens | `A` |
|---|---|---|
| root | its turns | `β_root·z`, with `β_root = (1 − λ)/√2` |
| student, group `student:<point>` | its turns after the restored prefix | `β_br·z`; a resolved student of an `FF` task adds `λ·ω_s/c_s` |
| repair continuation of candidate `m`, group `repair:<point>:<m>` | its turns after the prefix and the inserted turn | `β_br·z`, with `z` taken on `R′`; in an `FF` task, if `m` is in `E_s`, it adds `λ·R′·ω_s/(#E_s·K_m)` |
| â sample, one per candidate in `E_s` | the inserted turn alone (its reasoning and its calls), after the history it was inserted into | `ψ·λ·ω_s/#E_s`, never negative; reward 1 |

- `z` is the trajectory's z-score among the gradable trajectories of its group
  (the task's roots, `student:<point>`, `repair:<point>:<m>`):
  `(R − mean)/(std + 1e-6)` with the unbiased standard deviation, and 0 when
  fewer than two are gradable or all agree. A trajectory with nothing to train
  (every turn restored or inserted) still counts in its group and in `c_s` and
  `K_m`.
- `R` is the 0/1 verdict. `R′ = R`, except that a repair continuation whose patch
  changed a file of the task's test patch, or whose grade cannot say, has
  `R′ = 0` (below).
- `c_s` counts the students that resolved at point `s`, and `K_m` sums `R′` over
  candidate `m`'s gradable continuations.
- `E_s`, of size `#E_s`, holds the eligible candidates at `s`, and is empty
  unless a student was graded at `s` and none resolved. A candidate is eligible
  when `K_m ≥ 1`, its turn passes the leak gate and the inserted turn is
  canonical.
- A point is creditable when a student was graded there and either one resolved
  or `E_s` is not empty. Each creditable point of the task gets
  `ω_s = 1/(the task's creditable points)`; every other point gets 0.

Only an `FF` task carries λ credit. In `TT`, `TF` and the partial cases every
`A` is `β·z` alone. At a point where a student resolved, the credit goes to
those students: the policy's own success from that state, an exact signal.
Only where none did does the reviewer's turn earn credit, through its resolved
continuations and its â sample; that signal is biased, since the policy did
not sample the turn. The restored prefix only ever serves as context.
`--sprout-rollout-gsml-c-pre` (credit for the prefix) and
`--sprout-rollout-gsml-kappa-plus` (guided credit when a root resolved) are not
implemented; anything but 0 is refused. Samples whose `A` is 0 stay in the
batch.

At λ = 0.4 and β_br = ψ = 1, a `TF` task's roots (1, 0) get ±0.30 and its
students (1, 0) ±0.7071. In an `FF` task with two points, both roots get 0, and:

- At P1 the students resolve (1, 0): 0.9071 and −0.7071. Candidate 1's
  continuations (1, 0) get ±0.7071; candidate 2's (0, 0) get 0.
- At P2 the students fail (0, 0): 0. Candidate 1's continuations resolve (1, 1)
  and its turn names nothing hidden: 0.10 each, and its â sample 0.20.
  Candidate 2's turn names `tests/test_hidden.py`: its continuations (1, 0) get
  ±0.7071, and it has no â sample.
- Both points are creditable, so ω = ½ at each.

### The leak gate and R′

The reviewer reads the grading verdict, so its turn can name what the policy
could not have known at that point. For every repair, Sprout lists in
`metadata.search.leak_terms` the hidden terms its inserted turn names that the
history up to the point did not show; a term counts as named when it occurs in
the turn's text. Hidden terms are the paths in the task's test patch of at
least 6 characters, and their basenames of that length unless every repository
has one (`conftest.py`, `setup.py` and the like), and the task's FAIL_TO_PASS
test names of at least 8 characters, with their `[A-Za-z0-9_]+` parts of at
least 8 characters that are not all digits. PASS_TO_PASS names are not hidden.
A candidate passes the gate only when the list is empty for each of its
gradable continuations. A `null` list (the grade had neither list to check
against) fails the gate too. A candidate that fails the gate keeps its
continuations' `β_br·z` and gets no â sample.

Each grade also reports `touched_test_paths`, the files of the task's test patch
that the graded patch changed. A repair continuation that changed any resolved
nothing it can be credited for: `R′ = 0`, in its z-score and in `K_m`. So does
one whose grade cannot say (`touched_test_paths` not a list). Students and roots
keep their `R`.

### Canonical rendering

Sprout inserts the reviewer's turn as the assistant's reasoning, not as its
reply:
`{"role": "assistant", "content": "", "reasoning_content": "<the reviewer's text>", "tool_calls": [...]}`
(`assistant_turn_rendering: "reasoning"`). The â sample then trains that turn
as the policy produces one, its thinking and then its calls. A candidate whose
inserted turn has blank reasoning, or content of more than one line, is not
eligible. On its first step, before it submits anything, a search renders a
probe turn of that form and refuses a chat template and `--loss-mask-type`
under which the probe's reasoning is not trained
(`gsml.check_reasoning_is_trained`). With Qwen3.8's template, `qwen3` passes and
`qwen` is refused.

### The loss

The advantage is final. `--advantage-estimator grpo` broadcasts it over the
sample's trainable tokens. `--normalize-advantages` is refused: whitening `A`
across the batch would move every zero and rescale the λ shares.

`--calculate-per-token-loss` makes the loss a sum over the step's trainable
tokens, each weighted by its sample's `A`, so every token weighs alike whatever
its sample's length. The sum is divided by the step's trainable-token count.
That count varies from step to step, and samples whose `A` is 0 (`TT` roots,
groups that agree) add to it, so the update is consistent, not exact. An exact
update needs a constant divisor, a CP-aware reducer that is not built yet. With
context parallelism (the launcher's CP = 4), Miles counts each sample's tokens
on every CP rank and scales the loss by the CP size, and Megatron reduces the
count over DP × CP, so the two cancel.

How many samples a step returns varies, so a search needs
`--use-dynamic-global-batch-size`: one optimizer step over every sample, rounded
down to a multiple of the data-parallel size (1 in the launcher). A fixed
`--global-batch-size` would trim the step to a multiple of itself.

### Flags

The launcher passes:

```text
--sprout-rollout-search-points 1                    # N: points per failed root
--sprout-rollout-search-candidates 1                # FF: repair turns per point and round
--sprout-rollout-search-student-continuations 1     # FF: students per point
--sprout-rollout-search-candidate-continuations 1   # FF: continuations per repair turn
--sprout-rollout-search-tf-student-continuations 2  # TF, FF_partial: students per point
--sprout-rollout-search-repair-rounds 3             # FF: rounds of repair, each after the last's failure
--sprout-rollout-search-review-seconds 900          # each round's review wall time
--sprout-rollout-search-wall-time-seconds 900       # each round's branch phase wall time
--custom-reward-post-process-path miles.rollout.sprout.rewards.post_process_gsml
--sprout-rollout-gsml-lambda 0.4                    # λ, in [0, 1]
--sprout-rollout-gsml-beta-branch 1.0               # β_br, at least 0
--sprout-rollout-gsml-psi 1.0                       # ψ, at least 0
--sprout-rollout-gsml-c-pre 0.0                     # not implemented: must be 0
--sprout-rollout-gsml-kappa-plus 0.0                # not implemented: must be 0
--advantage-estimator grpo --calculate-per-token-loss --use-dynamic-global-batch-size
```

Miles's own defaults differ for `--sprout-rollout-search-candidates` (4), the
student and candidate continuations (2), `--sprout-rollout-search-repair-rounds`
(1) and both wall times (3600 s). More than one repair round needs one candidate
and one candidate continuation: a round reviews the previous round's failed
continuation on its own, picks one point in its history and inserts one more
turn there, and runs only where neither the point's student nor an earlier
round's repair resolved. A chain that resolves trains every round's turn as an
â sample, the candidate's share split among them. Construction refuses a search without
`--n-samples-per-prompt 2`, that post-process path, `--advantage-estimator grpo`,
`--calculate-per-token-loss` or `--use-dynamic-global-batch-size`, or with
`--normalize-advantages`.

A request carries a slot for every branch of the outcome that branches most:
`roots + N·max(roots·(students + candidates·continuations·rounds), (roots − 1)·tf_students)`,
which is 10 with the launcher's values (2 + max(8, 2)). Slots left unfilled are
not failures. `--sprout-rollout-search-review-seconds` and
`--sprout-rollout-search-wall-time-seconds` are added to what Miles waits for a
group. On the Sprout side the driver config needs the `review` section
`prepare_data.py` writes (see [Prepare the data](#prepare-the-data)).

### Metrics

Each step sums, over its tasks, under `rollout/sprout/gsml/`:

- `cases/<outcome>`: how many tasks had each outcome.
- For `FF` tasks, `points`, split into `points_exact`, `points_guided` and
  `points_no_mass` by how they were credited.
- For `FF` tasks, `candidates`, with the gates they passed or failed:
  `candidates_eligible`, `candidates_leaked`, `candidates_unverified` (a `null`
  leak list), `candidates_noncanonical` and `candidates_unconfirmed` (`K_m` = 0).
- `ahat`, the â samples made, and `ahat_untrainable`, the eligible candidates
  whose turn rendered nothing to train.
- `continuations_touching_tests`, the repair continuations whose `R′` is 0
  because they changed the task's test files or their grade cannot say.
- `estimation_only`, the gradable trajectories with nothing to train.
- `lambda_tasks`, the tasks that carried λ credit.

`rollout/sprout/ungradable` and `rollout/sprout/failed_samples` count the
missing trajectories, and `rollout/sprout/reward_mean` is taken over the
gradable ones.

## Without a GPU

Everything up to the optimizer step runs on a CPU host. `rollout_only.py` drives
`SproutRolloutFn` once against any Chat Completions endpoint that serves the
policy. It imports the trajectories as the trainer would and gives them the
advantages `train.py` would: Miles's GRPO reward normalization, or, with
`--sprout-rollout-search-points`, GSML's credit. For a search it sets the
post-process path itself, and it defaults to the launcher's
`--calculate-per-token-loss`, `--use-dynamic-global-batch-size` and
`--loss-mask-type qwen3`. It prints one line per sample, with its role, its
task's outcome, `z`, λ share and `A`. It then saves the samples in the format
`--load-debug-rollout-data` reads:

```bash
python examples/swe-rebench-sprout/rollout_only.py \
  --prompt-data /root/swe-inputs/miles.jsonl --hf-checkpoint Qwen/Qwen3.8-27B \
  --model-name Qwen/Qwen3.8-27B \
  --sprout-rollout-base-url http://SPROUT_HOST:11001 \
  --sprout-rollout-model-endpoint http://MODEL_HOST:PORT \
  --rollout-batch-size 1 --n-samples-per-prompt 2 \
  --sprout-rollout-search-points 1 --sprout-rollout-search-candidates 2 \
  --save-debug-rollout-data /root/swe-rollouts/rollout_data/{rollout_id}.pt
```

Only `hf_checkpoint`'s tokenizer, chat template and processor are needed on the
CPU side; a Qwen3.8 checkpoint's processor needs `torchvision` installed.

`run_qwen3_8_27b.py --load-rollout-data <file>` trains on a saved rollout
without SGLang or Sprout, but not yet on one `rollout_only.py` saved. The
launcher passes `--use-dynamic-global-batch-size`, and Miles then requires the
rollout's `dynamic_global_batch_size` metadata. Miles records it only when it
trims a step to the trainer's data-parallel size, which `rollout_only.py` does
not do, having no trainer. `convert_samples_to_train_data` refuses such a file
with an assertion.

The unit tests (`tests/fast/rollout/sprout/`) run on CPU too. They check the
request Miles builds against the example Sprout ships, the importer against
Sprout's shipped result example, and the rollout function against a fake driver
over `httpx.MockTransport`. They check GSML's credit on synthetic search groups,
the examples above among them. They also feed imported samples through Miles's
own GRPO reward normalization and advantage computation, and check that the
token-sum loss weighs each sample by its trainable tokens.
