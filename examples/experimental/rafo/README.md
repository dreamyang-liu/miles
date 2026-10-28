# RAFO: reviewer-guided forest RL for agentic coding

RAFO trains a tool-using agent on long-horizon tasks. When every attempt of the agent at a task fails, a fixed
reviewer model reads a failed attempt and writes one assistant turn **in the agent's first person** at the
state right after the first failed check. The turn says what to check and why, and contains the tool calls that
run the check. It does not contain the fix. The environment runs the check, and the agent continues on its own.
The agent learns from its own attempts with PPO, and learns the reviewer's turn with cross-entropy, weighted by
how often the continuations after it succeed.

## One task = one forest = one rollout

`RafoGenerate` builds the whole forest of a task in one generate call:

1. **Roots.** M episodes of the agent. Each turn is sampled from SGLang. Its tool calls run in the environment
   with `step(calls, check_only=False)`. The first step that reports `failed_check` gives the episode's
   **checkpoint**: the token stream and transcript up to that tool output, and `env.fork()` as the snapshot.
2. **Gate.** Only a task whose M roots all fail goes to review; otherwise the root advantages already carry a
   signal.
3. **Event.** The reviewer reads the full failed root (`ReviewRequest`) and returns a `ReviewerTurn`, or `None`
   to abstain. The turn is rendered with the agent's chat template and appended at the checkpoint. Its check runs
   on a fork of the snapshot with `check_only=True`. K continuations then start from forks of that state.
4. **Nested review.** If all K continuations fail, the next depth reviews again. With `branch`, it reviews inside
   the first failed continuation, after its own first failed check, so that sequence carries one reviewer turn
   for each depth. With `checkpoint`, the new turn replaces the old one at the same checkpoint. Each event uses
   one of the task's S slots; depth stops at D.

All samples of the forest share the task's `rollout_id`. The Megatron backend therefore trains them in the same
optimizer step, and divides that step's loss by its number of rollouts, which is the number of tasks.

## What goes to the loss

| Sample | Tokens with loss mask 1 | Loss | `rafo_coef` |
|---|---|---|---|
| root | every token the root generated | PPO, local RLOO over the M roots | 1 / (M C) |
| repair | every token the continuation generated after the check's output | PPO, local RLOO over the K continuations | 1 / (S K C) |
| reflect | the reviewer turn (reasoning, tool calls, end-of-turn token) | CE | w / (S C_h) |

Prompts, tool outputs, reviewer turns inside PPO samples, and inherited prefixes have no loss. Each token is
trained once, by the episode that generated it. A reviewer turn is trained only in its own event's CE sample.

The weight w of a reviewer turn is the mean outcome of its continuations. With `subtree` credit, a continuation
that a deeper review rescued counts with that review's weight. RAFO does not ask whether a continuation followed
the turn, because that is not observable in long-horizon tasks.

`loss.py` computes PPO with the miles clip (`--eps-clip`, `--eps-clip-high`) and dual clip (`--eps-clip-c`) on
root and repair tokens, and CE on reflect tokens. It sums the tokens with these fixed coefficients and never
divides by a sampled count. PPO samples with A = 0 and reviewer turns with w = 0 have exactly zero gradient and
are not sent to the trainer. A task without any signal keeps one zero-weight sample, so the per-step rollout
count does not change. π_old comes from the trainer, which recomputes it with the same micro-batches as the
update.

## Plugging in a task

- **Environment** (`--rafo-env-path`): an async factory `make_env(sample) -> RafoEnv` (see `protocols.py`).
  - `step` returns one output for each tool call, `failed_check` (for example a failing test run), and `done`.
  - `fork` returns an independent copy of the current state: snapshots and continuations depend on it.
    Container snapshots work, and so does replaying the tool calls into a fresh sandbox.
  - `reward` scores the final state, and `close` releases the environment.
  - With `check_only`, the environment must refuse edits.
- **Reviewer** (`--rafo-reviewer-path`): an async `review(ReviewRequest) -> ReviewerTurn | None`. The request
  holds the task sample, the full failed transcript, the checkpoint position, the depth and the tool specs.
  - Put the evidence and the reasoning that lead to the check in `reasoning`. In the reference experiments, turns
    that only ran the check taught the form of a check but not what to check.
  - Thinking-model templates render `reasoning` in their reasoning block. Other templates put it before the
    content.
  - `toy_env.toy_reviewer` is an oracle for the toy task; for real tasks, wrap the model or API of your choice.
- **Tools** (`--rafo-tool-specs-path`, `--rafo-tool-call-parser`): the agent's tool specs and the SGLang
  function-call parser for its output. Do not pass `--apply-chat-template`: RAFO renders the chat prompt with
  these tools.

All flags, their defaults and their checks are in `config.py`.

## Required miles settings

- `--train-backend megatron`: only Megatron keeps all samples of a rollout in one step and forwards
  `Sample.train_metadata` to the loss.
- `--n-samples-per-prompt 1`: the forest makes its own M roots.
- `--global-batch-size`: counts tasks per optimizer step.
- `--disable-rewards-normalization`: the samples of one task have different rewards.
- `--loss-type custom_loss --custom-loss-function-path examples.experimental.rafo.loss.rafo_loss_function`.
- Keep the default advantage computation. RAFO does not use its advantages, but that pass computes π_old.

The generate function and the loss reject what they would otherwise silently ignore:

- partial rollouts and group reward models;
- routing, indexer or sampling-support replay, the output store, and top-k log-probs;
- KL or entropy terms, TIS, OPSM, and a custom pg reducer;
- rollout log-probs as π_old, and the per-token loss normalization.

The reference recipe used B = 16 tasks for each iteration with two optimizer steps, and coefficients normalized
by B. In miles, each step is normalized by its own number of tasks (B/2). Adam does not see this factor of 2;
the gradient-norm clip does.

## Monitor values

`--custom-rollout-log-function-path examples.experimental.rafo.metrics.log_rollout_data` adds `rafo/*` to the
default rollout metrics:

- root success: the agent without review, also logged as miles' episode reward;
- reviewed-task fraction, events and reviewer reads for each task, and abstention rate;
- continuation success;
- samples for each task, by kind.

The loss logs `rafo_*` sums for each task: PPO and CE losses, PPO and CE tokens, clipped tokens, and the k3 KL.

## Smoke recipe

```bash
python examples/experimental/rafo/run_qwen3_0_6b_rafo.py \
    --model-dir /models --data-dir /datasets --output-dir /output
```

The recipe trains Qwen3-0.6B on the toy bug-finding task in `toy_env.py`. A patch of the wrong file fails the
tests and shows the failing line; `inspect` is the check. The default reviewer is the toy oracle reviewer.

## Status

The CPU tests in `tests/fast/examples/experimental/rafo/` cover these parts:

- the forest, the samples and their token ownership;
- the loss against a per-token reference, including 2-rank context parallelism;
- miles' argument parsing and a mock SGLang server;
- the train-data contract.

The algorithm was validated outside miles (BugLocator tasks with Qwen3-0.6B). This integration has not yet run end
to end on GPUs.
