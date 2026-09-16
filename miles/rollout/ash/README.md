# Ash rollout backend

## Hint-free message training (v3)

For Ash-generated branches whose hints are removed before training, select
`miles.rollout.ash.message_rollout.AshMessageRolloutFn`. The additive
`ash-rollout-v3` contract sends `task_id`, `image`, prompt, slots and native
sampling controls; Ash returns cleaned messages, tool schemas, branch provenance
and reward. Miles constructs fresh tokens/loss masks and uses the trainer's
normal logprob forward pass. It does not reuse rollout token positions or
hint-conditioned logprobs.

Set `--ash-rollout-max-turns N` (default 64) to limit each trajectory to N
model invocations. Tools requested by those invocations do not consume turns
and have no separate call-count cap. The limit is independent of prompt-group
size: eight samples each receive N turns. The v3 wire request contains
`"max_turns": N` and `"budgets": {"max_wall_time_seconds": ...}`.
`--ash-rollout-max-model-calls` and `--ash-rollout-max-tool-calls` are v2-only;
v3 rejects them and the corresponding old wire fields.
V2 retains its 100/100 CLI defaults and explicit `unbounded`/JSON `null`
semantics. An Ash v2 deployment must also support nullable call budgets before
using those values; v3 uses its own `max_turns` contract instead.
The group execution deadline, actor timeout, and output-token limit remain
independent. On an execution timeout or turn cutoff, Ash captures the final
quiescent sandbox, grades it, and returns `status="truncated"` with
`stop_reason="timeout"` or `"max_turns_reached"`. Miles preserves this reason
in the sample's `ash_rollout` metadata and recomputes logprobs as usual.
`--ash-rollout-finalization-timeout-seconds` (default 1800) gives snapshot
export and grading additional time after the execution deadline. Uncertain
tool execution and missing/invalid training histories remain explicit errors.

The existing v2 interface below remains available for exact-token trajectories.
See `examples/swe-rebench-ash/README.md` for data preparation, the matching Ash
configuration, supported sampling controls and training options. V3 does not
require a Session Server token-recording bridge or a Miles-owned environment
catalog. Optional review-guided branching is described below; it does not change
shared-prefix credit assignment.

### Review-guided branching

With a compatible Ash driver and configured reviewer, set
`--ash-rollout-branching --n-samples-per-prompt 2`. Ash runs and grades one root
per prompt, then reviews failed roots to seek a resolved child and successful
roots to seek an unresolved child. Its default limit is two rounds with one
branch per round, stopping on the first opposite signal.

Ash returns the root and selected child as a fixed pair. Intermediate attempts
remain in Ash's execution records. If both children have the root's reward, the
last child is selected. Missing recovery points or invalid execution can leave
an incomplete pair; Miles rejects that shortfall rather than padding samples.
Prompts progress independently.

Injected branch hints are removed before export. Miles rebuilds tokens/masks
and recomputes logprobs as for ordinary v3 samples; this does not introduce
tree-specific weighting. The flag defaults off and is omitted from the wire
when disabled, preserving compatibility with existing v3 drivers.

## Exact-token trajectories (v2)

The Ash rollout backend delegates one complete prompt group to an external
agent rollout service while Miles remains responsible for training. It is
intended for rollout strategies that need to grow branches, checkpoint and
restore environments, or otherwise coordinate a group of related agent
trajectories outside the standard per-sample generation loop.

## Ownership boundary

Miles owns:

- prompt-group and sample-slot allocation;
- the task identity and immutable logical environment reference copied from
  sample metadata;
- the SGLang model and Session Server endpoints exposed to the rollout service;
- the expected policy weight version and sampling configuration;
- validation and conversion of returned trajectories into `Sample` objects;
- reward calculation when the rollout service does not provide a reward; and
- advantage calculation, loss construction, optimization, and weight updates.

The Ash-compatible service owns:

- the agent loop and tool execution;
- environment creation, checkpointing, restoration, and cleanup;
- branch selection and search scheduling within the supplied budgets; and
- assembly of complete token-aligned trajectories for the allocated sample
  slots.

`ash-runtime` is below this boundary. It is the small tool server running
inside each Ash sandbox and provides `shell`, process control, file editing,
search/fetch, artifact download, and event waiting. Ash and AgentENV arrange
for it to be present and running; Miles communicates only with the rollout
service and its own model/Session Server endpoints.

## Request lifecycle

`AshRolloutFn` implements the class-based Miles rollout interface. For each
prompt group it:

1. allocates the sample slots already reserved by Miles;
2. submits an `AshRolloutRequest` with `POST /rollout-groups`;
3. polls `GET /rollout-groups/{rollout_job_id}` until the job is terminal;
4. validates group identity, slot ownership, policy version, messages, and
   token boundaries;
5. imports each returned trajectory into a Miles `Sample`; and
6. sends `DELETE /rollout-groups/{rollout_job_id}` after consuming the result,
   or when local processing fails or times out, so Ash can cancel unfinished
   work and release the complete trajectory record.

For SessionTree-backed agent calls, cancellation continues past the Ash job:
Ash deletes the active Miles session, Miles cancels that session's in-flight
upstream HTTP task, and the closed connection lets SGLang abort the request in
its scheduler. Keeping this translation in Miles avoids coupling Ash to an
SGLang-specific abort API or to a particular router topology.

`POST` is idempotent. A retry after a lost response may observe `queued`,
`running`, or a terminal status if Ash already finished the job; trajectories
are still read from `GET`, not from the POST acknowledgement.

The service receives a model endpoint for ordinary generation. When a Miles
Session Server endpoint is supplied, the service can use it for multi-turn
generation with exact token and trajectory tracking.

Every sample must provide the same environment identity within its prompt
group:

```json
{
  "metadata": {
    "task_id": "swebench__repo-123",
    "environment_ref": {
      "kind": "image",
      "id": "docker.io/library/ubuntu",
      "revision": "sha256:224a1869083a311ef3f13648a154ba79832fbef6364d31493642ca03082da254",
      "resource_profile": "standard"
    }
  }
}
```

Miles never sends a provider endpoint or credential. For `kind: template` and
`kind: snapshot`, Ash resolves the exact four-field identity through its
deployment-owned catalog. An Ash deployment may also opt into resolving
`kind: image`: `id` is then an OCI repository, `revision` must be its immutable
`sha256` digest, and `resource_profile` selects an Ash-approved resource class.
Ash prepares and caches a runtime-ready AgentENV snapshot before creating the
sandbox; Miles does not need a separate environment-resolution request.

`AshRolloutClient.list_environments()` calls `GET /rollout-environments` to
discover deployment-approved static catalog identities that may be
copied into sample metadata. Its response is:

```json
{
  "protocol_version": "ash-rollout-v2",
  "environments": [
    {
      "kind": "template",
      "id": "swebench-runtime",
      "revision": "sha256:immutable-revision",
      "resource_profile": "standard"
    }
  ]
}
```

It intentionally omits backend `spawn_ref` values and credentials. Dynamic OCI
images do not need one catalog entry per digest; the repository, immutable
digest, and resource profile are checked against Ash deployment policy when
the group is submitted. Miles never guesses an AgentENV template or snapshot
ID.

`ash-rollout-v2` is a strict contract on both sides: unknown fields are
rejected instead of ignored. Any wire-format extension therefore requires a
coordinated protocol update rather than a one-sided optional field.

## Wire request and response

The request sent by `AshRolloutFn` contains one prompt group, the slots already
reserved by Miles, the expected policy version, sampling controls, work
budgets, and the trusted environment identity:

```json
{
  "protocol_version": "ash-rollout-v2",
  "rollout_job_id": "miles-17-group-42-<unique>",
  "rollout_id": 17,
  "prompt_group_id": "group-42",
  "task_id": "swebench__repo-123",
  "environment_ref": {
    "kind": "image",
    "id": "docker.io/library/ubuntu",
    "revision": "sha256:224a1869083a311ef3f13648a154ba79832fbef6364d31493642ca03082da254",
    "resource_profile": "standard"
  },
  "sample_slots": [
    {"sample_slot_id": "...:slot:0", "sample_index": 0},
    {"sample_slot_id": "...:slot:1", "sample_index": 1}
  ],
  "max_samples": 2,
  "minimum_returned_samples": 2,
  "prompt": "<text or ordered messages>",
  "prompt_token_ids": [1, 2, 3],
  "model_endpoint": "http://model-router:port",
  "session_server_endpoint": "http://session-server:port",
  "model": "local-model",
  "expected_weight_version": "1",
  "return_rollout_logprobs": false,
  "sampling_params": {"temperature": 0.6, "max_new_tokens": 2048},
  "budgets": {
    "max_model_calls": null,
    "max_tool_calls": null,
    "max_wall_time_seconds": 10800
  }
}
```

`max_model_calls` and `max_tool_calls` may be `null`, meaning that Ash does
not terminate the episode by call count. This is still bounded by
`max_wall_time_seconds`, cancellation, the model context window, per-turn
generation limits, and sandbox lifecycle policy. Miles CLI accepts
`unbounded` for either call-limit flag and serializes it as JSON `null`.

The first `POST /rollout-groups` normally returns a `queued`
acknowledgement. An idempotent retry returns the job's current status, which
may already be terminal. Miles polls the job with
`GET /rollout-groups/{rollout_job_id}`. A completed response contains
`actual_samples`, `search_branches`, `consumed_budget` and one complete
trajectory per returned slot:

For a queued or running job, Ash may also return a `progress` object with the
current coarse phase, cumulative model/tool call counts, completed sample
count, active sample slot, elapsed and remaining wall time, and the Unix time
of the last activity update. Miles validates this optional telemetry but does
not use it to construct training samples or compute loss.

Cancelled and failed terminal results should retain the latest model/tool call
counts plus elapsed time in `consumed_budget`. This lets Miles and profiling
consumers distinguish a long right-censored rollout from a failure before any
useful work was performed.

```json
{
  "protocol_version": "ash-rollout-v2",
  "rollout_job_id": "miles-17-group-42-<unique>",
  "status": "queued"
}
```

```json
{
  "protocol_version": "ash-rollout-v2",
  "rollout_job_id": "miles-17-group-42-<unique>",
  "prompt_group_id": "group-42",
  "status": "completed",
  "max_samples": 2,
  "actual_samples": 2,
  "stop_reason": null,
  "search_branches": 1,
  "consumed_budget": {"model_calls": 3, "tool_calls": 1, "session_tree_leaves": 2},
  "trajectories": [
    {
      "sample_slot_id": "...:slot:0",
      "branch_id": "...:root:0",
      "parent_branch_id": null,
      "branch_point_token_count": null,
      "messages": [
        {"role": "user", "content": "Inspect the workspace."},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "call-1", "type": "function", "function": {"name": "shell", "arguments": "{\"command\":\"pwd\"}"}}]},
        {"role": "tool", "tool_call_id": "call-1", "content": "/workspace"},
        {"role": "assistant", "content": "Parent completed."}
      ],
      "token_ids": [1, 2, 3, 4, 5, 6],
      "prompt_length": 3,
      "generated_spans": [
        {
          "response_id": "response-tool",
          "start": 3,
          "end": 4,
          "input_token_ids": [1, 2, 3],
          "output_token_ids": [4],
          "output_token_log_probs": null,
          "weight_version": "1",
          "finish_reason": "tool_calls"
        },
        {
          "response_id": "response-parent",
          "start": 5,
          "end": 6,
          "input_token_ids": [1, 2, 3, 4, 5],
          "output_token_ids": [6],
          "output_token_log_probs": null,
          "weight_version": "1",
          "finish_reason": "stop"
        }
      ],
      "response_text": "Parent completed.",
      "reward": null,
      "status": "completed",
      "prompt_token_alignment": "request_exact",
      "metadata": {"environment_checkpoint_id": "opaque-to-Miles"}
    },
    {
      "sample_slot_id": "...:slot:1",
      "branch_id": "...:child:1",
      "parent_branch_id": "...:root:0",
      "branch_point_token_count": 5,
      "messages": [
        {"role": "user", "content": "Inspect the workspace."},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "call-1", "type": "function", "function": {"name": "shell", "arguments": "{\"command\":\"pwd\"}"}}]},
        {"role": "tool", "tool_call_id": "call-1", "content": "/workspace"},
        {"role": "assistant", "content": "Child completed."}
      ],
      "token_ids": [1, 2, 3, 4, 5, 7],
      "prompt_length": 3,
      "generated_spans": [
        {
          "response_id": "response-tool",
          "start": 3,
          "end": 4,
          "input_token_ids": [1, 2, 3],
          "output_token_ids": [4],
          "output_token_log_probs": null,
          "weight_version": "1",
          "finish_reason": "tool_calls"
        },
        {
          "response_id": "response-child",
          "start": 5,
          "end": 6,
          "input_token_ids": [1, 2, 3, 4, 5],
          "output_token_ids": [7],
          "output_token_log_probs": null,
          "weight_version": "1",
          "finish_reason": "stop"
        }
      ],
      "response_text": "Child completed.",
      "reward": null,
      "status": "completed",
      "prompt_token_alignment": "request_exact",
      "metadata": {"environment_checkpoint_id": "opaque-to-Miles"}
    }
  ]
}
```

The actual response includes the complete ordered message history (including
tool calls and tool results), all exact token IDs and one generated span per
model call. `output_token_log_probs` is present when
`return_rollout_logprobs=true`. Miles validates these fields and converts the
trajectory into a training `Sample`; it does not use the opaque environment
handles to recreate a sandbox. After import, it calls
`DELETE /rollout-groups/{rollout_job_id}` to cancel any unfinished work and
release Ash's retained result. A successful deletion response is:

```json
{
  "protocol_version": "ash-rollout-v2",
  "rollout_job_id": "miles-17-group-42-<unique>",
  "status": "completed"
}
```

## Trajectory contract

The wire models live in `protocol.py`. A returned trajectory must include:

- the complete ordered message history, including matched assistant tool calls
  and tool results;
- the exact token sequence and the prompt boundary;
- one `generated_span` for every assistant generation call;
- the input and output token IDs for each generated span;
- the policy weight version used by every generation call; and
- rollout log-probabilities when Miles requests them.

Only assistant-generated spans receive `loss_mask=1`. Prompt tokens, tool
results, and other environment-provided context remain part of the model input
but do not contribute directly to the policy loss.

`prompt_token_alignment` defaults to `request_exact`: Miles verifies that the
trajectory begins with the token IDs allocated by its data source. A trusted
external harness may instead return `harness_rendered` when it adds a system
prompt or runtime context before the task. In that mode the first exact token
prefix captured by the Miles Session Server is used for training, while the
request's original prompt token IDs remain in
`metadata.ash_rollout.request_prompt_token_ids` for provenance. This flag does
not permit Ash to synthesize token IDs; every token and generated span must
still come from the Session Server.

## Configuration

Select the backend and provide the rollout service URL:

```bash
--rollout-function-path miles.rollout.ash.rollout_fn.AshRolloutFn \
--ash-rollout-base-url http://ash-rollout-service:PORT
```

Miles derives the model and Session Server endpoints from the active rollout
deployment by default. They can be overridden when the Ash service must use
separately advertised addresses:

```bash
--ash-rollout-model-endpoint http://model-router:PORT \
--ash-rollout-session-server-endpoint http://session-server:PORT
```

The `--ash-rollout-max-model-calls`, `--ash-rollout-max-tool-calls`, and
`--ash-rollout-timeout-seconds` options bound the work allowed for one prompt
group. The wall-time default is 10,800 seconds so long-context agent turns and
the benchmark verifier can finish without a one-hour right-censoring bias.
`--ash-rollout-client-grace-seconds` only gives Ash time to publish the
cancelled terminal result after that server-side deadline; it does not extend
the rollout budget. HTTP request timeout and polling cadence are configured
separately.

## Current scope

The current protocol supports training rollout with a fixed number of
returned samples equal to `n_samples_per_prompt`. Evaluation, multimodal
prompts, and early return with fewer samples are not yet supported. Environment
selection is per prompt group; each sample in the group must carry the same
`task_id` and `environment_ref`.

For a public OCI source, sample metadata uses the same wire object:

```json
{
  "task_id": "task-123",
  "environment_ref": {
  "kind": "image",
  "id": "docker.io/library/ubuntu",
  "revision": "sha256:224a1869083a311ef3f13648a154ba79832fbef6364d31493642ca03082da254",
  "resource_profile": "standard"
}
}
```

Whether `kind: image` is accepted is an Ash deployment policy. Miles passes the
immutable identity unchanged; Ash controls allowed registries, resource
profiles, credentials, conversion to a runtime-ready template/snapshot and
cache lifecycle. A rejected source fails the rollout submission before any
sandbox is created.

`task_id` supplies task/trace identity and may be consumed by an Ash task
adapter. It does not by itself select a sandbox artifact; the four-field
`environment_ref` does that. Repository initialization, working directory,
tool policy and reward remain Ash/task-adapter responsibilities.

Miles serializes its usual sampling controls into `sampling_params`. The
recommended Claude Agent SDK backend maps supported controls into
`ClaudeAgentOptions` and sends Anthropic Messages requests through Miles'
session endpoint. Legacy AshAgent strategies retain their OpenAI-compatible
mapping. A new control still needs a coordinated adapter change before Miles
can assume it affects generation.

## Claude/Anthropic token semantics

Claude Agent SDK adds a stable system/tool protocol and runtime context. Ash
therefore returns `prompt_token_alignment=harness_rendered`, and Miles treats
the token sequence actually recorded by SessionTree as authoritative. System,
user, tool-result and synthetic continuation tokens receive `loss_mask=0`;
only assistant `generated_spans` receive `loss_mask=1`.

The Anthropic adapter preserves ordinary system content. It removes only a
fully matched, per-turn `<total_tokens>...` reminder when the request also
carries Claude Code's billing marker, preventing a volatile counter from
rewriting an otherwise reusable SessionTree prefix.

## Functional validation

The current recorded full training run (2026-09-14) used Claude Agent SDK,
Ash MCP, AgentENV/Firecracker, Miles SessionTree, SGLang/Qwen3.8-27B and one
GRPO/Megatron step. One environment checkpoint and Claude transcript fork
produced two trajectories with 3 model calls, 1 tool call and two SessionTree
leaves. Miles recomputed old-policy log-probabilities, obtained rewards
`[0, 1]` and advantages `[-1, +1]`, completed a valid optimizer step
(`grad_norm=14.268937110900879`), and updated SGLang weight version 1 -> 2.

This is an interface and correctness check, not evidence of rollout
performance, training quality, or a dynamic branch-selection policy.

Do not enable rollout routing replay, indexer replay, or rollout sampling-mask
features with this backend until their per-token payloads are represented in
the Ash protocol. The exact-token and generated-span requirements are part of
the correctness boundary and must not be replaced by text retokenization.
