# SWE-rebench with Ash message rollouts

Ash executes the agent, prepares the image, removes marked branch hints, and
grades the completed snapshot. Miles retokenizes the returned conversation,
masks user/system/tool text, and recomputes logprobs on the resulting **hint-free**
training sequence. This does not recover the probabilities of the original
hint-conditioned sampling process. The existing loss and fixed prompt-group size
are preserved.

Prepare a small dataset:

```bash
python examples/swe-rebench-ash/prepare_data.py \
  --output-dir /tmp/swe-inputs \
  --ash-data-dir /home/ec2-user/swe-inputs \
  --model local-model --limit 3
```

`miles.jsonl` is the Miles prompt dataset. Its metadata contains `task_id` and
`image` (the original dataset image name); Miles performs no image resolution.
`tasks.jsonl` and `log_parsers.py` belong on the Ash worker host at the specified
directory. Their hashes are frozen in `ash-driver.json`. The parser is downloaded
from a pinned PrimeIntellect source revision. Reference solution patches are not
included in the training input or grading bundle.

Configure an existing Ash Run Store profile with the microVM backend, runtime
binary, native agent, model credentials, and grading backend. Review the generated
profile name, model alias, resources and verifier networking for that deployment.
Start the existing Run Store and worker, then use the generated driver config:

```bash
cd /path/to/Ash
PYTHONPATH=.:sdk python3.12 -m rl_driver --config /home/ec2-user/swe-inputs/ash-driver.json
```

The default driver binds localhost. Use the deployment's tunnel or advertised
address for `--ash-rollout-base-url`. No service is started by data preparation.

Add these options to a normal Miles RL launcher:

```text
--prompt-data /tmp/swe-inputs/miles.jsonl
--input-key prompt
--metadata-key metadata
--rollout-function-path miles.rollout.ash.message_rollout.AshMessageRolloutFn
--ash-rollout-base-url http://ASH_HOST:11001
--rollout-batch-size 1
--n-samples-per-prompt 2
--ash-rollout-max-turns 64
--rollout-max-response-len 8192
--loss-mask-type qwen3
```

Select `loss-mask-type` for the checkpoint's chat template; the example uses
Qwen3. Preserve message lists (do not enable `--apply-chat-template` on the prompt
dataset). Do not enable `--use-rollout-logprobs`, `--skip-actor-forward-only`,
rollout-token replay, or TIS for this path: the trainer must score the newly
constructed sequence. No Session Server token-recording bridge is required.

The native model endpoint must serve the configured agent's API: Responses for
Codex or Messages for Claude. Sampling supports temperature, top-p, top-k,
text stop sequences and output length. Native APIs own their response formatting;
v2's raw-token detokenization settings are not part of v3. Token-id stopping and
chat-template overrides are rejected rather than silently ignored.

Ash v3 requires a real grader result. Turn-limit and execution-time cutoffs
produce a final quiescent snapshot for grading, with `status="truncated"` and
`stop_reason="max_turns_reached"` or `"timeout"` on the returned trajectory.
The driver and Miles wait for grading beyond the execution deadline, bounded
by `--ash-rollout-finalization-timeout-seconds` (1800 by default; this launcher
exposes `--finalization-timeout`). It returns no invented zero reward when a
grader is absent or infrastructure fails. SWE-rebench grading restores a separate
snapshot, resets only files touched by the held-out test patch, applies that patch
at grading time, and uses the pinned language-specific parser to check every
FAIL_TO_PASS and PASS_TO_PASS test.

Branch continuation prompts in v3 workers are wrapped with reserved
`<ash_training_hint>...</ash_training_hint>` markers before reaching the agent.
Those marked instructions are removed only from user/system/developer messages
during training export. Tool outputs and assistant responses are retained,
including any literal mention of a hint. Historical unmarked hints cannot be
reliably identified automatically. Incomplete tool histories, compaction and
unsupported native content fail export explicitly.

The driver still allocates independent samples by default; this change does not
introduce a branch-search policy or alter shared-prefix credit/weighting. Branches
produced by Ash's existing execution path retain their provenance. Every returned
sample still occupies a Miles-allocated slot.

For a single real Qwen3.8-27B update on eight GPUs, convert the checkpoint with
the existing `qwen3.8-27B` model definition, then run:

```bash
python examples/swe-rebench-ash/run_qwen3_8_27b.py \
  --model-dir /root/models --data-dir /tmp/swe-inputs \
  --output-dir /root/shared_data/ash-tp8 \
  --ash-url http://ASH_DRIVER:11001 \
  --model-endpoint https://MODEL_HOST
```

The launcher colocates a TP=8 trainer and a TP=8 rollout engine, uses two
trajectories per prompt, and saves a checkpoint after one update. Ash must reach
the Miles-owned model router through the advertised endpoint. A deployment that
requires HTTPS 443 needs its existing authenticated TLS frontend to forward to
that router (`--router-port`, default 18081). Run the launcher in its own process
and Ray namespace: `execute_train` stops matching Miles/SGLang/Ray processes
during startup.

This recipe uses `--use-miles-router` to forward native Responses requests
without an intermediate typed input parser. The deployed Rust router rejected
the agent's follow-up input after its first tool call.

The default run allows 64 turns per trajectory, 16,384 output tokens per model
call, and 3,600 execution seconds per group, plus 1,800 seconds for finalization.
Use `--max-turns N` on this launcher or
`--ash-rollout-max-turns N` on `train.py`. One turn is one model invocation;
its tool calls do not consume extra turns and have no separate count cap.
Changing `--n-samples-per-prompt` does not change the limit for each trajectory.
The old model/tool call-budget options are rejected by the v3 message path.
These are runtime ceilings,
not guarantees that tasks finish or produce different rewards. It retains the
existing GRPO loss and requires Ash grading; failed infrastructure is not scored
as a zero reward.

Long agent trajectories can contain many more tokens than the per-call output
limit. The launcher computes logprobs in 1,024-token chunks by default
(`--log-probs-chunk-size`) to avoid a full-response FP32 temporary tensor.
This preserves the complete trajectory and loss masks.
Loss computation also uses activation checkpointing (`--no-recompute-loss`
disables it). The training allocator reserves 1 GiB by default; an isolated
training-only retry can set `--train-memory-margin-bytes 0` to use that headroom.

To resume training from a saved real rollout after a training-side failure, add
`--load-rollout-data /path/to/dump_details/rollout_data/0.pt` and use a new output
directory. Miles loads those samples and rewards without starting SGLang or
submitting another Ash rollout; it still recomputes logprobs and runs the
optimizer. Keep the original model and training configuration when resuming.

For LoRA on a saved Ash rollout, add `--lora-rank 16 --lora-alpha 32
--lora-dropout 0`. The recipe loads the original HF weights through Megatron
Bridge, disables MTP, and targets decoder attention, MLP and GDN projections.
The base weights stay frozen. This mode currently requires a recorded rollout;
live Ash requests still need adapter-selection validation before using updated
LoRA weights. The initial optimizer step can use a recorded base-model rollout
when the adapters are initialized with zero output.
