"""Shared pieces for the Sprout rollout tests.

The JSON fixtures are byte copies of the examples Sprout ships in its
``examples/rl_driver/`` (``SPROUT_EXAMPLES``): the request Miles must send, the
result of a group without a search, and the result of a RAFO search.
``forest_result`` builds the result of a RAFO search the way Sprout annotates
one.
"""

import json
import os
from argparse import Namespace
from copy import deepcopy
from pathlib import Path

import pytest
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from miles.rollout.sprout.forest import search_outcome
from miles.utils.mask_utils import MultiTurnLossMaskGenerator
from miles.utils.types import Sample

FIXTURES = Path(__file__).parent / "fixtures"
#: Each fixture and the example in Sprout's ``examples/rl_driver/`` that it copies byte for byte.
SPROUT_EXAMPLES = {
    "sprout_request.example.json": "request.example.json",
    "sprout_result.example.json": "result.example.json",
    "sprout_rafo_search_result.example.json": "rafo_search_result.example.json",
}

WORDS = [
    "<unk>",
    "<u>",
    "<a>",
    "<t>",
    "<s>",
    "</m>",
    "FOR",
    "TESTING",
    "ONLY",
    "CALCULATING",
    "LOSS",
    "MASK",
    "fix",
    "task",
    "fixed",
    "/workspace",
    "inspect",
    "done",
    "hint",
    "tests",
    "pass",
    "<think>",
    "</think>",
    "sprout",
    "reasoning",
    "probe",
    "parser",
    "broken",
]

#: The toy chat template: one tag per role, an assistant turn's reasoning in
#: think tags before its content. Without reasoning a message renders as the
#: template did before it knew of reasoning.
TEMPLATE = (
    "{% for m in messages %}{{ {'user':'<u>', 'assistant':'<a>', 'tool':'<t>', 'system':'<s>'}[m.role] }} "
    "{% if m.reasoning_content %}<think> {{ m.reasoning_content }} </think> {% endif %}"
    "{{ m.content }}{% if m.tool_calls is defined %} {{ m.tool_calls | tojson }}{% endif %} </m> "
    "{% endfor %}{% if add_generation_prompt %}<a> {% endif %}"
)


def sprout_request() -> dict:
    return json.loads((FIXTURES / "sprout_request.example.json").read_text())


def sprout_result() -> dict:
    return json.loads((FIXTURES / "sprout_result.example.json").read_text())


def toy_mask_generator(template: str = TEMPLATE, tokenizer_type: str = "qwen") -> MultiTurnLossMaskGenerator:
    raw = Tokenizer(models.WordLevel({word: i for i, word in enumerate(WORDS)}, unk_token="<unk>"))
    raw.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=raw, unk_token="<unk>")
    tokenizer.chat_template = template
    return MultiTurnLossMaskGenerator(tokenizer, tokenizer_type=tokenizer_type)


@pytest.fixture
def mask_generator() -> MultiTurnLossMaskGenerator:
    """A word-level tokenizer with a one-tag-per-role chat template: enough to
    see which words of which role end up trainable."""
    return toy_mask_generator()


def args(**updates) -> Namespace:
    """The Miles arguments SproutRolloutFn reads, set so that a built request
    equals Sprout's shipped request example."""
    fields = {
        "sprout_rollout_base_url": "http://driver:11001",
        "sprout_rollout_model_endpoint": "http://model:30000",
        "sprout_rollout_token_env": "SPROUT_RL_DRIVER_TOKEN",
        "sprout_rollout_max_turns": 10,
        "sprout_rollout_timeout_seconds": 10.0,
        "sprout_rollout_finalization_timeout_seconds": 1800.0,
        "sprout_rollout_poll_interval_seconds": 0.001,
        "sprout_rollout_http_timeout_seconds": 1.0,
        "rollout_batch_size": 1,
        "n_samples_per_prompt": 1,
        "rollout_temperature": 0.6,
        "rollout_top_p": 1.0,
        "rollout_top_k": -1,
        "rollout_max_response_len": 512,
        "rollout_stop": ["stop-here"],
        "rollout_stop_token_ids": None,
        "apply_chat_template_kwargs": {},
        "hf_checkpoint": "unused",
        "chat_template_path": None,
        "loss_mask_type": "qwen",
        "model_name": "policy",
        "sglang_served_model_name": None,
        "sglang_router_ip": None,
        "sglang_router_port": None,
        "use_rollout_logprobs": False,
        "skip_actor_forward_only": False,
        "use_tis": False,
        "use_rollout_routing_replay": False,
        "use_rollout_indexer_replay": False,
        "use_sampling_support_replay": False,
        "train_backend": "megatron",
        "rewards_normalization": True,
        "custom_reward_post_process_path": None,
        "use_dynamic_global_batch_size": False,
        "compute_advantages_and_returns": True,
        "loss_type": "policy_loss",
        "custom_loss_function_path": None,
        "sprout_rollout_rafo": False,
        "sprout_rollout_rafo_continuations": 2,
        "sprout_rollout_rafo_depth": 3,
        "sprout_rollout_rafo_events": 4,
        "sprout_rollout_rafo_review_seconds": 5.0,
        "sprout_rollout_rafo_wall_time_seconds": 7.0,
        "sprout_rollout_rafo_c": 64.0,
        "sprout_rollout_rafo_c_h": 32.0,
        "sprout_rollout_rafo_turn_credit": "subtree",
    }
    return Namespace(**{**fields, **updates})


def group(size: int = 1, *, group_index: int = 3, first_index: int = 11) -> list[Sample]:
    """A prompt group the way RolloutDataSource hands one out: copies of one
    prompt, consecutive sample indices, one group index."""
    example = sprout_request()
    return [
        Sample(
            prompt=deepcopy(example["prompt"]),
            index=first_index + offset,
            group_index=group_index,
            metadata={"task_id": example["task_id"], "image": example["image"]},
        )
        for offset in range(size)
    ]


#: The history that an event of depth 1 restores: three messages, none of them its own.
PREFIX = [
    {"role": "user", "content": "fix task"},
    {
        "role": "assistant",
        "content": "inspect",
        "tool_calls": [
            {"id": "parent-1", "type": "function", "function": {"name": "shell", "arguments": '{"command":"pwd"}'}}
        ],
    },
    {"role": "tool", "tool_call_id": "parent-1", "content": "/workspace"},
]
#: The reasoning of each event's turn, by event number.
TURNS = {1: "parser broken", 2: "inspect tests", 3: "tests pass", 4: "inspect parser"}
#: A slot whose actor failed: Sprout names it in ``failed_samples``, with no trajectory.
LOST = "lost"


def inserted_turn(event: int) -> dict:
    """The turn of ``event`` as Sprout inserts it in a continuation: the reviewer's text as the reasoning,
    the content empty (``assistant_turn_rendering`` "reasoning"), then the check's command."""
    call = {
        "id": f"inserted-{event}",
        "type": "function",
        "function": {"name": "shell", "arguments": '{"command":"ls"}'},
    }
    return {"role": "assistant", "content": "", "reasoning_content": TURNS[event], "tool_calls": [call]}


#: The turn of event 1.
INSERTED_TURN = inserted_turn(1)


def forest_result(roots: list, events: list | None = None) -> tuple[dict, dict[str, Sample]]:
    """A RAFO search's result body, annotated the way Sprout annotates one, and the slots Miles allocated.

    ``roots`` holds each root's reward in slot order. ``events`` holds each event, numbered from 1 in
    the order given, as ``(parent, rewards)``: ``parent`` is where its turn went, ``"root:<i>"`` for
    the i-th root or ``"event:<n>:<j>"`` for the j-th continuation of event n (both from 0), and
    ``rewards`` are its continuations' in slot order. A reward of None is a trajectory whose grade gave
    no verdict, ``LOST`` a slot whose actor failed. Every turn is in reasoning form and checked clean.
    """
    graded = [reward for reward in roots if reward not in (None, LOST)]
    counts = {"planned": len(roots), "graded": len(graded), "resolved": int(sum(graded))}
    outcome = search_outcome(**counts)
    samples = group(len(roots), group_index=3, first_index=11)
    slots = {f"job:slot:{sample.index}": sample for sample in samples}
    template = sprout_result()["trajectories"][0]
    planned = [
        (None, "root", [*template["messages"][:3]], {"prefix_messages": 0, "inserted_messages": []}, reward)
        for reward in roots
    ]
    # Where each event's continuations stand: (branch id, history up to the turn, inserted turns, depth).
    continuations: dict[tuple[int, int], tuple[str, list, list, int]] = {}
    for number, (parent, rewards) in enumerate(events or [], 1):
        kind, *where = parent.split(":")
        if kind == "root":
            parent_id, prefix, inserted, depth = f"job-{11 + int(where[0])}", PREFIX, [], 1
        else:
            parent_id, history, inserted, depth = continuations[(int(where[0]), int(where[1]))]
            # The parent's history to the end of the check, and a step of its own, restored at its point.
            step = {"id": f"own-{parent_id}", "type": "function", "function": {"name": "shell", "arguments": "{}"}}
            prefix = [
                *history,
                {"role": "assistant", "content": "inspect", "tool_calls": [step]},
                {"role": "tool", "tool_call_id": step["id"], "content": "tests"},
            ]
            depth += 1
        history = [
            *deepcopy(prefix),
            inserted_turn(number),
            {"role": "tool", "tool_call_id": f"inserted-{number}", "content": "tests"},
        ]
        about = {
            "kind": "repair",
            "event": number,
            "depth": depth,
            "parent_job_id": parent_id,
            "point_id": f"{parent_id}-step2",
            "step": 2,
            "message_step": 1,
            "leak_terms": [],
            "assistant_turn_rendering": "reasoning",
        }
        provenance = {"prefix_messages": len(prefix), "inserted_messages": [*inserted, len(prefix)]}
        first = 11 + len(planned)
        for j in range(len(rewards)):
            continuations[(number, j)] = (f"job-{first + j}", history, provenance["inserted_messages"], depth)
        planned += [((parent_id, about), f"event:{number}", history, provenance, reward) for reward in rewards]
    trajectories, failed = [], []
    for index, (branch, condition, history, provenance, reward) in enumerate(planned, 11):
        slot = f"job:slot:{index}"
        if slot not in slots:
            slots[slot] = deepcopy(samples[0])
            slots[slot].index, slots[slot].group_index = index, None
        if reward == LOST:
            failed.append({"sample_slot_id": slot, "reason": "Actor did not complete"})
            continue
        about = {"kind": "root"} if branch is None else branch[1]
        search = {**about, "outcome": outcome, "root_counts": dict(counts)}
        parent = None if branch is None else branch[0]
        trajectories.append(_trajectory(slot, index, condition, history, provenance, search, parent, reward))
    body = {
        **sprout_result(),
        "rollout_job_id": "job",
        "max_samples": len(slots),
        "actual_samples": len(trajectories),
        "trajectories": trajectories,
        "failed_samples": failed,
        "search_branches": sum(trajectory["group"] != "root" for trajectory in trajectories),
        "status": "failed" if not trajectories else "early_stopped" if failed else "completed",
    }
    return body, slots


def _trajectory(
    slot: str,
    index: int,
    condition: str,
    history: list,
    provenance: dict,
    search: dict,
    parent: str | None,
    reward: float | None,
) -> dict:
    template = sprout_result()["trajectories"][0]
    own = {"role": "assistant", "content": "fixed" if reward == 1 else "done"}
    graded = reward is not None
    metadata = {
        **template["metadata"],
        "job_id": f"job-{index}",
        "attempt_id": f"attempt-job-{index}",
        "progress": float(reward) if graded else None,
        "provenance": provenance,
        "search": search,
        "zero_reason": None if graded else "grade_failed",
        "touched_test_paths": [] if graded else None,
    }
    return deepcopy(
        {
            **template,
            "sample_slot_id": slot,
            "branch_id": f"job-{index}",
            "parent_branch_id": parent,
            "group": condition,
            "messages": [*history, own],
            "reward": float(reward) if graded else 0.0,
            "metadata": metadata,
        }
    )


def members(body: dict, condition: str) -> list[dict]:
    """The trajectories of one condition (``Trajectory.group``) in a result body, to edit in place."""
    return [trajectory for trajectory in body["trajectories"] if trajectory["group"] == condition]


def sprout_examples() -> Path | None:
    """Sprout's ``examples/rl_driver``: of ``$SPROUT_REPO``, else of a checkout beside this one; None without."""
    configured = os.environ.get("SPROUT_REPO")
    siblings = Path(__file__).resolve().parents[5]  # the directory holding this checkout
    for checkout in [Path(configured)] if configured else [siblings / "sprout", siblings / "SPROUT"]:
        if (checkout / "examples" / "rl_driver").is_dir():
            return checkout / "examples" / "rl_driver"
    if configured:
        pytest.fail(f"SPROUT_REPO={configured} is not a Sprout checkout")
    return None
