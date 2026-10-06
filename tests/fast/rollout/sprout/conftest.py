"""Shared pieces for the Sprout rollout tests.

The two JSON fixtures are copies of the examples Sprout ships beside its RL
driver (``src/sprout/rl_driver/request.example.json`` and
``result.example.json``): what Miles must send, and what it receives back.
``search_result`` builds the result of a search group the way Sprout
annotates one.
"""

import itertools
import json
from argparse import Namespace
from copy import deepcopy
from pathlib import Path

import pytest
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from miles.rollout.sprout.gsml import assemble_search_group, search_outcome
from miles.rollout.sprout.protocol import RolloutResult
from miles.utils.mask_utils import MultiTurnLossMaskGenerator
from miles.utils.types import Sample

FIXTURES = Path(__file__).parent / "fixtures"

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
        "compute_advantages_and_returns": True,
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


#: The parent's history a branch restores: three messages, none of them the branch's own.
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
#: A reviewer's turn as Sprout inserts it in a repair continuation: its text as
#: the reasoning, the content empty (``assistant_turn_rendering`` "reasoning").
INSERTED_TURN = {
    "role": "assistant",
    "content": "",
    "reasoning_content": "parser broken",
    "tool_calls": [
        {"id": "inserted-1", "type": "function", "function": {"name": "shell", "arguments": '{"command":"ls"}'}}
    ],
}
#: A slot whose actor failed: Sprout names it in ``failed_samples``, with no trajectory.
LOST = "lost"


def search_result(roots: list, points: dict | None = None) -> tuple[dict, dict[str, Sample]]:
    """A search group's result body, annotated the way Sprout annotates one, and the slots Miles allocated.

    ``roots`` holds each root's reward in slot order; ``points`` maps a point id
    to ``{"students": rewards, "candidates": [rewards of each candidate]}``. A
    reward of None is a trajectory whose grade gave no verdict, ``LOST`` a slot
    whose actor failed. Every repair continuation is canonical, leaks nothing
    and touched no test; tests edit the body for anything else.
    """
    graded = [reward for reward in roots if reward not in (None, LOST)]
    counts = {"planned": len(roots), "graded": len(graded), "resolved": int(sum(graded))}
    outcome = search_outcome(**counts)
    samples = group(len(roots), group_index=3, first_index=11)
    slots = {f"job:slot:{sample.index}": sample for sample in samples}
    planned = [("root", {"kind": "root"}, reward) for reward in roots]
    for point, (point_id, branches) in enumerate((points or {}).items(), 1):
        about = {
            "kind": "student",
            "point": point,
            "point_id": point_id,
            "step": 2,
            "message_step": 1,
            "parent_job_id": "job-11",
        }
        planned += [(f"student:{point_id}", about, reward) for reward in branches.get("students", [])]
        for candidate, rewards in enumerate(branches.get("candidates", []), 1):
            repair = {
                **about,
                "kind": "repair",
                "candidate": candidate,
                "multiplicity": 1,
                "samples": None,
                "leak_terms": [],
                "assistant_turn_rendering": "reasoning",
            }
            planned += [(f"repair:{point_id}:{candidate}", repair, reward) for reward in rewards]
    trajectories, failed = [], []
    for index, (condition, about, reward) in enumerate(planned, 11):
        slot = f"job:slot:{index}"
        if slot not in slots:
            slots[slot] = deepcopy(samples[0])
            slots[slot].index, slots[slot].group_index = index, None
        if reward == LOST:
            failed.append({"sample_slot_id": slot, "reason": "Actor did not complete"})
            continue
        search = {**about, "outcome": outcome, "root_counts": dict(counts)}
        trajectories.append(_trajectory(slot, index, condition, search, reward))
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


def _trajectory(slot: str, index: int, condition: str, search: dict, reward: float | None) -> dict:
    template = sprout_result()["trajectories"][0]
    own = {"role": "assistant", "content": "fixed" if reward == 1 else "done"}
    if condition == "root":
        messages, provenance = [*template["messages"][:3], own], {"prefix_messages": 0, "inserted_messages": []}
    elif condition.startswith("student:"):
        messages, provenance = [*PREFIX, own], {"prefix_messages": 3, "inserted_messages": []}
    else:
        tool = {"role": "tool", "tool_call_id": "inserted-1", "content": "tests"}
        messages, provenance = [*PREFIX, INSERTED_TURN, tool, own], {"prefix_messages": 3, "inserted_messages": [3]}
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
            "parent_branch_id": None if condition == "root" else "job-11",
            "group": condition,
            "messages": messages,
            "reward": float(reward) if graded else 0.0,
            "metadata": metadata,
        }
    )


def members(body: dict, condition: str) -> list[dict]:
    """The trajectories of one condition (``Trajectory.group``) in a result body, to edit in place."""
    return [trajectory for trajectory in body["trajectories"] if trajectory["group"] == condition]


def ff_example() -> tuple[dict, dict[str, Sample]]:
    """The shared FF example: both roots fail. At P1 a student resolves; at P2 none does, candidate 1's
    continuations both resolve and candidate 2's turn names a hidden test file."""
    body, slots = search_result(
        [0.0, 0.0],
        {
            "p1": {"students": [1.0, 0.0], "candidates": [[1.0, 0.0], [0.0, 0.0]]},
            "p2": {"students": [0.0, 0.0], "candidates": [[1.0, 1.0], [1.0, 0.0]]},
        },
    )
    for trajectory in members(body, "repair:p2:2"):
        trajectory["metadata"]["search"]["leak_terms"] = ["tests/test_hidden.py"]
    return body, slots


def gsml_args(**updates) -> Namespace:
    """The flags ``post_process_gsml`` reads, at the values of the shared example (λ 0.4, β_br = ψ = 1)."""
    fields = {
        "sprout_rollout_gsml_lambda": 0.4,
        "sprout_rollout_gsml_beta_branch": 1.0,
        "sprout_rollout_gsml_psi": 1.0,
        "sprout_rollout_gsml_c_pre": 0.0,
        "sprout_rollout_gsml_kappa_plus": 0.0,
        "reward_key": None,
    }
    return Namespace(**{**fields, **updates})


def assemble(mask_generator, body: dict, slots: dict[str, Sample], *, planned_roots: int = 2):
    """``assemble_search_group`` on a result body, with fresh identities after every slot's."""
    return assemble_search_group(
        RolloutResult.model_validate(body),
        slots,
        mask_generator,
        planned_roots=planned_roots,
        weight_version=7,
        indices=itertools.count(100),
        group_indices=itertools.count(50),
    )
