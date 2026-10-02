"""Shared pieces for the Sprout rollout tests.

The two JSON fixtures are copies of the examples Sprout ships beside its RL
driver (``src/sprout/rl_driver/request.example.json`` and
``result.example.json``): what Miles must send, and what it receives back.
"""

import json
from argparse import Namespace
from copy import deepcopy
from pathlib import Path

import pytest
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

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
]


def sprout_request() -> dict:
    return json.loads((FIXTURES / "sprout_request.example.json").read_text())


def sprout_result() -> dict:
    return json.loads((FIXTURES / "sprout_result.example.json").read_text())


@pytest.fixture
def mask_generator() -> MultiTurnLossMaskGenerator:
    """A word-level tokenizer with a one-tag-per-role chat template: enough to
    see which words of which role end up trainable."""
    raw = Tokenizer(models.WordLevel({word: i for i, word in enumerate(WORDS)}, unk_token="<unk>"))
    raw.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=raw, unk_token="<unk>")
    tokenizer.chat_template = (
        "{% for m in messages %}{{ {'user':'<u>', 'assistant':'<a>', 'tool':'<t>', 'system':'<s>'}[m.role] }} "
        "{{ m.content }}{% if m.tool_calls is defined %} {{ m.tool_calls | tojson }}{% endif %} </m> "
        "{% endfor %}{% if add_generation_prompt %}<a> {% endif %}"
    )
    return MultiTurnLossMaskGenerator(tokenizer)


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
        "rollout_top_p": 0.95,
        "rollout_top_k": 20,
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
