import argparse
import asyncio
import json
import sys
from argparse import Namespace
from copy import deepcopy
from unittest.mock import patch

import httpx
import pytest
from pydantic import ValidationError
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from miles.rollout.ash.message_importer import import_ash_messages
from miles.rollout.ash.message_protocol import AshMessageRequest, AshMessageResult
from miles.rollout.ash.message_rollout import AshMessageClient, AshMessageRolloutFn, UnusableAshGroup
from miles.rollout.base_types import RolloutFnConstructorInput, RolloutFnTrainInput
from miles.utils.mask_utils import MultiTurnLossMaskGenerator
from miles.utils.types import Sample


@pytest.fixture
def mask_generator():
    words = [
        "<unk>",
        "<u>",
        "<a>",
        "<t>",
        "<s>",
        "</m>",
        "FOR",
        "TESTING",
        "ONLY",
        "fix",
        "task",
        "inspect",
        "shell",
        "pwd",
        "workspace",
        "done",
        "hint",
    ]
    raw = Tokenizer(models.WordLevel({word: i for i, word in enumerate(words)}, unk_token="<unk>"))
    raw.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=raw, unk_token="<unk>")
    tokenizer.chat_template = "{% for m in messages %}{{ {'user':'<u>', 'assistant':'<a>', 'tool':'<t>', 'system':'<s>'}[m.role] }} {{ m.content }}{% if m.tool_calls is defined %} {{ m.tool_calls | tojson }}{% endif %} </m> {% endfor %}{% if add_generation_prompt %}<a> {% endif %}"
    return MultiTurnLossMaskGenerator(tokenizer)


def payload():
    return {
        "protocol_version": "ash-rollout-v3",
        "rollout_job_id": "job",
        "prompt_group_id": "3",
        "max_samples": 1,
        "actual_samples": 1,
        "status": "completed",
        "trajectories": [
            {
                "sample_slot_id": "slot",
                "branch_id": "child",
                "parent_branch_id": "parent",
                "messages": [
                    {"role": "user", "content": "fix task"},
                    {
                        "role": "assistant",
                        "content": "inspect",
                        "tool_calls": [
                            {
                                "id": "call",
                                "type": "function",
                                "function": {"name": "shell", "arguments": '{"command":"pwd"}'},
                            }
                        ],
                    },
                    {"role": "tool", "tool_call_id": "call", "content": "workspace"},
                    {"role": "assistant", "content": "done"},
                ],
                "reward": 1.0,
                "status": "completed",
                "hints_removed": True,
            }
        ],
    }


def args(**updates):
    fields = {
        "ash_rollout_base_url": "http://ash",
        "ash_rollout_poll_interval_seconds": 0.001,
        "ash_rollout_timeout_seconds": 10.0,
        "ash_rollout_client_grace_seconds": 0.1,
        "ash_rollout_http_timeout_seconds": 1.0,
        "ash_rollout_max_turns": 10,
        "ash_rollout_max_model_calls": None,
        "ash_rollout_max_tool_calls": None,
        "rollout_batch_size": 1,
        "n_samples_per_prompt": 1,
        "rollout_temperature": 0.6,
        "rollout_top_p": 0.95,
        "rollout_top_k": 20,
        "rollout_max_response_len": 512,
        "rollout_stop": ["stop-here"],
        "rollout_stop_token_ids": None,
        "rollout_skip_special_tokens": False,
        "hf_checkpoint": "unused",
        "chat_template_path": None,
        "use_rollout_logprobs": False,
        "sglang_router_ip": "model",
        "sglang_router_port": 30000,
        "model_name": "policy",
        "loss_mask_type": "qwen",
    }
    return Namespace(**{**fields, **updates})


def group():
    return [
        Sample(
            prompt=[{"role": "user", "content": "fix task"}],
            index=11,
            group_index=3,
            metadata={"task_id": "task-1", "image": "prime/primeintellect/task:base"},
        )
    ]


def test_import_rebuilds_tokens_and_masks_and_discards_execution_logprobs(mask_generator):
    source = group()[0]
    source.tokens = [100, 200]
    source.rollout_log_probs = [-0.2]
    (sample,) = import_ash_messages(AshMessageResult.model_validate(payload()), {"slot": source}, mask_generator)
    assert sample.tokens != source.tokens
    assert source.tokens == [100, 200]
    assert sample.reward == 1.0
    assert sample.rollout_log_probs is None and sample.weight_versions == []
    response = sample.tokens[-sample.response_length :]
    trained = mask_generator.tokenizer.decode([t for t, m in zip(response, sample.loss_mask, strict=True) if m])
    assert "inspect" in trained and "done" in trained
    assert "workspace" not in trained
    assert "workspace" in mask_generator.tokenizer.decode(sample.tokens)
    assert sample.metadata["ash_rollout"]["parent_branch_id"] == "parent"


@pytest.mark.parametrize("reason", ["timeout", "max_turns_reached"])
def test_scored_cutoff_is_imported_as_truncated_with_reason(mask_generator, reason):
    body = payload()
    body["trajectories"][0].update(status="truncated", stop_reason=reason, reward=0.0)
    (sample,) = import_ash_messages(AshMessageResult.model_validate(body), {"slot": group()[0]}, mask_generator)
    assert sample.status == Sample.Status.TRUNCATED
    assert sample.reward == 0.0
    assert sample.metadata["ash_rollout"]["stop_reason"] == reason
    assert sample.train_metadata["ash_rollout"]["stop_reason"] == reason
    assert sample.rollout_log_probs is None and any(sample.loss_mask)


def test_cutoff_requires_a_reason():
    body = payload()
    body["trajectories"][0]["status"] = "truncated"
    with pytest.raises(ValidationError, match="stop_reason"):
        AshMessageResult.model_validate(body)


@pytest.mark.parametrize(
    "change",
    [
        lambda body: body["trajectories"][0].update(hints_removed=False),
        lambda body: body["trajectories"][0].update(reward=None),
        lambda body: body["trajectories"][0].update(reward=float("nan")),
        lambda body: body["trajectories"][0].update(token_ids=[1, 2]),
    ],
)
def test_message_wire_rejects_wrong_representation(change):
    body = payload()
    change(body)
    with pytest.raises(ValidationError):
        AshMessageResult.model_validate(body)


def test_unpaired_tool_results_are_rejected(mask_generator):
    body = payload()
    body["trajectories"][0]["messages"][2]["tool_call_id"] = "wrong"
    with pytest.raises(ValueError, match="unknown tool_call_id"):
        import_ash_messages(AshMessageResult.model_validate(body), {"slot": group()[0]}, mask_generator)


def test_hint_attestation_does_not_hide_an_unremoved_hint(mask_generator):
    body = payload()
    body["trajectories"][0]["messages"][0]["content"] += "<ash_training_hint>private</ash_training_hint>"
    with pytest.raises(ValueError, match="unremoved"):
        import_ash_messages(AshMessageResult.model_validate(body), {"slot": group()[0]}, mask_generator)


def test_native_tool_arguments_are_mappings_for_the_chat_template(mask_generator):
    body = payload()
    wire_arguments = body["trajectories"][0]["messages"][1]["tool_calls"][0]["function"]["arguments"]
    (sample,) = import_ash_messages(AshMessageResult.model_validate(body), {"slot": group()[0]}, mask_generator)
    arguments = sample.metadata["ash_rollout"]["messages"][1]["tool_calls"][0]["function"]["arguments"]
    assert arguments == {"command": "pwd"}
    assert isinstance(wire_arguments, str)


def test_non_object_tool_arguments_are_rejected(mask_generator):
    body = payload()
    body["trajectories"][0]["messages"][1]["tool_calls"][0]["function"]["arguments"] = "[1, 2]"
    with pytest.raises(ValueError, match="JSON object"):
        import_ash_messages(AshMessageResult.model_validate(body), {"slot": group()[0]}, mask_generator)


@pytest.mark.parametrize("explicit_system", [False, True])
def test_qwen3_mask_preserves_context_with_implicit_system_preamble(mask_generator, explicit_system):
    tokenizer = mask_generator.tokenizer
    tokenizer.chat_template = (
        "{% if messages[0].role != 'system' %}<s> fix task inspect workspace </m> {% endif %}"
        + tokenizer.chat_template
    )
    generator = MultiTurnLossMaskGenerator(tokenizer, tokenizer_type="qwen3")
    assert generator.system_message_length > 0
    messages = [
        {"role": "user", "content": "fix task"},
        {"role": "assistant", "content": "inspect"},
        {"role": "tool", "content": "workspace"},
        {"role": "assistant", "content": "done"},
    ]
    if explicit_system:
        messages.insert(0, {"role": "system", "content": "fix task inspect workspace"})
    actual, mask = generator.get_loss_mask(messages)
    expected = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False, return_dict=False)
    assert actual == expected
    trained = tokenizer.decode([token for token, keep in zip(actual, mask, strict=True) if keep])
    assert "inspect" in trained and "done" in trained and "workspace" not in trained


@pytest.mark.parametrize("mask_type", ["qwen", "qwen3"])
@pytest.mark.parametrize("implicit_system", [False, True])
def test_adjacent_assistants_preserve_tokens_and_both_loss_spans(mask_generator, mask_type, implicit_system):
    tokenizer = mask_generator.tokenizer
    if implicit_system:
        tokenizer.chat_template = (
            "{% if messages[0].role != 'system' %}<s> fix task </m> {% endif %}" + tokenizer.chat_template
        )
    generator = MultiTurnLossMaskGenerator(tokenizer, tokenizer_type=mask_type)
    messages = [
        {"role": "user", "content": "fix task"},
        {"role": "assistant", "content": "inspect"},
        {"role": "assistant", "content": "done"},
    ]
    body = payload()
    body["trajectories"][0]["messages"] = messages
    (sample,) = import_ash_messages(AshMessageResult.model_validate(body), {"slot": group()[0]}, generator)
    assert sample.tokens == tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=False, return_dict=False
    )
    response = sample.tokens[-sample.response_length :]
    trained = tokenizer.decode([token for token, keep in zip(response, sample.loss_mask, strict=True) if keep])
    assert "inspect" in trained and "done" in trained
    assert "fix" not in trained and "task" not in trained
    assert sample.metadata["ash_rollout"]["messages"] == messages


def _args_from_real_budget_cli(options):
    from miles.utils.arguments import get_miles_extra_args_provider

    argv = ["--rollout-batch-size", "1", *options]
    with patch.object(sys, "argv", ["test", *argv]):
        parser = argparse.ArgumentParser()
        get_miles_extra_args_provider()(parser)
        parsed = parser.parse_args(argv)
    result = args(ash_rollout_max_turns=None)
    for name in (
        "ash_rollout_max_turns",
        "ash_rollout_max_model_calls",
        "ash_rollout_max_tool_calls",
        "_ash_rollout_max_model_calls_explicit",
        "_ash_rollout_max_tool_calls_explicit",
    ):
        setattr(result, name, getattr(parsed, name))
    return result


def test_v3_accepts_unset_cli_count_flags_without_changing_v2_defaults():
    parsed = _args_from_real_budget_cli([])
    assert parsed.ash_rollout_max_model_calls == parsed.ash_rollout_max_tool_calls == 100
    fn = AshMessageRolloutFn(RolloutFnConstructorInput(args=parsed, data_source=None))
    request, _ = fn._build_request(group=group(), rollout_id=0, weight_version=0)
    assert request.max_turns == 64
    assert request.model_dump()["budgets"] == {"max_wall_time_seconds": 10.0}


@pytest.mark.parametrize("flag", ["--ash-rollout-max-model-calls", "--ash-rollout-max-tool-calls"])
@pytest.mark.parametrize("value", ["100", "unbounded"])
def test_v3_rejects_explicit_cli_counts_including_unbounded(flag, value):
    parsed = _args_from_real_budget_cli([flag, value])
    if value == "unbounded":
        assert getattr(parsed, flag[2:].replace("-", "_")) is None
    with pytest.raises(ValueError, match="v2-only"):
        AshMessageRolloutFn(RolloutFnConstructorInput(args=parsed, data_source=None))


@pytest.mark.parametrize("flag", ["use_rollout_logprobs", "skip_actor_forward_only", "use_tis"])
def test_configuration_requires_trainer_recomputation(flag):
    with pytest.raises(ValueError, match="incompatible"):
        AshMessageRolloutFn(RolloutFnConstructorInput(args=args(**{flag: True}), data_source=None))


def test_unset_output_limit_leaves_native_default():
    fn = AshMessageRolloutFn(RolloutFnConstructorInput(args=args(rollout_max_response_len=None), data_source=None))
    request, _ = fn._build_request(group=group(), rollout_id=1, weight_version=7)
    assert "max_new_tokens" not in request.sampling_params


@pytest.mark.parametrize("train_only,expected", [(False, "policy:miles_lora"), (True, "policy")])
def test_live_lora_uses_explicit_native_adapter_model(train_only, expected):
    fn = AshMessageRolloutFn(RolloutFnConstructorInput(args=args(
        lora_rank=32, lora_train_only=train_only, sglang_served_model_name="policy",
        ash_rollout_max_sequence_tokens=81920, ash_rollout_truncated_reward_scale=0.5,
    ), data_source=None))
    request, _ = fn._build_request(group=group(), rollout_id=1, weight_version=7)
    assert request.model == expected
    assert request.to_wire()["max_sequence_tokens"] == 81920
    assert request.to_wire()["truncated_reward_scale"] == 0.5


def test_sequence_truncation_reward_and_reason_survive_import(mask_generator):
    body = payload()
    body["trajectories"][0].update(status="truncated", stop_reason="max_sequence_tokens", reward=0.5)
    (sample,) = import_ash_messages(AshMessageResult.model_validate(body), {"slot": group()[0]}, mask_generator)
    assert sample.reward == 0.5 and sample.status == Sample.Status.TRUNCATED
    assert sample.metadata["ash_rollout"]["stop_reason"] == "max_sequence_tokens"


def test_unusable_group_is_replaced_with_a_new_prompt_without_padding():
    replacement = deepcopy(group())
    replacement[0].index = 99
    replacement[0].group_index = 4

    class Data:
        calls = 0

        def get_samples(self, count):
            self.calls += 1
            assert count == 1 and self.calls == 1
            return [replacement]

    data = Data()
    fn = AshMessageRolloutFn(RolloutFnConstructorInput(
        args=args(ash_rollout_max_unusable_groups=1), data_source=data,
    ))
    seen = []

    async def attempt(**kwargs):
        seen.append(kwargs["group"][0].index)
        if len(seen) == 1:
            raise UnusableAshGroup("no available recovery point")
        return kwargs["group"], None

    fn._run_attempt = attempt
    samples, _ = asyncio.run(fn._run_group(client=None, group=group(), rollout_id=0, weight_version=0))
    assert seen == [11, 99] and samples[0].group_index == 4
    assert fn._discarded_groups == 1


def test_protocol_errors_are_not_hidden_by_prompt_replacement():
    class Data:
        def get_samples(self, count):
            pytest.fail("Protocol errors must not consume replacement prompts")

    fn = AshMessageRolloutFn(RolloutFnConstructorInput(
        args=args(ash_rollout_max_unusable_groups=8), data_source=Data(),
    ))

    async def attempt(**kwargs):
        raise ValueError("result identity mismatch")

    fn._run_attempt = attempt
    with pytest.raises(ValueError, match="identity mismatch"):
        asyncio.run(fn._run_group(client=None, group=group(), rollout_id=0, weight_version=0))


def test_branching_off_preserves_the_old_wire_and_real_cli_defaults():
    fn = AshMessageRolloutFn(RolloutFnConstructorInput(args=args(), data_source=None))
    request, _ = fn._build_request(group=group(), rollout_id=1, weight_version=7)
    assert not request.branching and "branching" not in request.to_wire()
    from miles.utils.arguments import get_miles_extra_args_provider

    parser = argparse.ArgumentParser()
    with patch.object(sys, "argv", ["test", "--rollout-batch-size", "1"]):
        get_miles_extra_args_provider()(parser)
        assert not parser.parse_args(["--rollout-batch-size", "1"]).ash_rollout_branching
        assert parser.parse_args(["--rollout-batch-size", "1", "--ash-rollout-branching"]).ash_rollout_branching


@pytest.mark.parametrize("count", [1, 3, 8])
def test_branching_requires_a_fixed_pair(count):
    with pytest.raises(ValueError, match="n-samples-per-prompt 2"):
        AshMessageRolloutFn(RolloutFnConstructorInput(
            args=args(ash_rollout_branching=True, n_samples_per_prompt=count), data_source=None,
        ))


@pytest.mark.parametrize("root_reward,search_branches", [(0.0, 4), (0.0, 7), (1.0, 2)])
def test_branching_request_and_opposite_pair_import(mask_generator, root_reward, search_branches):
    sent, deleted = [], []

    def handler(request):
        if request.method == "POST":
            body = json.loads(request.content)
            sent.append(body)
            assert body["branching"] is True
            assert body["max_samples"] == body["minimum_returned_samples"] == 2
            return httpx.Response(202, json={
                "protocol_version": "ash-rollout-v3", "rollout_job_id": body["rollout_job_id"], "status": "queued",
            })
        if request.method == "DELETE":
            deleted.append(request.url.path)
            return httpx.Response(200, json={
                "protocol_version": "ash-rollout-v3",
                "rollout_job_id": sent[0]["rollout_job_id"], "status": "completed",
            })
        body = payload()
        child = deepcopy(body["trajectories"][0])
        body.update(
            rollout_job_id=sent[0]["rollout_job_id"], max_samples=2, actual_samples=2,
            search_branches=search_branches,
        )
        body["trajectories"].append(child)
        for i, trajectory in enumerate(body["trajectories"]):
            trajectory.update(
                sample_slot_id=sent[0]["sample_slots"][i]["sample_slot_id"],
                branch_id="root" if i == 0 else "child",
                parent_branch_id=None if i == 0 else "root",
                reward=root_reward if i == 0 else 1.0 - root_reward,
                metadata={"branching": {"round": i, "stop_reason": "target_found"}},
            )
        return httpx.Response(200, json=body)

    async def run():
        fn = AshMessageRolloutFn(RolloutFnConstructorInput(
            args=args(ash_rollout_branching=True, n_samples_per_prompt=2), data_source=None,
        ))
        fn._mask_generator = mask_generator
        samples = [group()[0], deepcopy(group()[0])]
        samples[1].index += 1
        async with AshMessageClient(
            "http://ash", client=httpx.AsyncClient(base_url="http://ash", transport=httpx.MockTransport(handler)),
        ) as client:
            return await fn._run_group(client=client, group=samples, rollout_id=1, weight_version=7)

    samples, _result = asyncio.run(run())
    assert len(deleted) == 1 and len(samples) == 2
    assert _result.search_branches == search_branches
    assert [sample.reward for sample in samples] == [root_reward, 1.0 - root_reward]
    assert samples[1].metadata["ash_rollout"]["parent_branch_id"] == "root"
    assert all(sample.rollout_log_probs is None and any(sample.loss_mask) for sample in samples)
    assert len({sample.index for sample in samples}) == 2


@pytest.mark.parametrize("actual_count", [1, 3, 8])
def test_all_branching_imports_only_actual_graded_trajectories(mask_generator, actual_count):
    sent, deleted = [], []

    def handler(request):
        if request.method == "POST":
            body = json.loads(request.content)
            sent.append(body)
            assert body["branching"] and body["max_samples"] == 8
            assert body["minimum_returned_samples"] == 1
            return httpx.Response(202, json={
                "protocol_version": "ash-rollout-v3", "rollout_job_id": body["rollout_job_id"], "status": "queued",
            })
        if request.method == "DELETE":
            deleted.append(request.url.path)
            return httpx.Response(200, json={
                "protocol_version": "ash-rollout-v3", "rollout_job_id": sent[0]["rollout_job_id"], "status": "completed",
            })
        body = payload()
        template = body["trajectories"][0]
        body.update(rollout_job_id=sent[0]["rollout_job_id"], max_samples=8, actual_samples=actual_count)
        body["trajectories"] = [
            {
                **deepcopy(template), "sample_slot_id": sent[0]["sample_slots"][i]["sample_slot_id"],
                "branch_id": f"branch-{i}", "parent_branch_id": None if i == 0 else "branch-0",
                "reward": float(i % 2),
            }
            for i in range(actual_count)
        ]
        return httpx.Response(200, json=body)

    async def run():
        fn = AshMessageRolloutFn(RolloutFnConstructorInput(
            args=args(
                ash_rollout_branching=True, ash_rollout_branching_return_mode="all",
                use_dynamic_global_batch_size=True, n_samples_per_prompt=8,
            ),
            data_source=None,
        ))
        fn._mask_generator = mask_generator
        allocated = [deepcopy(group()[0]) for _ in range(8)]
        for i, sample in enumerate(allocated):
            sample.index += i
        async with AshMessageClient(
            "http://ash", client=httpx.AsyncClient(base_url="http://ash", transport=httpx.MockTransport(handler)),
        ) as client:
            return await fn._run_group(client=client, group=allocated, rollout_id=1, weight_version=7)

    samples, _ = asyncio.run(run())
    assert len(deleted) == 1
    assert len(samples) == actual_count
    assert len({sample.index for sample in samples}) == actual_count
    assert {sample.group_index for sample in samples} == {3}
    assert [sample.reward for sample in samples] == [float(i % 2) for i in range(actual_count)]
    assert all(sample.rollout_log_probs is None and any(sample.loss_mask) for sample in samples)


@pytest.mark.parametrize("group_size", [1, 2, 8])
def test_turn_limit_is_per_trajectory_and_wire_has_no_call_budgets(group_size):
    samples = [
        Sample(prompt=group()[0].prompt, index=i, group_index=3, metadata=group()[0].metadata)
        for i in range(group_size)
    ]
    fn = AshMessageRolloutFn(
        RolloutFnConstructorInput(
            args=args(n_samples_per_prompt=group_size, ash_rollout_max_turns=2), data_source=None
        )
    )
    request, _ = fn._build_request(group=samples, rollout_id=1, weight_version=7)
    wire = request.model_dump(mode="json")
    assert wire["max_samples"] == group_size
    assert wire["max_turns"] == 2
    assert wire["budgets"] == {"max_wall_time_seconds": 10.0}
    assert AshMessageRequest.model_validate(wire) == request


@pytest.mark.parametrize("name", ["ash_rollout_max_model_calls", "ash_rollout_max_tool_calls"])
def test_legacy_cli_budgets_are_rejected_for_messages(name):
    with pytest.raises(ValueError, match="v2-only"):
        AshMessageRolloutFn(RolloutFnConstructorInput(args=args(**{name: 100}), data_source=None))


@pytest.mark.parametrize("limit", [0, -1, True, 2.5])
def test_turn_limit_must_be_a_positive_integer(limit):
    with pytest.raises(ValueError, match="positive integer"):
        AshMessageRolloutFn(RolloutFnConstructorInput(args=args(ash_rollout_max_turns=limit), data_source=None))


def test_omitted_cli_turn_limit_defaults_to_64():
    fn = AshMessageRolloutFn(RolloutFnConstructorInput(args=args(ash_rollout_max_turns=None), data_source=None))
    request, _ = fn._build_request(group=group(), rollout_id=1, weight_version=7)
    assert request.max_turns == 64
    body = request.model_dump(mode="json")
    for name in ("max_model_calls", "max_tool_calls"):
        with pytest.raises(ValidationError):
            AshMessageRequest.model_validate({**body, "budgets": {**body["budgets"], name: 100}})
    with pytest.raises(ValidationError):
        AshMessageRequest.model_validate({**body, "max_turns": None})


def test_message_training_requires_rl_forward_path():
    with pytest.raises(ValueError, match="normal RL"):
        AshMessageRolloutFn(
            RolloutFnConstructorInput(args=args(compute_advantages_and_returns=False), data_source=None)
        )


def test_group_roundtrip_uses_messages_without_session_server_or_token_metadata(mask_generator):
    sent, deleted = [], []

    def handler(request):
        if request.method == "POST":
            sent.append(json.loads(request.content))
            return httpx.Response(
                202,
                json={
                    "protocol_version": "ash-rollout-v3",
                    "rollout_job_id": sent[0]["rollout_job_id"],
                    "status": "queued",
                },
            )
        if request.method == "DELETE":
            deleted.append(request.url.path)
            return httpx.Response(
                200,
                json={
                    "protocol_version": "ash-rollout-v3",
                    "rollout_job_id": sent[0]["rollout_job_id"],
                    "status": "completed",
                },
            )
        body = payload()
        body["rollout_job_id"] = sent[0]["rollout_job_id"]
        body["trajectories"][0]["sample_slot_id"] = sent[0]["sample_slots"][0]["sample_slot_id"]
        return httpx.Response(200, json=body)

    class Data:
        def get_samples(self, count):
            assert count == 1
            return [group()]

    def factory(*_args, **_kwargs):
        return AshMessageClient(
            "http://ash",
            client=httpx.AsyncClient(
                base_url="http://ash",
                transport=httpx.MockTransport(handler),
            ),
        )

    fn = AshMessageRolloutFn(RolloutFnConstructorInput(args=args(), data_source=Data()), client_factory=factory)
    fn._mask_generator = mask_generator
    result = asyncio.run(fn(RolloutFnTrainInput(rollout_id=5, weight_version=7)))
    assert result.samples[0][0].rollout_log_probs is None
    assert result.samples[0][0].reward == 1
    assert len(deleted) == 1
    request = AshMessageRequest.model_validate(sent[0])
    assert request.image == "prime/primeintellect/task:base"
    assert request.sampling_params["top_k"] == 20
    assert request.max_turns == 10
    assert sent[0]["budgets"] == {"max_wall_time_seconds": 10.0}
    assert not {"prompt_token_ids", "environment_ref", "session_server_endpoint"} & sent[0].keys()


def test_group_wait_includes_time_for_grading_after_execution_limit(mask_generator):
    requested_timeouts = []
    real_wait_for = asyncio.wait_for

    async def wait_for(awaitable, timeout):
        requested_timeouts.append(timeout)
        return await real_wait_for(awaitable, timeout=timeout)

    class Client:
        async def submit(self, request):
            self.request = request
            return Namespace(rollout_job_id=request.rollout_job_id)

        async def wait_for_result(self, job_id, **kwargs):
            await asyncio.sleep(0.02)
            body = payload()
            body.update(rollout_job_id=job_id, prompt_group_id=self.request.prompt_group_id)
            body["trajectories"][0].update(
                sample_slot_id=self.request.sample_slots[0].sample_slot_id,
                status="truncated",
                stop_reason="timeout",
            )
            return AshMessageResult.model_validate(body)

        async def delete(self, job_id):
            pass

    fn = AshMessageRolloutFn(
        RolloutFnConstructorInput(
            args=args(ash_rollout_timeout_seconds=0.005, ash_rollout_finalization_timeout_seconds=1.0),
            data_source=None,
        )
    )
    fn._mask_generator = mask_generator
    with patch("miles.rollout.ash.message_rollout.asyncio.wait_for", wait_for):
        samples, _ = asyncio.run(fn._run_group(client=Client(), group=group(), rollout_id=0, weight_version=1))
    assert samples[0].status == Sample.Status.TRUNCATED
    assert requested_timeouts == [1.005]


def test_message_request_uses_served_name_and_accepts_text_only_vlm_metadata():
    fn = AshMessageRolloutFn(
        RolloutFnConstructorInput(
            args=args(model_name="qwen3_5config", sglang_served_model_name="Qwen/Qwen3.8-27B"),
            data_source=None,
        )
    )
    samples = group()
    samples[0].multimodal_inputs = {"images": None, "videos": None}
    request, _ = fn._build_request(group=samples, rollout_id=1, weight_version=7)
    assert request.model == "Qwen/Qwen3.8-27B"
    samples[0].multimodal_inputs["images"] = [object()]
    with pytest.raises(ValueError, match="text messages only"):
        fn._build_request(group=samples, rollout_id=1, weight_version=7)
