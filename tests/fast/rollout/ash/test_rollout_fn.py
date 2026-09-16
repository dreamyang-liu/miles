import asyncio
import json
from argparse import Namespace

import httpx
import pytest

from miles.rollout.ash.client import AshRolloutClient
from miles.rollout.ash.rollout_fn import AshRolloutFn
from miles.rollout.base_types import RolloutFnConstructorInput, RolloutFnEvalInput, RolloutFnTrainInput
from miles.utils.types import Sample


class FakeDataSource:
    def __init__(self, groups):
        self.groups = groups
        self.requests = []

    def get_samples(self, num_samples):
        self.requests.append(num_samples)
        return self.groups


def _args(**overrides):
    values = {
        "rollout_batch_size": 1,
        "n_samples_per_prompt": 2,
        "sglang_router_ip": "miles-router",
        "sglang_router_port": 30000,
        "rollout_temperature": 0.8,
        "rollout_top_p": 0.95,
        "rollout_top_k": 20,
        "rollout_max_response_len": 128,
        "rollout_stop": None,
        "rollout_stop_token_ids": None,
        "rollout_skip_special_tokens": False,
        "ash_rollout_base_url": "http://ash",
        "ash_rollout_model_endpoint": None,
        "ash_rollout_poll_interval_seconds": 0.001,
        "ash_rollout_timeout_seconds": 1.0,
        "ash_rollout_client_grace_seconds": 0.1,
        "ash_rollout_http_timeout_seconds": 2.0,
        "ash_rollout_max_model_calls": 12,
        "ash_rollout_max_tool_calls": 8,
        "use_rollout_logprobs": False,
        "hf_checkpoint": "unused-when-samples-have-tokens",
        "chat_template_path": None,
        "apply_chat_template_kwargs": {},
    }
    values.update(overrides)
    return Namespace(**values)


def _group():
    metadata = {
        "task_id": "swebench__repo-123",
        "environment_ref": {
            "kind": "template",
            "id": "swebench-runtime",
            "revision": "sha256:test",
            "resource_profile": "standard",
        },
    }
    return [
        Sample(
            prompt=[{"role": "user", "content": "fix it"}],
            tokens=[10],
            group_index=3,
            index=index,
            metadata=metadata,
        )
        for index in (11, 12)
    ]


def test_v2_retains_legacy_default_call_budgets():
    args = _args()
    del args.ash_rollout_max_model_calls
    del args.ash_rollout_max_tool_calls
    fn = AshRolloutFn(RolloutFnConstructorInput(args=args, data_source=None))
    request, _ = fn._build_request(group=_group(), rollout_id=0, weight_version=7)
    assert request.budgets.max_model_calls == request.budgets.max_tool_calls == 100


def test_v2_preserves_explicit_unbounded_call_budgets():
    fn = AshRolloutFn(
        RolloutFnConstructorInput(
            args=_args(ash_rollout_max_model_calls=None, ash_rollout_max_tool_calls=None),
            data_source=None,
        )
    )
    request, _ = fn._build_request(group=_group(), rollout_id=0, weight_version=7)
    assert request.budgets.max_model_calls is None
    assert request.budgets.max_tool_calls is None


def test_v2_rejects_the_v3_turn_limit_flag():
    with pytest.raises(ValueError, match="requires AshMessageRolloutFn"):
        AshRolloutFn(RolloutFnConstructorInput(args=_args(ash_rollout_max_turns=2), data_source=None))


def _trajectory(slot, *, token, include_log_probs):
    span = {
        "response_id": f"response-{slot['sample_index']}",
        "start": 1,
        "end": 2,
        "input_token_ids": [10],
        "output_token_ids": [token],
        "weight_version": "7",
        "finish_reason": "stop",
    }
    if include_log_probs:
        span["output_token_log_probs"] = [-0.25]
    return {
        "sample_slot_id": slot["sample_slot_id"],
        "branch_id": f"branch-{slot['sample_index']}",
        "messages": [
            {"role": "user", "content": "fix it"},
            {"role": "assistant", "content": "fixed"},
        ],
        "token_ids": [10, token],
        "prompt_length": 1,
        "generated_spans": [span],
        "response_text": "fixed",
        "reward": 1.0,
        "status": "completed",
    }


def test_rollout_fn_submits_group_polls_and_imports_samples():
    submitted_request = None
    polls = 0
    deleted = []

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal submitted_request, polls
        if request.method == "POST":
            submitted_request = json.loads(request.content)
            return httpx.Response(
                202,
                json={
                    "protocol_version": "ash-rollout-v2",
                    "rollout_job_id": submitted_request["rollout_job_id"],
                    "status": "queued",
                },
            )

        if request.method == "DELETE":
            deleted.append(request.url.path)
            return httpx.Response(200, json=_deletion(submitted_request, status="completed"))
        polls += 1
        if polls == 1:
            return httpx.Response(200, json=_result(submitted_request, status="running"))
        return httpx.Response(200, json=_result(submitted_request, status="completed", with_trajectories=True))

    output, data_source = _run_rollout(handler)

    assert data_source.requests == [1]
    assert submitted_request["prompt_group_id"] == "3"
    assert submitted_request["task_id"] == "swebench__repo-123"
    assert submitted_request["environment_ref"] == {
        "kind": "template",
        "id": "swebench-runtime",
        "revision": "sha256:test",
        "resource_profile": "standard",
    }
    assert submitted_request["max_samples"] == 2
    assert submitted_request["minimum_returned_samples"] == 2
    assert submitted_request["prompt_token_ids"] == [10]
    assert submitted_request["model_endpoint"] == "http://miles-router:30000"
    assert submitted_request["expected_weight_version"] == "7"
    assert submitted_request["return_rollout_logprobs"] is False
    assert submitted_request["sampling_params"]["temperature"] == 0.8
    assert [sample.index for sample in output.samples[0]] == [11, 12]
    assert all(sample.status == Sample.Status.COMPLETED for sample in output.samples[0])
    assert all(sample.rollout_log_probs is None for sample in output.samples[0])
    assert deleted == [f"/rollout-groups/{submitted_request['rollout_job_id']}"]
    assert output.metrics == {
        "rollout/ash/groups": 1,
        "rollout/ash/samples": 2,
        "rollout/ash/search_branches": 3,
        "rollout/ash/model_calls": 2,
        "rollout/ash/tool_calls": 1,
    }


def test_rollout_fn_requests_and_imports_rollout_log_probs_when_enabled():
    submitted_request = None

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal submitted_request
        if request.method == "POST":
            submitted_request = json.loads(request.content)
            return httpx.Response(
                202,
                json={
                    "protocol_version": "ash-rollout-v2",
                    "rollout_job_id": submitted_request["rollout_job_id"],
                    "status": "queued",
                },
            )
        if request.method == "DELETE":
            return httpx.Response(200, json=_deletion(submitted_request, status="completed"))
        return httpx.Response(200, json=_result(submitted_request, status="completed", with_trajectories=True))

    output, _data_source = _run_rollout(handler, use_rollout_logprobs=True)

    assert submitted_request["return_rollout_logprobs"] is True
    assert [sample.rollout_log_probs for sample in output.samples[0]] == [[-0.25], [-0.25]]


def test_rollout_fn_forwards_chat_template_kwargs_to_ash():
    submitted_request = None

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal submitted_request
        if request.method == "POST":
            submitted_request = json.loads(request.content)
            return httpx.Response(
                202,
                json={
                    "protocol_version": "ash-rollout-v2",
                    "rollout_job_id": submitted_request["rollout_job_id"],
                    "status": "queued",
                },
            )
        if request.method == "DELETE":
            return httpx.Response(200, json=_deletion(submitted_request, status="completed"))
        return httpx.Response(200, json=_result(submitted_request, status="completed", with_trajectories=True))

    _run_rollout(handler, apply_chat_template_kwargs={"enable_thinking": False})

    assert submitted_request["sampling_params"]["chat_template_kwargs"] == {
        "enable_thinking": False
    }


def test_rollout_fn_scores_only_trajectories_without_ash_reward(monkeypatch):
    submitted_request = None
    scored_indices = []

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal submitted_request
        if request.method == "POST":
            submitted_request = json.loads(request.content)
            return httpx.Response(
                202,
                json={
                    "protocol_version": "ash-rollout-v2",
                    "rollout_job_id": submitted_request["rollout_job_id"],
                    "status": "queued",
                },
            )
        if request.method == "DELETE":
            return httpx.Response(200, json=_deletion(submitted_request, status="completed"))
        result = _result(submitted_request, status="completed", with_trajectories=True)
        result["trajectories"][0]["reward"] = None
        result["trajectories"][1]["reward"] = 0.75
        return httpx.Response(200, json=result)

    async def fake_batched_async_rm(_args, samples, *, inplace_set_reward_field):
        assert inplace_set_reward_field is True
        scored_indices.extend(sample.index for sample in samples)
        for sample in samples:
            sample.reward = 0.25

    monkeypatch.setattr("miles.rollout.ash.rollout_fn.batched_async_rm", fake_batched_async_rm)

    output, _data_source = _run_rollout(handler)

    assert scored_indices == [11]
    assert [sample.reward for sample in output.samples[0]] == [0.25, 0.75]


def test_rollout_fn_rejects_missing_requested_rollout_log_probs():
    submitted_request = None

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal submitted_request
        if request.method == "POST":
            submitted_request = json.loads(request.content)
            return httpx.Response(
                202,
                json={
                    "protocol_version": "ash-rollout-v2",
                    "rollout_job_id": submitted_request["rollout_job_id"],
                    "status": "queued",
                },
            )
        if request.method == "DELETE":
            return httpx.Response(200, json=_deletion(submitted_request, status="cancelled"))
        result = _result(submitted_request, status="completed", with_trajectories=True)
        for trajectory in result["trajectories"]:
            for span in trajectory["generated_spans"]:
                span.pop("output_token_log_probs")
        return httpx.Response(200, json=result)

    with pytest.raises(ValueError, match="omitted output_token_log_probs requested by Miles"):
        _run_rollout(handler, use_rollout_logprobs=True)


def test_rollout_fn_cancels_remote_job_after_timeout():
    submitted_request = None
    cancelled = []

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal submitted_request
        if request.method == "POST":
            submitted_request = json.loads(request.content)
            return httpx.Response(
                202,
                json={
                    "protocol_version": "ash-rollout-v2",
                    "rollout_job_id": submitted_request["rollout_job_id"],
                    "status": "running",
                },
            )
        if request.method == "DELETE":
            cancelled.append(request.url.path)
            return httpx.Response(200, json=_deletion(submitted_request, status="cancelled"))
        return httpx.Response(200, json=_result(submitted_request, status="running"))

    with pytest.raises(TimeoutError):
        _run_rollout(handler, ash_rollout_timeout_seconds=0.01)

    assert cancelled == [f"/rollout-groups/{submitted_request['rollout_job_id']}"]


def test_rollout_fn_deletes_job_when_submit_response_is_lost():
    submitted_request = None
    deleted = []

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal submitted_request
        if request.method == "POST":
            submitted_request = json.loads(request.content)
            raise httpx.ReadError("response lost after server accepted job", request=request)
        if request.method == "DELETE":
            deleted.append(request.url.path)
            return httpx.Response(200, json=_deletion(submitted_request, status="cancelled"))
        raise AssertionError(f"unexpected request: {request.method} {request.url.path}")

    with pytest.raises(httpx.ReadError, match="response lost"):
        _run_rollout(handler)

    assert deleted == [f"/rollout-groups/{submitted_request['rollout_job_id']}"]


def test_rollout_fn_reports_remote_failure_reason():
    submitted_request = None

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal submitted_request
        if request.method == "POST":
            submitted_request = json.loads(request.content)
            return httpx.Response(
                202,
                json={
                    "protocol_version": "ash-rollout-v2",
                    "rollout_job_id": submitted_request["rollout_job_id"],
                    "status": "queued",
                },
            )
        if request.method == "DELETE":
            return httpx.Response(200, json=_deletion(submitted_request, status="failed"))
        result = _result(submitted_request, status="failed")
        result["stop_reason"] = "RuntimeError: sandbox startup failed"
        return httpx.Response(200, json=result)

    with pytest.raises(RuntimeError, match="sandbox startup failed"):
        _run_rollout(handler)


def test_rollout_fn_rejects_partial_group_until_training_supports_it():
    submitted_request = None

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal submitted_request
        if request.method == "POST":
            submitted_request = json.loads(request.content)
            return httpx.Response(
                202,
                json={
                    "protocol_version": "ash-rollout-v2",
                    "rollout_job_id": submitted_request["rollout_job_id"],
                    "status": "queued",
                },
            )
        if request.method == "DELETE":
            return httpx.Response(200, json=_deletion(submitted_request, status="cancelled"))
        result = _result(submitted_request, status="early_stopped", with_trajectories=True)
        result["actual_samples"] = 1
        result["trajectories"] = result["trajectories"][:1]
        return httpx.Response(200, json=result)

    with pytest.raises(ValueError, match="K < N support is not enabled"):
        _run_rollout(handler)


def test_rollout_fn_rejects_a_different_weight_version():
    submitted_request = None

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal submitted_request
        if request.method == "POST":
            submitted_request = json.loads(request.content)
            return httpx.Response(
                202,
                json={
                    "protocol_version": "ash-rollout-v2",
                    "rollout_job_id": submitted_request["rollout_job_id"],
                    "status": "queued",
                },
            )
        if request.method == "DELETE":
            return httpx.Response(200, json=_deletion(submitted_request, status="cancelled"))
        result = _result(submitted_request, status="completed", with_trajectories=True)
        result["trajectories"][0]["generated_spans"][0]["weight_version"] = "8"
        return httpx.Response(200, json=result)

    with pytest.raises(ValueError, match="expected only '7'"):
        _run_rollout(handler)


def test_rollout_fn_rejects_a_different_job_id():
    submitted_request = None

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal submitted_request
        if request.method == "POST":
            submitted_request = json.loads(request.content)
            return httpx.Response(
                202,
                json={
                    "protocol_version": "ash-rollout-v2",
                    "rollout_job_id": submitted_request["rollout_job_id"],
                    "status": "queued",
                },
            )
        if request.method == "DELETE":
            return httpx.Response(200, json=_deletion(submitted_request, status="cancelled"))
        result = _result(submitted_request, status="completed", with_trajectories=True)
        result["rollout_job_id"] = "another-job"
        return httpx.Response(200, json=result)

    with pytest.raises(ValueError, match="returned rollout_job_id='another-job'"):
        _run_rollout(handler)


def test_rollout_fn_cancels_remote_job_when_local_task_is_cancelled():
    submitted = asyncio.Event()
    cancelled = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            payload = json.loads(request.content)
            submitted.request = payload
            submitted.set()
            return httpx.Response(
                202,
                json={
                    "protocol_version": "ash-rollout-v2",
                    "rollout_job_id": payload["rollout_job_id"],
                    "status": "running",
                },
            )
        if request.method == "DELETE":
            cancelled.append(request.url.path)
            return httpx.Response(200, json=_deletion(submitted.request, status="cancelled"))
        return httpx.Response(200, json=_result(submitted.request, status="running"))

    async def exercise_cancellation():
        data_source = FakeDataSource([_group()])
        async_client = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://ash")

        def client_factory(base_url, **_kwargs):
            return AshRolloutClient(base_url, client=async_client)

        fn = AshRolloutFn(
            RolloutFnConstructorInput(args=_args(), data_source=data_source),
            client_factory=client_factory,
        )
        task = asyncio.create_task(fn(RolloutFnTrainInput(rollout_id=4, weight_version=7)))
        try:
            await submitted.wait()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        finally:
            await async_client.aclose()

    asyncio.run(exercise_cancellation())

    assert cancelled == [f"/rollout-groups/{submitted.request['rollout_job_id']}"]


def test_rollout_fn_requires_separate_evaluation_function():
    fn = AshRolloutFn(RolloutFnConstructorInput(args=_args(), data_source=FakeDataSource([_group()])))

    with pytest.raises(NotImplementedError, match="eval-function-path"):
        asyncio.run(fn(RolloutFnEvalInput(rollout_id=1)))


def test_rollout_fn_rejects_mismatched_prompt_token_ids():
    group = _group()
    group[1].tokens = [11]
    fn = AshRolloutFn(RolloutFnConstructorInput(args=_args(), data_source=FakeDataSource([group])))

    with pytest.raises(ValueError, match="same prompt token IDs"):
        fn._build_request(group=group, rollout_id=4, weight_version=7)


def test_rollout_fn_requires_trusted_environment_metadata():
    group = _group()
    group[0].metadata = {}
    fn = AshRolloutFn(RolloutFnConstructorInput(args=_args(), data_source=FakeDataSource([group])))

    with pytest.raises(ValueError, match=r"metadata\['task_id'\]"):
        fn._build_request(group=group, rollout_id=4, weight_version=7)


def test_rollout_fn_rejects_group_environment_mismatch():
    group = _group()
    group[1].metadata = {
        **group[1].metadata,
        "environment_ref": {
            **group[1].metadata["environment_ref"],
            "revision": "sha256:other",
        },
    }
    fn = AshRolloutFn(RolloutFnConstructorInput(args=_args(), data_source=FakeDataSource([group])))

    with pytest.raises(ValueError, match="same task_id and environment_ref"):
        fn._build_request(group=group, rollout_id=4, weight_version=7)


def test_rollout_fn_tokenizes_prompt_when_data_source_has_no_tokens(monkeypatch):
    class FakeTokenizer:
        def encode(self, text, *, add_special_tokens):
            assert text == "rendered prompt"
            assert add_special_tokens is False
            return [31, 32]

    monkeypatch.setattr("miles.rollout.ash.rollout_fn.load_tokenizer", lambda *_args, **_kwargs: FakeTokenizer())
    group = [
        Sample(
            prompt="rendered prompt",
            group_index=3,
            index=index,
            metadata=_group()[0].metadata,
        )
        for index in (11, 12)
    ]
    fn = AshRolloutFn(RolloutFnConstructorInput(args=_args(), data_source=FakeDataSource([group])))

    request, _slot_samples = fn._build_request(group=group, rollout_id=4, weight_version=7)

    assert request.prompt_token_ids == [31, 32]


def test_rollout_fn_tokenizes_text_messages_with_empty_multimodal_placeholders(monkeypatch):
    class FakeTokenizer:
        pass

    def fake_apply_chat_template(
        messages,
        *,
        tokenizer,
        tools,
        tokenize,
        add_generation_prompt,
        enable_thinking,
    ):
        assert messages == [{"role": "user", "content": "fix it"}]
        assert isinstance(tokenizer, FakeTokenizer)
        assert tools is None
        assert tokenize is True
        assert add_generation_prompt is True
        assert enable_thinking is False
        return [41, 42]

    monkeypatch.setattr("miles.rollout.ash.rollout_fn.load_tokenizer", lambda *_args, **_kwargs: FakeTokenizer())
    monkeypatch.setattr(
        "miles.utils.chat_template_utils.apply_chat_template",
        fake_apply_chat_template,
    )
    group = _group()
    for sample in group:
        sample.tokens = []
        sample.multimodal_inputs = {"images": None, "videos": None}
    fn = AshRolloutFn(
        RolloutFnConstructorInput(
            args=_args(apply_chat_template_kwargs={"enable_thinking": False}),
            data_source=FakeDataSource([group]),
        )
    )

    request, _slot_samples = fn._build_request(group=group, rollout_id=4, weight_version=7)

    assert request.prompt_token_ids == [41, 42]


def test_rollout_fn_rejects_actual_multimodal_prompt_without_tokens():
    group = _group()
    for sample in group:
        sample.tokens = []
        sample.multimodal_inputs = {"images": [object()], "videos": None}
    fn = AshRolloutFn(RolloutFnConstructorInput(args=_args(), data_source=FakeDataSource([group])))

    with pytest.raises(NotImplementedError, match="multimodal"):
        fn._build_request(group=group, rollout_id=4, weight_version=7)


def test_rollout_fn_resolves_model_endpoint_after_router_startup():
    args = _args(sglang_router_ip=None, sglang_router_port=None)
    group = _group()
    fn = AshRolloutFn(RolloutFnConstructorInput(args=args, data_source=FakeDataSource([group])))

    args.sglang_router_ip = "10.0.0.4"
    args.sglang_router_port = 3210
    request, _slot_samples = fn._build_request(group=group, rollout_id=4, weight_version=7)

    assert request.model_endpoint == "http://10.0.0.4:3210"


def test_rollout_fn_carries_v2_session_server_endpoint():
    args = _args(
        use_session_server="v2",
        session_server_ip="10.0.0.8",
        session_server_ports=[31001, 31002],
    )
    fn = AshRolloutFn(RolloutFnConstructorInput(args=args, data_source=FakeDataSource([_group()])))
    request, _slot_samples = fn._build_request(group=_group(), rollout_id=4, weight_version=7)

    assert request.session_server_endpoint == "http://10.0.0.8:31001"


def _run_rollout(handler, **arg_overrides):
    data_source = FakeDataSource([_group()])
    async_client = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://ash")

    def client_factory(base_url, **_kwargs):
        return AshRolloutClient(base_url, client=async_client)

    fn = AshRolloutFn(
        RolloutFnConstructorInput(args=_args(**arg_overrides), data_source=data_source),
        client_factory=client_factory,
    )

    async def run():
        try:
            return await fn(RolloutFnTrainInput(rollout_id=4, weight_version=7))
        finally:
            await async_client.aclose()

    return asyncio.run(run()), data_source


def _result(request, *, status, with_trajectories=False):
    trajectories = []
    if with_trajectories:
        trajectories = [
            _trajectory(
                slot,
                token=20 + position,
                include_log_probs=request["return_rollout_logprobs"],
            )
            for position, slot in enumerate(request["sample_slots"])
        ]
    return {
        "protocol_version": "ash-rollout-v2",
        "rollout_job_id": request["rollout_job_id"],
        "prompt_group_id": request["prompt_group_id"],
        "status": status,
        "max_samples": request["max_samples"],
        "actual_samples": len(trajectories),
        "search_branches": 3,
        "consumed_budget": {"model_calls": 2, "tool_calls": 1},
        "trajectories": trajectories,
    }


def _deletion(request, *, status):
    return {
        "protocol_version": "ash-rollout-v2",
        "rollout_job_id": request["rollout_job_id"],
        "status": status,
    }
