import argparse
import asyncio
import json
import sys
from argparse import Namespace
from unittest.mock import patch

import httpx
import pytest
from tests.fast.rollout.sprout.conftest import args, group, sprout_request, sprout_result

from miles.rollout.base_types import RolloutFnConstructorInput, RolloutFnEvalInput, RolloutFnTrainInput
from miles.rollout.sprout.client import SproutDriverError, SproutRolloutClient
from miles.rollout.sprout.protocol import RolloutRequest, RolloutResult
from miles.rollout.sprout.rollout_fn import GSML_POST_PROCESS, SproutRolloutFn, collect_metrics
from miles.utils.types import Sample


def construct(namespace: Namespace, data_source=None, **kwargs) -> SproutRolloutFn:
    return SproutRolloutFn(RolloutFnConstructorInput(args=namespace, data_source=data_source), **kwargs)


def without_ids(wire: dict) -> dict:
    return {key: value for key, value in wire.items() if key not in {"rollout_job_id", "sample_slots"}}


def test_built_request_is_sprouts_documented_request():
    """Everything but the fresh job id and the slot ids derived from it equals
    the example Sprout ships, so the two repositories document one wire."""
    request, slots = construct(args())._build_request(group=group(), rollout_id=1)
    wire = request.wire()
    example = sprout_request()
    assert without_ids(wire) == without_ids({**example, "finalization_timeout_seconds": 1800.0})
    assert wire["rollout_job_id"].startswith("miles-1-3-")
    assert wire["sample_slots"] == [{"sample_slot_id": wire["rollout_job_id"] + ":slot:11", "sample_index": 11}]
    assert (
        list(slots) == [wire["rollout_job_id"] + ":slot:11"]
        and slots[wire["sample_slots"][0]["sample_slot_id"]].index == 11
    )


@pytest.mark.parametrize("size", [1, 2, 8])
def test_a_prompt_group_is_one_fixed_size_request(size):
    request, slots = construct(args(n_samples_per_prompt=size))._build_request(group=group(size), rollout_id=4)
    assert request.max_samples == request.minimum_returned_samples == len(slots) == size
    assert [slot.sample_index for slot in request.sample_slots] == list(range(11, 11 + size))
    assert request.max_turns == 10 and request.budgets.max_wall_time_seconds == 10.0


def test_group_shape_is_checked_against_miles_arguments():
    with pytest.raises(ValueError, match="n_samples_per_prompt=2"):
        construct(args(n_samples_per_prompt=2))._build_request(group=group(1), rollout_id=0)
    samples = group(2)
    samples[1].group_index = 4
    with pytest.raises(ValueError, match="group_index"):
        construct(args(n_samples_per_prompt=2))._build_request(group=samples, rollout_id=0)
    samples = group(2)
    samples[1].metadata["image"] = "other:image"
    with pytest.raises(ValueError, match="share task_id and image"):
        construct(args(n_samples_per_prompt=2))._build_request(group=samples, rollout_id=0)
    samples = group()
    samples[0].metadata = {}
    with pytest.raises(ValueError, match=r"metadata\['task_id'\]"):
        construct(args())._build_request(group=samples, rollout_id=0)


def test_sampling_params_carry_what_miles_set_and_always_top_k():
    request, _ = construct(args(rollout_stop=None, rollout_max_response_len=None))._build_request(
        group=group(), rollout_id=0
    )
    assert request.sampling_params == {
        "temperature": 0.6,
        "top_p": 1.0,
        "top_k": -1,
    }, "off is sent too: left out, the engine takes generation_config's"


def test_model_endpoint_defaults_to_the_rollout_router():
    fn = construct(args(sprout_rollout_model_endpoint=None, sglang_router_ip="router", sglang_router_port=30000))
    assert fn.model_endpoint() == "http://router:30000"
    with pytest.raises(RuntimeError, match="router address is not known"):
        construct(args(sprout_rollout_model_endpoint=None)).model_endpoint()
    assert construct(args(sprout_rollout_model_endpoint="http://m:1/")).model_endpoint() == "http://m:1"


def test_model_name_prefers_the_served_name():
    assert construct(args(sglang_served_model_name="Qwen/Qwen3-8B")).model_name() == "Qwen/Qwen3-8B"
    assert construct(args()).model_name() == "policy"
    assert construct(args(model_name=None)).model_name() is None


@pytest.mark.parametrize(
    "updates, message",
    [
        ({"sprout_rollout_base_url": None}, "base-url is required"),
        ({"use_rollout_logprobs": True}, "incompatible with retokenized"),
        ({"skip_actor_forward_only": True}, "incompatible with retokenized"),
        ({"use_tis": True}, "incompatible with retokenized"),
        ({"use_rollout_routing_replay": True}, "incompatible with retokenized"),
        ({"use_sampling_support_replay": True}, "rollout-top-k\\) is incompatible with retokenized"),
        ({"compute_advantages_and_returns": False}, "advantage path"),
        ({"rollout_stop_token_ids": [7]}, "stop token ids"),
        ({"apply_chat_template_kwargs": {"enable_thinking": False}}, "cannot reach it"),
        ({"sprout_rollout_max_turns": 0}, "positive integer"),
        ({"sprout_rollout_max_turns": 2.5}, "positive integer"),
        ({"sprout_rollout_timeout_seconds": 0}, "finite and positive"),
        ({"sprout_rollout_poll_interval_seconds": float("nan")}, "finite and positive"),
        ({"n_samples_per_prompt": 0}, "positive integer"),
        ({"custom_reward_post_process_path": GSML_POST_PROCESS}, "credits search groups only"),
    ],
)
def test_configuration_that_cannot_be_honoured_is_refused_at_construction(updates, message):
    with pytest.raises(ValueError, match=message):
        construct(args(**updates))


def test_evaluation_is_not_served():
    fn = construct(args())
    with pytest.raises(NotImplementedError):
        asyncio.run(fn(RolloutFnEvalInput(rollout_id=0)))


class FakeDriver:
    """Sprout's RL driver over httpx.MockTransport: accepts, runs for a few
    polls, then returns the shipped result example in the slots Miles named."""

    def __init__(self, *, polls_until_done=2, terminal_status="completed", stop_reason=None, reward=1.0):
        self.requests, self.deleted, self.polls = [], [], 0
        self.polls_until_done, self.terminal_status, self.stop_reason, self.reward = (
            polls_until_done,
            terminal_status,
            stop_reason,
            reward,
        )
        self.headers = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.headers.append(dict(request.headers))
        if request.method == "POST":
            body = json.loads(request.content)
            self.requests.append(body)
            return httpx.Response(202, json={"rollout_job_id": body["rollout_job_id"], "status": "queued"})
        job_id = request.url.path.rsplit("/", 1)[1]
        if request.method == "DELETE":
            self.deleted.append(job_id)
            return httpx.Response(200, json={"rollout_job_id": job_id, "status": self.terminal_status})
        submitted = next(body for body in self.requests if body["rollout_job_id"] == job_id)
        self.polls += 1
        result = sprout_result()
        result.update(
            rollout_job_id=job_id,
            prompt_group_id=submitted["prompt_group_id"],
            max_samples=submitted["max_samples"],
            trajectories=[],
            actual_samples=0,
        )
        if self.polls < self.polls_until_done:
            result["status"] = "running"
            return httpx.Response(200, json=result)
        result["status"] = self.terminal_status
        result["stop_reason"] = self.stop_reason
        if self.terminal_status in {"completed", "early_stopped"}:
            template = sprout_result()["trajectories"][0]
            result["trajectories"] = [
                {
                    **template,
                    "sample_slot_id": slot["sample_slot_id"],
                    "branch_id": f"job-{slot['sample_index']}",
                    "reward": self.reward,
                }
                for slot in submitted["sample_slots"]
            ]
            result["actual_samples"] = len(result["trajectories"])
            result["consumed_budget"] = {"model_calls": 2 * len(result["trajectories"]), "tool_calls": 1}
        return httpx.Response(200, json=result)

    def factory(self, base_url, *, timeout, token=None):
        headers = {"Authorization": f"Bearer {token}"} if token else None
        return SproutRolloutClient(
            base_url,
            client=httpx.AsyncClient(base_url=base_url, transport=httpx.MockTransport(self.handler), headers=headers),
        )


class Data:
    def __init__(self, groups):
        self.groups, self.asked = groups, []

    def get_samples(self, count):
        self.asked.append(count)
        return self.groups


def test_group_round_trip_submits_waits_imports_and_releases(mask_generator, monkeypatch):
    monkeypatch.setenv("SPROUT_RL_DRIVER_TOKEN", "secret")
    driver = FakeDriver()
    data = Data([group(2)])
    fn = construct(args(n_samples_per_prompt=2), data, client_factory=driver.factory)
    fn._mask_generator = mask_generator
    output = asyncio.run(fn(RolloutFnTrainInput(rollout_id=5, weight_version=7)))
    assert data.asked == [1]
    assert len(output.samples) == 1 and len(output.samples[0]) == 2, "one group, n_samples_per_prompt samples"
    first, second = output.samples[0]
    assert first.index == 11 and second.index == 12 and first.group_index == second.group_index == 3
    assert first.reward == 1.0 and first.status == Sample.Status.COMPLETED and first.rollout_log_probs is None
    assert (
        first.metadata["sprout_rollout"]["branch_id"] == "job-11"
        and first.metadata["sprout_rollout"]["weight_version"] == 7
    )
    assert output.metrics["rollout/sprout/samples"] == 2 and output.metrics["rollout/sprout/model_calls"] == 4
    assert output.metrics["rollout/sprout/reward_mean"] == 1.0 and output.metrics["rollout/sprout/truncated"] == 0
    [sent] = driver.requests
    request = RolloutRequest.model_validate(sent)
    assert request.rollout_id == 5 and request.max_samples == 2 and request.image == "prime/primeintellect/task:base"
    assert driver.deleted == [sent["rollout_job_id"]], "the group is released exactly once"
    assert all(header.get("authorization") == "Bearer secret" for header in driver.headers)


def test_failed_group_raises_with_the_drivers_reason_and_is_still_released(mask_generator):
    driver = FakeDriver(terminal_status="failed", stop_reason="slot: Actor did not complete")
    fn = construct(args(), Data([group()]), client_factory=driver.factory)
    fn._mask_generator = mask_generator
    with pytest.raises(RuntimeError, match="status='failed': slot: Actor did not complete"):
        asyncio.run(fn(RolloutFnTrainInput(rollout_id=0)))
    assert len(driver.deleted) == 1


def test_driver_rejection_surfaces_its_body(mask_generator):
    def handler(request):
        return httpx.Response(
            400, json={"detail": "Configure Sprout tasks['task-1'].grade for message rollout rewards"}
        )

    def factory(base_url, *, timeout, token=None):
        return SproutRolloutClient(
            base_url, client=httpx.AsyncClient(base_url=base_url, transport=httpx.MockTransport(handler))
        )

    fn = construct(args(), Data([group()]), client_factory=factory)
    fn._mask_generator = mask_generator
    with pytest.raises(SproutDriverError, match="HTTP 400.*tasks\\['task-1'\\]"):
        asyncio.run(fn(RolloutFnTrainInput(rollout_id=0)))


class OneGroupFails(FakeDriver):
    """Prompt group 4 fails: Sprout rejects its submission, or its rollout ends ``failed``."""

    def __init__(self, failure: str) -> None:
        super().__init__(polls_until_done=1)
        self.failure = failure

    def handler(self, request: httpx.Request) -> httpx.Response:
        if request.method == "POST" and self.failure == "rejected":
            body = json.loads(request.content)
            if body["prompt_group_id"] == "4":
                self.requests.append(body)
                return httpx.Response(400, json={"detail": "Configure Sprout tasks['task-1'].grade"})
        response = super().handler(request)
        if request.method == "GET" and self.failure == "failed" and response.json()["prompt_group_id"] == "4":
            failed = {"status": "failed", "stop_reason": "slot: Actor did not complete"}
            return httpx.Response(200, json={**response.json(), **failed, "trajectories": [], "actual_samples": 0})
        return response


@pytest.mark.parametrize("failure", ["rejected", "failed"])
def test_a_failed_group_is_left_out_and_the_step_trains_the_others(mask_generator, caplog, failure):
    driver = OneGroupFails(failure)
    groups = [group(2, group_index=3, first_index=11), group(2, group_index=4, first_index=13)]
    fn = construct(args(rollout_batch_size=2, n_samples_per_prompt=2), Data(groups), client_factory=driver.factory)
    fn._mask_generator = mask_generator
    output = asyncio.run(fn(RolloutFnTrainInput(rollout_id=2)))
    assert [[s.index for s in g] for g in output.samples] == [[11, 12], []], "a failed group contributes nothing"
    assert output.metrics["rollout/sprout/groups"] == 1 and output.metrics["rollout/sprout/groups_failed"] == 1
    assert output.metrics["rollout/sprout/samples"] == 2 and output.metrics["rollout/sprout/reward_mean"] == 1.0
    assert sorted(driver.deleted) == sorted(body["rollout_job_id"] for body in driver.requests), "both released"
    assert len(driver.deleted) == 2
    (lost,) = [body["rollout_job_id"] for body in driver.requests if body["prompt_group_id"] == "4"]
    assert any(lost in record.getMessage() for record in caplog.records), "the log names the group's job"


def test_when_every_group_fails_the_first_groups_error_is_raised(mask_generator):
    driver = FakeDriver(polls_until_done=1, terminal_status="failed", stop_reason="slot: Actor did not complete")
    groups = [group(2, group_index=3, first_index=11), group(2, group_index=4, first_index=13)]
    fn = construct(args(rollout_batch_size=2, n_samples_per_prompt=2), Data(groups), client_factory=driver.factory)
    fn._mask_generator = mask_generator
    with pytest.raises(RuntimeError, match="'miles-0-3-[0-9a-f]+' ended with status='failed': slot: Actor did not"):
        asyncio.run(fn(RolloutFnTrainInput(rollout_id=0)))
    assert len(driver.requests) == len(driver.deleted) == 2


def test_metrics_leave_what_was_not_graded_out_of_the_reward_mean():
    body = sprout_result()
    graded = body["trajectories"][0]
    missing_grade = {"progress": None, "zero_reason": "grade_failed", "touched_test_paths": None}
    ungraded = {
        **graded,
        "sample_slot_id": "slot-2",
        "branch_id": "job-2",
        "reward": 0.0,
        "metadata": {**graded["metadata"], **missing_grade},
    }
    body.update(trajectories=[graded, ungraded], actual_samples=2, max_samples=3)
    body["failed_samples"] = [{"sample_slot_id": "slot-3", "reason": "Actor did not complete"}]
    metrics = collect_metrics([RolloutResult.model_validate(body)], groups_failed=1)
    assert metrics["rollout/sprout/reward_mean"] == 1.0, "a missing grade is no 0"
    assert metrics["rollout/sprout/ungradable"] == 1 and metrics["rollout/sprout/failed_samples"] == 1
    assert metrics["rollout/sprout/groups"] == 1 and metrics["rollout/sprout/groups_failed"] == 1
    assert metrics["rollout/sprout/samples"] == 2
    body.update(trajectories=[ungraded], actual_samples=1)
    assert "rollout/sprout/reward_mean" not in collect_metrics([RolloutResult.model_validate(body)])


class Abort(BaseException):
    """Not an ``Exception``: what a group must not swallow (an interrupt, a cancellation)."""


#: Waits that run out in seconds, for steps whose groups never finish: a regression fails, it does not hang.
QUICK = {"sprout_rollout_timeout_seconds": 2.5, "sprout_rollout_finalization_timeout_seconds": 2.5}


def test_an_interrupt_in_one_group_stops_every_group(mask_generator):
    class Driver(FakeDriver):
        def handler(self, request):
            if request.method == "POST" and json.loads(request.content)["prompt_group_id"] == "4":
                raise Abort("interrupted")
            return super().handler(request)

    driver = Driver(polls_until_done=10**9)
    groups = [group(2, group_index=3, first_index=11), group(2, group_index=4, first_index=13)]
    # Group 3 never finishes: should the interrupt be swallowed, its wait runs out in seconds, not an hour.
    namespace = args(rollout_batch_size=2, n_samples_per_prompt=2, **QUICK)
    fn = construct(namespace, Data(groups), client_factory=driver.factory)
    fn._mask_generator = mask_generator
    with pytest.raises(Abort):
        asyncio.run(fn(RolloutFnTrainInput(rollout_id=0)))
    # Group 3 was still running and is cancelled; group 4's submission may have reached Sprout.
    assert sorted(job.split("-")[2] for job in driver.deleted) == ["3", "4"], "both released"


def test_a_cancelled_step_releases_every_group(mask_generator):
    driver = FakeDriver(polls_until_done=10**9)
    groups = [group(2, group_index=3, first_index=11), group(2, group_index=4, first_index=13)]
    namespace = args(rollout_batch_size=2, n_samples_per_prompt=2, **QUICK)
    fn = construct(namespace, Data(groups), client_factory=driver.factory)
    fn._mask_generator = mask_generator

    async def cancel_once_both_are_running():
        step = asyncio.create_task(fn(RolloutFnTrainInput(rollout_id=0)))
        while len(driver.requests) < 2 or driver.polls < 2:
            await asyncio.sleep(0.001)
        step.cancel()
        with pytest.raises(asyncio.CancelledError):
            await step

    asyncio.run(cancel_once_both_are_running())
    assert sorted(driver.deleted) == sorted(body["rollout_job_id"] for body in driver.requests)
    assert len(driver.deleted) == 2


def test_wait_covers_execution_and_finalization(mask_generator):
    timeouts = []
    real_wait_for = asyncio.wait_for

    async def wait_for(awaitable, timeout):
        timeouts.append(timeout)
        return await real_wait_for(awaitable, timeout=timeout)

    driver = FakeDriver()
    fn = construct(
        args(sprout_rollout_timeout_seconds=10.0, sprout_rollout_finalization_timeout_seconds=5.0),
        Data([group()]),
        client_factory=driver.factory,
    )
    fn._mask_generator = mask_generator
    with patch("miles.rollout.sprout.rollout_fn.asyncio.wait_for", wait_for):
        asyncio.run(fn(RolloutFnTrainInput(rollout_id=0)))
    assert timeouts == [15.0]
    assert driver.requests[0]["finalization_timeout_seconds"] == 5.0


def test_batch_of_groups_keeps_group_order_and_identities(mask_generator):
    driver = FakeDriver(polls_until_done=1, reward=0.0)
    groups = [group(2, group_index=3, first_index=11), group(2, group_index=4, first_index=13)]
    fn = construct(args(rollout_batch_size=2, n_samples_per_prompt=2), Data(groups), client_factory=driver.factory)
    fn._mask_generator = mask_generator
    output = asyncio.run(fn(RolloutFnTrainInput(rollout_id=2)))
    assert [[s.group_index for s in g] for g in output.samples] == [[3, 3], [4, 4]]
    assert [[s.index for s in g] for g in output.samples] == [[11, 12], [13, 14]]
    assert output.metrics["rollout/sprout/groups"] == 2 and output.metrics["rollout/sprout/reward_mean"] == 0.0
    assert len(driver.requests) == 2 and len(driver.deleted) == 2


def test_miles_parser_exposes_the_sprout_flags_once_the_function_is_selected():
    """The wiring: Miles asks the selected rollout function for its
    arguments, so these flags exist without touching arguments.py."""
    from miles.utils.arguments import get_miles_extra_args_provider

    argv = [
        "--rollout-batch-size",
        "1",
        "--rollout-function-path",
        "miles.rollout.sprout.rollout_fn.SproutRolloutFn",
        "--sprout-rollout-base-url",
        "http://driver:11001",
        "--sprout-rollout-max-turns",
        "7",
    ]
    with patch.object(sys, "argv", ["test", *argv]):
        parser = argparse.ArgumentParser()
        get_miles_extra_args_provider()(parser)
        parsed, unknown = parser.parse_known_args(argv)
    assert unknown == []
    assert parsed.sprout_rollout_base_url == "http://driver:11001" and parsed.sprout_rollout_max_turns == 7
    assert (
        parsed.sprout_rollout_timeout_seconds == 3600.0
        and parsed.sprout_rollout_finalization_timeout_seconds == 1800.0
    )
    assert parsed.sprout_rollout_search_tf_student_continuations == 2
    gsml = {k.removeprefix("sprout_rollout_gsml_"): v for k, v in vars(parsed).items() if "_gsml_" in k}
    assert gsml == {"lambda": 0.4, "beta_branch": 1.0, "psi": 1.0, "c_pre": 0.0, "kappa_plus": 0.0}
    removed = ("branch_advantage_scale", "distill_weight", "difficulty_beta")
    assert not any(hasattr(parsed, f"sprout_rollout_{name}") for name in removed)
    _, unknown = parser.parse_known_args([*argv, "--sprout-rollout-distill-weight", "0.5"])
    assert unknown == ["--sprout-rollout-distill-weight", "0.5"], "the legacy credit flags are gone"
    given = {k: v for k, v in vars(parsed).items() if k.startswith("sprout_") and v is not None}
    fn = construct(Namespace(**{**vars(args()), **given}))
    request, _ = fn._build_request(group=group(), rollout_id=0)
    assert request.max_turns == 7 and request.budgets.max_wall_time_seconds == 3600.0
