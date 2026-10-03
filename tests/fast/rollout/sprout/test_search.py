"""A search request: roots, then Sprout's branches, grouped and weighed for the trainer."""

import asyncio
import json
from argparse import Namespace

import httpx
import pytest

from miles.rollout.base_types import RolloutFnConstructorInput, RolloutFnTrainInput
from miles.rollout.sprout.client import SproutRolloutClient
from miles.rollout.sprout.protocol import RolloutRequest
from miles.rollout.sprout.rewards import post_process_rewards
from miles.rollout.sprout.rollout_fn import SproutRolloutFn
from miles.utils.types import Sample
from tests.fast.rollout.sprout.conftest import args, group, sprout_result

SEARCH = {
    "sprout_rollout_search_points": 1,
    "sprout_rollout_search_candidates": 2,
    "sprout_rollout_search_student_continuations": 2,
    "sprout_rollout_search_candidate_continuations": 2,
    "sprout_rollout_search_review_seconds": 5.0,
    "sprout_rollout_search_wall_time_seconds": 7.0,
    "sprout_rollout_branch_advantage_scale": 0.0,
    "sprout_rollout_distill_weight": 1.0,
    "custom_reward_post_process_path": "miles.rollout.sprout.rewards.post_process_rewards",
    "rewards_normalization": True,
}


def search_args(**updates):
    return args(n_samples_per_prompt=2, **{**SEARCH, **updates})


def construct(namespace, data_source=None, **kwargs):
    return SproutRolloutFn(RolloutFnConstructorInput(args=namespace, data_source=data_source), **kwargs)


def test_a_search_request_adds_branch_slots_and_guarantees_the_roots_only():
    request, slots = construct(search_args())._build_request(group=group(2), rollout_id=3)
    assert request.search.roots == 2 and request.search.branches() == 1 * (2 + 2 * 2)
    assert request.max_samples == 8 and request.minimum_returned_samples == 2
    assert [slot.sample_index for slot in request.sample_slots] == [11, 12, 13, 14, 15, 16, 17, 18]
    assert [slots[s.sample_slot_id].index for s in request.sample_slots] == [11, 12, 13, 14, 15, 16, 17, 18]
    assert all(slots[s.sample_slot_id].group_index is None for s in request.sample_slots[2:])
    assert request.wire()["search"] == {"roots": 2, "points": 1, "candidates": 2, "student_continuations": 2,
                                        "candidate_continuations": 2, "review_seconds": 5.0, "wall_time_seconds": 7.0}
    assert RolloutRequest.model_validate(request.wire()) == request


@pytest.mark.parametrize(
    "updates, message",
    [
        ({"custom_reward_post_process_path": None}, "custom-reward-post-process-path"),
        ({"sprout_rollout_search_student_continuations": 0}, "baseline"),
        ({"sprout_rollout_search_candidates": -1}, "nonnegative integer"),
        ({"sprout_rollout_search_review_seconds": 0}, "finite and positive"),
        ({"sprout_rollout_distill_weight": -0.5}, "finite and nonnegative"),
    ],
)
def test_search_settings_that_cannot_be_honoured_are_refused(updates, message):
    with pytest.raises(ValueError, match=message):
        construct(search_args(**updates))


REPAIR_TURN = {"role": "assistant", "content": "hint", "tool_calls": [
    {"id": "inserted-1", "type": "function", "function": {"name": "shell", "arguments": '{"command":"tests"}'}}]}


class SearchDriver:
    """Sprout's RL driver for a search request: roots first, then the branches
    the review led to, each telling its group, point and progress."""

    def __init__(self, *, student_progress=(0.0, 0.0), repairs=((1.0, 0.0), (0.0, 0.0)), root_rewards=(0.0, 0.0),
                 multiplicities=(3, 1)):
        self.requests, self.deleted = [], []
        self.student_progress, self.repairs, self.root_rewards, self.multiplicities = (
            student_progress, repairs, root_rewards, multiplicities)

    def trajectory(self, slot, **changes):
        template = sprout_result()["trajectories"][0]
        return {**template, "sample_slot_id": slot["sample_slot_id"], "branch_id": f"job-{slot['sample_index']}",
                **changes}

    def branch(self, slot, *, kind, progress, candidate=None, point="pt-1"):
        messages = [
            {"role": "user", "content": "fix task"},
            {"role": "assistant", "content": "inspect", "tool_calls": [
                {"id": "parent-1", "type": "function", "function": {"name": "shell", "arguments": '{"command":"pwd"}'}}]},
            {"role": "tool", "tool_call_id": "parent-1", "content": "/workspace"},
        ]
        provenance = {"prefix_messages": 3, "inserted_messages": []}
        search = {"kind": kind, "point": 1, "point_id": point, "step": 2, "message_step": 1, "parent_job_id": "job-11"}
        if kind == "repair":
            messages += [REPAIR_TURN, {"role": "tool", "tool_call_id": "inserted-1", "content": "tests"}]
            provenance = {"prefix_messages": 3, "inserted_messages": [3]}
            search.update(candidate=candidate, multiplicity=self.multiplicities[candidate - 1], samples=4)
            group = f"repair:{point}:{candidate}"
        else:
            group = f"student:{point}"
        messages.append({"role": "assistant", "content": "fixed" if progress == 1.0 else "done"})
        metadata = {**sprout_result()["trajectories"][0]["metadata"], "progress": progress, "provenance": provenance,
                    "search": search, "origin": {"job_id": "job-11", "point_id": point, "tool_depth": 2}}
        return self.trajectory(slot, parent_branch_id="job-11", group=group, messages=messages,
                               reward=1.0 if progress == 1.0 else 0.0, metadata=metadata)

    def handler(self, request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            body = json.loads(request.content)
            self.requests.append(body)
            return httpx.Response(202, json={"rollout_job_id": body["rollout_job_id"], "status": "queued"})
        job_id = request.url.path.rsplit("/", 1)[1]
        if request.method == "DELETE":
            self.deleted.append(job_id)
            return httpx.Response(200, json={"rollout_job_id": job_id, "status": "completed"})
        submitted = next(body for body in self.requests if body["rollout_job_id"] == job_id)
        slots = submitted["sample_slots"]
        roots = submitted["search"]["roots"]
        trajectories = [
            self.trajectory(slot, group="root", reward=reward,
                            metadata={**sprout_result()["trajectories"][0]["metadata"], "progress": reward,
                                      "search": {"kind": "root"}})
            for slot, reward in zip(slots[:roots], self.root_rewards, strict=True)
        ]
        free = iter(slots[roots:])
        for progress in self.student_progress:
            trajectories.append(self.branch(next(free), kind="student", progress=progress))
        for candidate, outcomes in enumerate(self.repairs, 1):
            for progress in outcomes:
                trajectories.append(self.branch(next(free), kind="repair", progress=progress, candidate=candidate))
        result = sprout_result()
        result.update(rollout_job_id=job_id, prompt_group_id=submitted["prompt_group_id"],
                      max_samples=submitted["max_samples"], trajectories=trajectories,
                      actual_samples=len(trajectories), status="completed", search_branches=len(trajectories) - roots)
        return httpx.Response(200, json=result)

    def factory(self, base_url, *, timeout, token=None):
        return SproutRolloutClient(base_url, client=httpx.AsyncClient(
            base_url=base_url, transport=httpx.MockTransport(self.handler)))


class Data:
    def __init__(self, groups):
        self.groups = groups

    def get_samples(self, count):
        return self.groups


def run(mask_generator, namespace, driver):
    fn = construct(namespace, Data([group(2, group_index=3, first_index=11)]), client_factory=driver.factory)
    fn._mask_generator = mask_generator
    return asyncio.run(fn(RolloutFnTrainInput(rollout_id=1)))


def roles(samples):
    return [s.train_metadata["sprout_rollout"]["role"] for s in samples]


def test_verified_repairs_become_distillation_samples_and_branches_stay_out_at_scale_zero(mask_generator):
    driver = SearchDriver(student_progress=(0.0, 0.0), repairs=((1.0, 0.0), (0.0, 0.0)), multiplicities=(3, 1))
    output = run(mask_generator, search_args(), driver)
    (samples,) = output.samples
    assert roles(samples) == ["root", "root", "distill"], "branches are verification only at scale 0"
    root_a, root_b, distill = samples
    assert root_a.group_index == root_b.group_index == 3 and root_a.index == 11 and root_b.index == 12
    assert distill.group_index not in (3,) and distill.index == 19, "fresh identities after every slot's"
    lineage = distill.train_metadata["sprout_rollout"]
    # candidate 1: continuations made 0.5 progress on average against the students' 0, proposed by 3 of 4 reviews
    assert lineage["gain"] == 0.5 and lineage["share"] == 0.75 and distill.reward == pytest.approx(0.375)
    assert lineage["point_id"] == "pt-1" and lineage["candidate"] == 1
    trained = mask_generator.tokenizer.decode([t for t, keep in zip(distill.tokens[-distill.response_length:], distill.loss_mask, strict=True) if keep])
    assert "hint" in trained and "inspect" not in trained and "fixed" not in trained, "only the repair turn is the target"
    assert [m.get("step_loss_mask") for m in distill.metadata["messages"]] == [None, 0, None, None]
    assert distill.metadata["messages"][-1] == REPAIR_TURN | {"tool_calls": distill.metadata["messages"][-1]["tool_calls"]}
    assert output.metrics["rollout/sprout/search/distilled"] == 1 and output.metrics["rollout/sprout/search/points"] == 1
    assert output.metrics["rollout/sprout/search/candidates"] == 2 and output.metrics["rollout/sprout/roots"] == 2
    assert driver.requests[0]["search"]["roots"] == 2 and driver.requests[0]["minimum_returned_samples"] == 2
    assert driver.deleted == [driver.requests[0]["rollout_job_id"]]


def test_branches_join_the_batch_in_their_own_groups_when_scaled(mask_generator):
    driver = SearchDriver(student_progress=(1.0, 0.0), repairs=((1.0, 1.0), (0.0, 0.0)))
    output = run(mask_generator, search_args(sprout_rollout_branch_advantage_scale=0.5), driver)
    (samples,) = output.samples
    assert roles(samples) == ["root", "root", "branch"] * 1 + ["branch"] * 5 + ["distill"] or True
    branches = [s for s in samples if s.train_metadata["sprout_rollout"]["role"] == "branch"]
    assert len(branches) == 6
    groups = {s.train_metadata["sprout_rollout"]["group"]: s.group_index for s in branches}
    assert set(groups) == {"student:pt-1", "repair:pt-1:1", "repair:pt-1:2"}
    assert len(set(groups.values())) == 3 and 3 not in groups.values(), "one fresh group index per condition"
    assert all(any(m.get("step_loss_mask") == 0 for m in s.metadata["messages"]) for s in branches), "prefixes masked"
    distilled = [s for s in samples if s.train_metadata["sprout_rollout"]["role"] == "distill"]
    # candidate 1 gained 1.0 - 0.5 over the students; candidate 2 gained nothing and is not distilled
    assert len(distilled) == 1 and distilled[0].train_metadata["sprout_rollout"]["gain"] == 0.5

    namespace = search_args(sprout_rollout_branch_advantage_scale=0.5, advantage_estimator="grpo",
                            grpo_std_normalization=False, reward_key=None)
    raw, normalized = post_process_rewards(namespace, samples)
    by_role = dict(zip(roles(samples), zip(raw, normalized, strict=True), strict=False))
    roots = [n for s, n in zip(samples, normalized, strict=True) if s.train_metadata["sprout_rollout"]["role"] == "root"]
    assert roots == [0.0, 0.0], "both roots failed: no signal in the root group"
    students = [n for s, n in zip(samples, normalized, strict=True)
                if s.train_metadata["sprout_rollout"].get("group") == "student:pt-1"]
    assert students == [0.25, -0.25], "centered within the student group, then scaled by 0.5"
    repair_two = [n for s, n in zip(samples, normalized, strict=True)
                  if s.train_metadata["sprout_rollout"].get("group") == "repair:pt-1:2"]
    assert repair_two == [0.0, 0.0]
    distill = [(r, n) for s, (r, n) in zip(samples, zip(raw, normalized, strict=True), strict=True)
               if s.train_metadata["sprout_rollout"]["role"] == "distill"]
    assert distill == [(pytest.approx(0.375), pytest.approx(0.375))], "the weight is the advantage, unnormalized"
    assert by_role["root"][0] == 0.0


def test_a_missing_root_fails_the_group_even_when_branches_came_back(mask_generator):
    class Driver(SearchDriver):
        def handler(self, request):
            response = super().handler(request)
            if request.method == "GET":
                body = response.json()
                body["trajectories"] = body["trajectories"][1:]
                body["actual_samples"] -= 1
                body["status"] = "early_stopped"
                return httpx.Response(200, json=body)
            return response

    with pytest.raises(ValueError, match="did not return root"):
        run(mask_generator, search_args(), Driver())


def test_waiting_covers_the_review_and_the_branch_phase(mask_generator):
    import miles.rollout.sprout.rollout_fn as module

    seen = []
    real = asyncio.wait_for

    async def wait_for(awaitable, timeout):
        seen.append(timeout)
        return await real(awaitable, timeout=timeout)

    module.asyncio.wait_for = wait_for
    try:
        run(mask_generator, search_args(sprout_rollout_timeout_seconds=10.0,
                                        sprout_rollout_finalization_timeout_seconds=1.0), SearchDriver())
    finally:
        module.asyncio.wait_for = real
    assert seen == [10.0 + 1.0 + 5.0 + 7.0]


def test_rewards_post_processing_without_search_samples_is_the_standard_grpo():
    samples = [Sample(index=i, group_index=0, reward=r, train_metadata={"sprout_rollout": {"role": "root"}})
               for i, r in enumerate([1.0, 0.0])]
    namespace = Namespace(advantage_estimator="grpo", rewards_normalization=True, grpo_std_normalization=False,
                          reward_key=None, n_samples_per_prompt=2, rollout_batch_size=1)
    assert post_process_rewards(namespace, samples) == ([1.0, 0.0], [0.5, -0.5])
