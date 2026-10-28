"""A RAFO search request through SproutRolloutFn: its slots, its wait, and how its groups come back.

The fake driver serves result bodies annotated the way Sprout annotates a RAFO search
(``conftest.forest_result``), moved into the slots Miles submitted. What each sample is credited with
is ``test_rafo.py``'s subject; here it is that the rollout function asks for the right search and
hands every group to ``rafo.assemble_rafo_group`` on its own.
"""

import asyncio
import json
from argparse import Namespace
from collections import Counter
from copy import deepcopy
from unittest.mock import patch

import httpx
import pytest
from tests.fast.rollout.sprout.conftest import LOST, TEMPLATE, args, forest_result, group, toy_mask_generator

from miles.rollout.base_types import RolloutFnConstructorInput, RolloutFnTrainInput
from miles.rollout.sprout.client import SproutRolloutClient
from miles.rollout.sprout.protocol import RolloutRequest
from miles.rollout.sprout.rafo_role import RafoRole
from miles.rollout.sprout.rollout_fn import RAFO_LOSS, SproutRolloutFn

#: A RAFO search of two continuations per event, at most three events, two deep, and what it needs of the trainer.
RAFO = {
    "sprout_rollout_rafo": True,
    "sprout_rollout_rafo_continuations": 2,
    "sprout_rollout_rafo_depth": 2,
    "sprout_rollout_rafo_events": 3,
    "loss_type": "custom_loss",
    "custom_loss_function_path": RAFO_LOSS,
    "train_backend": "megatron",
    "rewards_normalization": False,
    "custom_reward_post_process_path": None,
    "use_dynamic_global_batch_size": False,
}
M = "rollout/sprout/rafo/"
#: What the toy template renders of an assistant turn's reasoning.
THINKING = "{% if m.reasoning_content %}<think> {{ m.reasoning_content }} </think> {% endif %}"
#: Both roots fail; event 1 in root 1 splits.
SPLIT_EVENT = [("root:0", [1.0, 0.0])]


def rafo_args(**updates) -> Namespace:
    return args(**{"n_samples_per_prompt": 2, **RAFO, **updates})


def construct(namespace: Namespace, data_source=None, **kwargs) -> SproutRolloutFn:
    return SproutRolloutFn(RolloutFnConstructorInput(args=namespace, data_source=data_source), **kwargs)


def in_submitted_slots(body: dict, submitted: dict) -> dict:
    """A ``forest_result`` body (slots ``job:slot:11`` on, in slot order) in the slots of the request submitted."""
    slots = {f"job:slot:{11 + k}": slot["sample_slot_id"] for k, slot in enumerate(submitted["sample_slots"])}
    body = deepcopy(body)
    for entry in [*body["trajectories"], *body["failed_samples"]]:
        entry["sample_slot_id"] = slots[entry["sample_slot_id"]]
    body.update(
        rollout_job_id=submitted["rollout_job_id"],
        prompt_group_id=submitted["prompt_group_id"],
        max_samples=submitted["max_samples"],
    )
    return body


class SearchDriver:
    """Sprout's RL driver for search requests: each prompt group's result body, in the slots Miles named."""

    def __init__(self, bodies: dict[str, dict]) -> None:
        self.bodies, self.requests, self.deleted = bodies, [], []

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
        return httpx.Response(200, json=self.answer(submitted))

    def answer(self, submitted: dict) -> dict:
        return in_submitted_slots(self.bodies[submitted["prompt_group_id"]], submitted)

    def factory(self, base_url, *, timeout, token=None):
        transport = httpx.MockTransport(self.handler)
        return SproutRolloutClient(base_url, client=httpx.AsyncClient(base_url=base_url, transport=transport))


class Data:
    def __init__(self, groups):
        self.groups = groups

    def get_samples(self, count):
        return self.groups


def two_groups():
    return [group(2, group_index=3, first_index=11), group(2, group_index=4, first_index=13)]


def run(mask_generator, driver, namespace=None, groups=None):
    """One training step over ``groups`` (one prompt group of two roots by default)."""
    groups = groups or [group(2)]
    namespace = namespace or rafo_args(rollout_batch_size=len(groups))
    fn = construct(namespace, Data(groups), client_factory=driver.factory)
    fn._mask_generator = mask_generator
    return asyncio.run(fn(RolloutFnTrainInput(rollout_id=1, weight_version=7)))


def kinds(samples) -> Counter:
    return Counter(RafoRole.of(sample.train_metadata).kind for sample in samples)


def test_a_rafo_request_has_a_slot_for_every_continuation_its_events_may_make():
    request, slots = construct(rafo_args())._build_request(group=group(2), rollout_id=3)

    assert request.search.roots == 2 and request.search.branches() == 6, "3 events of 2 continuations"
    # Only the roots are guaranteed, and a root lost to a failed actor is named in failed_samples.
    assert request.max_samples == len(request.sample_slots) == 8 and request.minimum_returned_samples == 1
    assert [slot.sample_index for slot in request.sample_slots] == list(range(11, 19))
    assert [slots[s.sample_slot_id].index for s in request.sample_slots] == list(range(11, 19))
    assert all(slots[s.sample_slot_id].group_index is None for s in request.sample_slots[2:])
    assert request.wire()["search"] == {
        "mode": "rafo",
        "roots": 2,
        "continuations": 2,
        "depth": 2,
        "events": 3,
        "review_seconds": 5.0,
        "wall_time_seconds": 7.0,
    }
    assert RolloutRequest.model_validate(request.wire()) == request


def test_the_wait_covers_every_depths_review_and_continuations(mask_generator):
    seen = []
    real_wait_for = asyncio.wait_for

    async def wait_for(awaitable, timeout):
        seen.append(timeout)
        return await real_wait_for(awaitable, timeout=timeout)

    namespace = rafo_args(sprout_rollout_timeout_seconds=10.0, sprout_rollout_finalization_timeout_seconds=1.0)
    body, _ = forest_result([0.0, 0.0], SPLIT_EVENT)
    with patch("miles.rollout.sprout.rollout_fn.asyncio.wait_for", wait_for):
        run(mask_generator, SearchDriver({"3": body}), namespace)
    assert seen == [10.0 + 1.0 + 2 * (5.0 + 7.0) + 1.0], "the roots, then two depths and the grades between"


def test_a_rafo_group_comes_back_credited_as_one_task(mask_generator):
    """Event 1 fails twice; event 2, inside its second continuation, splits."""
    body, _ = forest_result([0.0, 0.0], [("root:0", [0.0, 0.0]), ("event:1:1", [1.0, 0.0])])
    driver = SearchDriver({"3": body})

    output = run(mask_generator, driver)

    (samples,) = output.samples
    assert kinds(samples) == {"repair": 2, "reflect": 2}
    assert {s.rollout_id for s in samples} == {3} and {s.group_index for s in samples} == {3}
    assert sorted(s.index for s in samples if RafoRole.of(s.train_metadata).kind == "reflect") == [19, 20]
    assert {s.train_metadata["sprout_rollout"]["weight_version"] for s in samples} == {7}
    assert output.metrics[f"{M}cases/FF"] == 1 and output.metrics[f"{M}nested_events"] == 1
    assert output.metrics["rollout/sprout/samples"] == 6 and output.metrics["rollout/sprout/reward_mean"] == 1 / 6
    [sent] = driver.requests
    assert sent["search"]["mode"] == "rafo" and sent["max_samples"] == 8 and sent["minimum_returned_samples"] == 1
    assert driver.deleted == [sent["rollout_job_id"]], "released exactly once"


def test_a_root_lost_to_a_failed_actor_is_missing_and_the_task_still_trains(mask_generator):
    body, _ = forest_result([LOST, 0.0], [("root:1", [1.0, 0.0])])

    output = run(mask_generator, SearchDriver({"3": body}))

    (samples,) = output.samples
    assert kinds(samples) == {"repair": 2, "reflect": 1}
    assert output.metrics["rollout/sprout/failed_samples"] == 1 and output.metrics[f"{M}cases/FF_partial"] == 1


def unnamed_root() -> tuple[dict, str]:
    body, _ = forest_result([0.0, 0.0], SPLIT_EVENT)
    # The first root came back neither as a trajectory nor as a failed sample.
    body["trajectories"] = body["trajectories"][1:]
    body.update(actual_samples=len(body["trajectories"]), status="early_stopped")
    return body, "did not return root"


def mislabelled() -> tuple[dict, str]:
    body, _ = forest_result([0.0, 0.0], SPLIT_EVENT)
    body["trajectories"][2]["metadata"]["search"]["outcome"] = "TF"
    return body, "Sprout labels"


@pytest.mark.parametrize("broken", [unnamed_root, mislabelled], ids=["unnamed-missing-root", "mislabelled"])
def test_a_group_returned_broken_fails_and_only_its_group(mask_generator, broken):
    body, message = broken()
    with pytest.raises(ValueError, match=message):
        run(mask_generator, SearchDriver({"3": body}))
    split, _ = forest_result([1.0, 0.0])
    driver = SearchDriver({"3": body, "4": split})

    output = run(mask_generator, driver, groups=two_groups())

    assert output.samples[0] == [] and kinds(output.samples[1]) == {"root": 2}
    assert output.metrics["rollout/sprout/groups"] == 1 and output.metrics["rollout/sprout/groups_failed"] == 1
    assert output.metrics[f"{M}cases/TF"] == 1, "only the groups that came back are measured"
    assert len(driver.deleted) == 2, "the failed group is released too"


def test_a_group_whose_every_root_failed_returns_nothing_beside_the_others(mask_generator):
    lost, _ = forest_result([LOST, LOST])
    body, _ = forest_result([0.0, 0.0], SPLIT_EVENT)

    output = run(mask_generator, SearchDriver({"3": lost, "4": body}), groups=two_groups())

    assert output.samples[0] == [] and kinds(output.samples[1]) == {"repair": 2, "reflect": 1}
    assert output.metrics["rollout/sprout/groups"] == 2 and output.metrics["rollout/sprout/groups_failed"] == 0
    assert output.metrics["rollout/sprout/failed_samples"] == 2
    assert output.metrics[f"{M}cases/ungraded"] == 1 and output.metrics[f"{M}cases/FF"] == 1, "summed over groups"


def test_a_step_with_nothing_to_train_fails(mask_generator):
    lost, _ = forest_result([LOST, LOST])
    with pytest.raises(RuntimeError, match="no trainable sample in this step: 0 of 1 groups failed"):
        run(mask_generator, SearchDriver({"3": lost}))
    broken, _ = unnamed_root()
    with pytest.raises(RuntimeError, match="no trainable sample in this step: 1 of 2 groups failed") as raised:
        run(mask_generator, SearchDriver({"3": lost, "4": broken}), groups=two_groups())
    assert isinstance(raised.value.__cause__, ValueError), "the failed group's error is the cause"


def test_a_rafo_search_probes_the_chat_template_before_it_submits_anything(monkeypatch):
    dropping = TEMPLATE.replace(THINKING, "")
    assert dropping != TEMPLATE
    tokenizer = toy_mask_generator(dropping).tokenizer
    monkeypatch.setattr("miles.rollout.sprout.rollout_fn.load_tokenizer", lambda *_, **__: tokenizer)
    body, _ = forest_result([0.0, 0.0], SPLIT_EVENT)
    driver = SearchDriver({"3": body})
    fn = construct(rafo_args(), Data([group(2)]), client_factory=driver.factory)
    with pytest.raises(ValueError, match="do not train an assistant turn's reasoning_content"):
        asyncio.run(fn(RolloutFnTrainInput(rollout_id=1)))
    assert driver.requests == [], "no sandbox is spent on a step whose reviewer turns could not be trained"
    injected = toy_mask_generator(dropping)
    fn._mask_generator = injected
    assert fn._get_mask_generator() is injected, "a generator set on the instance is used as it is"
    assert construct(args())._get_mask_generator().tokenizer is tokenizer, "without a search nothing is probed"
    trained = toy_mask_generator().tokenizer
    monkeypatch.setattr("miles.rollout.sprout.rollout_fn.load_tokenizer", lambda *_, **__: trained)
    assert construct(rafo_args())._get_mask_generator().tokenizer is trained
