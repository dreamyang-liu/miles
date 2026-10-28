"""A search request through SproutRolloutFn: its slots, what construction refuses, and how its groups come back.

The fake driver serves result bodies annotated the way Sprout annotates a
search (``conftest.search_result``), moved into the slots Miles submitted. What
each sample is credited with is ``test_search.py``'s subject; here it is that
the rollout function asks for the right search, refuses what GSML cannot
credit, and hands every group to ``gsml.assemble_search_group`` on its own.
"""

import asyncio
import json
from argparse import Namespace
from collections import Counter
from copy import deepcopy
from unittest.mock import patch

import httpx
import pytest
from tests.fast.rollout.sprout.conftest import (
    LOST,
    TEMPLATE,
    args,
    ff_example,
    group,
    gsml_args,
    members,
    search_result,
    toy_mask_generator,
)

from miles.ray.rollout.rollout_data_conversion import postprocess_rollout_data
from miles.rollout.base_types import RolloutFnConstructorInput, RolloutFnTrainInput
from miles.rollout.sprout.client import SproutRolloutClient
from miles.rollout.sprout.protocol import RolloutRequest
from miles.rollout.sprout.rewards import post_process_gsml
from miles.rollout.sprout.rollout_fn import GSML_POST_PROCESS, SproutRolloutFn

#: Two roots and a point in each failed one, and what GSML needs of the trainer, at the shared example's λ, β, ψ.
SEARCH = {
    "sprout_rollout_search_points": 1,
    "sprout_rollout_search_candidates": 2,
    "sprout_rollout_search_student_continuations": 2,
    "sprout_rollout_search_candidate_continuations": 2,
    "sprout_rollout_search_tf_student_continuations": 2,
    "sprout_rollout_search_repair_rounds": 1,
    "sprout_rollout_search_review_seconds": 5.0,
    "sprout_rollout_search_wall_time_seconds": 7.0,
    "custom_reward_post_process_path": GSML_POST_PROCESS,
    "advantage_estimator": "grpo",
    "calculate_per_token_loss": True,
    "normalize_advantages": False,
    "use_dynamic_global_batch_size": True,
    **vars(gsml_args()),
}
M = "rollout/sprout/gsml/"
#: What the toy template renders of an assistant turn's reasoning.
THINKING = "{% if m.reasoning_content %}<think> {{ m.reasoning_content }} </think> {% endif %}"


def search_args(**updates) -> Namespace:
    return args(**{"n_samples_per_prompt": 2, **SEARCH, **updates})


def construct(namespace: Namespace, data_source=None, **kwargs) -> SproutRolloutFn:
    return SproutRolloutFn(RolloutFnConstructorInput(args=namespace, data_source=data_source), **kwargs)


def in_submitted_slots(body: dict, submitted: dict) -> dict:
    """A ``search_result`` body (slots ``job:slot:11`` on, in slot order) in the slots of the request submitted."""
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
        return httpx.Response(200, json=in_submitted_slots(self.bodies[submitted["prompt_group_id"]], submitted))

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
    namespace = namespace or search_args(rollout_batch_size=len(groups))
    fn = construct(namespace, Data(groups), client_factory=driver.factory)
    fn._mask_generator = mask_generator
    return asyncio.run(fn(RolloutFnTrainInput(rollout_id=1, weight_version=7)))


def role(sample):
    return sample.train_metadata["sprout_rollout"]["role"]


def split_task():
    """One root resolved, one did not: a point in the failed one, two students there."""
    return search_result([1.0, 0.0], {"p1": {"students": [1.0, 0.0]}})[0]


def test_a_search_request_has_a_slot_for_every_branch_its_outcome_may_need():
    request, slots = construct(search_args())._build_request(group=group(2), rollout_id=3)
    assert request.search.roots == 2 and request.search.branches() == 12, "both roots failed: 2 * 1 * (2 + 2 * 2)"
    # Only the roots are guaranteed, and a root lost to a failed actor is named in failed_samples.
    assert request.max_samples == len(request.sample_slots) == 14 and request.minimum_returned_samples == 1
    assert [slot.sample_index for slot in request.sample_slots] == list(range(11, 25))
    assert [slots[s.sample_slot_id].index for s in request.sample_slots] == list(range(11, 25))
    assert all(slots[s.sample_slot_id].group_index is None for s in request.sample_slots[2:])
    assert request.wire()["search"] == {
        "roots": 2,
        "points": 1,
        "candidates": 2,
        "student_continuations": 2,
        "candidate_continuations": 2,
        "tf_student_continuations": 2,
        "repair_rounds": 1,
        "review_seconds": 5.0,
        "wall_time_seconds": 7.0,
    }
    assert RolloutRequest.model_validate(request.wire()) == request
    students_only = search_args(
        sprout_rollout_search_candidates=0,
        sprout_rollout_search_candidate_continuations=0,
        sprout_rollout_search_student_continuations=1,
        sprout_rollout_search_tf_student_continuations=4,
    )
    request, _ = construct(students_only)._build_request(group=group(2), rollout_id=3)
    assert request.search.branches() == 4 and request.max_samples == 6, "the split task's students set the size"


@pytest.mark.parametrize(
    "updates, message",
    [
        ({"sprout_rollout_gsml_lambda": -0.1}, r"gsml-lambda must be finite and in \[0, 1\]"),
        ({"sprout_rollout_gsml_lambda": 1.5}, r"gsml-lambda must be finite and in \[0, 1\]"),
        ({"sprout_rollout_gsml_beta_branch": -1.0}, "gsml-beta-branch must be finite and nonnegative"),
        ({"sprout_rollout_gsml_psi": float("nan")}, "gsml-psi must be finite and nonnegative"),
        ({"sprout_rollout_gsml_c_pre": 0.5}, "gsml-c-pre is not implemented"),
        ({"sprout_rollout_gsml_kappa_plus": 0.1}, "gsml-kappa-plus is not implemented"),
        ({"calculate_per_token_loss": False}, "needs --calculate-per-token-loss"),
        (
            {"custom_reward_post_process_path": "miles.rollout.sprout.rewards.post_process_rewards"},
            "needs --custom-reward-post-process-path miles.rollout.sprout.rewards.post_process_gsml",
        ),
        ({"custom_reward_post_process_path": None}, "needs --custom-reward-post-process-path"),
        ({"normalize_advantages": True}, "refuses --normalize-advantages"),
        ({"n_samples_per_prompt": 3}, "needs --n-samples-per-prompt 2"),
        ({"advantage_estimator": "ppo"}, "needs --advantage-estimator grpo"),
        ({"sprout_rollout_search_tf_student_continuations": -1}, "tf-student-continuations must be a nonnegative"),
        ({"sprout_rollout_search_student_continuations": 0}, "they are the baseline"),
        ({"sprout_rollout_search_candidates": 0}, "both 0 or both positive"),
        ({"sprout_rollout_search_review_seconds": 0}, "review-seconds must be finite and positive"),
        ({"use_dynamic_global_batch_size": False}, "needs --use-dynamic-global-batch-size"),
    ],
    ids=[
        "lambda-negative",
        "lambda-above-one",
        "beta-branch-negative",
        "psi-nan",
        "c-pre",
        "kappa-plus",
        "per-sample-loss",
        "legacy-rewards",
        "no-post-process",
        "normalized-advantages",
        "three-roots",
        "ppo",
        "tf-negative",
        "no-students",
        "candidates-without-continuations",
        "no-review-time",
        "fixed-global-batch",
    ],
)
def test_a_search_gsml_cannot_credit_is_refused_at_construction(updates, message):
    with pytest.raises(ValueError, match=message):
        construct(search_args(**updates))


def test_the_wait_covers_the_review_and_the_branch_phase(mask_generator):
    seen = []
    real_wait_for = asyncio.wait_for

    async def wait_for(awaitable, timeout):
        seen.append(timeout)
        return await real_wait_for(awaitable, timeout=timeout)

    namespace = search_args(sprout_rollout_timeout_seconds=10.0, sprout_rollout_finalization_timeout_seconds=1.0)
    with patch("miles.rollout.sprout.rollout_fn.asyncio.wait_for", wait_for):
        run(mask_generator, SearchDriver({"3": split_task()}), namespace)
    assert seen == [10.0 + 1.0 + 5.0 + 7.0], "the roots, the grades, the review, the branches"


def test_a_search_group_comes_back_credited_by_gsml(mask_generator):
    driver = SearchDriver({"3": ff_example()[0]})
    output = run(mask_generator, driver)
    (samples,) = output.samples
    assert Counter(role(s) for s in samples) == {"root": 2, "student": 4, "repair": 8, "ahat": 1}
    roots = [s for s in samples if role(s) == "root"]
    assert [s.index for s in roots] == [11, 12] and {s.group_index for s in roots} == {3}
    branches = [s for s in samples if role(s) in ("student", "repair")]
    conditions = {s.train_metadata["sprout_rollout"]["group"]: s.group_index for s in branches}
    assert len(conditions) == len(set(conditions.values())) == 6 and min(conditions.values()) == 4
    (ahat,) = [s for s in samples if role(s) == "ahat"]
    assert ahat.index == 25, "after the twelve branch slots' indices"
    assert ahat.group_index not in {3, *conditions.values()}
    assert {s.train_metadata["sprout_rollout"]["weight_version"] for s in samples} == {7}
    _, advantages = post_process_gsml(search_args(), samples)
    assert [a for s, a in zip(samples, advantages, strict=True) if role(s) == "ahat"] == pytest.approx([0.2])
    assert output.metrics[f"{M}cases/FF"] == 1 and output.metrics[f"{M}ahat"] == 1
    assert output.metrics[f"{M}lambda_tasks"] == 1 and output.metrics[f"{M}candidates_leaked"] == 1
    assert output.metrics["rollout/sprout/samples"] == 14 and output.metrics["rollout/sprout/reward_mean"] == 5 / 14
    [sent] = driver.requests
    assert sent["search"]["roots"] == 2 and sent["max_samples"] == 14 and sent["minimum_returned_samples"] == 1
    assert driver.deleted == [sent["rollout_job_id"]], "released exactly once"


def test_a_root_lost_to_a_failed_actor_is_missing_and_its_sibling_trains(mask_generator):
    body, _ = search_result([LOST, 0.0], {"p1": {"students": [1.0, 0.0]}})
    output = run(mask_generator, SearchDriver({"3": body}))
    (samples,) = output.samples
    assert Counter(role(s) for s in samples) == {"root": 1, "student": 2}
    cases = {s.train_metadata["sprout_rollout"]["credit"]["case"] for s in samples}
    assert cases == {"FF_partial"}, "one of the two planned roots graded, none resolved: not FF"
    assert output.metrics["rollout/sprout/failed_samples"] == 1 and output.metrics[f"{M}cases/FF_partial"] == 1


def unnamed_root() -> tuple[dict, str]:
    body, _ = search_result([0.0, 0.0], {"p1": {"students": [1.0, 0.0]}})
    # The first root came back neither as a trajectory nor as a failed sample.
    body["trajectories"] = body["trajectories"][1:]
    body.update(actual_samples=len(body["trajectories"]), status="early_stopped")
    return body, "did not return root"


def mislabelled() -> tuple[dict, str]:
    body, _ = search_result([0.0, 0.0], {"p1": {"students": [1.0, 0.0]}})
    members(body, "student:p1")[0]["metadata"]["search"]["outcome"] = "TF"
    return body, "Sprout labels"


@pytest.mark.parametrize("broken", [unnamed_root, mislabelled], ids=["unnamed-missing-root", "mislabelled"])
def test_a_group_returned_broken_fails_and_only_its_group(mask_generator, broken):
    body, message = broken()
    with pytest.raises(ValueError, match=message):
        run(mask_generator, SearchDriver({"3": body}))
    driver = SearchDriver({"3": body, "4": split_task()})
    output = run(mask_generator, driver, groups=two_groups())
    assert output.samples[0] == [] and Counter(role(s) for s in output.samples[1]) == {"root": 2, "student": 2}
    assert output.metrics["rollout/sprout/groups"] == 1 and output.metrics["rollout/sprout/groups_failed"] == 1
    assert output.metrics[f"{M}cases/TF"] == 1, "only the groups that came back are measured"
    assert len(driver.deleted) == 2, "the failed group is released too"


def test_a_group_whose_every_root_failed_returns_nothing_beside_the_others(mask_generator):
    lost, _ = search_result([LOST, LOST])
    output = run(mask_generator, SearchDriver({"3": lost, "4": ff_example()[0]}), groups=two_groups())
    assert output.samples[0] == [] and len(output.samples[1]) == 15
    assert output.metrics["rollout/sprout/groups"] == 2 and output.metrics["rollout/sprout/groups_failed"] == 0
    assert output.metrics["rollout/sprout/failed_samples"] == 2
    assert output.metrics[f"{M}cases/ungraded"] == 1 and output.metrics[f"{M}cases/FF"] == 1, "summed over groups"
    # How many samples a search step returns varies: one optimizer step trains all of them.
    batch = Namespace(disable_rollout_trim_samples=False, use_dynamic_global_batch_size=True, global_batch_size=4)
    kept, metadata = postprocess_rollout_data(batch, output.samples, train_parallel_config={"dp_size": 1})
    assert len(kept) == 15 and metadata["dynamic_global_batch_size"] == 15
    fixed = Namespace(**{**vars(batch), "use_dynamic_global_batch_size": False})
    trimmed, _ = postprocess_rollout_data(fixed, output.samples, train_parallel_config=None)
    assert len(trimmed) == 12, "a fixed global batch drops the step's last samples"


def test_a_step_with_nothing_to_train_fails(mask_generator):
    lost, _ = search_result([LOST, LOST])
    with pytest.raises(RuntimeError, match="no trainable sample in this step: 0 of 1 groups failed"):
        run(mask_generator, SearchDriver({"3": lost}))
    broken, _ = unnamed_root()
    with pytest.raises(RuntimeError, match="no trainable sample in this step: 1 of 2 groups failed") as raised:
        run(mask_generator, SearchDriver({"3": lost, "4": broken}), groups=two_groups())
    assert isinstance(raised.value.__cause__, ValueError), "the failed group's error is the cause"


def test_a_search_probes_the_chat_template_before_it_submits_anything(monkeypatch):
    dropping = TEMPLATE.replace(THINKING, "")
    assert dropping != TEMPLATE
    tokenizer = toy_mask_generator(dropping).tokenizer
    monkeypatch.setattr("miles.rollout.sprout.rollout_fn.load_tokenizer", lambda *_, **__: tokenizer)
    driver = SearchDriver({"3": ff_example()[0]})
    fn = construct(search_args(), Data([group(2)]), client_factory=driver.factory)
    with pytest.raises(ValueError, match="do not train an assistant turn's reasoning_content"):
        asyncio.run(fn(RolloutFnTrainInput(rollout_id=1)))
    assert driver.requests == [], "no sandbox is spent on a step whose repair turns could not be trained"
    injected = toy_mask_generator(dropping)
    fn._mask_generator = injected
    assert fn._get_mask_generator() is injected, "a generator set on the instance is used as it is"
    assert construct(args())._get_mask_generator().tokenizer is tokenizer, "without a search nothing is probed"
    trained = toy_mask_generator().tokenizer
    monkeypatch.setattr("miles.rollout.sprout.rollout_fn.load_tokenizer", lambda *_, **__: trained)
    assert construct(search_args())._get_mask_generator().tokenizer is trained
