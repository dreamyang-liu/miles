"""Sprout's shared GSML example through Miles: the credit, the advantages, and what wires them in.

``fixtures/sprout_search_result.example.json`` is a byte copy of Sprout's
``src/sprout/rl_driver/search_result.example.json``, the result Sprout's driver
and adapter return for the shared FF example of the credit rules
(``tests/rl_driver/test_gsml_wiring.py`` there): both roots failed; at P1 a
student resolved; at P2 none did, candidate 1's continuations both resolved
and candidate 2's turn names a hidden test file. Unlike ``conftest.ff_example``,
nothing in it is annotated by hand: GSML reads what Sprout's export wrote. The
flags are the example's, λ 0.4 and β_br = ψ = 1, unless a test parses others.
"""

import argparse
import asyncio
import json
import math
import os
import sys
import uuid
from collections import Counter
from pathlib import Path
from unittest.mock import patch

import httpx
import pytest
import torch
from tests.fast.rollout.sprout.conftest import (
    FIXTURES,
    TEMPLATE,
    assemble,
    group,
    gsml_args,
    search_result,
    toy_mask_generator,
)
from tests.fast.rollout.sprout.test_search_request import THINKING, Data, SearchDriver, construct, search_args

from miles.backends.training_utils.loss.hub.advantages import compute_advantages
from miles.ray.rollout.rollout_data_conversion import postprocess_rollout_data
from miles.ray.rollout.train_data_conversion import _post_process_rewards
from miles.rollout.base_types import RolloutFnTrainInput
from miles.rollout.sprout.protocol import RolloutResult
from miles.rollout.sprout.rewards import coefficients, post_process_gsml
from miles.utils.function_registry import load_function
from miles.utils.types import Sample

EXAMPLE = FIXTURES / "sprout_search_result.example.json"
#: Each fixture and the example Sprout ships beside its RL driver that it copies byte for byte.
SPROUT_EXAMPLES = {
    "sprout_request.example.json": "request.example.json",
    "sprout_result.example.json": "result.example.json",
    "sprout_search_result.example.json": "search_result.example.json",
}
#: The z-score of the success in a (1, 0) group, give or take the 1e-6 under it.
R = 1 / math.sqrt(2)
M = "rollout/sprout/gsml/"


def example() -> dict:
    return json.loads(EXAMPLE.read_text())


def example_slots(body: dict) -> dict[str, Sample]:
    """The slots Miles allocates for the example's request: ``SproutRolloutFn._build_request`` under its job id."""
    job_hex = body["rollout_job_id"].rsplit("-", 1)[1]
    with patch("miles.rollout.sprout.rollout_fn.uuid.uuid4", return_value=uuid.UUID(job_hex)):
        request, slots = construct(search_args())._build_request(group=group(2), rollout_id=1)
    assert request.rollout_job_id == body["rollout_job_id"]
    return slots


def role(sample: Sample) -> str:
    return sample.train_metadata["sprout_rollout"]["role"]


def credit(sample: Sample) -> dict:
    return sample.train_metadata["sprout_rollout"]["credit"]


def condition(sample: Sample) -> str:
    """Where a sample stands in its task: ``root``, or its role at the point and candidate Sprout numbered
    (``student:P1``, ``repair:P2:1``, ``ahat:P2:1``)."""
    if role(sample) == "root":
        return "root"
    search = sample.metadata["sprout_rollout"]["trajectory_metadata"]["search"]
    label = f"{role(sample)}:P{search['point']}"
    return label if role(sample) == "student" else f"{label}:{search['candidate']}"


def numbered(trajectory: dict) -> str:
    """A returned repair's point and candidate as Sprout numbered them (``P2:1``)."""
    search = trajectory["metadata"]["search"]
    return f"P{search['point']}:{search['candidate']}"


def by_condition(samples: list[Sample], advantages: list[float]) -> dict[str, list[float]]:
    grouped = {}
    for sample, advantage in zip(samples, advantages, strict=True):
        grouped.setdefault(condition(sample), []).append(advantage)
    return grouped


def trained_tokens(sample: Sample) -> list[int]:
    return [t for t, keep in zip(sample.tokens[-sample.response_length :], sample.loss_mask, strict=True) if keep]


def assert_the_shared_example(grouped: dict[str, list[float]]) -> None:
    """The shared FF example's advantages, by condition in slot order."""
    assert grouped["root"] == [0.0, 0.0], "both roots failed: their group carries no signal"
    # P1: a student resolved, so P1's λ goes to it, exactly; its candidates get their z-scores only.
    assert grouped["student:P1"] == pytest.approx([0.9071, -0.7071], abs=1e-4), "β_br·z, + λ·ω/c_s = 0.4·½/1"
    assert grouped["repair:P1:1"] == pytest.approx([R, -R], abs=1e-5)
    assert grouped["repair:P1:2"] == [0.0, 0.0]
    # P2: no student resolved; candidate 1 is eligible, candidate 2 named a hidden test file.
    assert grouped["student:P2"] == [0.0, 0.0]
    assert grouped["repair:P2:1"] == pytest.approx([0.10, 0.10]), "λ·ω/(|E|·K) = 0.4·½/(1·2)"
    assert grouped["ahat:P2:1"] == pytest.approx([0.20]), "λ·ψ·ω/|E| = 0.4·½/1"
    assert grouped["repair:P2:2"] == pytest.approx([R, -R], abs=1e-5), "leaked: β_br·z, nothing from λ"
    assert set(grouped) == {
        "root",
        "student:P1",
        "repair:P1:1",
        "repair:P1:2",
        "student:P2",
        "repair:P2:1",
        "ahat:P2:1",
        "repair:P2:2",
    }, "one â, for the one eligible candidate where no student resolved"


class ExampleDriver(SearchDriver):
    """Sprout's RL driver answering a search request with the shared example, in the slots Miles submitted."""

    def __init__(self) -> None:
        super().__init__({})

    def handler(self, request: httpx.Request) -> httpx.Response:
        if request.method != "GET":
            return super().handler(request)
        job_id = request.url.path.rsplit("/", 1)[1]
        submitted = next(body for body in self.requests if body["rollout_job_id"] == job_id)
        return httpx.Response(200, json=in_submitted_slots(example(), submitted))


def in_submitted_slots(body: dict, submitted: dict) -> dict:
    """``body`` re-keyed to a submitted request: its k-th trajectory in the k-th slot (the example fills
    every slot, in order), under the submitted group's identity."""
    assert body["failed_samples"] == []
    for trajectory, slot in zip(body["trajectories"], submitted["sample_slots"], strict=True):
        trajectory["sample_slot_id"] = slot["sample_slot_id"]
    body.update(
        rollout_job_id=submitted["rollout_job_id"],
        prompt_group_id=submitted["prompt_group_id"],
        max_samples=submitted["max_samples"],
    )
    return body


def parsed_gsml_flags(*argv: str) -> argparse.Namespace:
    """The GSML flags as Miles's parser reads them once SproutRolloutFn is the rollout function."""
    from miles.utils.arguments import get_miles_extra_args_provider

    argv = (
        "--rollout-batch-size",
        "1",
        "--rollout-function-path",
        "miles.rollout.sprout.rollout_fn.SproutRolloutFn",
        "--sprout-rollout-base-url",
        "http://driver:11001",
        *argv,
    )
    with patch.object(sys, "argv", ["test", *argv]):
        parser = argparse.ArgumentParser()
        get_miles_extra_args_provider()(parser)
        parsed, unknown = parser.parse_known_args(argv)
    assert unknown == []
    return gsml_args(**{key: value for key, value in vars(parsed).items() if key.startswith("sprout_rollout_gsml_")})


def sprout_examples() -> Path | None:
    """Sprout's ``src/sprout/rl_driver``: of ``$SPROUT_REPO``, else of a checkout beside this one; None without."""
    configured = os.environ.get("SPROUT_REPO")
    siblings = Path(__file__).resolve().parents[5]  # the directory holding this checkout
    for checkout in [Path(configured)] if configured else [siblings / "sprout", siblings / "SPROUT"]:
        if (checkout / "src" / "sprout" / "rl_driver").is_dir():
            return checkout / "src" / "sprout" / "rl_driver"
    if configured:
        pytest.fail(f"SPROUT_REPO={configured} is not a Sprout checkout")
    return None


def test_the_example_is_a_result_miles_accepts_as_it_is():
    body = example()
    result = RolloutResult.model_validate(body)
    assert result.model_dump(mode="json") == body
    assert result.status == "completed" and result.actual_samples == result.max_samples == 14
    counts = {"planned": 2, "graded": 2, "resolved": 0}
    assert all(
        t.metadata["search"]["outcome"] == "FF" and t.metadata["search"]["root_counts"] == counts
        for t in result.trajectories
    )


def test_the_shared_ff_example_is_credited_as_the_rules_say(mask_generator):
    body = example()
    slots = example_slots(body)
    assert list(slots) == [t["sample_slot_id"] for t in body["trajectories"]], "it fills the slots Miles allocates"
    samples, metrics = assemble(mask_generator, body, slots)
    _, advantages = post_process_gsml(gsml_args(), samples)
    assert_the_shared_example(by_condition(samples, advantages))
    assert {credit(s)["omega"] for s in samples if role(s) != "root"} == {0.5}, "two points with mass: ω = ½ each"
    repairs = [t for t in body["trajectories"] if t["metadata"]["search"]["kind"] == "repair"]
    assert {numbered(t): t["metadata"]["search"]["leak_terms"] for t in repairs} == {
        "P1:1": [],
        "P1:2": [],
        "P2:1": [],
        "P2:2": ["tests/test_hidden.py"],
    }, "as Sprout's review found them"

    # The â: P2 candidate 1's first continuation up to the inserted turn, of which only the turn trains.
    (ahat,) = [s for s in samples if role(s) == "ahat"]
    source = next(t for t in repairs if numbered(t) == "P2:1")
    turn = ahat.metadata["messages"][-1]
    assert [m["role"] for m in ahat.metadata["messages"]] == ["user", "assistant", "tool", "assistant"]
    assert turn["reasoning_content"] == source["messages"][3]["reasoning_content"] and turn["content"] == ""
    assert [call["id"] for call in turn["tool_calls"]] == [call["id"] for call in source["messages"][3]["tool_calls"]]
    rendered = mask_generator.tokenizer.apply_chat_template([turn], tokenize=True, return_dict=False)
    assert trained_tokens(ahat) == rendered[mask_generator.gen_token_length :], "the turn's reasoning and its call"
    trained = mask_generator.tokenizer.decode(trained_tokens(ahat))
    assert trained.startswith(f"<think> {turn['reasoning_content']} </think>")
    assert ahat.train_metadata["sprout_rollout"]["sample_slot_id"] == source["sample_slot_id"]
    assert (credit(ahat)["k_m"], credit(ahat)["n_eligible"], credit(ahat)["lambda_share"]) == (2, 1, 0.5)

    assert {key: value for key, value in metrics.items() if value} == {
        f"{M}cases/FF": 1,
        f"{M}points": 2,
        f"{M}points_exact": 1,
        f"{M}points_guided": 1,
        f"{M}candidates": 4,
        # P1's candidate 1 passes every gate too; only where no student resolved does that earn credit.
        f"{M}candidates_eligible": 2,
        f"{M}candidates_leaked": 1,
        f"{M}candidates_unconfirmed": 1,
        f"{M}ahat": 1,
        f"{M}lambda_tasks": 1,
    }


def test_the_example_trains_end_to_end_each_sample_at_its_a(mask_generator):
    driver = ExampleDriver()
    namespace = search_args(disable_rollout_trim_samples=False, global_batch_size=4)
    fn = construct(namespace, Data([group(2)]), client_factory=driver.factory)
    fn._mask_generator = mask_generator
    output = asyncio.run(fn(RolloutFnTrainInput(rollout_id=1, weight_version=7)))
    (samples,) = output.samples
    assert Counter(role(s) for s in samples) == {"root": 2, "student": 4, "repair": 8, "ahat": 1}
    assert output.metrics[f"{M}cases/FF"] == 1 and output.metrics[f"{M}lambda_tasks"] == 1
    [sent] = driver.requests
    assert sent["max_samples"] == 14 and driver.deleted == [sent["rollout_job_id"]]

    flat, metadata = postprocess_rollout_data(namespace, output.samples, train_parallel_config={"dp_size": 1})
    assert len(flat) == 15 and metadata["dynamic_global_batch_size"] == 15, "one optimizer step trains them all"
    # As the trainer applies it: the post-process the rollout function requires, loaded from its path.
    post_process = load_function(namespace.custom_reward_post_process_path)
    assert post_process is post_process_gsml
    _, rewards = _post_process_rewards(namespace, flat, post_process)
    assert_the_shared_example(by_condition(flat, rewards))
    loss_masks = [torch.tensor(sample.loss_mask, dtype=torch.int32) for sample in flat]
    advantages, _ = compute_advantages(
        namespace,
        kl=[torch.zeros(len(mask), dtype=torch.float32) for mask in loss_masks],
        rewards=rewards,
        log_probs=None,
        loss_masks=loss_masks,
        total_lengths=[len(sample.tokens) for sample in flat],
        response_lengths=[sample.response_length for sample in flat],
    )
    for sample, advantage, a in zip(flat, advantages, rewards, strict=True):
        assert advantage.shape == (sample.response_length,)
        assert torch.allclose(advantage, torch.full_like(advantage, a)), "A on every response position, as it is"


def test_the_parsed_lambda_moves_the_root_and_lambda_terms_as_it_should(mask_generator):
    pytest.importorskip("sglang")
    default, flags = parsed_gsml_flags(), parsed_gsml_flags("--sprout-rollout-gsml-lambda", "0.3")
    for namespace, lam in ((default, 0.4), (flags, 0.3)):
        assert coefficients(namespace) == (
            lam,
            {"root": pytest.approx((1 - lam) / math.sqrt(2)), "branch": 1.0, None: 0.0},
            1.0,
        ), "β_root = (1 - λ)/√2, β_br and ψ at their defaults"

    body = example()
    samples, _ = assemble(mask_generator, body, example_slots(body))
    _, advantages = post_process_gsml(default, samples)
    assert_the_shared_example(by_condition(samples, advantages))
    _, advantages = post_process_gsml(flags, samples)
    grouped = by_condition(samples, advantages)
    assert grouped["student:P1"] == pytest.approx([R + 0.3 * 0.5, -R], abs=1e-5), "λ·ω/c_s = 0.3·½/1"
    assert grouped["repair:P2:1"] == pytest.approx([0.075, 0.075]), "λ·ω/(|E|·K) = 0.3·½/(1·2)"
    assert grouped["ahat:P2:1"] == pytest.approx([0.15]), "λ·ψ·ω/|E| = 0.3·½/1"
    assert grouped["repair:P2:2"] == pytest.approx([R, -R], abs=1e-5) and grouped["root"] == [0.0, 0.0]
    # The example's roots both failed (z = 0); a split pair shows the root term: ±β_root/√2 = ±(1 - λ)/2.
    split, slots = search_result([1.0, 0.0], {"p1": {"students": [1.0, 0.0]}})
    split_samples, _ = assemble(mask_generator, split, slots)
    for namespace, expected in ((default, 0.30), (flags, 0.35)):
        _, advantages = post_process_gsml(namespace, split_samples)
        assert by_condition(split_samples, advantages)["root"] == pytest.approx([expected, -expected], abs=1e-5)


def test_a_template_that_drops_reasoning_fails_the_first_step_before_anything_is_sent(monkeypatch):
    dropping = TEMPLATE.replace(THINKING, "")
    assert dropping != TEMPLATE
    tokenizer = toy_mask_generator(dropping).tokenizer
    monkeypatch.setattr("miles.rollout.sprout.rollout_fn.load_tokenizer", lambda *_, **__: tokenizer)
    driver = ExampleDriver()
    fn = construct(search_args(), Data([group(2)]), client_factory=driver.factory)  # no mask generator injected
    with pytest.raises(ValueError, match="do not train an assistant turn's reasoning_content"):
        asyncio.run(fn(RolloutFnTrainInput(rollout_id=1)))
    assert driver.requests == [], "no POST /rollout-groups: no sandbox is spent on a step that cannot train"


@pytest.mark.parametrize("fixture", sorted(SPROUT_EXAMPLES))
def test_the_fixtures_are_byte_copies_of_the_examples_sprout_ships(fixture):
    directory = sprout_examples()
    if directory is None:
        pytest.skip("no Sprout checkout beside this one: set SPROUT_REPO to compare the fixtures with its examples")
    shipped = directory / SPROUT_EXAMPLES[fixture]
    assert (FIXTURES / fixture).read_bytes() == shipped.read_bytes(), f"copy {shipped} over fixtures/{fixture}"
