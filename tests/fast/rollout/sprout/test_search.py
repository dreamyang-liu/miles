"""GSML credit for a search group: ``assemble_search_group``, then ``post_process_gsml``.

The results are synthetic and annotated the way Sprout annotates a search
(``conftest.search_result``); the flags are those of the shared example of the
credit rules, λ 0.4 and β_br = ψ = 1.
"""

import io
import math

import msgpack
import pytest
import torch
from tests.fast.rollout.sprout.conftest import (
    LOST,
    TEMPLATE,
    assemble,
    ff_example,
    gsml_args,
    members,
    search_result,
    toy_mask_generator,
)

from miles.rollout.sprout.gsml import check_reasoning_is_trained, group_z, search_outcome
from miles.rollout.sprout.rewards import post_process_gsml
from miles.utils.types import Sample

#: The z-score of the success in a (1, 0) group, give or take the 1e-6 under it.
R = 1 / math.sqrt(2)
M = "rollout/sprout/gsml/"


def role(sample):
    return sample.train_metadata["sprout_rollout"]["role"]


def credit(sample):
    return sample.train_metadata["sprout_rollout"]["credit"]


def condition(sample):
    """The group a sample was sampled under; an â sample by its point and candidate."""
    if role(sample) == "ahat":
        return f"ahat:{credit(sample)['point_id']}:{credit(sample)['candidate']}"
    return sample.train_metadata["sprout_rollout"]["group"]


def credited(mask_generator, body, slots, **kwargs):
    """A group's advantages by condition, its samples and its metrics."""
    samples, metrics = assemble(mask_generator, body, slots, **kwargs)
    _, advantages = post_process_gsml(gsml_args(), samples)
    by_condition = {}
    for sample, advantage in zip(samples, advantages, strict=True):
        by_condition.setdefault(condition(sample), []).append(advantage)
    return by_condition, samples, metrics


def trained_tokens(sample):
    return [t for t, keep in zip(sample.tokens[-sample.response_length :], sample.loss_mask, strict=True) if keep]


def test_the_shared_ff_example(mask_generator):
    by_condition, samples, metrics = credited(mask_generator, *ff_example())
    assert by_condition["root"] == [0.0, 0.0], "both roots failed: their group carries no signal"
    # P1: a student resolved, so P1's λ goes to it, exactly; its candidates get their z-scores only.
    assert by_condition["student:p1"] == pytest.approx([0.9071, -0.7071], abs=1e-4), "β_br·z, + λ·ω/c_s = 0.4·½/1"
    assert by_condition["repair:p1:1"] == pytest.approx([R, -R], abs=1e-5)
    assert by_condition["repair:p1:2"] == [0.0, 0.0]
    # P2: no student resolved; candidate 1 is eligible, candidate 2 named a hidden test file.
    assert by_condition["student:p2"] == [0.0, 0.0]
    assert by_condition["repair:p2:1"] == pytest.approx([0.10, 0.10]), "λ·ω/(|E|·K) = 0.4·½/(1·2)"
    assert by_condition["ahat:p2:1"] == pytest.approx([0.20]), "λ·ψ·ω/|E| = 0.4·½/1"
    assert by_condition["repair:p2:2"] == pytest.approx([R, -R], abs=1e-5), "leaked: β_br·z, nothing from λ"
    assert set(by_condition) == {
        "root",
        "student:p1",
        "repair:p1:1",
        "repair:p1:2",
        "student:p2",
        "repair:p2:1",
        "ahat:p2:1",
        "repair:p2:2",
    }, "one â, for the one eligible candidate where no student resolved"
    assert {credit(s)["omega"] for s in samples if role(s) != "root"} == {0.5}, "two points with mass: ω = ½ each"
    assert all(credit(s)["lambda_share"] == 0 for s in samples if condition(s) == "repair:p2:2")

    (ahat,) = [s for s in samples if role(s) == "ahat"]
    inserted = mask_generator.tokenizer.apply_chat_template(
        [ahat.metadata["messages"][-1]], tokenize=True, return_dict=False
    )
    assert trained_tokens(ahat) == inserted[mask_generator.gen_token_length :], "the turn's reasoning and its call"
    assert mask_generator.tokenizer.decode(trained_tokens(ahat)).startswith("<think> parser broken </think>")
    assert credit(ahat) == {
        "kind": "ahat",
        "beta": None,
        "z": 0.0,
        "lambda_share": 0.5,
        "case": "FF",
        "point_id": "p2",
        "candidate": 1,
        "c_s": 0,
        "k_m": 2,
        "n_eligible": 1,
        "omega": 0.5,
        "r_prime": None,
    }

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


def test_a_points_share_is_split_among_its_resolved_students_or_its_eligible_candidates(mask_generator):
    body, slots = search_result(
        [0.0, 0.0],
        {
            "p1": {"students": [1.0, 1.0]},
            "p2": {"students": [0.0, 0.0], "candidates": [[1.0, 1.0], [1.0, 0.0]]},
        },
    )
    by_condition, samples, metrics = credited(mask_generator, body, slots)
    assert by_condition["student:p1"] == pytest.approx([0.1, 0.1]), "λ·ω/c_s = 0.4·½/2"
    assert by_condition["repair:p2:1"] == pytest.approx([0.05, 0.05]), "λ·ω/(|E|·K) = 0.4·½/(2·2)"
    assert by_condition["repair:p2:2"] == pytest.approx([R + 0.1, -R], abs=1e-5), "λ·ω/(|E|·K) = 0.4·½/(2·1)"
    assert by_condition["ahat:p2:1"] == by_condition["ahat:p2:2"] == pytest.approx([0.1]), "λ·ψ·ω/|E| = 0.4·½/2"
    assert metrics[f"{M}ahat"] == 2 and {credit(s)["n_eligible"] for s in samples if role(s) == "ahat"} == {2}


@pytest.mark.parametrize(
    "students, expected",
    [([1.0, 0.0], [R, -R]), ([1.0, 0.0, 0.0, 0.0], [1.5, -0.5, -0.5, -0.5])],
    ids=["S_TF=2", "S_TF=4"],
)
def test_a_split_task_is_trained_on_policy_without_lambda(mask_generator, students, expected):
    body, slots = search_result([1.0, 0.0], {"p1": {"students": students}})
    by_condition, samples, metrics = credited(mask_generator, body, slots)
    assert by_condition["root"] == pytest.approx([0.30, -0.30], abs=1e-6), "β_root·z = (1 - λ)/2"
    assert by_condition["student:p1"] == pytest.approx(expected, abs=1e-5), "β_br·z at the restart point"
    assert all(credit(s)["lambda_share"] == 0 and credit(s)["case"] == "TF" for s in samples)
    assert not any(role(s) == "ahat" for s in samples)
    assert metrics[f"{M}cases/TF"] == 1 and metrics[f"{M}lambda_tasks"] == 0 and metrics[f"{M}points"] == 0


def test_three_roots_one_of_them_missing_and_one_resolved_are_tf(mask_generator):
    body, slots = search_result([1.0, 0.0, None], {"p1": {"students": [1.0, 0.0]}})
    by_condition, samples, metrics = credited(mask_generator, body, slots, planned_roots=3)
    assert by_condition["root"] == pytest.approx([0.30, -0.30], abs=1e-6), "the missing root is not a 0"
    assert metrics[f"{M}cases/TF"] == 1


@pytest.mark.parametrize(
    "leak_terms, gate",
    [(["tests/test_hidden.py"], "candidates_leaked"), (None, "candidates_unverified")],
    ids=["leaked", "unchecked"],
)
def test_a_candidate_the_leak_gate_refuses_gets_its_z_score_and_no_ahat(mask_generator, leak_terms, gate):
    body, slots = search_result([0.0, 0.0], {"p1": {"students": [0.0, 0.0], "candidates": [[1.0, 0.0]]}})
    for trajectory in members(body, "repair:p1:1"):
        trajectory["metadata"]["search"]["leak_terms"] = leak_terms
    by_condition, samples, metrics = credited(mask_generator, body, slots)
    assert by_condition["repair:p1:1"] == pytest.approx([R, -R], abs=1e-5)
    assert not any(role(s) == "ahat" for s in samples) and all(credit(s)["lambda_share"] == 0 for s in samples)
    assert metrics[M + gate] == 1 and metrics[f"{M}points_no_mass"] == 1 and metrics[f"{M}lambda_tasks"] == 0


def test_a_continuation_that_touched_the_tests_resolved_nothing_in_z_or_k(mask_generator):
    body, slots = search_result([0.0, 0.0], {"p1": {"students": [0.0, 0.0], "candidates": [[1.0, 1.0], [1.0, 1.0]]}})
    members(body, "repair:p1:1")[0]["metadata"]["touched_test_paths"] = ["tests/test_parser.py"]
    for trajectory, touched in zip(members(body, "repair:p1:2"), (["tests/test_parser.py"], None), strict=True):
        trajectory["metadata"]["touched_test_paths"] = touched  # None: the grade could not say
    by_condition, samples, metrics = credited(mask_generator, body, slots)
    # Candidate 1: R' = (0, 1), so K = 1 and all of its λ goes to the clean continuation.
    assert by_condition["repair:p1:1"] == pytest.approx([-R, R + 0.4], abs=1e-5)
    assert by_condition["ahat:p1:1"] == pytest.approx([0.4]), "the only creditable point: ω = 1"
    # Candidate 2: R' = (0, 0), so K = 0: no spread, no credit, no â.
    assert by_condition["repair:p1:2"] == [0.0, 0.0] and "ahat:p1:2" not in by_condition
    assert [credit(s)["r_prime"] for s in samples if condition(s) == "repair:p1:1"] == [0.0, 1.0]
    assert [s.reward for s in samples if condition(s) == "repair:p1:1"] == [1.0, 1.0], "the verdict itself stands"
    assert metrics[f"{M}continuations_touching_tests"] == 3 and metrics[f"{M}candidates_unconfirmed"] == 1


def test_a_continuation_with_nothing_to_train_counts_in_z_and_k_but_is_no_sample(mask_generator):
    body, slots = search_result([0.0, 0.0], {"p1": {"students": [0.0, 0.0], "candidates": [[1.0, 0.0]]}})
    resolved = members(body, "repair:p1:1")[0]
    # Cut off right after the reviewer's turn, which resolved the task: no turn of its own.
    resolved["messages"] = resolved["messages"][:5]
    resolved.update(status="truncated", stop_reason="max_turns_reached")
    by_condition, samples, metrics = credited(mask_generator, body, slots)
    assert by_condition["repair:p1:1"] == pytest.approx([-R], abs=1e-5), "its R = 1 sets the other's z-score"
    assert by_condition["ahat:p1:1"] == pytest.approx([0.4]), "and K = 1 makes its candidate eligible"
    assert metrics[f"{M}estimation_only"] == 1 and len(samples) == 2 + 2 + 1 + 1


def test_a_point_without_a_gradable_student_has_no_lambda_and_no_share_of_omega(mask_generator):
    body, slots = search_result(
        [0.0, 0.0],
        {
            "p1": {"students": [None, LOST], "candidates": [[1.0, 1.0]]},
            "p2": {"students": [0.0, 0.0], "candidates": [[1.0, 1.0]]},
        },
    )
    by_condition, samples, metrics = credited(mask_generator, body, slots)
    assert "student:p1" not in by_condition, "a missing student is no sample and no 0"
    assert by_condition["repair:p1:1"] == [0.0, 0.0] and "ahat:p1:1" not in by_condition
    assert by_condition["repair:p2:1"] == pytest.approx([0.2, 0.2]), "P2 is the only point with mass: ω = 1"
    assert by_condition["ahat:p2:1"] == pytest.approx([0.4])
    assert metrics[f"{M}points_no_mass"] == 1 and metrics[f"{M}points_guided"] == 1


def test_ff_partial_has_no_lambda_and_tt_no_signal(mask_generator):
    body, slots = search_result([0.0, None], {"p1": {"students": [1.0, 0.0], "candidates": [[1.0, 1.0]]}})
    by_condition, samples, metrics = credited(mask_generator, body, slots)
    assert by_condition["root"] == [0.0], "one gradable root: z = 0; the one without a grade is no sample"
    assert by_condition["student:p1"] == pytest.approx([R, -R], abs=1e-5)
    assert by_condition["repair:p1:1"] == [0.0, 0.0]
    assert all(credit(s)["lambda_share"] == 0 for s in samples) and not any(role(s) == "ahat" for s in samples)
    assert metrics[f"{M}cases/FF_partial"] == 1 and metrics[f"{M}lambda_tasks"] == 0

    by_condition, samples, metrics = credited(mask_generator, *search_result([1.0, 1.0]))
    assert by_condition == {"root": [0.0, 0.0]} and metrics[f"{M}cases/TT"] == 1


@pytest.mark.parametrize(
    "change",
    [
        lambda body: members(body, "student:p1")[0]["metadata"]["search"].update(outcome="TF"),
        lambda body: members(body, "root")[0]["metadata"]["search"].update(outcome=None),
        lambda body: members(body, "root")[1]["metadata"]["search"]["root_counts"].update(resolved=1),
    ],
    ids=["student-labelled-tf", "root-unlabelled", "root-counts"],
)
def test_a_task_sprout_labelled_otherwise_is_refused(mask_generator, change):
    body, slots = search_result([0.0, 0.0], {"p1": {"students": [1.0, 0.0]}})
    change(body)
    with pytest.raises(ValueError, match="Sprout labels"):
        assemble(mask_generator, body, slots)


def test_roots_counted_against_another_plan_are_refused(mask_generator):
    with pytest.raises(ValueError, match="make the task 'FF_partial'"):
        assemble(mask_generator, *search_result([0.0, 0.0]), planned_roots=3)


@pytest.mark.parametrize(
    "render",
    [
        lambda turn: turn.update(content=turn.pop("reasoning_content")),
        lambda turn: turn.update(content="parser\nbroken"),
        lambda turn: turn.update(reasoning_content=" "),
    ],
    ids=["text-as-content", "multi-line-content", "blank-reasoning"],
)
def test_a_noncanonical_inserted_turn_makes_its_candidate_ineligible(mask_generator, render):
    body, slots = search_result([0.0, 0.0], {"p1": {"students": [0.0, 0.0], "candidates": [[1.0, 1.0]]}})
    for trajectory in members(body, "repair:p1:1"):
        render(trajectory["messages"][3])
    by_condition, samples, metrics = credited(mask_generator, body, slots)
    assert by_condition["repair:p1:1"] == [0.0, 0.0] and "ahat:p1:1" not in by_condition
    assert metrics[f"{M}candidates_noncanonical"] == 1 and metrics[f"{M}points_no_mass"] == 1


@pytest.mark.parametrize(
    "change, message",
    [
        (lambda body: members(body, "student:p1")[0]["metadata"].update(search=None), "search annotations"),
        (lambda body: members(body, "student:p1")[0]["metadata"].pop("zero_reason"), "search annotations"),
        (lambda body: members(body, "student:p1")[0]["metadata"]["search"].pop("point_id"), "name its point"),
        (lambda body: members(body, "repair:p1:1")[0]["metadata"]["search"].pop("candidate"), "its candidate"),
        (lambda body: members(body, "repair:p1:1")[0].update(group="repair:p1:2"), "its annotation puts it"),
        (lambda body: members(body, "student:p1")[0].update(reward=0.5), "0/1 verdict"),
    ],
)
def test_missing_annotations_fail_closed(mask_generator, change, message):
    body, slots = search_result([0.0, 0.0], {"p1": {"students": [1.0, 0.0], "candidates": [[1.0]]}})
    change(body)
    with pytest.raises(ValueError, match=message):
        assemble(mask_generator, body, slots)


def test_roots_keep_their_group_and_each_condition_and_ahat_gets_its_own(mask_generator):
    samples, _ = assemble(mask_generator, *ff_example())
    groups = {}
    for sample in samples:
        groups.setdefault(condition(sample), set()).add(sample.group_index)
    assert groups.pop("root") == {3}
    assert all(len(indices) == 1 for indices in groups.values()), "a condition is one group"
    fresh = set().union(*groups.values())
    assert len(fresh) == len(groups) and min(fresh) >= 50, "fresh indices, one per condition and per â"
    assert [s.index for s in samples if role(s) == "ahat"] == [100], "the â's identity comes after every slot's"
    assert all(credit(s) == s.metadata["sprout_rollout"]["credit"] for s in samples)
    assert all(s.train_metadata["sprout_rollout"]["rollout_job_id"] == "job" for s in samples)


def test_a_group_whose_every_root_failed_credits_nothing(mask_generator):
    samples, metrics = assemble(mask_generator, *search_result([LOST, LOST]))
    assert samples == [] and metrics[f"{M}cases/ungraded"] == 1
    # One root's actor failed, the other's grade gave no verdict: a root came back, nothing to credit.
    samples, metrics = assemble(mask_generator, *search_result([LOST, None]))
    assert samples == [] and metrics[f"{M}cases/ungraded"] == 1


def test_a_saved_rollout_is_re_credited_alike_and_under_new_flags(mask_generator):
    samples, _ = assemble(mask_generator, *ff_example())
    raw, advantages = post_process_gsml(gsml_args(), samples)
    assert all(a >= 0 for s, a in zip(samples, advantages, strict=True) if role(s) == "ahat")
    # What --save-debug-rollout-data writes and --load-debug-rollout-data reads.
    buffer = io.BytesIO()
    torch.save({"samples": [sample.to_dict() for sample in samples]}, buffer)
    buffer.seek(0)
    restored = [Sample.from_dict(data) for data in torch.load(buffer, weights_only=False)["samples"]]
    assert post_process_gsml(gsml_args(), restored) == (raw, advantages)
    # The trainer receives train_metadata as msgpack.
    assert all(msgpack.unpackb(msgpack.packb(s.train_metadata)) == s.train_metadata for s in samples)
    _, relaxed = post_process_gsml(gsml_args(sprout_rollout_gsml_lambda=0.3, sprout_rollout_gsml_psi=0.5), restored)
    assert [a for s, a in zip(restored, relaxed, strict=True) if role(s) == "ahat"] == pytest.approx([0.3 * 0.5 * 0.5])
    assert [a for s, a in zip(restored, relaxed, strict=True) if condition(s) == "student:p1"] == pytest.approx(
        [0.7071 + 0.3 * 0.5, -0.7071], abs=1e-4
    )


def test_post_processing_refuses_a_sample_without_credit_and_flags_it_cannot_apply(mask_generator):
    plain = Sample(index=0, reward=1.0, train_metadata={"sprout_rollout": {"role": "root"}})
    with pytest.raises(ValueError, match="no GSML credit"):
        post_process_gsml(gsml_args(), [plain])
    samples, _ = assemble(mask_generator, *ff_example())
    for flags, message in [
        ({"sprout_rollout_gsml_lambda": -0.1}, "lambda must be finite and in"),
        ({"sprout_rollout_gsml_lambda": 1.5}, "lambda must be finite and in"),
        ({"sprout_rollout_gsml_beta_branch": -1.0}, "beta-branch must be finite and nonnegative"),
        ({"sprout_rollout_gsml_psi": float("nan")}, "psi must be finite"),
        ({"sprout_rollout_gsml_c_pre": 0.5}, "c-pre is not implemented"),
        ({"sprout_rollout_gsml_kappa_plus": 0.1}, "kappa-plus is not implemented"),
    ]:
        with pytest.raises(ValueError, match=message):
            post_process_gsml(gsml_args(**flags), samples)


@pytest.mark.parametrize(
    "counts, label",
    [
        ((2, 2, 2), "TT"),
        ((2, 1, 1), "TT_partial"),
        ((2, 2, 1), "TF"),
        ((2, 2, 0), "FF"),
        ((2, 1, 0), "FF_partial"),
        ((2, 0, 0), "ungraded"),
        ((1, 1, 0), "FF"),
        ((1, 1, 1), "TT"),
        ((3, 2, 1), "TF"),
        ((3, 2, 2), "TT_partial"),
        ((3, 2, 0), "FF_partial"),
    ],
)
def test_search_outcome(counts, label):
    assert search_outcome(*counts) == label


@pytest.mark.parametrize("counts", [(2, 1, 2), (2, 3, 0), (2, 0, -1), (2.0, 2, 0), (2, True, 0)])
def test_search_outcome_refuses_impossible_counts(counts):
    with pytest.raises(ValueError, match="resolved <= graded <= planned"):
        search_outcome(*counts)


def test_group_z():
    assert group_z([]) == [] and group_z([1.0]) == [0.0] and group_z([1.0, 1.0, 1.0]) == [0.0, 0.0, 0.0]
    assert group_z([1.0, 0.0]) == pytest.approx([R, -R], abs=1e-5)
    assert group_z([1.0, 0.0, 0.0, 0.0]) == pytest.approx([1.5, -0.5, -0.5, -0.5], abs=1e-5), "unbiased std"


def test_the_reasoning_probe_passes_a_template_that_trains_reasoning_and_refuses_one_that_drops_it():
    check_reasoning_is_trained(toy_mask_generator())
    dropping = TEMPLATE.replace(
        "{% if m.reasoning_content %}<think> {{ m.reasoning_content }} </think> {% endif %}", ""
    )
    assert dropping != TEMPLATE
    with pytest.raises(ValueError, match="do not train an assistant turn's reasoning_content"):
        check_reasoning_is_trained(toy_mask_generator(dropping))
    with pytest.raises(ValueError, match="distill_qwen do not train"):
        check_reasoning_is_trained(toy_mask_generator(tokenizer_type="distill_qwen"))
    # Qwen3.8's template, for one, refuses an assistant turn rendered without a user message before it.
    refusing = TEMPLATE.replace(
        "{% for m in messages %}",
        "{% for m in messages %}{% if m.role == 'assistant' %}{{ raise_exception('No user query found') }}{% endif %}",
    )
    with pytest.raises(ValueError, match="cannot render an inserted turn.*No user query found"):
        check_reasoning_is_trained(toy_mask_generator(refusing))


def _chained(body, slot_index, turns):
    """Make the repair in slot ``job:slot:<slot_index>`` a round-2 continuation of a repair chain: its
    history holds the round-1 turn, the actor's turn after it, then the round-2 turn, each inserted."""
    from copy import deepcopy

    from tests.fast.rollout.sprout.conftest import INSERTED_TURN, PREFIX

    (trajectory,) = [t for t in body["trajectories"] if t["sample_slot_id"] == f"job:slot:{slot_index}"]
    second = deepcopy(INSERTED_TURN)
    second["reasoning_content"] = turns
    second["tool_calls"][0]["id"] = "inserted-2"
    actor = {
        "role": "assistant",
        "content": "inspect",
        "tool_calls": [{"id": "actor-1", "type": "function", "function": {"name": "shell", "arguments": '{"command":"pwd"}'}}],
    }
    trajectory["messages"] = [
        *PREFIX,
        INSERTED_TURN,
        {"role": "tool", "tool_call_id": "inserted-1", "content": "tests"},
        actor,
        {"role": "tool", "tool_call_id": "actor-1", "content": "/workspace"},
        second,
        {"role": "tool", "tool_call_id": "inserted-2", "content": "tests"},
        {"role": "assistant", "content": "fixed"},
    ]
    trajectory["metadata"]["provenance"] = {"prefix_messages": 7, "inserted_messages": [3, 7]}
    trajectory["metadata"]["search"]["round"] = 2
    return trajectory


def test_a_resolved_repair_chain_trains_every_round_s_turn_as_an_ahat_sample(mask_generator):
    # P1: its student failed, round 1's repair failed, round 2's (candidate 2, after round 1's turn) resolved.
    body, slots = search_result([0.0, 0.0], {"p1": {"students": [0.0], "candidates": [[0.0], [1.0]]}})
    _chained(body, 15, "inspect tests")
    by_condition, samples, metrics = credited(mask_generator, body, slots)
    assert by_condition["repair:p1:1"] == [0.0], "round 1's repair failed: one continuation, z = 0, no λ"
    assert by_condition["repair:p1:2"] == pytest.approx([0.4]), "λ·ω/(|E|·K) = 0.4·1/(1·1)"
    assert by_condition["ahat:p1:2"] == pytest.approx([0.2, 0.2]), "one â per round's turn: λ·ψ·ω/(|E|·2)"
    ahats = [s for s in samples if role(s) == "ahat"]
    decoded = [mask_generator.tokenizer.decode(trained_tokens(s)) for s in ahats]
    assert decoded[0].startswith("<think> parser broken </think>"), "round 1's turn, after the root's prefix"
    assert decoded[1].startswith("<think> inspect tests </think>"), "round 2's turn, after round 1's history"
    assert metrics[f"{M}ahat"] == 2


def test_a_chain_whose_earlier_turn_is_not_canonical_is_not_eligible(mask_generator):
    body, slots = search_result([0.0, 0.0], {"p1": {"students": [0.0], "candidates": [[0.0], [1.0]]}})
    trajectory = _chained(body, 15, "inspect tests")
    trajectory["messages"][3] = {**trajectory["messages"][3], "content": "two\nlines"}
    by_condition, _, metrics = credited(mask_generator, body, slots)
    assert "ahat:p1:2" not in by_condition and by_condition["repair:p1:2"] == [0.0]
    assert metrics[f"{M}candidates_noncanonical"] == 1
