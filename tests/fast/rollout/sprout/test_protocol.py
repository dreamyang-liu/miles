import pytest
from pydantic import ValidationError
from tests.fast.rollout.sprout.conftest import sprout_request, sprout_result

from miles.rollout.sprout.protocol import Acknowledgement, RolloutRequest, RolloutResult, SearchSpec

#: Two roots, one point in each failed one; a split task's point gets two students.
SEARCH = {
    "roots": 2,
    "points": 1,
    "candidates": 2,
    "student_continuations": 2,
    "candidate_continuations": 2,
    "tf_student_continuations": 2,
    "repair_rounds": 1,
    "review_seconds": 1200.0,
    "wall_time_seconds": 1200.0,
}


def search_request(search: dict, slots: int = 14) -> dict:
    """Sprout's request example as a search: the roots' slots, then one for each branch."""
    body = sprout_request()
    job = body["rollout_job_id"]
    body["sample_slots"] = [{"sample_slot_id": f"{job}:slot:{11 + k}", "sample_index": 11 + k} for k in range(slots)]
    return {**body, "max_samples": slots, "search": search}


def test_sprout_request_example_validates_and_round_trips():
    """The request Sprout documents is a request Miles's model accepts
    unchanged; serializing it back adds only the documented default."""
    example = sprout_request()
    request = RolloutRequest.model_validate(example)
    assert request.model_dump(mode="json", exclude_unset=True) == example
    assert request.wire() == {**example, "finalization_timeout_seconds": 1800.0}, "no search: no search key on the wire"


def test_a_search_request_round_trips_with_its_split_task_students():
    body = search_request(SEARCH)
    request = RolloutRequest.model_validate(body)
    assert request.search.tf_student_continuations == 2 and request.search.branches() == 12
    assert request.wire() == {**body, "finalization_timeout_seconds": 1800.0}
    assert list(request.wire()["search"]) == list(SEARCH), "Sprout's key order"
    assert RolloutRequest.model_validate(request.wire()) == request


@pytest.mark.parametrize(
    "change",
    [
        lambda search: search.pop("tf_student_continuations"),
        lambda search: search.update(tf_student_continuations=-1),
        lambda search: search.update(tf_student_continuations=2.0),
        lambda search: search.update(tf_student_continuations=True),
        lambda search: search.update(tf_students=2),
    ],
    ids=["missing", "negative", "float", "bool", "unknown-key"],
)
def test_a_search_without_a_valid_split_task_student_count_is_refused(change):
    search = dict(SEARCH)
    change(search)
    with pytest.raises(ValidationError):
        RolloutRequest.model_validate(search_request(search))


def test_branches_cover_the_outcome_that_makes_the_most():
    assert SearchSpec(**SEARCH).branches() == 1 * max(2 * (2 + 2 * 2), (2 - 1) * 2) == 12, "both roots failed"
    students_only = {**SEARCH, "candidates": 0, "candidate_continuations": 0, "student_continuations": 1}
    assert SearchSpec(**{**students_only, "tf_student_continuations": 4}).branches() == 4, "the roots split"
    assert SearchSpec(**{**students_only, "tf_student_continuations": 0}).branches() == 2
    assert SearchSpec(**{**SEARCH, "points": 3}).branches() == 36
    assert SearchSpec(**{**SEARCH, "roots": 1, "tf_student_continuations": 9}).branches() == 6, "one root cannot split"
    # The request needs a slot for each: 2 roots and 13 branches when the split task's students dominate.
    dominating = {**SEARCH, "tf_student_continuations": 13}
    assert RolloutRequest.model_validate(search_request(dominating, slots=15)).search.branches() == 13
    with pytest.raises(ValidationError, match="max_samples must cover the roots and every branch"):
        RolloutRequest.model_validate(search_request(dominating, slots=14))


def test_sprout_result_example_validates_and_round_trips():
    example = sprout_result()
    result = RolloutResult.model_validate(example)
    assert result.model_dump(mode="json") == example
    assert result.trajectories[0].reward == 1.0 and result.trajectories[0].status == "completed"


@pytest.mark.parametrize(
    "change",
    [
        lambda body: body.update(protocol_version="ash-rollout-v3"),
        lambda body: body.update(max_model_calls=100),
        lambda body: body["budgets"].update(max_tool_calls=100),
        lambda body: body.update(max_turns=None),
        lambda body: body.update(max_turns=0),
        lambda body: body.update(minimum_returned_samples=2),
        lambda body: body.update(sample_slots=body["sample_slots"] * 2),
        lambda body: body.update(prompt=""),
        lambda body: body.update(rollout_id=-1),
        lambda body: body["budgets"].update(max_wall_time_seconds=0),
        lambda body: body.update(finalization_timeout_seconds=float("inf")),
    ],
)
def test_request_rejects_what_sprout_rejects(change):
    """Sprout refuses unknown fields and these bounds; Miles refuses them
    before the request leaves, so the error names the field, not an HTTP 400."""
    body = sprout_request()
    change(body)
    with pytest.raises(ValidationError):
        RolloutRequest.model_validate(body)


@pytest.mark.parametrize(
    "change",
    [
        lambda body: body["trajectories"][0].update(status="truncated"),
        lambda body: body["trajectories"][0].update(stop_reason="timeout"),
        lambda body: body["trajectories"][0].update(hints_removed=False),
        lambda body: body["trajectories"][0].update(reward=None),
        lambda body: body["trajectories"][0].update(reward="1"),
        lambda body: body["trajectories"][0].update(reward={"score": 1}),
        lambda body: body["trajectories"][0].update(token_ids=[1, 2]),
        lambda body: body["trajectories"][0]["messages"][0].update(content=[{"type": "text", "text": "x"}]),
        lambda body: body.update(actual_samples=2),
        lambda body: body.update(status="done"),
    ],
)
def test_result_rejects_wrong_representations(change):
    body = sprout_result()
    change(body)
    with pytest.raises(ValidationError):
        RolloutResult.model_validate(body)


def test_truncated_trajectory_carries_its_reason():
    body = sprout_result()
    body["trajectories"][0].update(status="truncated", stop_reason="max_turns_reached", reward=0)
    result = RolloutResult.model_validate(body)
    assert result.trajectories[0].stop_reason == "max_turns_reached"
    assert result.trajectories[0].reward == 0


def test_acknowledgement_is_id_and_status_only():
    assert Acknowledgement.model_validate({"rollout_job_id": "job", "status": "queued"}).status == "queued"
    with pytest.raises(ValidationError):
        Acknowledgement.model_validate({"rollout_job_id": "job", "status": "queued", "protocol_version": "v3"})
