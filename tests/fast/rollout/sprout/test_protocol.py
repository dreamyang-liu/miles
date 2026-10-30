import pytest
from pydantic import ValidationError
from tests.fast.rollout.sprout.conftest import sprout_request, sprout_result

from miles.rollout.sprout.protocol import Acknowledgement, RafoSearchSpec, RolloutRequest, RolloutResult

#: Two roots; at most three events of two continuations, two deep.
SEARCH = {
    "mode": "rafo",
    "roots": 2,
    "continuations": 2,
    "depth": 2,
    "events": 3,
    "review_seconds": 1200.0,
    "wall_time_seconds": 1200.0,
}


def search_request(search: dict, slots: int = 8) -> dict:
    """Sprout's request example as a RAFO search: the roots' slots, then one for each continuation."""
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
    assert request.wire() == {
        **example,
        "finalization_timeout_seconds": 1800.0,
    }, "no search: no search key on the wire"


def test_a_rafo_search_request_round_trips_with_its_mode():
    body = search_request(SEARCH)
    request = RolloutRequest.model_validate(body)
    assert request.search.branches() == 6 and request.search.mode == "rafo"
    assert request.wire() == {**body, "finalization_timeout_seconds": 1800.0}
    assert list(request.wire()["search"]) == list(SEARCH), "Sprout's key order"
    assert RolloutRequest.model_validate(request.wire()) == request
    built = RafoSearchSpec(**{key: value for key, value in SEARCH.items() if key != "mode"})
    assert built.model_dump(mode="json") == SEARCH, "a spec Miles builds names its mode on the wire"


@pytest.mark.parametrize(
    "change",
    [
        lambda search: search.pop("events"),
        lambda search: search.update(events=0),
        lambda search: search.update(continuations=2.0),
        lambda search: search.update(depth=True),
        lambda search: search.update(mode="hindsight"),
        lambda search: search.update(points=1),
    ],
    ids=["missing", "zero", "float", "bool", "another-mode", "unknown-key"],
)
def test_a_rafo_search_with_a_missing_unknown_or_invalid_field_is_refused(change):
    search = dict(SEARCH)
    change(search)
    with pytest.raises(ValidationError):
        RolloutRequest.model_validate(search_request(search))


def test_the_slots_cover_a_continuation_group_for_each_event_and_the_wait_each_depth():
    spec = RafoSearchSpec(**{**SEARCH, "events": 4, "continuations": 3, "depth": 3})
    assert spec.branches() == 12
    assert spec.search_seconds(600.0) == 3 * (1200.0 + 1200.0) + 2 * 600.0
    assert RafoSearchSpec(**{**SEARCH, "events": 2, "depth": 3}).search_seconds(600.0) == 2 * 2400.0 + 600.0
    with pytest.raises(ValidationError, match="max_samples must cover the roots and every branch"):
        RolloutRequest.model_validate(search_request(SEARCH, slots=7))


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


@pytest.mark.parametrize("reason", ["format_invalid", "response_truncated", "repetition"])
def test_an_invalid_attempt_is_a_truncated_trajectory_with_reward_minus_one(reason):
    body = sprout_result()
    body["trajectories"][0].update(status="truncated", stop_reason=reason, reward=-1.0)
    result = RolloutResult.model_validate(body)
    assert (result.trajectories[0].stop_reason, result.trajectories[0].reward) == (reason, -1.0)


def test_acknowledgement_is_id_and_status_only():
    assert Acknowledgement.model_validate({"rollout_job_id": "job", "status": "queued"}).status == "queued"
    with pytest.raises(ValidationError):
        Acknowledgement.model_validate({"rollout_job_id": "job", "status": "queued", "protocol_version": "v3"})
