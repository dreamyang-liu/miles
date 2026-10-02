import pytest
from pydantic import ValidationError
from tests.fast.rollout.sprout.conftest import sprout_request, sprout_result

from miles.rollout.sprout.protocol import Acknowledgement, RolloutRequest, RolloutResult


def test_sprout_request_example_validates_and_round_trips():
    """The request Sprout documents is a request Miles's model accepts
    unchanged; serializing it back adds only the documented default."""
    example = sprout_request()
    request = RolloutRequest.model_validate(example)
    assert request.model_dump(mode="json", exclude_unset=True) == example
    assert request.model_dump(mode="json") == {**example, "finalization_timeout_seconds": 1800.0}


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
