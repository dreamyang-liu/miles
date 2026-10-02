import pytest
from tests.fast.rollout.sprout.conftest import group, sprout_result

from miles.rollout.sprout.importer import import_trajectories
from miles.rollout.sprout.protocol import RolloutResult
from miles.utils.types import Sample


def slots(samples):
    return {sprout_result()["trajectories"][0]["sample_slot_id"]: samples[0]}


def trained_text(mask_generator, sample):
    response = sample.tokens[-sample.response_length :]
    return mask_generator.tokenizer.decode([t for t, keep in zip(response, sample.loss_mask, strict=True) if keep])


def test_import_rebuilds_tokens_from_messages_and_trains_assistant_words_only(mask_generator):
    source = group()[0]
    source.tokens = [100, 200]
    source.rollout_log_probs = [-0.2]
    result = RolloutResult.model_validate(sprout_result())
    (sample,) = import_trajectories(result, slots([source]), mask_generator, weight_version=7)
    assert source.tokens == [100, 200], "the slot sample is copied, not edited"
    assert sample.tokens != source.tokens and sample.rollout_log_probs is None and sample.weight_versions == []
    assert sample.reward == 1.0 and sample.status == Sample.Status.COMPLETED
    assert len(sample.loss_mask) == sample.response_length and any(sample.loss_mask)
    trained = trained_text(mask_generator, sample)
    assert "fixed" in trained
    assert "/workspace" not in trained, "tool output is context, never a target"
    assert "/workspace" in mask_generator.tokenizer.decode(sample.tokens)
    assert sample.prompt == [{"role": "user", "content": "fix task"}]
    assert sample.group_index == 3 and sample.index == 11, "GRPO groups by the slot's identity"
    lineage = sample.metadata["sprout_rollout"]
    assert lineage["branch_id"] == "job-0" and lineage["parent_branch_id"] is None
    assert lineage["weight_version"] == 7 and lineage["logprob_context"] == "hint_free_messages"
    assert sample.train_metadata["sprout_rollout"]["rollout_job_id"] == result.rollout_job_id
    assert sample.metadata["messages"][-1] == {"role": "assistant", "content": "fixed"}


def test_tool_arguments_become_mappings_for_the_chat_template(mask_generator):
    body = sprout_result()
    assert isinstance(body["trajectories"][0]["messages"][1]["tool_calls"][0]["function"]["arguments"], str)
    (sample,) = import_trajectories(RolloutResult.model_validate(body), slots(group()), mask_generator)
    call = sample.metadata["messages"][1]["tool_calls"][0]
    assert call["function"]["arguments"] == {"command": "pwd"}


@pytest.mark.parametrize("reason", ["timeout", "max_turns_reached"])
def test_graded_cutoff_is_truncated_with_its_reason(mask_generator, reason):
    body = sprout_result()
    body["trajectories"][0].update(status="truncated", stop_reason=reason, reward=0.0)
    (sample,) = import_trajectories(RolloutResult.model_validate(body), slots(group()), mask_generator)
    assert sample.status == Sample.Status.TRUNCATED and sample.reward == 0.0
    assert sample.metadata["sprout_rollout"]["stop_reason"] == reason
    assert sample.train_metadata["sprout_rollout"]["stop_reason"] == reason
    assert any(sample.loss_mask), "a truncated trajectory is still trained on what it did"


def test_unremoved_hint_is_an_error_not_a_training_target(mask_generator):
    body = sprout_result()
    body["trajectories"][0]["messages"][0][
        "content"
    ] += " <sprout_training_hint>look at the parser</sprout_training_hint>"
    with pytest.raises(ValueError, match="unremoved training hint"):
        import_trajectories(RolloutResult.model_validate(body), slots(group()), mask_generator)


@pytest.mark.parametrize(
    "change, message",
    [
        (lambda m: m[2].update(tool_call_id="other"), "unknown tool_call_id"),
        (lambda m: m[1]["tool_calls"][0]["function"].update(arguments="[1, 2]"), "JSON object"),
        (lambda m: m.pop(2), "unanswered tool calls"),
        (lambda m: m[0].update(role="developer"), "unsupported message role"),
    ],
)
def test_broken_conversations_are_rejected(mask_generator, change, message):
    body = sprout_result()
    change(body["trajectories"][0]["messages"])
    with pytest.raises(ValueError, match=message):
        import_trajectories(RolloutResult.model_validate(body), slots(group()), mask_generator)


def test_unknown_slot_and_unfinished_group_are_rejected(mask_generator):
    body = sprout_result()
    body["trajectories"][0]["sample_slot_id"] = "someone-else"
    with pytest.raises(ValueError, match="unknown sample_slot_id"):
        import_trajectories(RolloutResult.model_validate(body), slots(group()), mask_generator)
    body = sprout_result()
    body.update(status="failed", stop_reason="slot: Sprout grading did not produce a resolved verdict")
    with pytest.raises(ValueError, match="status='failed'.*resolved verdict"):
        import_trajectories(RolloutResult.model_validate(body), slots(group()), mask_generator)


def branch_body():
    """A branch trajectory: the parent's restored turn, an inserted turn, then the policy's own."""
    body = sprout_result()
    body["trajectories"][0]["messages"] = [
        {"role": "user", "content": "fix task"},
        {"role": "assistant", "content": "inspect", "tool_calls": [
            {"id": "parent-1", "type": "function", "function": {"name": "shell", "arguments": '{"command":"pwd"}'}}]},
        {"role": "tool", "tool_call_id": "parent-1", "content": "/workspace"},
        {"role": "assistant", "content": "hint", "tool_calls": [
            {"id": "inserted-1", "type": "function", "function": {"name": "shell", "arguments": '{"command":"ls"}'}}]},
        {"role": "tool", "tool_call_id": "inserted-1", "content": "tests"},
        {"role": "assistant", "content": "fixed"},
    ]
    body["trajectories"][0]["parent_branch_id"] = "parent-job"
    body["trajectories"][0]["metadata"]["provenance"] = {"prefix_messages": 3, "inserted_messages": [3]}
    return body


def test_a_branch_trains_only_what_its_policy_sampled(mask_generator):
    """The restored prefix and the inserted turn are rendered as context and kept
    out of the loss; the turn the policy produced after them is the target."""
    (sample,) = import_trajectories(RolloutResult.model_validate(branch_body()), slots(group()), mask_generator)
    trained = trained_text(mask_generator, sample)
    assert "fixed" in trained
    assert "inspect" not in trained and "hint" not in trained
    context = mask_generator.tokenizer.decode(sample.tokens)
    assert "inspect" in context and "hint" in context and "tests" in context
    assert [message.get("step_loss_mask") for message in sample.metadata["messages"]] == [0, 0, 0, 0, None, None]


@pytest.mark.parametrize("provenance, message", [
    (None, "say which of its messages"),
    ({"prefix_messages": 0}, "say which of its messages"),
    ({"prefix_messages": 7, "inserted_messages": []}, "prefix_messages"),
    ({"prefix_messages": 0, "inserted_messages": [2]}, "index assistant messages"),
    ({"prefix_messages": 0, "inserted_messages": [9]}, "index assistant messages"),
])
def test_a_trajectory_must_say_which_messages_the_policy_sampled(mask_generator, provenance, message):
    body = branch_body()
    metadata = body["trajectories"][0]["metadata"]
    if provenance is None:
        del metadata["provenance"]
    else:
        metadata["provenance"] = provenance
    with pytest.raises(ValueError, match=message):
        import_trajectories(RolloutResult.model_validate(body), slots(group()), mask_generator)


def test_a_branch_whose_own_turns_are_all_unsampled_has_nothing_to_train(mask_generator):
    body = branch_body()
    body["trajectories"][0]["metadata"]["provenance"] = {"prefix_messages": 3, "inserted_messages": [3, 5]}
    with pytest.raises(ValueError, match="no trainable assistant tokens"):
        import_trajectories(RolloutResult.model_validate(body), slots(group()), mask_generator)
