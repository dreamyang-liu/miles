"""Training samples from Sprout's hint-free messages.

Sprout returns the conversation the agent actually had, minus the branch hints
it marked, and says nothing about tokens: Miles renders the messages with the
policy's own chat template, so the tokens, the loss mask and later the
log-probs all describe one sequence the trainer can score. Nothing from the
sampling process survives on purpose -- a hint-conditioned rollout's token
log-probs would not describe this sequence.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from copy import deepcopy

from miles.rollout.sprout.protocol import HINT_END, HINT_START, RolloutResult, Trajectory
from miles.utils.mask_utils import MultiTurnLossMaskGenerator
from miles.utils.types import Sample

SAMPLE_STATUS = {"completed": Sample.Status.COMPLETED, "truncated": Sample.Status.TRUNCATED}
ROLES = {"system", "user", "assistant", "tool"}
#: What a returned trajectory is in a search (``metadata.search.kind``); a root without one.
SEARCH_KINDS = {"root", "student", "repair"}


def import_trajectories(
    result: RolloutResult,
    slot_samples: Mapping[str, Sample],
    mask_generator: MultiTurnLossMaskGenerator,
    *,
    weight_version: int | None = None,
) -> list[Sample]:
    """One Miles sample per returned trajectory, in the slots Miles allocated."""
    if result.status == "failed" and not result.trajectories:
        return []   # every slot's actor failed: nothing to train on (``validate_result`` decides if that is allowed)
    if result.status not in {"completed", "early_stopped"}:
        raise ValueError(f"cannot import a rollout group with status={result.status!r}: {result.stop_reason}")
    if result.max_samples > len(slot_samples):
        raise ValueError("the driver's max_samples exceeds the slots Miles allocated")
    samples = []
    for trajectory in result.trajectories:
        if trajectory.sample_slot_id not in slot_samples:
            raise ValueError(f"unknown sample_slot_id: {trajectory.sample_slot_id!r}")
        sample = import_trajectory(
            result, trajectory, slot_samples[trajectory.sample_slot_id], mask_generator, weight_version=weight_version
        )
        if sample is None:
            # Without a search every returned trajectory is a GRPO sample of a fixed-size group.
            raise ValueError("the rendered trajectory has no trainable assistant tokens")
        samples.append(sample)
    return samples


def prepare_messages(trajectory: Trajectory) -> list[dict]:
    """The trajectory's messages as the chat template wants them, checked."""
    messages = deepcopy(trajectory.messages)
    pending: set[str] = set()
    seen: set[str] = set()
    for message in messages:
        role = message.get("role")
        if role not in ROLES:
            raise ValueError(f"unsupported message role {role!r}")
        if role in {"user", "system"} and any(
            marker in (message.get("content") or "") for marker in (HINT_START, HINT_END)
        ):
            raise ValueError("Sprout returned an unremoved training hint")
        for call in message.get("tool_calls") or []:
            if role != "assistant":
                raise ValueError("only assistant messages carry tool calls")
            identifier = call.get("id")
            if not identifier or identifier in seen:
                raise ValueError("tool calls need distinct ids")
            seen.add(identifier)
            pending.add(identifier)
            function = call.get("function") or {}
            arguments = function.get("arguments", {})
            if isinstance(arguments, str):
                arguments = json.loads(arguments)
            if not isinstance(arguments, dict):
                raise ValueError("tool arguments must decode to a JSON object")
            # Chat Completions carries arguments as a JSON string; HF tool
            # templates render a mapping.
            function["arguments"] = arguments
        if role == "tool":
            if message.get("tool_call_id") not in pending:
                raise ValueError(f"unknown tool_call_id {message.get('tool_call_id')!r}")
            pending.remove(message["tool_call_id"])
    if pending:
        raise ValueError("the conversation ends with unanswered tool calls")
    if not any(message["role"] == "assistant" for message in messages):
        raise ValueError("a trajectory needs at least one assistant message")
    return messages


def unsampled_messages(trajectory: Trajectory) -> set[int]:
    """Indices of the messages no policy sampled in this trajectory, from Sprout's provenance.

    A branch restores its parent's history (``prefix_messages``) and may insert
    an assistant turn written outside the actor (``inserted_messages``). Both
    stay in the context the trainer scores, but neither is the policy's output
    to learn from.
    """
    provenance = trajectory.metadata.get("provenance")
    if not isinstance(provenance, dict) or set(provenance) != {"prefix_messages", "inserted_messages"}:
        raise ValueError("a Sprout trajectory must say which of its messages the policy sampled")
    prefix, inserted = provenance["prefix_messages"], provenance["inserted_messages"]
    count = len(trajectory.messages)
    if type(prefix) is not int or not 0 <= prefix <= count:
        raise ValueError("provenance prefix_messages is not a message count of this trajectory")
    if not isinstance(inserted, list) or any(
        type(index) is not int or not 0 <= index < count or trajectory.messages[index].get("role") != "assistant"
        for index in inserted
    ):
        raise ValueError("provenance inserted_messages must index assistant messages of this trajectory")
    return set(range(prefix)) | set(inserted)


def import_trajectory(
    result: RolloutResult,
    trajectory: Trajectory,
    slot_sample: Sample,
    mask_generator: MultiTurnLossMaskGenerator,
    *,
    weight_version: int | None,
) -> Sample | None:
    """The trajectory as a sample of its slot; None when no token of it is the policy's to learn.

    A branch whose own turns were all restored or inserted still tells what
    happened from its state, so a search counts its reward (it is
    estimation-only, ``gsml.assemble_search_group``); there is nothing in it
    to train.
    """
    messages = prepare_messages(trajectory)
    for index in unsampled_messages(trajectory):
        # Rendered as context, excluded from the loss (MultiTurnLossMaskGenerator).
        messages[index]["step_loss_mask"] = 0
    search = trajectory.metadata.get("search")
    if search is not None and (not isinstance(search, dict) or search.get("kind") not in SEARCH_KINDS):
        raise ValueError(f"metadata.search of {trajectory.sample_slot_id!r} does not say what the trajectory is")
    lineage = {
        "rollout_job_id": result.rollout_job_id,
        "prompt_group_id": result.prompt_group_id,
        # Its place in a search, which decides how it is credited (gsml.Credit).
        **_trajectory_lineage(
            trajectory, weight_version, stop_reason=trajectory.stop_reason, role=search["kind"] if search else "root"
        ),
    }
    return _rendered_sample(
        slot_sample,
        trajectory,
        messages,
        mask_generator,
        reward=float(trajectory.reward),
        status=SAMPLE_STATUS[trajectory.status],
        lineage=lineage,
    )


def ahat_sample(
    trajectory: Trajectory,
    slot_sample: Sample,
    mask_generator: MultiTurnLossMaskGenerator,
    *,
    index: int,
    group_index: int,
    weight_version: int | None,
    turn: int | None = None,
) -> Sample | None:
    """A repair continuation's inserted turn as a one-turn target: GSML's â sample.

    ``turn`` is the message index of an inserted turn, by default the last
    one (a repair chain inserts one per round). The context is the
    continuation's history up to that turn, as the policy would see it there;
    only the turn itself -- its reasoning and its calls -- is trained, with
    reward 1.0. None when the turn renders no trainable token.
    """
    unsampled_messages(trajectory)  # the provenance is checked before it is read
    inserted = trajectory.metadata["provenance"]["inserted_messages"]
    if not inserted:
        raise ValueError(f"{trajectory.sample_slot_id!r} has no inserted turn to make an â sample of")
    if turn is None:
        turn = max(inserted)
    elif turn not in inserted:
        raise ValueError(f"message {turn} of {trajectory.sample_slot_id!r} is not an inserted turn")
    messages = prepare_messages(trajectory)[: turn + 1]
    for position, message in enumerate(messages):
        if message["role"] == "assistant" and position != turn:
            message["step_loss_mask"] = 0
    lineage = _trajectory_lineage(trajectory, weight_version, stop_reason=None, role="ahat")
    sample = _rendered_sample(
        slot_sample, trajectory, messages, mask_generator, reward=1.0, status=Sample.Status.COMPLETED, lineage=lineage
    )
    if sample is not None:
        sample.index, sample.group_index = index, group_index
    return sample


def _trajectory_lineage(
    trajectory: Trajectory, weight_version: int | None, *, stop_reason: str | None, role: str
) -> dict:
    return {
        "sample_slot_id": trajectory.sample_slot_id,
        "branch_id": trajectory.branch_id,
        "parent_branch_id": trajectory.parent_branch_id,
        "stop_reason": stop_reason,
        "weight_version": weight_version,
        "group": trajectory.group,
        "role": role,
    }


def _rendered_sample(
    slot_sample: Sample,
    trajectory: Trajectory,
    messages: list[dict],
    mask_generator: MultiTurnLossMaskGenerator,
    *,
    reward: float,
    status: Sample.Status,
    lineage: dict,
) -> Sample | None:
    """``messages`` rendered into a copy of the slot's sample; None when no token is trainable."""
    token_ids, full_mask = mask_generator.get_loss_mask(messages, tools=trajectory.tools or None)
    if len(token_ids) != len(full_mask):
        raise ValueError("the loss mask does not cover the rendered tokens")
    if not any(full_mask):
        return None
    response_length = mask_generator.get_response_lengths([full_mask])[0]
    first_assistant = next(i for i, message in enumerate(messages) if message["role"] == "assistant")

    sample = deepcopy(slot_sample)
    sample.tokens = list(token_ids)
    sample.response_length = response_length
    sample.loss_mask = list(full_mask[-response_length:])
    sample.response = mask_generator.tokenizer.decode(token_ids[-response_length:])
    sample.prompt = messages[:first_assistant]
    sample.reward = reward
    sample.status = status
    # Every token-position-dependent record of the sampling process describes
    # a different sequence; the trainer recomputes what it needs from these tokens.
    sample.rollout_log_probs = None
    sample.multimodal_inputs = None
    sample.multimodal_train_inputs = None
    sample.rollout_sampling_mask = None
    sample.rollout_routed_experts = None
    sample.rollout_indexer_topk = None
    sample.teacher_log_probs = None
    sample.opd_reverse_kl = None
    sample.weight_versions = []

    sample.metadata = {
        **sample.metadata,
        # ``messages`` is where Miles's trajectory dumps and dashboard look.
        "messages": messages,
        "tools": trajectory.tools,
        "sprout_rollout": {
            **lineage,
            "hints_removed": True,
            "logprob_context": "hint_free_messages",
            "trajectory_metadata": deepcopy(trajectory.metadata),
        },
    }
    sample.train_metadata = {**(sample.train_metadata or {}), "sprout_rollout": lineage}
    sample.validate()
    return sample
