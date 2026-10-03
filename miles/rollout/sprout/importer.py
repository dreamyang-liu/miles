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


def import_trajectories(
    result: RolloutResult,
    slot_samples: Mapping[str, Sample],
    mask_generator: MultiTurnLossMaskGenerator,
    *,
    weight_version: int | None = None,
) -> list[Sample]:
    """One Miles sample per returned trajectory, in the slots Miles allocated."""
    if result.status not in {"completed", "early_stopped"}:
        raise ValueError(f"cannot import a rollout group with status={result.status!r}: {result.stop_reason}")
    if result.max_samples > len(slot_samples):
        raise ValueError("the driver's max_samples exceeds the slots Miles allocated")
    samples = []
    for trajectory in result.trajectories:
        if trajectory.sample_slot_id not in slot_samples:
            raise ValueError(f"unknown sample_slot_id: {trajectory.sample_slot_id!r}")
        samples.append(
            _import_trajectory(
                result, trajectory, slot_samples[trajectory.sample_slot_id], mask_generator, weight_version
            )
        )
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


def _import_trajectory(
    result: RolloutResult,
    trajectory: Trajectory,
    slot_sample: Sample,
    mask_generator: MultiTurnLossMaskGenerator,
    weight_version: int | None,
) -> Sample:
    messages = prepare_messages(trajectory)
    for index in unsampled_messages(trajectory):
        # Rendered as context, excluded from the loss (MultiTurnLossMaskGenerator).
        messages[index]["step_loss_mask"] = 0
    token_ids, full_mask = mask_generator.get_loss_mask(messages, tools=trajectory.tools or None)
    if len(token_ids) != len(full_mask) or not any(full_mask):
        raise ValueError("the rendered trajectory has no trainable assistant tokens")
    response_length = mask_generator.get_response_lengths([full_mask])[0]
    first_assistant = next(i for i, message in enumerate(messages) if message["role"] == "assistant")

    sample = deepcopy(slot_sample)
    sample.tokens = list(token_ids)
    sample.response_length = response_length
    sample.loss_mask = list(full_mask[-response_length:])
    sample.response = mask_generator.tokenizer.decode(token_ids[-response_length:])
    sample.prompt = messages[:first_assistant]
    sample.reward = float(trajectory.reward)
    sample.status = SAMPLE_STATUS[trajectory.status]
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

    lineage = {
        "rollout_job_id": result.rollout_job_id,
        "prompt_group_id": result.prompt_group_id,
        "sample_slot_id": trajectory.sample_slot_id,
        "branch_id": trajectory.branch_id,
        "parent_branch_id": trajectory.parent_branch_id,
        "stop_reason": trajectory.stop_reason,
        "weight_version": weight_version,
        "group": trajectory.group,
        # What the trainer does with the sample: a root or a branch is a GRPO
        # sample; a distillation sample's reward is its weight (see rewards.py).
        "role": "root" if trajectory.group in (None, "root") else "branch",
    }
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


def search_of(sample: Sample) -> dict:
    """What Sprout said about a search sample's place in the tree (``metadata.search``)."""
    return (sample.metadata.get("sprout_rollout", {}).get("trajectory_metadata") or {}).get("search") or {}


def progress_of(sample: Sample) -> float | None:
    value = (sample.metadata.get("sprout_rollout", {}).get("trajectory_metadata") or {}).get("progress")
    return float(value) if value is not None else None


def distillation_sample(repair: Sample, mask_generator: MultiTurnLossMaskGenerator, *, weight: float, index: int,
                        group_index: int, measured: dict) -> Sample:
    """The repair turn of a repair continuation, as a one-shot target.

    The context is the history up to the inserted turn, as the policy would
    see it at inference; only the inserted turn is trained, and the weight --
    how much more progress its continuations made than the student's own --
    is the sample's reward, which ``rewards.post_process_rewards`` passes
    through as the advantage.
    """
    inserted = max(repair.metadata["sprout_rollout"]["trajectory_metadata"]["provenance"]["inserted_messages"])
    messages = deepcopy(repair.metadata["messages"][: inserted + 1])
    for position, message in enumerate(messages):
        message.pop("step_loss_mask", None)
        if message["role"] == "assistant" and position != inserted:
            message["step_loss_mask"] = 0
    token_ids, full_mask = mask_generator.get_loss_mask(messages, tools=repair.metadata.get("tools") or None)
    if len(token_ids) != len(full_mask) or not any(full_mask):
        raise ValueError("the repair turn rendered no trainable tokens")
    response_length = mask_generator.get_response_lengths([full_mask])[0]
    sample = deepcopy(repair)
    sample.index, sample.group_index = index, group_index
    sample.tokens = list(token_ids)
    sample.response_length = response_length
    sample.loss_mask = list(full_mask[-response_length:])
    sample.response = mask_generator.tokenizer.decode(token_ids[-response_length:])
    sample.reward = float(weight)
    sample.status = Sample.Status.COMPLETED
    sample.metadata = {**sample.metadata, "messages": messages}
    lineage = {**sample.train_metadata["sprout_rollout"], "role": "distill", "weight": float(weight), **measured}
    sample.metadata["sprout_rollout"] = {**sample.metadata["sprout_rollout"], **lineage}
    sample.train_metadata = {**sample.train_metadata, "sprout_rollout": lineage}
    sample.validate()
    return sample


def arrange_search_samples(samples: list[Sample], result: RolloutResult, mask_generator: MultiTurnLossMaskGenerator,
                           *, indices, group_indices, branch_scale: float, distill_weight: float
                           ) -> tuple[list[Sample], dict[str, float]]:
    """A search group's imported samples -> what the trainer gets, and what was measured.

    Roots keep their group. Branches are grouped by the condition they were
    sampled under (a fresh ``group_index`` per ``Trajectory.group``) and kept
    only when ``branch_scale`` is positive. Each candidate repair whose
    continuations made more progress than the student continuations at the
    same point yields one distillation sample, weighted by that gain times the
    share of independent reviews that proposed it, times ``distill_weight``.
    """
    by_slot = {trajectory.sample_slot_id: trajectory for trajectory in result.trajectories}
    groups: dict[str, int] = {}
    trained: list[Sample] = []
    students: dict[str, list[float]] = {}
    repairs: dict[tuple[str, int], list[Sample]] = {}
    for sample in samples:
        trajectory = by_slot[sample.metadata["sprout_rollout"]["sample_slot_id"]]
        if trajectory.group in (None, "root"):
            trained.append(sample)
            continue
        if trajectory.group not in groups:
            groups[trajectory.group] = next(group_indices)
        sample.group_index = groups[trajectory.group]
        search = search_of(sample)
        progress = progress_of(sample)
        if search.get("kind") == "student" and progress is not None:
            students.setdefault(search["point_id"], []).append(progress)
        elif search.get("kind") == "repair":
            repairs.setdefault((search["point_id"], search["candidate"]), []).append(sample)
        if branch_scale > 0:
            trained.append(sample)
    measured = {"rollout/sprout/search/points": len(students), "rollout/sprout/search/candidates": len(repairs),
                "rollout/sprout/search/distilled": 0, "rollout/sprout/search/student_progress": 0.0,
                "rollout/sprout/search/repair_progress": 0.0}
    for point, values in students.items():
        measured["rollout/sprout/search/student_progress"] += sum(values) / len(values) / max(1, len(students))
    for (point, candidate), group in repairs.items():
        progress = [p for p in (progress_of(sample) for sample in group) if p is not None]
        baseline = students.get(point)
        if not progress or not baseline or distill_weight <= 0:
            continue
        gain = sum(progress) / len(progress) - sum(baseline) / len(baseline)
        measured["rollout/sprout/search/repair_progress"] += sum(progress) / len(progress) / len(repairs)
        search = search_of(group[0])
        share = search.get("multiplicity", 1) / (search.get("samples") or search.get("multiplicity", 1))
        weight = max(0.0, gain) * share * distill_weight
        if weight <= 0:
            continue
        trained.append(distillation_sample(
            group[0], mask_generator, weight=weight, index=next(indices), group_index=next(group_indices),
            measured={"gain": gain, "share": share, "repair_progress": sum(progress) / len(progress),
                      "student_progress": sum(baseline) / len(baseline), "point_id": point, "candidate": candidate}))
        measured["rollout/sprout/search/distilled"] += 1
    return trained, measured
