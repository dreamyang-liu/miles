"""GSML credit for a Sprout search group: what each sample of a task is credited with.

A search task returns its roots and the branches Sprout ran from points its
review chose in the failed ones: student continuations, with nothing added,
and repair continuations, after a reviewer's turn inserted as reasoning
(``protocol.SearchSpec``). ``assemble_search_group`` labels the task by how its
roots did (``search_outcome``) and z-scores every trajectory within the group
it was sampled under. Only when every root was graded and none resolved (FF)
does it spread a mass λ over the points where something resolved: to the
students that resolved, where any did -- exact, the policy's own success --
and otherwise to the reviewer's turns whose continuations resolved without
touching the task's tests or naming what the review should not have seen:
their continuations, and the turn itself as a one-turn target (the â sample).

Each sample carries one ``Credit``; ``rewards.post_process_gsml`` turns it into
the advantage under λ, β and ψ. A trajectory whose grade is missing counts for
nothing, not as a 0; a gradable one with no token the policy sampled counts in
every statistic and trains nothing (estimation-only).
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass

from miles.rollout.sprout.importer import SEARCH_KINDS, ahat_sample, import_trajectory, unsampled_messages
from miles.rollout.sprout.protocol import RolloutResult, Trajectory
from miles.utils.mask_utils import MultiTurnLossMaskGenerator
from miles.utils.types import Sample

#: Task labels from the roots (``search_outcome``); only FF earns λ credit.
OUTCOMES = ("TT", "TT_partial", "TF", "FF", "FF_partial", "ungraded")
KINDS = ("root", "student", "repair", "ahat")
#: Which coefficient a credit's z-score takes: (1 - λ)/√2, β_br, or none.
BETAS = ("root", "branch", None)
METRICS = (
    "points",
    "points_exact",
    "points_guided",
    "points_no_mass",
    "candidates",
    "candidates_eligible",
    "candidates_leaked",
    "candidates_unverified",
    "candidates_noncanonical",
    "candidates_unconfirmed",
    "continuations_touching_tests",
    "ahat",
    "ahat_untrainable",
    "estimation_only",
    "lambda_tasks",
)
#: The reasoning of the inserted turn ``check_reasoning_is_trained`` renders.
REASONING_PROBE = "sprout reasoning probe"
PROBE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Run a command.",
            "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]},
        },
    }
]


def search_outcome(planned: int, graded: int, resolved: int) -> str:
    """A search task's label from its roots: planned, graded (a verdict came back) and resolved.

    Sprout chooses what to sample after the roots with the same function
    (``sprout.rl_driver.outcome.search_outcome``); Miles recomputes it from the
    returned roots and refuses a group Sprout labelled otherwise.
    """
    if any(type(count) is not int for count in (planned, graded, resolved)) or not (
        0 <= resolved <= graded <= planned
    ):
        raise ValueError(f"expected 0 <= resolved <= graded <= planned, got {planned=}, {graded=}, {resolved=}")
    if graded == 0:
        return "ungraded"
    if resolved == graded:
        return "TT" if graded == planned else "TT_partial"
    if resolved >= 1:
        return "TF"
    return "FF" if graded == planned else "FF_partial"


def group_z(values: Sequence[float]) -> list[float]:
    """GRPO's z-scores within one group; all 0 below two values or when they do not differ (unbiased std)."""
    if len(values) < 2:
        return [0.0] * len(values)
    mean = sum(values) / len(values)
    std = math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1))
    if std == 0:
        return [0.0] * len(values)
    return [(value - mean) / (std + 1e-6) for value in values]


@dataclass(frozen=True)
class Credit:
    """What one sample of a search group is credited with, before λ, β and ψ are applied.

    ``kind`` is its place in the tree and ``beta`` the coefficient its z-score
    takes (``root``: (1 - λ)/√2, ``branch``: β_br, None: none); ``lambda_share``
    is its share of the task's λ mass. The rest says how the share was reached:
    the point and candidate, c_s (students resolved there), K_m (the
    candidate's continuations resolved, R'), |E_s| (eligible candidates), ω_s
    and the repair's R'. Rides in ``train_metadata``, which reaches the trainer
    as msgpack: plain values only.
    """

    kind: str
    beta: str | None
    z: float
    lambda_share: float
    case: str
    point_id: str | None = None
    candidate: int | None = None
    c_s: int | None = None
    k_m: int | None = None
    n_eligible: int | None = None
    omega: float | None = None
    r_prime: float | None = None

    def __post_init__(self) -> None:
        if self.kind not in KINDS or self.beta not in BETAS or self.case not in OUTCOMES:
            raise ValueError(f"not a GSML credit: kind={self.kind!r}, beta={self.beta!r}, case={self.case!r}")
        if any(type(value) not in (int, float) or not math.isfinite(value) for value in (self.z, self.lambda_share)):
            raise ValueError(
                f"a credit's z and lambda_share are finite numbers, got {self.z!r}, {self.lambda_share!r}"
            )
        if self.lambda_share < 0:
            raise ValueError(f"a share of the λ mass is never negative, got {self.lambda_share}")

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping) -> Credit:
        try:
            return cls(**data)
        except TypeError as error:
            raise ValueError(f"not a GSML credit: {data!r}") from error


def check_reasoning_is_trained(mask_generator: MultiTurnLossMaskGenerator) -> None:
    """Refuse a chat template and loss mask under which an inserted turn's reasoning is not trained.

    Sprout inserts a reviewer's turn with its text as ``reasoning_content`` and
    an empty ``content``, and the â sample trains that turn. A template that
    drops ``reasoning_content`` -- Qwen3's does before the last user message,
    so for every turn when each message is rendered alone -- would train the
    commands without the reasoning they follow from. The probe is such a turn
    after a user message, rendered the way a sample is.
    """
    probe = [
        {"role": "user", "content": "probe"},
        {
            "role": "assistant",
            "content": "",
            "reasoning_content": REASONING_PROBE,
            "tool_calls": [
                {"id": "probe-1", "type": "function", "function": {"name": "bash", "arguments": {"command": "true"}}}
            ],
        },
        {"role": "tool", "tool_call_id": "probe-1", "content": "done"},
    ]
    mask_type = getattr(mask_generator, "tokenizer_type", None)
    try:
        token_ids, loss_mask = mask_generator.get_loss_mask(probe, tools=PROBE_TOOLS)
        trained = mask_generator.tokenizer.decode([t for t, keep in zip(token_ids, loss_mask, strict=True) if keep])
    except Exception as error:
        raise ValueError(
            f"the chat template cannot render an inserted turn (--loss-mask-type {mask_type}): {error}"
        ) from error
    if REASONING_PROBE not in trained:
        raise ValueError(
            f"the chat template and --loss-mask-type {mask_type} do not train an assistant turn's reasoning_content "
            f"(trained: {trained!r}); an â sample would learn the commands of a repair without its reasoning"
        )


def assemble_search_group(
    result: RolloutResult,
    slot_samples: Mapping[str, Sample],
    mask_generator: MultiTurnLossMaskGenerator,
    *,
    planned_roots: int,
    weight_version: int | None,
    indices: Iterator[int],
    group_indices: Iterator[int],
) -> tuple[list[Sample], dict[str, float]]:
    """A search group's trajectories -> the samples the trainer gets, each with its credit, and what was measured.

    Every gradable trajectory is imported; a missing one produces nothing. The
    credit is at ``train_metadata["sprout_rollout"]["credit"]`` (and in
    ``metadata`` for dumps). Roots keep their prompt's ``group_index``; each
    condition (``Trajectory.group``) gets a fresh one from ``group_indices``,
    and each â sample a fresh index from ``indices`` and a group of its own.
    """
    counts = dict.fromkeys(METRICS, 0) | {f"cases/{label}": 0 for label in OUTCOMES}
    if result.status == "failed" and not result.trajectories:
        counts[f"cases/{search_outcome(planned_roots, 0, 0)}"] = 1  # every root's actor failed
        return [], _metric_names(counts)
    _check_group(result, slot_samples)
    members = [_member(trajectory) for trajectory in result.trajectories]
    case = _outcome(members, planned_roots)
    z = _z_scores(members)
    points = _points(members) if case == "FF" else {}
    credits = {member.slot: _credit(member, z[member.slot], case, points) for member in members if member.gradable}
    assembly = _Assembly(result, slot_samples, mask_generator, weight_version, counts)
    samples = _trajectory_samples(assembly, members, credits, group_indices)
    samples += _ahat_samples(assembly, points, case, indices, group_indices)
    _measure(counts, case, members, points)
    return samples, _metric_names(counts)


@dataclass(frozen=True)
class _Assembly:
    """What making a group's samples needs besides their credit; ``counts`` is updated as they are made."""

    result: RolloutResult
    slot_samples: Mapping[str, Sample]
    mask_generator: MultiTurnLossMaskGenerator
    weight_version: int | None
    counts: dict[str, int]


@dataclass(frozen=True)
class _Member:
    """A returned trajectory as the credit rules read it."""

    trajectory: Trajectory
    kind: str
    point_id: str | None
    candidate: int | None
    gradable: bool
    #: R; for a repair R', which is 0 when it touched the task's tests.
    value: float
    #: A gradable repair whose ``touched_test_paths`` made its R' 0 (non-empty, or not a list).
    touching: bool
    #: A repair's leak gate: the hidden terms its turn names (a list), or not a list when nothing was checked.
    leak_terms: object
    canonical: bool

    @property
    def slot(self) -> str:
        return self.trajectory.sample_slot_id


@dataclass(frozen=True)
class _Candidate:
    """A reviewer's turn at one point, by its gradable continuations."""

    continuations: tuple[_Member, ...]
    k_m: int
    #: leak_terms names terms the review could see but the history did not show.
    leaked: bool
    #: leak_terms is not a list: there was nothing to check the turn against.
    unverified: bool
    noncanonical: bool

    @property
    def eligible(self) -> bool:
        return self.k_m >= 1 and not (self.leaked or self.unverified or self.noncanonical)


@dataclass(frozen=True)
class _Point:
    """A branch point of an FF task, by its gradable students and its candidates."""

    point_id: str
    #: |S_s|, the gradable students, and c_s, how many of them resolved.
    students: int
    c_s: int
    candidates: dict[int, _Candidate]
    omega: float = 0.0

    @property
    def eligible(self) -> list[int]:
        """E_s: only where some student was graded and none resolved."""
        if self.students == 0 or self.c_s != 0:
            return []
        return [number for number, candidate in self.candidates.items() if candidate.eligible]

    @property
    def exact(self) -> bool:
        return self.students >= 1 and self.c_s >= 1

    @property
    def guided(self) -> bool:
        return bool(self.eligible)

    @property
    def creditable(self) -> bool:
        return self.exact or self.guided


def _check_group(result: RolloutResult, slot_samples: Mapping[str, Sample]) -> None:
    if result.status not in {"completed", "early_stopped"}:
        raise ValueError(f"cannot credit a rollout group with status={result.status!r}: {result.stop_reason}")
    if result.max_samples > len(slot_samples):
        raise ValueError("the driver's max_samples exceeds the slots Miles allocated")
    for trajectory in result.trajectories:
        if trajectory.sample_slot_id not in slot_samples:
            raise ValueError(f"unknown sample_slot_id: {trajectory.sample_slot_id!r}")


def _member(trajectory: Trajectory) -> _Member:
    """Read what the credit needs from Sprout's annotations, refusing a trajectory that lacks them."""
    slot, metadata = trajectory.sample_slot_id, trajectory.metadata
    search = metadata.get("search")
    if "zero_reason" not in metadata or not isinstance(search, dict) or search.get("kind") not in SEARCH_KINDS:
        raise ValueError(f"trajectory {slot!r} lacks the search annotations GSML credits it by")
    kind, point_id, candidate = search["kind"], search.get("point_id"), search.get("candidate")
    if kind != "root" and not (isinstance(point_id, str) and point_id):
        raise ValueError(f"branch {slot!r} does not name its point")
    if kind == "repair" and type(candidate) is not int:
        raise ValueError(f"repair {slot!r} does not name its candidate")
    group = {"root": "root", "student": f"student:{point_id}", "repair": f"repair:{point_id}:{candidate}"}[kind]
    if trajectory.group != group:
        raise ValueError(f"trajectory {slot!r} is in group {trajectory.group!r}; its annotation puts it in {group!r}")
    gradable = metadata["zero_reason"] is None
    if gradable and trajectory.reward not in (0, 1):
        raise ValueError(f"trajectory {slot!r} has reward {trajectory.reward!r}; GSML credits a 0/1 verdict")
    touched = metadata.get("touched_test_paths")
    # A continuation after the reviewer's turn that changed a file of the
    # task's test patch resolved nothing it can be credited for; neither did
    # one whose grade could not say which files it touched.
    touching = kind == "repair" and gradable and (not isinstance(touched, list) or bool(touched))
    return _Member(
        trajectory=trajectory,
        kind=kind,
        point_id=None if kind == "root" else point_id,
        candidate=candidate if kind == "repair" else None,
        gradable=gradable,
        value=0.0 if touching else float(trajectory.reward),
        touching=touching,
        leak_terms=search.get("leak_terms"),
        canonical=kind == "repair" and gradable and _canonical(trajectory),
    )


def _canonical(trajectory: Trajectory) -> bool:
    """Whether every inserted turn of a repair (one per round of its chain) is in the form an â sample
    trains: the reviewer's text as ``reasoning_content``, at most one line of ``content``."""
    unsampled_messages(trajectory)  # the provenance is checked before it is read
    inserted = trajectory.metadata["provenance"]["inserted_messages"]
    if not inserted:
        return False
    for position in inserted:
        turn = trajectory.messages[position]
        reasoning, content = turn.get("reasoning_content"), turn.get("content") or ""
        if not (isinstance(reasoning, str) and bool(reasoning.strip()) and "\n" not in content):
            return False
    return True


def _outcome(members: list[_Member], planned_roots: int) -> str:
    """Miles's own label for the task, which every trajectory's annotation must repeat."""
    roots = [member for member in members if member.kind == "root"]
    if len(roots) > planned_roots:
        raise ValueError(f"{len(roots)} root trajectories for {planned_roots} planned roots")
    graded = [member for member in roots if member.gradable]
    counts = {"planned": planned_roots, "graded": len(graded), "resolved": int(sum(m.value for m in graded))}
    case = search_outcome(**counts)
    for member in members:
        search = member.trajectory.metadata["search"]
        if search.get("outcome") != case or search.get("root_counts") != counts:
            raise ValueError(
                f"Sprout labels {member.slot!r} {search.get('outcome')!r} with root counts "
                f"{search.get('root_counts')}; the returned roots make the task {case!r} with {counts}"
            )
    return case


def _z_scores(members: list[_Member]) -> dict[str, float]:
    """Each gradable trajectory's z-score within its group, estimation-only ones included."""
    groups: dict[str, list[_Member]] = {}
    for member in members:
        if member.gradable:
            groups.setdefault(member.trajectory.group, []).append(member)
    return {
        member.slot: z
        for group in groups.values()
        for member, z in zip(group, group_z([member.value for member in group]), strict=True)
    }


def _points(members: list[_Member]) -> dict[str, _Point]:
    """An FF task's points: who resolved there, each candidate's K_m and gates, and ω_s,
    one share of the λ mass per point that can be credited."""
    students: dict[str, list[_Member]] = {}
    repairs: dict[str, dict[int, list[_Member]]] = {}
    for member in members:
        if member.kind == "student":
            students.setdefault(member.point_id, []).append(member)
        elif member.kind == "repair":
            repairs.setdefault(member.point_id, {}).setdefault(member.candidate, []).append(member)
    points = {}
    for point_id in dict.fromkeys([*students, *repairs]):
        graded = [member for member in students.get(point_id, []) if member.gradable]
        candidates = {number: _candidate(group) for number, group in repairs.get(point_id, {}).items()}
        points[point_id] = _Point(point_id, len(graded), int(sum(m.value for m in graded)), candidates)
    creditable = sum(point.creditable for point in points.values())
    return {
        point_id: dataclasses.replace(point, omega=1 / creditable if point.creditable else 0.0)
        for point_id, point in points.items()
    }


def _candidate(members: list[_Member]) -> _Candidate:
    graded = tuple(member for member in members if member.gradable)
    return _Candidate(
        continuations=graded,
        k_m=int(sum(member.value for member in graded)),
        leaked=any(isinstance(m.leak_terms, list) and bool(m.leak_terms) for m in graded),
        unverified=any(not isinstance(m.leak_terms, list) for m in graded),
        noncanonical=not all(member.canonical for member in graded),
    )


def _credit(member: _Member, z: float, case: str, points: dict[str, _Point]) -> Credit:
    """One gradable trajectory's credit; ``points`` is empty unless the task is FF."""
    if member.kind == "root":
        return Credit(kind="root", beta="root", z=z, lambda_share=0.0, case=case)
    point = points.get(member.point_id)
    about = {} if point is None else {"c_s": point.c_s, "n_eligible": len(point.eligible), "omega": point.omega}
    if member.kind == "student":
        # Exact: the policy resolved from this state on its own.
        share = point.omega / point.c_s if point is not None and point.exact and member.value == 1 else 0.0
        return Credit(
            kind="student", beta="branch", z=z, lambda_share=share, case=case, point_id=member.point_id, **about
        )
    candidate = point.candidates[member.candidate] if point is not None else None
    share = 0.0
    if point is not None and member.candidate in point.eligible:
        # Guided: no student resolved here, this candidate's continuations did.
        share = member.value * point.omega / (len(point.eligible) * candidate.k_m)
    return Credit(
        kind="repair",
        beta="branch",
        z=z,
        lambda_share=share,
        case=case,
        point_id=member.point_id,
        candidate=member.candidate,
        k_m=None if candidate is None else candidate.k_m,
        r_prime=member.value,
        **about,
    )


def _trajectory_samples(
    assembly: _Assembly, members: list[_Member], credits: dict[str, Credit], group_indices: Iterator[int]
) -> list[Sample]:
    """The gradable trajectories with something to train, each branch in its condition's group."""
    samples, groups = [], {}
    for member in members:
        if not member.gradable:
            continue
        trajectory = member.trajectory
        sample = import_trajectory(
            assembly.result,
            trajectory,
            assembly.slot_samples[member.slot],
            assembly.mask_generator,
            weight_version=assembly.weight_version,
        )
        if sample is None:
            assembly.counts["estimation_only"] += 1
            continue
        if member.kind != "root":
            if trajectory.group not in groups:
                groups[trajectory.group] = next(group_indices)
            sample.group_index = groups[trajectory.group]
        _annotate(sample, credits[member.slot], assembly.result)
        samples.append(sample)
    return samples


def _ahat_samples(
    assembly: _Assembly, points: dict[str, _Point], case: str, indices: Iterator[int], group_indices: Iterator[int]
) -> list[Sample]:
    """The â samples of each eligible candidate where no student resolved, from its first gradable
    continuation: one per inserted turn, which for a repair chain is one per round, sharing the
    candidate's part of the mass alike (the turns together made the path that resolved)."""
    samples = []
    for point in points.values():
        for number in point.eligible:
            candidate = point.candidates[number]
            source = candidate.continuations[0].trajectory
            turns = sorted(source.metadata["provenance"]["inserted_messages"])
            for turn in turns:
                sample = _ahat_turn(assembly, point, number, candidate, source, turn, len(turns), case,
                                    indices, group_indices)
                if sample is not None:
                    samples.append(sample)
    return samples


def _ahat_turn(
    assembly: _Assembly,
    point: _Point,
    number: int,
    candidate: _Candidate,
    source: Trajectory,
    turn: int,
    n_turns: int,
    case: str,
    indices: Iterator[int],
    group_indices: Iterator[int],
) -> Sample | None:
    """One inserted turn of an eligible candidate as an â sample, its share split over the chain's turns."""
    sample = ahat_sample(
        source,
        assembly.slot_samples[source.sample_slot_id],
        assembly.mask_generator,
        index=next(indices),
        group_index=next(group_indices),
        weight_version=assembly.weight_version,
        turn=turn,
    )
    if sample is None:
        assembly.counts["ahat_untrainable"] += 1
        return None
    credit = Credit(
        kind="ahat",
        beta=None,
        z=0.0,
        lambda_share=point.omega / (len(point.eligible) * n_turns),
        case=case,
        point_id=point.point_id,
        candidate=number,
        c_s=point.c_s,
        k_m=candidate.k_m,
        n_eligible=len(point.eligible),
        omega=point.omega,
    )
    _annotate(sample, credit, assembly.result)
    assembly.counts["ahat"] += 1
    return sample


def _annotate(sample: Sample, credit: Credit, result: RolloutResult) -> None:
    """The credit where ``post_process_gsml`` reads it and where trajectory dumps show it."""
    for lineage in (sample.train_metadata["sprout_rollout"], sample.metadata["sprout_rollout"]):
        # An â sample is made from a trajectory alone; its group's identity comes from here.
        lineage.update(
            rollout_job_id=result.rollout_job_id, prompt_group_id=result.prompt_group_id, credit=credit.to_dict()
        )


def _measure(counts: dict[str, int], case: str, members: list[_Member], points: dict[str, _Point]) -> None:
    """The task's label, its points by how they were credited and its candidates by the gates they failed
    (a candidate may fail several; ``candidates_eligible`` passed all, whether or not a student resolved)."""
    counts[f"cases/{case}"] = 1
    counts["continuations_touching_tests"] = sum(member.touching for member in members)
    for point in points.values():
        counts["points"] += 1
        counts["points_exact"] += point.exact
        counts["points_guided"] += point.guided
        counts["points_no_mass"] += not point.creditable
        for candidate in point.candidates.values():
            counts["candidates"] += 1
            counts["candidates_eligible"] += candidate.eligible
            counts["candidates_leaked"] += candidate.leaked
            counts["candidates_unverified"] += candidate.unverified
            counts["candidates_noncanonical"] += candidate.noncanonical
            counts["candidates_unconfirmed"] += candidate.k_m < 1
    counts["lambda_tasks"] = int(any(point.creditable for point in points.values()))


def _metric_names(counts: dict[str, int]) -> dict[str, float]:
    return {f"rollout/sprout/gsml/{name}": value for name, value in counts.items()}
