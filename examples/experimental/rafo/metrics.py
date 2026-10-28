"""Optional rollout log hook with the RAFO monitor values.

``--custom-rollout-log-function-path examples.experimental.rafo.metrics.log_rollout_data``. It logs ``rafo/*``
next to the default rollout metrics, which it keeps.
"""

from collections import Counter
from typing import Any

from miles.utils.metric_utils import compute_rollout_step, namespace_metrics
from miles.utils.tracking_utils import tracking
from miles.utils.types import Sample


def log_rollout_data(
    rollout_id: int, args: Any, samples: list[Sample], extra_metrics: Any, rollout_time: float
) -> bool:
    if values := compute_rafo_metrics(samples):
        log_dict, step_key = namespace_metrics(
            values, trainer_model_id=None, step_name="rollout/step", step=compute_rollout_step(args, rollout_id)
        )
        tracking.log(args, log_dict, step_key=step_key)
    return False  # keep the default rollout metrics


def compute_rafo_metrics(samples: list[Sample]) -> dict[str, float]:
    """Per-task means; every sample of a task carries the same task statistics, so each task counts once."""
    tasks: dict[Any, dict] = {}
    kinds: Counter = Counter()
    for sample in samples:
        if (rafo := sample.metadata.get("rafo")) is None:
            continue
        tasks[sample.rollout_id] = rafo["task"]
        kinds[rafo["kind"]] += 1
    if not tasks:
        return {}
    stats = list(tasks.values())
    reads = sum(task["reviewer_reads"] for task in stats)
    reviewed = [task for task in stats if task["cont_success"] is not None]
    values = {
        "rafo/root_success": _mean(task["root_success"] for task in stats),
        "rafo/reviewed_task_frac": _mean(float(task["reviewer_reads"] > 0) for task in stats),
        "rafo/events_per_task": _mean(task["events"] for task in stats),
        "rafo/reviewer_reads_per_task": _mean(task["reviewer_reads"] for task in stats),
        "rafo/abstention_rate": sum(task["abstentions"] for task in stats) / reads if reads else 0.0,
        **{f"rafo/samples_per_task/{kind}": count / len(stats) for kind, count in kinds.items()},
    }
    if reviewed:
        values["rafo/cont_success"] = _mean(task["cont_success"] for task in reviewed)
    return values


def _mean(values) -> float:
    values = list(values)
    return sum(values) / len(values)
