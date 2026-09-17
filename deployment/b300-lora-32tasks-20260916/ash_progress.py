"""Read the dedicated B300 Ash ledger without changing jobs or rewards."""

import argparse
import datetime
import json
from pathlib import Path
import sqlite3


ROOT = Path("/home/ec2-user/projects/LBP/deployments/miles-52.70.152.249")


def summarize(document):
    request = document.get("message_request") or {}
    branching = document.get("branching") or {}
    exported = document.get("message_result") or {}
    result = {
        "task": request.get("task_id"),
        "external_id": request.get("rollout_job_id"),
        "rollout_id": request.get("rollout_id"),
        "status": document.get("status"),
        "phase": branching.get("phase"),
        "return_mode": branching.get("config", {}).get("return_mode"),
        "stop_reason": branching.get("stop_reason"),
        "review_error": branching.get("error"),
        "export_status": exported.get("status"),
        "exported_trajectories": exported.get("actual_samples"),
        "samples": [],
    }
    for sample in document.get("samples", []):
        actor = sample.get("actor") or {}
        grade = sample.get("grade") or {}
        actor_result = actor.get("result") or {}
        grade_result = grade.get("result") or {}
        row = {
            "slot_id": sample.get("sample_slot_id"),
            "actor": actor.get("state"),
            "grade": grade.get("state"),
            "resolved": grade_result.get("resolved"),
            "error": actor.get("error") or grade.get("error"),
            "job_id": actor.get("job_id"),
            "attempt_id": actor.get("attempt_id"),
            "grade_job_id": grade.get("job_id"),
            "graded_snapshot_id": grade.get("snapshot_id"),
            "tokens": actor_result.get("training_token_count"),
            "calls": actor_result.get("rollout_usage"),
            "length_truncated": actor_result.get("stop_reason") == "max_sequence_tokens",
            "first_reasoning": next(
                (message.get("reasoning_content", "")[:300]
                 for message in actor_result.get("training_messages", [])
                 if message.get("reasoning_content")),
                "",
            ),
        }
        if actor.get("state") == "running" and actor.get("attempt_id"):
            journal = ROOT / "attempts" / actor["job_id"] / actor["attempt_id"] / "trajectory.jsonl"
            if journal.exists():
                events = []
                for line in journal.read_text().splitlines():
                    try:
                        events.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue  # A live writer may not have completed the final line.
                row["calls"] = {
                    "model_calls": sum(
                        event.get("type") == "gateway.request" and event.get("status") == "ok"
                        for event in events
                    ),
                    "tool_calls": sum(event.get("type") == "tool.finished" for event in events),
                }
                budgets = [event for event in events if event.get("type") == "rollout.sequence_budget"]
                if budgets:
                    row["tokens"] = budgets[-1].get("used_tokens")
                thinking = [event for event in events if event.get("type") == "agent.thinking" and event.get("text")]
                if thinking:
                    row["first_reasoning"] = thinking[0]["text"][:300]
        result["samples"].append(row)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=32)
    parser.add_argument("--since", type=float, default=0)
    parser.add_argument("--rollout-id", type=int)
    args = parser.parse_args()
    config = json.loads((ROOT / "driver.json").read_text())
    connection = sqlite3.connect("file:" + config["ledger"] + "?mode=ro", uri=True)
    rows = connection.execute(
        "select document from driver_groups where created_at >= ? order by created_at desc limit ?",
        (args.since, args.limit),
    )
    documents = [json.loads(row[0]) for row in rows]
    if args.rollout_id is not None:
        documents = [
            document for document in documents
            if document.get("message_request", {}).get("rollout_id") == args.rollout_id
        ]
    print(json.dumps({
        "checked_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "groups": [summarize(document) for document in documents],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
