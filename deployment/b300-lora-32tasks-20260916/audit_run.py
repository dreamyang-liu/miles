"""Verify actual task families, dynamic batches, and three real optimizer updates."""

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import torch


def load(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def normalize_names(tensors):
    result = {}
    for name, value in tensors.items():
        if name.startswith("0."):
            name = name[2:]
        while name.startswith("module."):
            name = name[len("module."):]
        result[name] = value
    return result


def digest(tensors):
    checksum = hashlib.sha256()
    for name, value in sorted(tensors.items()):
        checksum.update(name.encode())
        checksum.update(value.contiguous().view(torch.uint8).numpy().tobytes())
    return checksum.hexdigest()


def audit_batch(root, rollout_id, launch):
    data = load(root / f"dump_details/rollout_data/{rollout_id}.pt")
    samples = data["samples"]
    assert data["rollout_id"] == rollout_id and samples
    assert len({sample["index"] for sample in samples}) == len(samples)
    groups = defaultdict(list)
    for sample in samples:
        lineage = sample["metadata"]["ash_rollout"]
        metadata = lineage["trajectory_metadata"]
        assert lineage["protocol_version"] == "ash-rollout-v3" and lineage["hints_removed"]
        assert sample["rollout_log_probs"] is None
        assert len(sample["tokens"]) <= 131072
        assert len(sample["tokens"]) == metadata["training_token_count"]
        assert len(sample["loss_mask"]) == sample["response_length"] and sum(sample["loss_mask"]) > 0
        assert metadata["graded_snapshot_id"]
        assert sample["reward"] == metadata["raw_reward"] * metadata["truncated_reward_scale"]
        assert metadata["logprob_context"] == "hint_free_messages"
        groups[sample["group_index"]].append(sample)
    task_count = launch["tasks_per_update"]
    assert len(groups) == task_count
    assert all(1 <= len(group) <= 8 for group in groups.values())
    by_index = {sample["index"]: sample for sample in samples}
    counts = {group: len(values) for group, values in groups.items()}
    shard = load(root / f"dump_details/train_data/{rollout_id}_0.pt")["rollout_data"]
    sample_indices = [int(index) for index in shard["sample_indices"]]
    n = len(samples)
    assert shard["dynamic_global_batch_size"] == n
    assert shard["num_rollouts"] == [n]
    schedule = shard["micro_batch_indices"]
    assert shard["num_microbatches"] == [len(schedule)]
    assert all(1 <= len(indices) <= launch["micro_batch_size"] for indices in schedule)
    assert sorted(i for indices in schedule for i in indices) == list(range(n))
    assert sorted(sample_indices) == sorted(by_index)
    if launch["loss_weighting"] == "task":
        expected = [
            n / (task_count * counts[by_index[index]["group_index"]])
            for index in sample_indices
        ]
        torch.testing.assert_close(
            torch.tensor(shard["sample_loss_weights"], dtype=torch.float32),
            torch.tensor(expected, dtype=torch.float32),
        )
    else:
        assert "sample_loss_weights" not in shard
    for group in groups.values():
        raw = torch.tensor([sample["reward"] for sample in group])
        normalized = raw - raw.mean()
        if len(raw) > 1 and raw.std() > 0:
            normalized /= raw.std() + 1e-6
        locations = [sample_indices.index(sample["index"]) for sample in group]
        torch.testing.assert_close(
            torch.tensor([shard["rewards"][index] for index in locations]), normalized,
        )
    return {
        "rollout_id": rollout_id, "tasks": task_count, "trajectories": n,
        "task_trajectory_counts": [
            {"task": values[0]["metadata"]["task_id"], "count": len(values),
             "rewards": [sample["reward"] for sample in values]}
            for values in groups.values()
        ],
        "microbatches": len(schedule), "microbatch_sizes": [len(indices) for indices in schedule],
        "maximum_sequence_tokens": max(len(sample["tokens"]) for sample in samples),
        "positive_reward_trajectories": sum(sample["reward"] > 0 for sample in samples),
        "mixed_reward_tasks": sum(len({sample["reward"] for sample in values}) > 1 for values in groups.values()),
        "loss_weighting": launch["loss_weighting"],
    }


def audit_updates(root, batches):
    ranks = []
    for rank in range(8):
        events = [json.loads(line) for line in (root / f"audit-rank-{rank}.jsonl").read_text().splitlines()]
        inventory = next(event for event in events if event["event"] == "inventory")
        assert inventory["mtp_num_layers"] == 0 and inventory["lora_rank"] == 32
        rank_result = {"rank": rank, "steps": []}
        for batch in batches:
            step = batch["rollout_id"]
            selected = [event for event in events if event.get("rollout_id") == step]
            updates = [event for event in selected if event["event"] == "optimizer_step"]
            assert len(updates) == 1 and updates[0]["result"].startswith("(True,")
            assert any(event["event"] == "nonzero_adapter_gradient" for event in selected)
            assert sum(event["event"] == "train_microbatch" for event in selected) == batch["microbatches"]
            scheduler = [event for event in selected if event["event"] == "scheduler_step"]
            assert len(scheduler) == 1 and scheduler[0]["trajectory_increment"] == batch["trajectories"]
            before = normalize_names(load(root / f"adapters-before-rollout-{step}-step-0-rank-{rank}.pt"))
            after = normalize_names(load(
                root / f"checkpoints/iter_{step:07d}/adapter/adapter_megatron_rank{rank}.pt"
            ))
            assert before.keys() == after.keys()
            assert all(torch.isfinite(value).all() for value in after.values())
            changed = sum(not torch.equal(before[name], after[name]) for name in before)
            assert changed > 0
            if step == 0:
                b = [value for name, value in before.items() if ".adapter.linear_out." in name]
                assert b and all(torch.count_nonzero(value).item() == 0 for value in b)
            rank_result["steps"].append({
                "rollout_id": step, "changed_adapter_tensors": changed,
                "adapter_tensors": len(after), "adapter_sha256": digest(after),
                "scheduler_trajectory_increment": scheduler[0]["trajectory_increment"],
            })
        ranks.append(rank_result)
    for batch in batches:
        step = batch["rollout_id"]
        for tp_rank in range(2):
            assert len({rank["steps"][step]["adapter_sha256"] for rank in ranks if rank["rank"] % 2 == tp_rank}) == 1
        before = json.loads((root / f"frozen-prefix-before-rollout-{step}-step-0-rank-0.json").read_text())
        after = json.loads((root / f"frozen-prefix-after-rollout-{step}-rank-0.json").read_text())
        assert before == after
        assert (root / f"checkpoints/iter_{step:07d}/adapter/adapter_model.bin").is_file()
    return ranks


def main(args):
    torch.set_num_threads(4)
    root = args.run
    launch = json.loads((root / "launch-arguments.json").read_text())
    assert launch["tasks_per_update"] == 32 and launch["branching_return_mode"] == "all"
    assert launch["train_mtp_layers"] == 0 and not launch["frozen_mtp_draft"]
    assert not launch["speculative_decoding"]
    assert not launch["capacity_only"]
    assert "--load-debug-rollout-data" not in launch["arguments"]
    batches = [audit_batch(root, step, launch) for step in range(args.steps)]
    ranks = audit_updates(root, batches)
    report = {
        "status": "PASS", "scope": "real variable-trajectory task batches",
        "optimizer_updates": args.steps, "batches": batches, "ranks": ranks,
        "train_mtp_layers": 0, "frozen_draft": False,
        "loss_weighting": launch["loss_weighting"],
    }
    (root / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "ranks"}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=3)
    main(parser.parse_args())
