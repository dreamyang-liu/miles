"""Prepare Miles inputs and Ash-owned SWE-rebench grading configuration.

This downloads task data and a pinned reference log parser, not task images.
Copy tasks.jsonl and log_parsers.py to --ash-data-dir on the Ash worker host.
Configure that worker's existing microVM/runtime profile before running rollout.

Example:
    python examples/swe-rebench-ash/prepare_data.py --output-dir /tmp/swe-inputs \
        --ash-data-dir /home/ec2-user/swe-inputs --model local-model --limit 3
"""

import argparse
import hashlib
import itertools
import json
from pathlib import Path
from urllib.request import urlopen

from datasets import load_dataset

DATASET = "PrimeIntellect/SWE-rebench-V2-Filtered-Verified"
DATASET_REVISION = "03cc767ee33126b7fc7890ad57047e9dd6914cca"
PARSER_REVISION = "c4d04dfe212c153a587ea4ce072ae6753e74d6e9"
PARSER_URL = f"https://raw.githubusercontent.com/PrimeIntellect-ai/research-environments/{PARSER_REVISION}/environments/swe/swerebench_v2/swerebench_v2/log_parsers.py"
TASK_KEYS = ("instance_id", "repo", "base_commit", "test_patch", "FAIL_TO_PASS", "PASS_TO_PASS", "install_config")


def _prepare(rows, *, output: Path, ash_data: Path, parser_bytes: bytes, model: str, profile: str) -> int:
    output.mkdir(parents=True, exist_ok=True)
    names = ("miles.jsonl", "tasks.jsonl", "log_parsers.py", "ash-driver.json", "manifest.json")
    if any((output / name).exists() for name in names):
        raise FileExistsError("Use a fresh output directory to preserve frozen task data")
    tasks, seen = [], set()
    with (output / "miles.jsonl").open("w", encoding="utf-8") as miles:
        for row in rows:
            task_id = row["instance_id"]
            if task_id in seen:
                raise ValueError(f"Duplicate task id {task_id}")
            seen.add(task_id)
            workdir = "/" + row["repo"].split("/", 1)[1]
            record = {
                "prompt": [
                    {"role": "user", "content": f"Work in {workdir} and fix this issue:\n\n{row['problem_statement']}"}
                ],
                "metadata": {"task_id": task_id, "image": row["image_name"]},
            }
            miles.write(json.dumps(record, ensure_ascii=False) + "\n")
            tasks.append({key: row[key] for key in TASK_KEYS})
    if not tasks:
        raise ValueError("No tasks selected")
    task_data = "".join(json.dumps(task, ensure_ascii=False) + "\n" for task in tasks).encode()
    dataset_hash = hashlib.sha256(task_data).hexdigest()
    parser_hash = hashlib.sha256(parser_bytes).hexdigest()
    (output / "tasks.jsonl").write_bytes(task_data)
    (output / "log_parsers.py").write_bytes(parser_bytes)
    grades = {
        task["instance_id"]: {
            "grade": {
                "profile": profile,
                "spec": {
                    "benchmark": "swe-rebench-v2",
                    "instance_id": task["instance_id"],
                    "dataset_path": str(ash_data / "tasks.jsonl"),
                    "dataset_sha256": dataset_hash,
                    "parser_path": str(ash_data / "log_parsers.py"),
                    "grader_revision": "sha256:" + parser_hash,
                    "verifier_network": "allow",
                },
            }
        }
        for task in tasks
    }
    config = {
        "runstore_url": "http://127.0.0.1:18110",
        "runstore_token_env": "ASH_RUNSTORE_TOKEN",
        "driver_token_env": None,
        "ledger": str(ash_data / "driver.sqlite3"),
        "poll_interval_s": 1,
        "miles": {
            "profile": profile,
            "run_defaults": {"slot": "codex", "tools": "shell_only", "model": model},
            "image_resources": {"cpu": 2, "memory_mb": 12288},
            "tasks": grades,
        },
    }
    (output / "ash-driver.json").write_text(json.dumps(config, indent=2) + "\n")
    (output / "manifest.json").write_text(
        json.dumps(
            {
                "dataset": DATASET,
                "dataset_revision": DATASET_REVISION,
                "parser_source_revision": PARSER_REVISION,
                "task_file_sha256": dataset_hash,
                "parser_sha256": parser_hash,
                "tasks": len(tasks),
            },
            indent=2,
        )
        + "\n"
    )
    return len(tasks)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ash-data-dir", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--profile", default="codex")
    parser.add_argument("--limit", type=int, default=3, help="0 selects all rows")
    parser.add_argument("--source-jsonl", type=Path, help="Optional local raw task rows")
    parser.add_argument("--parser-file", type=Path, help="Optional deployment-provided parser; its bytes are pinned")
    args = parser.parse_args()
    if not args.ash_data_dir.is_absolute() or args.limit < 0:
        parser.error("--ash-data-dir must be absolute and --limit must be nonnegative")
    if args.source_jsonl:
        rows = (json.loads(line) for line in args.source_jsonl.read_text().splitlines() if line.strip())
    else:
        # Materialize the dataset before slicing. Abandoning a streaming
        # parquet iterator can leave remote-read workers alive at shutdown.
        rows = load_dataset(DATASET, revision=DATASET_REVISION, split="train")
    if args.limit:
        rows = itertools.islice(rows, args.limit)
    if args.parser_file:
        parser_bytes = args.parser_file.read_bytes()
    else:
        with urlopen(PARSER_URL, timeout=60) as response:
            parser_bytes = response.read()
    count = _prepare(
        rows,
        output=args.output_dir,
        ash_data=args.ash_data_dir,
        parser_bytes=parser_bytes,
        model=args.model,
        profile=args.profile,
    )
    print(f"Prepared {count} tasks in {args.output_dir}")


if __name__ == "__main__":
    main()
