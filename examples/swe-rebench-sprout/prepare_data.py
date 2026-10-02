"""Prepare Miles's prompt data and Sprout's grading configuration from one dataset.

Both sides read the same frozen task rows: Miles gets ``miles.jsonl`` (one
prompt per task, ``metadata`` naming Sprout's ``task_id`` and the sandbox
``image``), Sprout's RL driver gets ``sprout-driver.json`` whose ``tasks``
section pins ``tasks.jsonl`` and ``log_parsers.py`` by SHA-256 for its
SWE-rebench grader. Nothing here pulls task images or starts a service; copy
``tasks.jsonl`` and ``log_parsers.py`` to ``--sprout-data-dir`` on the Sprout
worker host.

Example:
    python examples/swe-rebench-sprout/prepare_data.py --output-dir /tmp/swe-inputs \
        --sprout-data-dir /home/ubuntu/swe-inputs --model Qwen/Qwen3-8B --limit 3
"""

import argparse
import hashlib
import itertools
import json
from pathlib import Path
from urllib.request import urlopen

DATASET = "PrimeIntellect/SWE-rebench-V2-Filtered-Verified"
DATASET_REVISION = "03cc767ee33126b7fc7890ad57047e9dd6914cca"
PARSER_REVISION = "c4d04dfe212c153a587ea4ce072ae6753e74d6e9"
PARSER_URL = (
    "https://raw.githubusercontent.com/PrimeIntellect-ai/research-environments/"
    f"{PARSER_REVISION}/environments/swe/swerebench_v2/swerebench_v2/log_parsers.py"
)
#: What Sprout's ``swe-rebench-v2`` grader reads from a task row.
TASK_KEYS = ("instance_id", "repo", "base_commit", "test_patch", "FAIL_TO_PASS", "PASS_TO_PASS", "install_config")
OUTPUTS = ("miles.jsonl", "tasks.jsonl", "log_parsers.py", "sprout-driver.json", "manifest.json")


def prompt_for(row: dict) -> str:
    workdir = "/" + row["repo"].split("/", 1)[1]
    return f"Work in {workdir} and fix this issue:\n\n{row['problem_statement']}"


def prepare(
    rows,
    *,
    output: Path,
    sprout_data: Path,
    parser_bytes: bytes,
    model: str,
    profile: str,
    api_key_env: str | None,
    cpu: int,
    memory_mb: int,
    verifier_network: str,
    runstore_url: str,
) -> int:
    output.mkdir(parents=True, exist_ok=True)
    if any((output / name).exists() for name in OUTPUTS):
        raise FileExistsError("Use a fresh output directory: frozen task data is never overwritten")
    tasks, seen = [], set()
    with (output / "miles.jsonl").open("w", encoding="utf-8") as miles:
        for row in rows:
            task_id = row["instance_id"]
            if task_id in seen:
                raise ValueError(f"Duplicate task id {task_id}")
            seen.add(task_id)
            record = {
                "prompt": [{"role": "user", "content": prompt_for(row)}],
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
                "profile": "grade",
                "spec": {
                    "benchmark": "swe-rebench-v2",
                    "instance_id": task["instance_id"],
                    "dataset_path": str(sprout_data / "tasks.jsonl"),
                    "dataset_sha256": dataset_hash,
                    "parser_path": str(sprout_data / "log_parsers.py"),
                    "grader_revision": "sha256:" + parser_hash,
                    "verifier_network": verifier_network,
                },
            }
        }
        for task in tasks
    }
    miles_section = {
        "profile": profile,
        "run_defaults": {"slot": "mini-swe-agent", "model": model},
        "image_resources": {"cpu": cpu, "memory_mb": memory_mb},
        "tasks": grades,
    }
    if api_key_env:
        miles_section["api_key_env"] = api_key_env
    config = {
        "runstore_url": runstore_url,
        "runstore_token_env": "SPROUT_RUNSTORE_TOKEN",
        "driver_token_env": None,
        "ledger": str(sprout_data / "driver.sqlite3"),
        "poll_interval_s": 1,
        "miles": miles_section,
    }
    (output / "sprout-driver.json").write_text(json.dumps(config, indent=2) + "\n")
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


def load_rows(args):
    if args.source_jsonl:
        rows = (json.loads(line) for line in args.source_jsonl.read_text().splitlines() if line.strip())
    elif args.source_parquet:
        import pyarrow.parquet as pq

        rows = iter(pq.read_table(args.source_parquet).to_pylist())
    else:
        from datasets import load_dataset

        # Materialize before slicing: an abandoned streaming parquet iterator
        # can leave remote-read workers alive at shutdown.
        rows = iter(load_dataset(DATASET, revision=DATASET_REVISION, split="train"))
    if args.select:
        wanted = set(args.select)
        rows = (row for row in rows if row["instance_id"] in wanted)
    if args.limit:
        rows = itertools.islice(rows, args.limit)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--sprout-data-dir",
        type=Path,
        required=True,
        help="absolute directory on the Sprout worker host that will hold tasks.jsonl and log_parsers.py",
    )
    parser.add_argument("--model", required=True, help="the model name Sprout's gateway sends to the policy endpoint")
    parser.add_argument("--profile", default="mini", help="Sprout Run Store worker profile for the rollouts")
    parser.add_argument(
        "--api-key-env",
        default=None,
        help="worker-side environment variable holding the policy endpoint's API key, if it takes one",
    )
    parser.add_argument("--cpu", type=int, default=2)
    parser.add_argument("--memory-mb", type=int, default=12288)
    parser.add_argument(
        "--verifier-network",
        choices=("allow", "deny"),
        default="allow",
        help="whether the grading sandbox may reach the network while running the tests",
    )
    parser.add_argument("--runstore-url", default="http://127.0.0.1:18110")
    parser.add_argument("--limit", type=int, default=3, help="0 selects all rows")
    parser.add_argument(
        "--select",
        action="append",
        default=None,
        metavar="INSTANCE_ID",
        help="keep only these instance ids (repeatable)",
    )
    parser.add_argument("--source-jsonl", type=Path, help="local raw task rows instead of the Hub dataset")
    parser.add_argument("--source-parquet", type=Path, help="a local copy of the dataset's parquet file")
    parser.add_argument("--parser-file", type=Path, help="a local copy of the log parser; its bytes are pinned")
    args = parser.parse_args()
    if not args.sprout_data_dir.is_absolute() or args.limit < 0 or args.cpu <= 0 or args.memory_mb <= 0:
        parser.error("--sprout-data-dir must be absolute; --limit, --cpu and --memory-mb nonnegative")
    if args.parser_file:
        parser_bytes = args.parser_file.read_bytes()
    else:
        with urlopen(PARSER_URL, timeout=60) as response:
            parser_bytes = response.read()
    count = prepare(
        load_rows(args),
        output=args.output_dir,
        sprout_data=args.sprout_data_dir,
        parser_bytes=parser_bytes,
        model=args.model,
        profile=args.profile,
        api_key_env=args.api_key_env,
        cpu=args.cpu,
        memory_mb=args.memory_mb,
        verifier_network=args.verifier_network,
        runstore_url=args.runstore_url,
    )
    print(f"Prepared {count} tasks in {args.output_dir}")


if __name__ == "__main__":
    main()
