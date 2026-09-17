"""Host-side progress summary for a B300 capacity or live training run."""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import datetime
import json
from pathlib import Path
import re
import subprocess
import urllib.request


ROOT = Path("/opt/dlami/nvme/projects/miles-stack")


def group_status(path):
    request = json.loads(path.read_text())
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    result = {"task": request["task_id"], "id": request["rollout_job_id"]}
    try:
        with opener.open(
            "http://172.17.0.1:19051/rollout-groups/" + request["rollout_job_id"],
            timeout=15,
        ) as response:
            value = json.load(response)
        result.update({
            key: value.get(key)
            for key in ("status", "actual_samples", "search_branches", "stop_reason")
        })
    except Exception as error:
        result["query_error"] = str(error)[:200]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--compact", action="store_true")
    args = parser.parse_args()
    assert "/" not in args.run and args.run not in (".", "..")
    root = ROOT / "runs/b300-lora-32tasks-20260916" / args.run
    lines = (root / "train.log").read_text(errors="replace").splitlines()
    ansi = re.compile(r"\x1b\[[0-9;]*m")
    steps = {}
    for line in lines:
        if "'train/loss':" in line:
            match = re.search(r"'train/step': (\d+)", line)
            if match:
                steps[int(match[1])] = ansi.sub("", line)
    result = {
        "checked_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "run": args.run, "optimizer_updates": len(steps),
        "last_training_metrics": steps[max(steps)] if steps else None,
        "tail": [ansi.sub("", line)[:600] for line in lines[-5:]],
        "log_error_markers": [ansi.sub("", line)[:1000] for line in lines if any(
            token in line for token in (
                "Traceback (most recent", "OutOfMemoryError:", "RuntimeError:",
                "AssertionError:", "ray.exceptions.RayTaskError",
            )
        )][-8:],
    }
    jobs = re.findall(r"Job '(raysubmit_[^']+)'", "\n".join(lines))
    if jobs:
        probe = (
            "import urllib.request,json; "
            "d=json.load(urllib.request.urlopen("
            + repr("http://127.0.0.1:8265/api/jobs/" + jobs[-1])
            + ",timeout=8)); print(json.dumps({k:d.get(k) for k in "
            "['submission_id','status','message','start_time','end_time']}))"
        )
        value = subprocess.run(
            ["docker", "exec", "miles-b300-dev", "python", "-c", probe],
            capture_output=True, text=True, timeout=15,
        )
        result["ray"] = json.loads(value.stdout) if value.returncode == 0 else {"query_error": value.stderr[-400:]}
    gpu = subprocess.run([
        "nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits",
    ], capture_output=True, text=True, timeout=10)
    result["gpus"] = gpu.stdout.strip().splitlines()
    paths = sorted(
        (root / "ash-requests").glob("*.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )[:32]
    with ThreadPoolExecutor(max_workers=32) as pool:
        result["ash"] = list(pool.map(group_status, paths))
    if (root / "run-exit.json").exists():
        result["exit"] = json.loads((root / "run-exit.json").read_text())
    (root / "status.json").write_text(json.dumps(result, indent=2) + "\n")
    if args.compact:
        print(json.dumps({
            "checked_at": result["checked_at"], "run": args.run,
            "ray_status": result.get("ray", {}).get("status"),
            "optimizer_updates": result["optimizer_updates"],
            "task_states": dict(Counter(group.get("status", "query_error") for group in result["ash"])),
            "last_training_metrics": result["last_training_metrics"],
            "gpus": result["gpus"], "tail": result["tail"][-2:],
            "exit": result.get("exit"),
        }, ensure_ascii=False, indent=2))
    else:
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
