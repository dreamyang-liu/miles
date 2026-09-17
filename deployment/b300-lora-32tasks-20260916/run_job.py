"""Host launcher with persistent logs, GPU sampling, and recorded exit status."""

import argparse
import datetime
import json
from pathlib import Path
import subprocess
import time


ROOT = Path("/opt/dlami/nvme/projects/miles-stack")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["live"], default="live")
    parser.add_argument("--micro-batch-size", type=int, choices=[1, 2], required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--num-rollout", type=int, default=3)
    parser.add_argument("--loss-weighting", choices=["task", "trajectory"], default="task")
    parser.add_argument("--source-root", default="/runs/b300-lora-32tasks-20260916/source")
    parser.add_argument("--inference-profile", choices=["original", "graph"], default="graph")
    parser.add_argument("--recover-requests")
    args = parser.parse_args()
    assert "/" not in args.name and args.name not in (".", "..")
    subprocess.run([
        "docker", "exec", "miles-b300-dev", "python", "-c",
        "import torch; assert torch.cuda.device_count()==8; "
        "x=torch.ones(1,device='cuda'); torch.cuda.synchronize(); "
        "print('Container CUDA preflight passed')",
    ], check=True)
    run = ROOT / "runs/b300-lora-32tasks-20260916" / args.name
    run.mkdir(parents=True, exist_ok=False)
    command = [
        "docker", "exec", "-w", args.source_root,
        "-e", f"PYTHONPATH={args.source_root}:/root/Megatron-LM",
        "miles-b300-dev", "python",
        "/deployment/b300-lora-32tasks-20260916/launch.py",
        "--output", "/runs/b300-lora-32tasks-20260916/" + args.name,
        "--micro-batch-size", str(args.micro_batch_size),
        "--num-rollout", str(args.num_rollout),
        "--inference-profile", args.inference_profile,
        "--loss-weighting", args.loss_weighting,
    ]
    if args.recover_requests:
        command += ["--recover-requests", args.recover_requests]
    started = datetime.datetime.now(datetime.timezone.utc).isoformat()
    with (run / "train.log").open("w") as stream:
        process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT)
    (run / "host-launch.json").write_text(json.dumps({
        "started_at": started, "pid": process.pid, "command": command,
        "mode": args.mode, "micro_batch_size": args.micro_batch_size,
        "inference_profile": args.inference_profile,
        "tasks_per_update": 32, "loss_weighting": args.loss_weighting,
        "source_root": args.source_root,
    }, indent=2) + "\n")
    peaks = [0] * 8
    with (run / "gpu-memory.jsonl").open("w", buffering=1) as stream:
        while process.poll() is None:
            result = subprocess.run([
                "nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
            ], capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                values = [[int(x.strip()) for x in line.split(",")] for line in result.stdout.splitlines()]
                for index, used, utilization in values:
                    peaks[index] = max(peaks[index], used)
                stream.write(json.dumps({"time": time.time(), "gpus": values}) + "\n")
            time.sleep(1)
    (run / "run-exit.json").write_text(json.dumps({
        "returncode": process.returncode,
        "finished_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "gpu_peak_mib": peaks,
    }, indent=2) + "\n")
    raise SystemExit(process.returncode)


if __name__ == "__main__":
    main()
