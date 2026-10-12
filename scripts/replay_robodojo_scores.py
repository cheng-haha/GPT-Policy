"""Replay a fixed manifest on its original GPUs; no model calls or result replacement."""

import argparse
import concurrent.futures
import fcntl
import json
import os
import subprocess
import time
import uuid
from collections import defaultdict
from pathlib import Path

from gpt_policy.simulation.robodojo.io import sha256, write_json
from gpt_policy.simulation.robodojo.runtime import gpu_inventory


def run_case(row, output, *, native_horizon=False):
    source, gpu = Path(row["source"]).resolve(), row["gpu"]
    config = json.loads((source / "run_config.json").read_text())
    original_launch = json.loads((source / "sim/launch.json").read_text())
    case = json.loads((source / "sim/case.json").read_text())
    directory = output / row["id"] / "sim"
    directory.mkdir(parents=True, exist_ok=False)
    for name in ("case.json", "budget-config.json"):
        (directory / name).write_bytes((source / "sim" / name).read_bytes())
    if native_horizon:
        budget_path = directory / "budget-config.json"
        budget = json.loads(budget_path.read_text())
        budget["control_step_limit"] = "native"
        write_json(budget_path, budget)
    env = dict(os.environ)
    env.update(config.get("simulation_environment", {}))
    env.update(
        CUDA_VISIBLE_DEVICES=str(gpu), PYTHONPATH=original_launch["pythonpath"],
        ROBODOJO_RUN_ID="gpt-policy-score-replay-" + uuid.uuid4().hex,
        XDG_RUNTIME_DIR=str(directory / "xdg"),
    )
    Path(env["XDG_RUNTIME_DIR"]).mkdir(mode=0o700)
    command = [
        config["simulation_python"], "-u", "-m", "gpt_policy.simulation.robodojo.server",
        "--task", case["task"], "--budget-file", str(directory / "budget-config.json"),
        "--case-file", str(directory / "case.json"), "--output", str(directory),
        "--replay-from", str(source), "--device", "cuda:0", "--render-gpu", str(gpu),
    ]
    started = time.time()
    with Path(f"/tmp/gpt-policy-gpu-{os.getuid()}-{gpu}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        inventory = gpu_inventory(gpu)
        if any(p["uuid"] == inventory["uuid"] for p in inventory["processes"]):
            raise RuntimeError(f"GPU {gpu} already has a compute process")
        write_json(directory / "launch.json", {
            "source": source, "gpu": gpu, "command": command,
            "cwd": config["robodojo_root"], "gpu_before": inventory,
            "started_at_s": started, "model_calls": 0,
        })
        with (directory / "server.log").open("x") as log:
            child = subprocess.Popen(command, cwd=config["robodojo_root"], env=env,
                                     stdout=log, stderr=subprocess.STDOUT)
            write_json(directory / "pid.json", {"pid": child.pid})
            exit_code = child.wait()
    result_path = directory / "replay-result.json"
    result = json.loads(result_path.read_text()) if result_path.exists() else None
    status = {
        "id": row["id"], "source": str(source), "gpu": gpu, "exit_code": exit_code,
        "result": result, "elapsed_s": time.time() - started,
    }
    write_json(directory.parent / "status.json", status)
    return status


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--native-horizon", action="store_true",
                        help="Replay only through the original native control horizon")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    rows = manifest["cases"]
    if args.output.exists():
        raise ValueError("Replay output must be new; preserve prior attempts")
    args.output.mkdir(parents=True)
    for row in rows:
        for name, digest in row["source_hashes"].items():
            if sha256(Path(row["source"]) / name) != digest:
                raise ValueError("Source changed before replay: " + row["id"])
    write_json(args.output / "manifest.json", manifest)
    groups = defaultdict(list)
    for row in rows:
        groups[row["gpu"]].append(row)

    def run_gpu(cases):
        completed = []
        for row in cases:
            try:
                completed.append(run_case(row, args.output, native_horizon=args.native_horizon))
            except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
                status = {"id": row["id"], "error": f"{type(error).__name__}: {error}"}
                write_json(args.output / row["id"] / "status.json", status)
                completed.append(status)
        return completed

    statuses = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(groups)) as pool:
        jobs = [pool.submit(run_gpu, cases) for cases in groups.values()]
        for job in concurrent.futures.as_completed(jobs):
            statuses.extend(job.result())
            write_json(args.output / "progress.json", {"finished": statuses, "planned": len(rows)})
    write_json(args.output / "completion.json", {
        "statuses": statuses, "planned": len(rows), "finished_at_s": time.time(),
        "all_scored": all(s.get("exit_code") == 0 and s.get("result") for s in statuses),
    })


if __name__ == "__main__":
    main()
