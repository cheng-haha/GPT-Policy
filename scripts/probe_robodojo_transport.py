#!/usr/bin/env python3
"""Run a preserved-source, model-free transport comparison on idle GPUs."""

import argparse
import concurrent.futures
import fcntl
import json
import os
import subprocess
import time
import uuid
from pathlib import Path

from gpt_policy.simulation.robodojo.io import sha256, write_json
from gpt_policy.simulation.robodojo.runtime import gpu_inventory


def run_one(args, gpu, variant):
    source = args.source.resolve()
    directory = args.output / f"gpu{gpu}_{variant}" / "sim"
    directory.mkdir(parents=True, exist_ok=False)
    config = json.loads((source / "run_config.json").read_text())
    old_launch = json.loads((source / "sim/launch.json").read_text())
    case = json.loads((source / "sim/case.json").read_text())
    for name in ("case.json", "budget-config.json", "control-config.json"):
        (directory / name).write_bytes((source / "sim" / name).read_bytes())
    specification = {
        "source": str(source), "prefix_controls": args.prefix_controls,
        "variant": variant, "arm": "left", "gpu": gpu,
    }
    write_json(directory / "probe-spec.json", specification)
    env = dict(os.environ)
    env.update(config.get("simulation_environment", {}))
    env.update(CUDA_VISIBLE_DEVICES=str(gpu), PYTHONPATH=old_launch["pythonpath"],
               ROBODOJO_RUN_ID="gpt-policy-transport-" + uuid.uuid4().hex,
               XDG_RUNTIME_DIR=str(directory / "xdg"))
    Path(env["XDG_RUNTIME_DIR"]).mkdir(mode=0o700)
    command = [
        config["simulation_python"], "-u", "-m", "gpt_policy.simulation.robodojo.server",
        "--task", case["task"], "--control-file", str(directory / "control-config.json"),
        "--budget-file", str(directory / "budget-config.json"),
        "--case-file", str(directory / "case.json"), "--output", str(directory),
        "--transport-probe", str(directory / "probe-spec.json"),
        "--device", "cuda:0", "--render-gpu", str(gpu),
        "--kit_args=--/log/file=" + str(directory / "kit.log"),
    ]
    started = time.time()
    with Path(f"/tmp/gpt-policy-gpu-{os.getuid()}-{gpu}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        inventory = gpu_inventory(gpu)
        if any(p["uuid"] == inventory["uuid"] for p in inventory["processes"]):
            raise RuntimeError(f"GPU {gpu} already has a compute process")
        write_json(directory / "launch.json", {
            "command": command, "cwd": config["robodojo_root"], "gpu_before": inventory,
            "started_at_s": started, "model_calls": 0, "specification": specification,
        })
        with (directory / "server.log").open("x") as log:
            child = subprocess.Popen(command, cwd=config["robodojo_root"], env=env,
                                     stdout=log, stderr=subprocess.STDOUT)
            write_json(directory / "pid.json", {"pid": child.pid})
            exit_code = child.wait()
    result_path = directory / "probe-result.json"
    status = {
        "variant": variant, "gpu": gpu, "exit_code": exit_code,
        "result": json.loads(result_path.read_text()) if result_path.exists() else None,
        "elapsed_s": time.time() - started,
    }
    write_json(directory.parent / "status.json", status)
    return status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prefix-controls", type=int, default=856)
    parser.add_argument("--gpus", default="1,2")
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    gpus = [int(g) for g in args.gpus.split(",")]
    if len(set(gpus)) != len(gpus):
        raise ValueError("GPU assignments must be unique")
    root = Path(__file__).resolve().parents[1]
    write_json(args.output / "manifest.json", {
        "source": args.source.resolve(), "prefix_controls": args.prefix_controls,
        "gpus": gpus, "model_calls": 0, "valid_for_success_rate": False,
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "git_status": subprocess.check_output(["git", "status", "--short"], cwd=root, text=True),
        "probe_sha256": sha256(root / "src/gpt_policy/simulation/robodojo/transport_probe.py"),
        "interpretation": "Local paired engineering probe; visually verify matched prefix grasp.",
    })

    def run_gpu(index, gpu):
        statuses = []
        variants = ["direct", "distributed"] if index % 2 == 0 else ["distributed", "direct"]
        for variant in variants:
            try:
                status = run_one(args, gpu, variant)
            except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
                status = {"gpu": gpu, "variant": variant, "error": f"{type(error).__name__}: {error}"}
                write_json(args.output / f"gpu{gpu}_{variant}" / "status.json", status)
            statuses.append(status)
            print(json.dumps({k: v for k, v in status.items() if k != "result"}), flush=True)
            # Do not repeat an infrastructure failure or an unreproduced grasp blindly.
            if status.get("exit_code") != 0 or not (status.get("result") or {}).get("prefix_gate_passed"):
                break
        return statuses

    statuses = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        jobs = [pool.submit(run_gpu, index, gpu) for index, gpu in enumerate(gpus)]
        for job in concurrent.futures.as_completed(jobs):
            statuses.extend(job.result())
            write_json(args.output / "progress.json", {"statuses": statuses})
    write_json(args.output / "completion.json", {"statuses": statuses, "finished_at_s": time.time()})
    if len(statuses) != 2 * len(gpus) or any(s.get("exit_code") != 0 for s in statuses):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
