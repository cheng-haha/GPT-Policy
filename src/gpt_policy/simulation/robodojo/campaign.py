"""Frozen, bounded process queue. Resume only unstarted cases; never replay attempts."""

import fcntl
import hashlib
import json
import math
import os
import signal
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

from .config import check
from .io import append_json, write_json
from .runtime import gpu_inventory


def fingerprint(value):
    frozen = json.loads(json.dumps(value))
    for item in frozen.get("evidence", []):
        source = item.get("source")
        if isinstance(source, dict):
            # Runtime code bytes are frozen. Documentation/progress commits may continue.
            item["source"] = {"python_sha256": source["python_sha256"]}
    return hashlib.sha256(json.dumps(frozen, sort_keys=True).encode()).hexdigest()


@contextmanager
def campaign_lock(output):
    with (Path(output) / ".lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another controller owns this campaign") from error
        yield


def live_pid(pid):
    if not pid:
        return False
    try:
        os.kill(pid, 0)
        stat = Path(f"/proc/{pid}/stat").read_text()
        return stat[stat.rindex(")") + 2] != "Z"
    except (ProcessLookupError, FileNotFoundError):
        return False


def reconcile(output, state):
    """No result is sufficient to skip an attempted case, even when invalid."""
    for job in state["jobs"]:
        folder = Path(output) / job["id"]
        result_path = folder / "result.json"
        if job["status"] == "pending":
            if folder.exists():
                raise RuntimeError(f"Unexpected output for unstarted case: {folder}")
            continue
        # Do not adopt a result while its worker/simulator still owns resources.
        if live_pid(job.get("pid")):
            raise RuntimeError(
                f"Prior worker is still alive: {job['id']} PID {job['pid']}"
            )
        pid_path = folder / "sim/pid.json"
        if pid_path.exists() and not (folder / "sim/cleanup.json").exists():
            sim_pid = json.loads(pid_path.read_text())["pid"]
            if live_pid(sim_pid):
                raise RuntimeError(f"Prior simulator is still alive: PID {sim_pid}")
        if result_path.exists():
            job.update(status="finished", result=json.loads(result_path.read_text()))
        elif job["status"] == "running":
            job.update(
                status="interrupted",
                result={
                    "complete": False,
                    "valid_for_success_rate": False,
                    "native_success": None,
                    "native_score": None,
                    "status": "controller_interrupted",
                    "reason": "No authoritative result; no automatic retry",
                },
            )
    return state


def report(state):
    jobs = state["jobs"]
    finished = [j for j in jobs if j["status"] in ("finished", "interrupted")]
    valid = [
        j for j in finished
        if (r := j.get("result", {})).get("complete") is True
        and r.get("valid_for_success_rate") is True
        and type(r.get("native_success")) is bool
    ]
    scores = [
        score for j in valid
        if type(score := j["result"].get("native_score")) in (int, float)
        and math.isfinite(score)
    ]
    mean_score = sum(scores) / len(scores) if scores else None
    success = sum(j["result"].get("native_success") is True for j in valid)
    return {
        "planned": len(jobs),
        "attempted": sum(j["status"] != "pending" for j in jobs),
        "valid_native_episodes": len(valid),
        "native_successes": success,
        "native_failures": len(valid) - success,
        "success_rate_among_valid": success / len(valid) if valid else None,
        "mean_score_among_valid": mean_score,
        "scored_valid": len(scores),
        "final_success_rate": success / len(jobs) if jobs and len(valid) == len(jobs) else None,
        "final_mean_score": mean_score if jobs and len(scores) == len(jobs) else None,
        "finished": len(finished),
        "running": sum(j["status"] == "running" for j in jobs),
        "incomplete_or_invalid": len(finished) - len(valid),
        "pending": sum(j["status"] == "pending" for j in jobs),
        "dispatch_stopped": state.get("dispatch_stopped"),
        "episodes": jobs,
    }


def run_campaign(config, specs, output, gpus, *, smoke=False, resume=False):
    if not 1 <= len(gpus) <= 8 or len(set(gpus)) != len(gpus) or min(gpus) < 0:
        raise ValueError("Select 1..8 distinct nonnegative GPU IDs")
    output = Path(output).resolve()
    if resume:
        if not output.is_dir():
            raise ValueError("Resume requires an existing campaign")
    else:
        output.mkdir(parents=True, exist_ok=False)
    with campaign_lock(output):
        freeze_path = output / "freeze.json"
        evidence = [check(config, s["task"], s["seed"], s["layout"]) for s in specs]
        if not smoke and any(e["context"] is None for e in evidence):
            raise ValueError("Every policy case needs a prepared expert context")
        candidate = {
            "config": config,
            "specs": specs,
            "evidence": evidence,
            "smoke": smoke,
        }
        if resume:
            frozen = json.loads(freeze_path.read_text())
            if fingerprint(candidate) != frozen["fingerprint"]:
                raise ValueError(
                    "Frozen source/config/context/assets/cases changed; resume refused"
                )
            append_json(
                output / "resume_checks.jsonl",
                {"at_s": time.time(), "evidence": evidence},
            )
            state = reconcile(output, json.loads((output / "state.json").read_text()))
            # A persistent quota stop is never reset automatically by resuming.
        else:
            write_json(freeze_path, dict(candidate, fingerprint=fingerprint(candidate)))
            state = {
                "jobs": [
                    {
                        "id": f"{i:03d}_{e['case']['case_id']}",
                        "status": "pending",
                        "spec": specs[i],
                    }
                    for i, e in enumerate(evidence)
                ],
                "dispatch_stopped": None,
            }
        write_json(output / "state.json", state)
        active = {}
        try:
            while True:
                for gpu, (process, job, handle) in list(active.items()):
                    if process.poll() is None:
                        continue
                    handle.close()
                    folder = output / job["id"]
                    result_path = folder / "result.json"
                    result = (
                        json.loads(result_path.read_text())
                        if result_path.exists()
                        else {
                            "status": "worker_error",
                            "native_success": None,
                            "native_score": None,
                            "complete": False,
                            "valid_for_success_rate": False,
                            "error": f"worker exit {process.returncode}",
                        }
                    )
                    job.update(
                        status="finished", result=result, exit_code=process.returncode
                    )
                    if result.get("status") == "quota_error":
                        state["dispatch_stopped"] = "quota_error"
                    del active[gpu]
                    write_json(output / "state.json", state)
                    write_json(output / "report.json", report(state))
                pending = [j for j in state["jobs"] if j["status"] == "pending"]
                if not state["dispatch_stopped"]:
                    for gpu in gpus:
                        if gpu in active or not pending:
                            continue
                        inventory = gpu_inventory(gpu)
                        if any(
                            p["uuid"] == inventory["uuid"]
                            for p in inventory["processes"]
                        ):
                            # Occupied cards wait without consuming a planned attempt.
                            continue
                        job = pending.pop(0)
                        index = state["jobs"].index(job)
                        payload = output / (job["id"] + ".job.json")
                        write_json(
                            payload,
                            {
                                "config": config,
                                "evidence": evidence[index],
                                "smoke": smoke,
                                "output": str(output / job["id"]),
                                "gpu": gpu,
                            },
                        )
                        handle = (output / (job["id"] + ".log")).open("x")
                        # Save an attempt BEFORE launch; even a controller crash here cannot replay it.
                        job.update(status="running", gpu=gpu, pid=None)
                        write_json(output / "state.json", state)
                        process = subprocess.Popen(
                            [
                                sys.executable,
                                "-u",
                                "-m",
                                "gpt_policy.simulation.robodojo.worker",
                                str(payload),
                            ],
                            stdout=handle,
                            stderr=subprocess.STDOUT,
                            start_new_session=True,
                        )
                        job["pid"] = process.pid
                        active[gpu] = (process, job, handle)
                        write_json(output / "state.json", state)
                if not active and (not pending or state["dispatch_stopped"]):
                    break
                time.sleep(1 if active else 5)
        finally:
            for process, job, handle in active.values():
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=45)
                except subprocess.TimeoutExpired:
                    # Never kill a simulator indirectly and claim clean completion.
                    job["cleanup_required"] = True
                handle.close()
            write_json(output / "state.json", state)
            write_json(output / "report.json", report(state))
        return report(state)
