#!/usr/bin/env python3
"""Inspect one frozen campaign; --recover-quota uses only its explicit free-reset authorization."""

import argparse
import json
import math
import time
from collections import Counter
from pathlib import Path

from gpt_policy.simulation.robodojo.io import write_json
from gpt_policy.simulation.robodojo.quota import read_quota, recover_quota


def valid_terminal(result):
    return (
        result.get("complete") is True
        and result.get("valid_for_success_rate") is True
        and type(result.get("native_success")) is bool
    )


def snapshot(output):
    frozen = json.loads((output / "freeze.json").read_text())
    state = json.loads((output / "state.json").read_text())
    rows = []
    for job in state["jobs"]:
        row = {k: job.get(k) for k in ("id", "spec", "status", "gpu", "pid")}
        progress = output / job["id"] / "progress.json"
        if progress.exists():
            row["progress"] = json.loads(progress.read_text())
            row["progress_age_s"] = round(time.time() - progress.stat().st_mtime, 1)
        result_path = output / job["id"] / "result.json"
        result = json.loads(result_path.read_text()) if result_path.exists() else job.get("result")
        if result:
            row["result"] = {k: result.get(k) for k in (
                "complete", "valid_for_success_rate", "native_success", "native_score",
                "reason", "status", "decisions", "provider_recoveries", "quota_recoveries", "error",
            )}
        rows.append(row)
    tasks = {}
    for task in sorted({j["spec"]["task"] for j in rows}):
        subset = [j for j in rows if j["spec"]["task"] == task]
        valid = [j for j in subset if valid_terminal(j.get("result", {}))]
        done = [j for j in subset if j["status"] in ("finished", "interrupted")]
        successes = sum(j["result"]["native_success"] is True for j in valid)
        scores = [
            score for j in valid
            if type(score := j["result"].get("native_score")) in (int, float)
            and math.isfinite(score)
        ]
        mean_score = sum(scores) / len(scores) if scores else None
        all_cases_valid = len(valid) == len(subset)
        tasks[task] = {
            "planned": len(subset), "finished": len(done), "valid": len(valid),
            "successes": successes, "native_failures": len(valid) - successes,
            "invalid_finished": sum(not valid_terminal(j.get("result", {})) for j in done),
            "invalid_finished_reasons": dict(Counter(
                j.get("result", {}).get("reason") or "missing_result"
                for j in done if not valid_terminal(j.get("result", {}))
            )),
            "running": sum(j["status"] == "running" for j in subset),
            "pending": sum(j["status"] == "pending" for j in subset),
            "success_rate_among_valid": successes / len(valid) if valid else None,
            "mean_score_among_valid": mean_score,
            "scored_valid": len(scores),
            "confirmed_success_fraction": successes / len(subset),
            "all_attempts_finished": len(done) == len(subset),
            "all_cases_valid": all_cases_valid,
            "final_success_rate": successes / len(subset) if all_cases_valid else None,
            "final_mean_score": mean_score if len(scores) == len(subset) else None,
        }
    return frozen["config"], {
        "checked_at_s": time.time(), "output": str(output),
        "states": dict(Counter(j["status"] for j in rows)),
        "dispatch_stopped": state.get("dispatch_stopped"),
        "tasks": tasks, "episodes": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--recover-quota", action="store_true")
    args = parser.parse_args()
    config, report = snapshot(args.output)
    if args.recover_quota and any(report["states"].get(s, 0) for s in ("pending", "running")):
        report["quota_available_after_authorized_recovery"] = recover_quota(config)
    try:
        report["quota"] = read_quota(config)
    except Exception as error:  # noqa: BLE001 - retain progress when quota transport is unavailable
        report["quota_error"] = f"{type(error).__name__}: {error}"
    write_json(args.output / "monitor.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
