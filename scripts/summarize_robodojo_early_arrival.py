#!/usr/bin/env python3
"""Compare fresh native-horizon B with historical unlimited A without replacing either."""

import argparse
import json
from pathlib import Path

from gpt_policy.simulation.robodojo.io import sha256, write_json


def read(path):
    return json.loads(path.read_text())


def summarize(campaign, baseline):
    frozen = read(campaign / "freeze.json")
    state = read(campaign / "state.json")
    old = read(baseline / "state.json")
    old_freeze = read(baseline / "freeze.json")
    rows = []
    for job in state["jobs"]:
        spec = job["spec"]
        source = next(j for j in old["jobs"] if j["spec"] == spec)
        old_dir, new_dir = baseline / source["id"], campaign / job["id"]
        a = read(old_dir / "result.json")
        b = read(new_dir / "result.json") if (new_dir / "result.json").exists() else {}
        eligible = (
            b.get("complete") is True and b.get("valid_for_success_rate") is True
            and type(b.get("native_success")) is bool
        )
        ledger = new_dir / "sim/waypoint-execution.jsonl"
        points = [json.loads(line) for line in ledger.read_text().splitlines()] if ledger.exists() else []
        executed = sum(p["executed_controls"] for p in points)
        controls = b.get("native_control_steps")
        if b and controls is not None and executed != controls:
            raise ValueError(f"Waypoint/ACK count mismatch: {new_dir}")
        rows.append({
            "layout": spec["layout"], "historical_a_path": str(old_dir),
            "b_path": str(new_dir), "historical_a_result_sha256": sha256(old_dir / "result.json"),
            "historical_a_controls_to_success": a["native_control_steps"],
            "historical_a_decisions": a["decisions"],
            "historical_a_success": a["native_success"],
            "historical_a_budget": a["evaluation_budget"],
            "b_status": b.get("status", job["status"]),
            "b_valid": eligible, "b_success": b.get("native_success") if eligible else None,
            "b_score_100": b["native_score"] * 100 if eligible and b.get("native_score") is not None else None,
            "b_controls": controls, "b_decisions": b.get("decisions"),
            "b_early_completed_waypoints": sum(p["stop_reason"] == "stable_arrival" for p in points),
            "b_controls_saved_within_processed_waypoints": sum(p["saved_controls"] for p in points),
            "b_processed_waypoints": len(points),
            "b_result_sha256": sha256(new_dir / "result.json") if b else None,
        })
    valid = [x for x in rows if x["b_valid"]]
    scores = [x["b_score_100"] for x in valid if x["b_score_100"] is not None]
    report = {
        "comparison": "historical_unlimited_A_vs_fresh_native_horizon_B",
        "limitation": (
            "Not a paired single-variable A/B: A had unlimited controls and different budget prompts; "
            "source/provider versions and model randomness also differ. Saved holds are direct execution "
            "accounting, not a counterfactual estimate of total episode savings or success."
        ),
        "historical_a_source_commits": sorted({e["source"]["commit"] for e in old_freeze["evidence"]}),
        "b_source_commits": sorted({e["source"]["commit"] for e in frozen["evidence"]}),
        "b_config": frozen["config"],
        "planned": len(rows), "valid": len(valid),
        "native_successes": sum(x["b_success"] for x in valid),
        "success_rate": sum(x["b_success"] for x in valid) / len(rows) if len(valid) == len(rows) else None,
        "mean_score_100": sum(scores) / len(rows) if len(scores) == len(rows) else None,
        "episodes": rows,
    }
    write_json(campaign / "comparison.json", report)
    lines = [
        "# Classify early-arrival B / historical A",
        "",
        report["limitation"],
        "",
        "| Layout | Historical A controls to success (unlimited) | B status | B success | B score /100 | B controls | B decisions | Holds saved |",
        "|---|---:|---|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        values = [row[k] for k in ("layout", "historical_a_controls_to_success", "b_status",
                  "b_success", "b_score_100", "b_controls", "b_decisions",
                  "b_controls_saved_within_processed_waypoints")]
        lines.append("| " + " | ".join("—" if v is None else str(v) for v in values) + " |")
    lines += ["", f"Valid native episodes: {len(valid)}/{len(rows)}.",
              f"Final success rate: {report['success_rate']}; final mean score /100: {report['mean_score_100']}.",
              "", "Historical records and public-release results are unchanged."]
    (campaign / "comparison.md").write_text("\n".join(lines) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    args = parser.parse_args()
    report = summarize(args.campaign, args.baseline)
    print(json.dumps({k: v for k, v in report.items() if k not in ("episodes", "b_config")}, indent=2))


if __name__ == "__main__":
    main()
