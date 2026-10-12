"""Rescore an existing panel at native horizons without changing source outcomes."""

import argparse
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from gpt_policy.simulation.robodojo.io import require, sha256, write_json


def score_at_cutoff(path, cutoff, expected_steps):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    require([r["native_control_step"] for r in rows] == list(range(expected_steps + 1)),
            "Score ledger is not contiguous: " + str(path))
    row = rows[cutoff]
    require(row["native_layout_valid"], "Invalid native layout at cutoff")
    require(not any(r["native_complete"] for r in rows[:cutoff + 1]),
            "Unexpected native termination before or at cutoff")
    score = row["score_100"]
    require(isinstance(score, (float, int)) and math.isfinite(score) and 0 <= score <= 100,
            "Invalid native process score at cutoff")
    return score, row


def check_replay(path, source, expected_controls, native_prefix):
    replay = json.loads(path.read_text())
    require(replay["source_artifacts_unchanged"], "Replay changed original artifacts")
    require(Path(replay["source"]).resolve() == source.resolve(), "Replay source mismatch")
    for key in ("max_joint_error_rad", "max_link6_position_error_m", "max_gripper_error"):
        require(replay[key] == 0, "Replay robot state diverged: " + key)
    if native_prefix:
        require(replay["replayed_requested_prefix"], "Replay did not reach requested cutoff")
        require(replay["controls_applied"] == expected_controls, "Replay cutoff mismatch")
        result = replay["native_result"]
        require(result["complete"] and result["valid_for_success_rate"],
                "Replay has no valid native terminal")
        require(result["native_success"] is False, "Replay contradicts source success timing")
        require(result["evaluation_budget"]["standard_native_horizon"],
                "Replay did not enforce native horizon")
    else:
        require(replay["replayed_all_controls"], "Historical score reconstruction incomplete")
    a = [json.loads(s) for s in (source / "sim/actions.jsonl").read_text().splitlines()]
    b = [json.loads(s) for s in (path.parent / "actions.jsonl").read_text().splitlines()]
    require(b == a[:replay["controls_applied"]], "Replay commands differ from original prefix")
    provenance = json.loads((path.parent / "replay-provenance.json").read_text())
    for original, digest in provenance["source_hashes"].items():
        require(sha256(Path(original)) == digest, "Replay source no longer matches its hash")
    return replay


def summarize(prior, audit):
    tasks, inputs = [], {}
    for task in prior["tasks"]:
        cases = []
        for case in task["cases"]:
            source = Path(case["source"]).parent
            require(sha256(source / "result.json") == case["source_sha256"],
                    "Source result changed")
            result = json.loads((source / "result.json").read_text())
            horizon = result["evaluation_budget"]["original_native_step_limit"]
            steps = result["native_control_steps"]
            success = bool(case["confirmed_native_success"] and steps <= horizon)
            evidence = None
            if steps <= horizon:
                require(success, "Short unsuccessful source needs a separate score audit")
                score, basis = 100.0, "original_success_within_native_horizon"
            elif task["task"] == "make_kong":
                # Pinned native task has no get_score; all non-successes receive zero.
                score, basis = 0.0, "native_task_has_no_partial_score"
            else:
                ledger = source / "sim/native-process-scores.jsonl"
                expected = steps
                if ledger.exists():
                    basis = "original_per_control_native_score"
                else:
                    if case.get("replay_result"):
                        replay_path = Path(case["replay_result"])
                        basis = "previous_offline_replay_prefix_score"
                        evidence = check_replay(replay_path, source, steps, False)
                    else:
                        replay_path = audit / "replays" / result["case"]["case_id"] / "sim/replay-result.json"
                        basis = "new_offline_native_horizon_replay"
                        evidence = check_replay(replay_path, source, horizon, True)
                        expected = horizon
                    ledger = replay_path.parent / "native-process-scores.jsonl"
                    inputs[str(replay_path)] = sha256(replay_path)
                if basis == "new_offline_native_horizon_replay":
                    ledger_rows = [json.loads(s) for s in ledger.read_text().splitlines()]
                    require([r["native_control_step"] for r in ledger_rows] == list(range(horizon + 1)),
                            "Native replay ledger is not contiguous")
                    cutoff_row = ledger_rows[-1]
                    score = cutoff_row["score_100"]
                    require(score == evidence["reconstructed_score_100"], "Replay score mismatch")
                    require(cutoff_row["native_complete"] and cutoff_row["native_layout_valid"],
                            "Replay ledger has no valid terminal")
                else:
                    score, cutoff_row = score_at_cutoff(ledger, horizon, expected)
                inputs[str(ledger)] = sha256(ledger)
                require(math.isfinite(score) and 0 <= score < 100,
                        "Failed prefix has an invalid or successful score")
            row = {
                "task": task["task"], "variant": case["variant"], "layout": case["layout"],
                "source": str(source), "source_result_sha256": case["source_sha256"],
                "original_controls": steps, "native_control_horizon": horizon,
                "cutoff_control_step": min(steps, horizon),
                "original_success": case["confirmed_native_success"],
                "success_within_native_horizon": success,
                "original_score_100": case["score_100"], "score_100": score, "basis": basis,
            }
            if steps > horizon and task["task"] != "make_kong":
                row.update(score_ledger=str(ledger), cutoff_score_record=cutoff_row)
            if evidence is not None:
                row["replay_verification"] = {k: evidence[k] for k in (
                    "max_joint_error_rad", "max_link6_position_error_m", "max_gripper_error",
                    "all_rgb_exact", "image_mae_255", "source_artifacts_unchanged",
                    "exact_original_score_claim")}
                row["replay_result"] = str(replay_path)
            cases.append(row)
        require(len(cases) == 5, "Expected five selected cases per task")
        successes = sum(c["success_within_native_horizon"] for c in cases)
        tasks.append({
            "task": task["task"], "name": task["name"], "cases": cases,
            "successes": successes, "count": len(cases),
            "success_rate_percent": successes / len(cases) * 100,
            "mean_score_100": sum(c["score_100"] for c in cases) / len(cases),
        })
    flat = [c for t in tasks for c in t["cases"]]
    require(len(flat) == 50, "Expected the entire selected 50-case panel")
    require(len({(c["task"], c["variant"], c["layout"]) for c in flat}) == 50,
            "Duplicate selected case")
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "type": "posthoc_native_control_horizon_prefix_rescore",
        "model_calls": 0, "source_results_modified": False,
        "exact_original_score_claim": False,
        "definition": "Keep original success only if reached by the native horizon; "
                      "otherwise use the official process score at exactly that control. "
                      "Scores from existing/new action replay are labeled reconstructions.",
        "limitations": [
            "Post-hoc truncation of existing policies, not a new budget-aware policy evaluation.",
            ("Replayed robot observations match exactly, but RGB need not be pixel-identical; "
             "original full physics states were not saved."),
            ("Scene selection, source runtime versions and ICL contexts remain those of the "
             "source panel; this audit does not establish full parity with another method."),
        ],
        "tasks": tasks,
        "overall": {
            "count": len(flat), "successes": sum(c["success_within_native_horizon"] for c in flat),
            "success_rate_percent": sum(c["success_within_native_horizon"] for c in flat) / len(flat) * 100,
            "mean_score_100": sum(c["score_100"] for c in flat) / len(flat),
            "score_sum_100": sum(c["score_100"] for c in flat),
        },
        "score_sources": dict(Counter(c["basis"] for c in flat)),
        "additional_evidence_hashes": inputs,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.audit / "manifest.json").read_text())
    original_hashes = json.loads((args.audit / "original_hashes.json").read_text())
    for path, digest in original_hashes.items():
        require(sha256(Path(path)) == digest, "Original artifact changed: " + path)
    report = summarize(json.loads(Path(manifest["source_report"]).read_text()), args.audit)
    report["replay_runtime_commit"] = manifest["runtime_commit"]
    report["source_report"] = manifest["source_report"]
    report["source_report_sha256"] = manifest["source_report_sha256"]
    report["original_hashes_verified"] = len(original_hashes)
    report["native_score_semantics"] = json.loads((args.audit / "native_score_semantics.json").read_text())
    write_json(args.audit / "NATIVE_HORIZON_RESULTS.json", report)
    lines = [
        "# 原生控制步数截断统计", "",
        "沿用当前选定的 50 条轨迹，所有任务都按各自原生步数上限截断。",
        "未成功条目取截止步的官方阶段分，不取全轨迹最高分或原终局分。",
        "这属于事后截断分析，不是重新调用策略进行预算感知评测。原始记录与公开发布包均保留。",
        "", "| 任务 | 原生步数上限 | 成功数 | Success rate | Mean score |",
        "|---|---:|---:|---:|---:|",
    ]
    for t in report["tasks"]:
        lines.append(f"| {t['name']} | {t['cases'][0]['native_control_horizon']} | "
                     f"{t['successes']}/{t['count']} | {t['success_rate_percent']:.2f}% | "
                     f"{t['mean_score_100']:.2f} |")
    o = report["overall"]
    lines += [(f"| 总体 | — | {o['successes']}/{o['count']} | {o['success_rate_percent']:.2f}% | "
              f"{o['mean_score_100']:.2f} |"), "", "## 分数来源", "",
              json.dumps(report["score_sources"], ensure_ascii=False, indent=2), "",
              ("回放分数明确标注为重建值：机器人状态逐步核对，RGB 并非像素级完全相同，"
               "原仿真完整物理状态未保存，因此不能宣称精确恢复历史阶段分。"),
              "", "## 逐条结果", "",
              "| Task | Variant | Layout | 原控制数 | 截断控制数 | 成功 | 分数 | 来源 |",
              "|---|---|---:|---:|---:|---|---:|---|"]
    for t in report["tasks"]:
        for c in t["cases"]:
            lines.append(f"| {c['task']} | {c['variant']} | {c['layout']} | "
                         f"{c['original_controls']} | {c['cutoff_control_step']} | "
                         f"{c['success_within_native_horizon']} | {c['score_100']:.2f} | {c['basis']} |")
    (args.audit / "NATIVE_HORIZON_RESULTS.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"overall": o, "score_sources": report["score_sources"]}, indent=2))


if __name__ == "__main__":
    main()
