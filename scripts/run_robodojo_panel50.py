#!/usr/bin/env python3
"""Run the 50 published GPT-as-Policy RoboDojo layouts with GPT-Policy.

This reproduces the *case selection*, not the other project's policy. Each
native RoboDojo invocation evaluates consecutive layout IDs for one task
variant. Task failure is a valid completed outcome; missing native results are
not counted as failures or silently replaced.
"""

from __future__ import annotations

import argparse
from collections import Counter, OrderedDict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
ROBODOJO = ROOT / "third_party/RoboDojo"
MANIFEST = ROOT / "configs/robodojo_panel50.json"
LAYOUT_ROOT = ROBODOJO / "Assets/Eval_Layout/RoboDojo/arx_x5"
DATASET_ROOT = Path(os.environ.get("ROBODOJO_ICL_ROOT", "/mnt/data/cpfs/b5/post_train_data/robodojo_sim"))
TASKS = (
    "organize_table", "classify_objects_by_language", "imitate_sorting_sequence",
    "arrange_largest_number", "pack_objects_into_box", "classify_objects",
    "build_tower", "make_kong", "fold_clothes", "put_bottles_into_dustbin",
)
GENERALIZATION = {"arrange_largest_number", "pack_objects_into_box", "fold_clothes"}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def layout_files(runtime_task: str, eval_seed: int) -> list[Path]:
    directory = LAYOUT_ROOT / str(eval_seed)
    require(directory.is_dir(), f"Missing layout group: {directory}")
    pattern = re.compile(rf"{re.escape(runtime_task)}_\d+\.json")
    return sorted(
        (p for p in directory.iterdir() if pattern.fullmatch(p.name)),
        key=lambda p: int(p.stem.rsplit("_", 1)[-1]),
    )


def validate_panel() -> tuple[dict, OrderedDict[str, list[dict]]]:
    panel = json.loads(MANIFEST.read_text(encoding="utf-8"))
    require(panel.get("schema") == "gpt_policy.robodojo.panel50.v1", "Unknown panel schema")
    support_files = panel.get("support_files")
    require(isinstance(support_files, list) and len(support_files) == 10,
            "Expected the ten published support trajectories")
    require(len({item["path"] for item in support_files}) == len(support_files),
            "Duplicate support trajectory")
    for item in support_files:
        relative = Path(item["path"])
        require(not relative.is_absolute() and ".." not in relative.parts,
                f"Invalid support trajectory path: {relative}")
        support = ROBODOJO / relative
        require(support.is_file(), f"Missing support trajectory: {support}")
        require(digest(support) == item["sha256"],
                f"Support trajectory hash differs from published file: {relative}")
    cases = panel.get("cases")
    require(isinstance(cases, list) and len(cases) == 50, "Expected exactly 50 cases")
    require(len({c["case_id"] for c in cases}) == 50, "Duplicate case ID")
    per_task = Counter(c["task"] for c in cases)
    require(set(per_task) == set(TASKS) and all(n == 5 for n in per_task.values()),
            "Expected five cases for each of the ten tasks")
    per_variant = Counter((c["task"], c["variant"]) for c in cases)
    for task in TASKS:
        expected = {"standard": 2, "random": 3} if task in GENERALIZATION else {"standard": 5}
        actual = {variant: n for (name, variant), n in per_variant.items() if name == task}
        require(actual == expected, f"Unexpected standard/random split for {task}: {actual}")

    groups: OrderedDict[str, list[dict]] = OrderedDict()
    for case in cases:
        task = case["task"]
        variant = case["variant"]
        runtime_task = task + ("_random" if variant == "random" else "")
        require(case["runtime_task"] == runtime_task, f"Wrong runtime task: {case['case_id']}")
        require(case["eval_seed"] == 0 and case["simulator_initial_seed"] == 0,
                f"Unsupported eval group/startup seed: {case['case_id']}")
        require(case["reset_seed"] == case["layout_id"], f"Wrong reset seed: {case['case_id']}")
        require(case["policy_rng_seed"] == 0, f"Unexpected policy seed: {case['case_id']}")
        expected_id = f"{task}__{variant}__g0__l{case['layout_id']}"
        require(case["case_id"] == expected_id, f"Wrong case ID: {case['case_id']}")
        groups.setdefault(runtime_task, []).append(case)

    for runtime_task, rows in groups.items():
        ids = sorted(c["layout_id"] for c in rows)
        require(ids == list(range(len(rows))), f"Nonconsecutive layouts for {runtime_task}: {ids}")
        files = layout_files(runtime_task, 0)
        for case in rows:
            index = case["layout_id"]
            require(index < len(files), f"Missing layout: {case['case_id']}")
            require(digest(files[index]) == case["layout_sha256"],
                    f"Layout hash differs from published case: {case['case_id']}")
        require((ROBODOJO / "task/RoboDojo/config" / f"{runtime_task}.yml").is_file(),
                f"Missing RoboDojo task config: {runtime_task}")
    return panel, groups


def source_fingerprints() -> dict[str, str]:
    paths = (
        "scripts/run_robodojo_panel50.py",
        "scripts/run_robodojo_eval.sh",
        "scripts/eval_robodojo_8gpu.sh",
        "third_party/RoboDojo/scripts/internal/smoke_all_tasks.sh",
        "third_party/RoboDojo/src/eval_client/main.py",
        "src/gpt_policy/robodojo/model.py",
        "src/gpt_policy/robodojo/deploy.py",
        "src/gpt_policy/robodojo/adapter.py",
        "src/gpt_policy/robodojo/execution.py",
        "third_party/RoboDojo/env_cfg/gpt_policy_x5.yml",
        "third_party/RoboDojo/src/eval_client/eval_env.py",
    )
    return {name: digest(ROOT / name) for name in paths}


def write_report(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_completed_group(summary_path: Path, runtime_task: str, cases: list[dict]) -> dict:
    require(summary_path.is_file(), f"Missing RoboDojo summary: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    rows = summary.get("results", [])
    require(len(rows) == 1 and rows[0].get("task") == runtime_task,
            f"Unexpected RoboDojo summary rows: {summary_path}")
    row = rows[0]
    require(row.get("status") == "PASS", f"Native evaluation did not finish: {runtime_task}")
    result_path = Path(row["result_path"]).resolve()
    require(result_path.is_relative_to(ROBODOJO.resolve()) and result_path.is_file(),
            f"Missing native _result.json: {result_path}")
    native = json.loads(result_path.read_text(encoding="utf-8"))
    require(native.get("eval_time") == len(cases), f"Wrong episode count for {runtime_task}")
    details = list(native.get("details", {}).values())
    require(len(details) == len(cases), f"Missing native case details for {runtime_task}")
    by_layout = {int(item["layout_id"]): item for item in details}
    require(len(by_layout) == len(cases), f"Duplicate native layout ID for {runtime_task}")
    case_results = []
    for case in cases:
        item = by_layout.get(case["layout_id"])
        require(item is not None, f"Missing native layout result: {case['case_id']}")
        require(type(item.get("success")) is bool, f"Invalid native success: {case['case_id']}")
        score = float(item["score"])
        require(0 <= score <= 1, f"Invalid native score: {case['case_id']}")
        case_results.append(dict(case_id=case["case_id"], runtime_task=runtime_task,
                                 layout_id=case["layout_id"], success=item["success"], score=score))
    require(abs(native["success_rate"] - sum(x["success"] for x in case_results) / len(cases)) < 1e-6,
            f"Native success aggregate differs: {runtime_task}")
    require(abs(native["score"] - 100 * sum(x["score"] for x in case_results) / len(cases)) < 1e-4,
            f"Native score aggregate differs: {runtime_task}")
    return dict(status="complete", summary_path=str(summary_path), result_path=str(result_path),
                cases=case_results)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--icl-mode", choices=("video+action", "video", "none"), default="video+action")
    parser.add_argument("--require-icl", action="store_true", help="Fail if any task lacks a demonstration")
    parser.add_argument("--run-id", default=None, help="Stable ID for reporting and resume")
    parser.add_argument("--resume", action="store_true", help="Resume a prior --run-id")
    parser.add_argument("--dry-run", action="store_true", help="Verify all layouts and print the 13 commands")
    args = parser.parse_args()
    require(args.gpu >= 0, "GPU ID must be nonnegative")
    require(not args.resume or args.run_id, "--resume requires --run-id")
    panel, groups = validate_panel()
    missing_demos = sorted(task for task in TASKS if not (DATASET_ROOT / task / "meta/tasks.parquet").is_file())
    if args.icl_mode != "none" and missing_demos:
        print("ICL demonstration missing for: " + ", ".join(missing_demos), file=sys.stderr)
        require(not args.require_icl, "Cannot run uniform ICL without all ten demonstrations")
    run_id = args.run_id or "xingwu_panel50_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    require(re.fullmatch(r"[A-Za-z0-9_-]+", run_id) is not None, "Invalid run ID")
    report_path = ROOT / "var/runs/robodojo/panel50" / f"{run_id}.json"
    fingerprints = source_fingerprints()
    manifest_hash = digest(MANIFEST)
    commands = {}
    for runtime_task, cases in groups.items():
        group_id = f"{run_id}_{runtime_task}"
        commands[runtime_task] = [str(ROOT / "scripts/run_robodojo_eval.sh"), "--skip-setup",
                                  "benchmark", "--icl-mode", args.icl_mode, "--only", runtime_task,
                                  "--eval-num", str(len(cases)), "--seed", "0", "--run-id", group_id]
    print(f"Verified {len(panel['cases'])} published cases, {len(TASKS)} tasks, {len(groups)} native runs")
    if args.dry_run:
        for runtime_task, cmd in commands.items():
            print(f"ROBODOJO_GPU_IDS={args.gpu} " + shlex.join(cmd))
        return 0

    if args.resume:
        require(report_path.is_file(), f"No report to resume: {report_path}")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        require(report.get("manifest_sha256") == manifest_hash and report.get("source_fingerprints") == fingerprints,
                "Source or panel changed since this run started")
        require(report.get("gpu") == args.gpu and report.get("icl_mode") == args.icl_mode,
                "GPU or ICL mode changed since this run started")
    else:
        require(not report_path.exists(), f"Run ID already exists: {run_id}; use --resume or another ID")
        report = dict(schema="gpt_policy.robodojo.panel50_result.v1", run_id=run_id,
                      source_panel=panel["source_repository"] + "/blob/" + panel["source_commit"] + "/" + panel["source_path"],
                      manifest_sha256=manifest_hash, source_fingerprints=fingerprints,
                      gpu=args.gpu, icl_mode=args.icl_mode, missing_icl_demonstrations=missing_demos,
                      status="running", groups={})
        write_report(report_path, report)

    env = os.environ.copy()
    env["ROBODOJO_GPU_IDS"] = str(args.gpu)
    for runtime_task, cases in groups.items():
        if report["groups"].get(runtime_task, {}).get("status") == "complete":
            completed = report["groups"][runtime_task]
            verified = read_completed_group(Path(completed["summary_path"]), runtime_task, cases)
            require(completed["cases"] == verified["cases"],
                    f"Previously completed result changed: {runtime_task}")
            print(f"Skipping completed {runtime_task}")
            continue
        command = commands[runtime_task]
        print(f"Running {runtime_task}: {len(cases)} layouts", flush=True)
        rc = subprocess.run(command, cwd=ROOT, env=env, check=False).returncode
        summary_path = ROBODOJO / "smoke_results" / f"{run_id}_{runtime_task}.json"
        try:
            require(rc == 0, f"RoboDojo exited {rc} for {runtime_task}")
            report["groups"][runtime_task] = read_completed_group(summary_path, runtime_task, cases)
        except (OSError, ValueError, KeyError, TypeError) as error:
            report["groups"][runtime_task] = dict(status="incomplete", error=str(error),
                                                   summary_path=str(summary_path))
            report["status"] = "incomplete"
            write_report(report_path, report)
            print(f"Stopped: {error}; partial report: {report_path}", file=sys.stderr)
            return 1
        write_report(report_path, report)

    results = [case for group in report["groups"].values() for case in group["cases"]]
    require(len(results) == 50 and len({x["case_id"] for x in results}) == 50,
            "Panel is incomplete despite all native runs returning")
    report["status"] = "complete"
    report["successes"] = sum(x["success"] for x in results)
    report["success_rate"] = report["successes"] / 50
    report["mean_score"] = 100 * sum(x["score"] for x in results) / 50
    write_report(report_path, report)
    print(f"Complete: {report['successes']}/50 success, Score {report['mean_score']:.2f}; {report_path}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"panel50: {error}", file=sys.stderr)
        sys.exit(2)
