#!/usr/bin/env python3
"""Run the 50 published GPT-as-Policy RoboDojo layouts with GPT-Policy.

This reproduces the published case selection and one-process-per-case reset
protocol, not the other project's policy. Task failure is a valid completed
outcome; missing native results are not counted as failures or silently replaced.
"""

from __future__ import annotations

import argparse
from collections import Counter, OrderedDict, deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import threading


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
    source_files = panel.get("native_source_files")
    overrides = panel.get("native_overrides")
    require(isinstance(source_files, list) and len(source_files) == 169,
            "Expected the 169 published native source fingerprints")
    require(isinstance(overrides, dict) and len(overrides) == 5,
            "Expected exactly five reviewed native integration changes")
    source_paths = [item["path"] for item in source_files]
    require(len(set(source_paths)) == len(source_paths) and set(overrides) <= set(source_paths),
            "Duplicate native source path or unknown integration change")
    for item in source_files:
        relative = Path(item["path"])
        require(not relative.is_absolute() and ".." not in relative.parts,
                f"Invalid native source path: {relative}")
        source = ROBODOJO / relative
        require(source.is_file(), f"Missing native source file: {source}")
        require(digest(source) == overrides.get(item["path"], item["sha256"]),
                f"Native source differs from frozen panel/integration: {relative}")
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
        "third_party/RoboDojo/env/seed_manager/seed_manager.py",
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
    parser.add_argument("--gpu", type=int, default=None, help="One GPU ID (default: 0)")
    parser.add_argument("--gpus", default=None, help="Comma-separated GPU IDs; one case process per GPU")
    parser.add_argument("--icl-mode", choices=("video+action", "video", "none"), default="video+action")
    parser.add_argument("--require-icl", action="store_true", help="Fail if any task lacks a demonstration")
    parser.add_argument("--run-id", default=None, help="Stable ID for reporting and resume")
    parser.add_argument("--resume", action="store_true", help="Resume a prior --run-id")
    parser.add_argument("--case-timeout-s", type=int, default=3600,
                        help="Maximum wall time for one case; timeout is incomplete, not failure")
    parser.add_argument("--dry-run", action="store_true", help="Verify layouts and print 50 case commands")
    args = parser.parse_args()
    require(args.gpu is None or args.gpus is None, "Use --gpu or --gpus, not both")
    gpu_text = args.gpus if args.gpus is not None else str(args.gpu if args.gpu is not None else 0)
    require(re.fullmatch(r"[0-9]+(,[0-9]+)*", gpu_text) is not None, "Invalid GPU list")
    gpus = [int(value) for value in gpu_text.split(",")]
    require(len(set(gpus)) == len(gpus), "Duplicate GPU ID")
    require(args.case_timeout_s > 0, "Case timeout must be positive")
    require(not args.resume or args.run_id, "--resume requires --run-id")
    panel, _groups = validate_panel()
    missing_demos = sorted(task for task in TASKS if not (DATASET_ROOT / task / "meta/tasks.parquet").is_file())
    if args.icl_mode != "none" and missing_demos:
        print("ICL demonstration missing for: " + ", ".join(missing_demos), file=sys.stderr)
        require(not args.require_icl, "Cannot run uniform ICL without all ten demonstrations")
    run_id = args.run_id or "xingwu_panel50_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    require(re.fullmatch(r"[A-Za-z0-9_-]+", run_id) is not None, "Invalid run ID")
    report_path = ROOT / "var/runs/robodojo/panel50" / f"{run_id}.json"
    fingerprints = source_fingerprints()
    manifest_hash = digest(MANIFEST)
    assignments: dict[int, list[dict]] = {gpu: [] for gpu in gpus}
    cases_by_task = {task: [case for case in panel["cases"] if case["task"] == task] for task in TASKS}
    for task_index, task in enumerate(TASKS):
        gpu = gpus[task_index % len(gpus)]
        assignments[gpu].extend(cases_by_task[task])

    def command_for(case: dict) -> list[str]:
        return ["timeout", "--signal=TERM", "--kill-after=20s", f"{args.case_timeout_s}s",
                str(ROOT / "scripts/run_robodojo_eval.sh"), "--skip-setup",
                "benchmark", "--icl-mode", args.icl_mode, "--only", case["runtime_task"],
                "--eval-num", "1", "--seed", "0", "--run-id", f"{run_id}_{case['case_id']}"]

    def summary_for(case: dict) -> Path:
        return ROBODOJO / "smoke_results" / f"{run_id}_{case['case_id']}.json"

    print(f"Verified {len(panel['cases'])} published cases, "
          f"{len(panel['native_source_files']) - len(panel['native_overrides'])} matching native files and "
          f"{len(panel['native_overrides'])} reviewed overrides, "
          f"{len(TASKS)} tasks, 50 isolated native runs on {len(gpus)} GPU(s)")
    if args.dry_run:
        print("GPU assignment below is a preview; live workers take the next task as they become free")
        for gpu, cases in assignments.items():
            for case in cases:
                print(f"ROBODOJO_GPU_IDS={gpu} ROBODOJO_LAYOUT_ID={case['layout_id']} " +
                      shlex.join(command_for(case)))
        return 0

    if args.resume:
        require(report_path.is_file(), f"No report to resume: {report_path}")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        require(report.get("schema") == "gpt_policy.robodojo.panel50_result.v2", "Incompatible report schema")
        require(report.get("manifest_sha256") == manifest_hash and report.get("source_fingerprints") == fingerprints,
                "Source or panel changed since this run started")
        require(report.get("gpus") == gpus and report.get("icl_mode") == args.icl_mode
                and report.get("case_timeout_s") == args.case_timeout_s,
                "GPU assignment, ICL mode, or timeout changed since this run started")
        for case in panel["cases"]:
            completed = report["cases"].get(case["case_id"])
            if completed and completed.get("status") == "complete":
                verified = read_completed_group(Path(completed["summary_path"]), case["runtime_task"], [case])
                require(completed["result"] == verified["cases"][0],
                        f"Previously completed result changed: {case['case_id']}")
    else:
        require(not report_path.exists(), f"Run ID already exists: {run_id}; use --resume or another ID")
        report = dict(schema="gpt_policy.robodojo.panel50_result.v2", run_id=run_id,
                      source_panel=panel["source_repository"] + "/blob/" + panel["source_commit"] + "/" + panel["source_path"],
                      manifest_sha256=manifest_hash, source_fingerprints=fingerprints,
                      gpus=gpus, icl_mode=args.icl_mode, missing_icl_demonstrations=missing_demos,
                      case_timeout_s=args.case_timeout_s,
                      published_policy_rng_seed_note="OpenPI JAX seed from source panel; not applied to GPT-Policy Codex",
                      status="running", cases={})
        write_report(report_path, report)

    lock = threading.Lock()
    task_lock = threading.Lock()
    stop = threading.Event()
    pending_tasks = deque(TASKS)

    def run_gpu_queue(gpu: int) -> None:
        while not stop.is_set():
            with task_lock:
                if not pending_tasks:
                    return
                task = pending_tasks.popleft()
            print(f"GPU {gpu}: assigned task {task}", flush=True)
            for case in cases_by_task[task]:
                if not run_case(gpu, case):
                    return

    def run_case(gpu: int, case: dict) -> bool:
        case_id = case["case_id"]
        if stop.is_set():
            return False
        if report["cases"].get(case_id, {}).get("status") == "complete":
            print(f"Skipping completed {case_id}", flush=True)
            return True
        env = os.environ.copy()
        env["ROBODOJO_GPU_IDS"] = str(gpu)
        env["ROBODOJO_LAYOUT_ID"] = str(case["layout_id"])
        print(f"GPU {gpu}: running {case_id}", flush=True)
        summary_path = summary_for(case)
        try:
            rc = subprocess.run(command_for(case), cwd=ROOT, env=env, check=False).returncode
            require(rc == 0, f"RoboDojo exited {rc} for {case_id}")
            native = read_completed_group(summary_path, case["runtime_task"], [case])
            entry = dict(status="complete", gpu=gpu, summary_path=str(summary_path),
                         result_path=native["result_path"], result=native["cases"][0])
        except (OSError, ValueError, KeyError, TypeError) as error:
            entry = dict(status="incomplete", gpu=gpu, summary_path=str(summary_path), error=str(error))
            stop.set()
            print(f"Stopped {case_id}: {error}", file=sys.stderr, flush=True)
        with lock:
            report["cases"][case_id] = entry
            if entry["status"] != "complete":
                report["status"] = "incomplete"
            write_report(report_path, report)
        if entry["status"] != "complete":
            return False
        return True

    with ThreadPoolExecutor(max_workers=len(gpus)) as executor:
        futures = [executor.submit(run_gpu_queue, gpu) for gpu in gpus]
        for future in futures:
            future.result()

    results = [report["cases"].get(case["case_id"]) for case in panel["cases"]]
    if any(item is None or item.get("status") != "complete" for item in results):
        report["status"] = "incomplete"
        write_report(report_path, report)
        print(f"Incomplete panel; partial report: {report_path}", file=sys.stderr)
        return 1
    case_results = [item["result"] for item in results]
    report["status"] = "complete"
    report["successes"] = sum(x["success"] for x in case_results)
    report["success_rate"] = report["successes"] / 50
    report["mean_score"] = 100 * sum(x["score"] for x in case_results) / 50
    write_report(report_path, report)
    print(f"Complete: {report['successes']}/50 success, Score {report['mean_score']:.2f}; {report_path}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"panel50: {error}", file=sys.stderr)
        sys.exit(2)
