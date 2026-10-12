"""Evaluate the next layout for one natively invalid slot; preserve all old evidence.

This is an explicit one-case operation, not a retry loop or a score-based selector.
The existing campaign owns simulation, GPU arbitration, freezing and API recovery.
"""
import argparse
import copy
import json
import math
from pathlib import Path

from gpt_policy.simulation.robodojo.campaign import campaign_lock, run_campaign
from gpt_policy.simulation.robodojo.config import git, load_config
from gpt_policy.simulation.robodojo.io import require, sha256, write_json


def read(path):
    return json.loads(Path(path).read_text())


def verify_sources(report):
    paths = {}
    for task in report["tasks"]:
        for row in task["cases"]:
            path = Path(row["source"])
            require(sha256(path) == row["source_sha256"],
                    f"Historical result changed: {path}")
            paths[str(path)] = row["source_sha256"]
    return paths


def select(report, task, layout, seed):
    rows = [t for t in report["tasks"] if t["task"] == task]
    require(len(rows) == 1, "Task must occur exactly once in the report")
    cases = rows[0]["cases"]
    selected = [c for c in cases if c["layout"] == layout]
    require(len(selected) == 1, "Original layout must occur exactly once")
    old = selected[0]
    require(sha256(old["source"]) == old["source_sha256"], "Original result changed")
    result = read(old["source"])
    require(result.get("status") == "invalid_native_layout"
            and result.get("valid_for_success_rate") is False
            and result.get("native_success") is None,
            "Only a native-invalid scene may be substituted")
    identity = result.get("case", {})
    require(result.get("task", identity.get("task")) == task
            and result.get("seed", identity.get("eval_seed")) == seed
            and result.get("layout", identity.get("layout_id")) == layout,
            "Native-invalid result identity does not match selection")
    used = [c["layout"] for c in cases]
    used += [c["layout"] for c in report.get("skipped_invalid_layouts", [])
             if c["task"] == task and c["seed"] == seed]
    return copy.deepcopy(old), {"task": task, "seed": seed, "layout": max(used) + 1}


def summarize(report):
    """Keep budget/infrastructure outcomes separate and retain official 0..100 scores."""
    excluded = []
    for task in report["tasks"]:
        scores, successes, failures, censored, invalid, errors = [], 0, 0, 0, 0, 0
        for row in task["cases"]:
            result = read(row["source"])
            eligible = (result.get("complete") is True
                        and result.get("valid_for_success_rate") is True)
            row["confirmed_native_success"] = eligible and result.get("native_success") is True
            successes += int(row["confirmed_native_success"])
            failures += int(eligible and result.get("native_success") is False)
            status = result["status"]
            censored += int(status == "budget_censored")
            invalid += int(status == "invalid_native_layout")
            errors += int(not eligible and status not in
                          ("budget_censored", "invalid_native_layout"))
            score = row["score_100"]
            if score is not None:
                require(type(score) in (int, float) and math.isfinite(score)
                        and 0 <= score <= 100, "Invalid native process score")
                scores.append(score)
            else:
                excluded.append({"task": task["task"], "layout": row["layout"],
                                 "status": status})
        mean = sum(scores) / len(scores) if scores else None
        n = len(task["cases"])
        task.update(counted_cases=len(scores), score_sum_100=sum(scores),
                    mean_score_100=mean, reconstruction_aggregate_mean_score_100=mean,
                    confirmed_successes=successes, planned_cases=n,
                    confirmed_success_fraction_of_planned=successes / n if n else None,
                    native_failures=failures, budget_censored=censored,
                    invalid_scenes=invalid, infrastructure_errors=errors,
                    evaluated_layouts=[r["layout"] for r in task["cases"]])
    tasks = report["tasks"]
    scored = sum(t["counted_cases"] for t in tasks)
    planned = sum(t["planned_cases"] for t in tasks)
    successes = sum(t["confirmed_successes"] for t in tasks)
    means = [t["mean_score_100"] for t in tasks]
    report.update(counted_cases=scored, planned_cases=planned,
                  confirmed_successes=successes,
                  confirmed_success_fraction_of_planned=successes / planned if planned else None,
                  episode_weighted_mean_score_100=(
                      sum(t["score_sum_100"] for t in tasks) / scored if scored else None),
                  task_macro_mean_score_100=(
                      sum(means) / len(means) if means and None not in means else None),
                  excluded=excluded)
    return report


def merge(prior, manifest, result_path):
    verify_sources(prior)
    spec = manifest["replacement_spec"]
    result_path = Path(result_path).resolve()
    result = read(result_path)
    identity = result.get("case", {})
    require(identity.get("task") == spec["task"]
            and identity.get("eval_seed") == spec["seed"]
            and identity.get("layout_id") == spec["layout"],
            "Replacement result identity does not match frozen case")
    merged = copy.deepcopy(prior)
    task = next(t for t in merged["tasks"] if t["task"] == spec["task"])
    index = next(i for i, r in enumerate(task["cases"])
                 if r["layout"] == manifest["original_case"]["layout"])
    require(task["cases"][index] == manifest["original_case"], "Original row changed")
    status = result["status"]
    score = (result.get("native_process_score_100")
             if status in ("native_completed", "budget_censored") else None)
    task["cases"][index] = {
        "layout": spec["layout"], "seed": spec["seed"],
        "source": str(result_path), "source_sha256": sha256(result_path),
        "original_status": status, "original_reason": result.get("reason"),
        "score_100": score,
        "basis": "stored_native_process_score" if score is not None else status,
        "attempt_kind": "native_invalid_layout_substitution",
        "substitutes_layout": manifest["original_case"]["layout"],
    }
    merged.setdefault("skipped_invalid_layouts", []).append({
        "task": spec["task"], "seed": spec["seed"],
        **manifest["original_case"], "replacement_layout": spec["layout"],
    })
    merged.update(previous_report=manifest["prior_report"],
                  source_commit=manifest["source_commit"],
                  policy="Next layout for a native-invalid slot; never select by score")
    return summarize(merged)


def finalize(output):
    output = Path(output)
    manifest = read(output / "substitution.json")
    prior = read(output / "PRIOR_RESULTS.json")
    require(sha256(output / "PRIOR_RESULTS.json") == manifest["prior_report_sha256"],
            "Saved prior report changed")
    freeze = read(output / "campaign/freeze.json")
    require(freeze["specs"] == [manifest["replacement_spec"]], "Campaign case changed")
    require(freeze["config"] == read(output / "config.json"), "Campaign config changed")
    state = read(output / "campaign/state.json")
    require(len(state["jobs"]) == 1 and state["jobs"][0]["status"] == "finished",
            "Replacement worker has not finished")
    folder = output / "campaign" / state["jobs"][0]["id"]
    result = read(folder / "result.json")
    # A worker failing before simulator initialization may have no case identity.
    # Preserve its error, and never invent a finalized result or launch another layout.
    merged = merge(prior, manifest, folder / "result.json")
    write_json(output / "FINAL_RESULTS.json", merged)
    lines = ["# 十任务合并结果（原生无效场景顺延）", "",
             "顺序由场景编号决定，不按成绩择优。旧结果及无效场景全部保留。",
             "成功按计划场景数报告；到时未完成和接口异常不冒充原生失败。",
             "Mean score 使用原生 0–100 阶段分；历史离线回放重建的标记继续保留。", "",
             "| 任务 | Layouts | Mean score | 成功 / 计划数 | 原生失败 | 到时未完成 | 无效 | 接口异常 |",
             "|---|---|---:|---:|---:|---:|---:|---:|"]
    for task in merged["tasks"]:
        mean = task["mean_score_100"]
        score_text = "N/A" if mean is None else f"{mean:.2f}"
        layouts = ",".join(map(str, task["evaluated_layouts"]))
        lines.append(f"| {task['name']} | {layouts} | {score_text} | "
                     f"{task['confirmed_successes']}/{task['planned_cases']} | "
                     f"{task['native_failures']} | {task['budget_censored']} | "
                     f"{task['invalid_scenes']} | {task['infrastructure_errors']} |")
    lines += ["", f"确认成功：{merged['confirmed_successes']}/{merged['planned_cases']}。",
              f"十任务等权 Mean score：{merged['task_macro_mean_score_100']}。",
              "", f"旧合并表：{manifest['prior_report']}",
              "该合并表仍跨历史运行版本；9000 秒、不限控制次数并非官方默认控制预算。",
              "九条历史阶段得分来自固定动作的离线回放重建。"]
    (output / "FINAL_RESULTS.md").write_text("\n".join(lines) + "\n")
    write_json(output / "history_integrity.json", {
        "all_prior_results_preserved": True, "results": verify_sources(prior)})
    write_json(output / "completion.json", {
        "all_cases_finished": True, "status": result["status"],
        "native_success": result.get("native_success"),
        "native_process_score_100": result.get("native_process_score_100"),
        "report": str(output / "FINAL_RESULTS.json"),
    })
    return merged


def run(args):
    prior_path = args.prior_report.resolve()
    prior = read(prior_path)
    verify_sources(prior)
    original, spec = select(prior, args.task, args.invalid_layout, args.seed)
    require(spec["layout"] == args.expected_layout, "Next layout differs from authorization")
    config = load_config(args.config)
    config["output_root"] = str(args.output.resolve())
    config["asset_audits"][args.task] = str(args.asset_audit.resolve())
    args.output.mkdir(parents=True, exist_ok=False)
    with campaign_lock(args.output):
        (args.output / "PRIOR_RESULTS.json").write_bytes(prior_path.read_bytes())
        write_json(args.output / "config.json", config)
        project = Path(__file__).resolve().parents[1]
        manifest = {
            "prior_report": str(prior_path),
            "prior_report_sha256": sha256(prior_path),
            "source_config": str(args.config.resolve()),
            "source_config_sha256": sha256(args.config),
            "source_commit": git(project, "rev-parse", "HEAD"),
            "script_sha256": sha256(__file__),
            "original_case": original, "replacement_spec": spec,
            "selection_rule": "One next layout, only for native invalidity; no automatic retries",
        }
        write_json(args.output / "substitution.json", manifest)
        run_campaign(config, [spec], args.output / "campaign", [args.gpu])
        return finalize(args.output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("run", "finalize"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prior-report", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--asset-audit", type=Path)
    parser.add_argument("--task", default="imitate_sorting_sequence")
    parser.add_argument("--invalid-layout", type=int, default=4)
    parser.add_argument("--expected-layout", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()
    if args.action == "run":
        if not all((args.prior_report, args.config, args.asset_audit)):
            parser.error("run requires --prior-report, --config and --asset-audit")
        result = run(args)
    else:
        with campaign_lock(args.output):
            result = finalize(args.output)
    print(json.dumps({k: result[k] for k in
                      ("confirmed_successes", "planned_cases", "task_macro_mean_score_100")}))


if __name__ == "__main__":
    main()
