"""Merge reconstructed scores into a separate report, retaining all source results."""

import argparse
import json
from pathlib import Path

from gpt_policy.simulation.robodojo.io import write_json


def summarize(prior, replay_root):
    tasks, reconstructed = [], []
    for task in prior["tasks"]:
        scores, cases = [], []
        for case in task["cases"]:
            row = dict(case)
            if case["basis"] == "missing_original_process_state":
                case_id = f"{task['task']}__g0__l{case['layout']}"
                path = replay_root / case_id / "sim/replay-result.json"
                replay = json.loads(path.read_text())
                if not replay["replayed_all_controls"]:
                    raise ValueError("Replay terminated early: " + case_id)
                if not replay["source_artifacts_unchanged"]:
                    raise ValueError("Original changed: " + case_id)
                score = replay["reconstructed_score_100"]
                if score is None:
                    raise ValueError("Replay native scene is invalid: " + case_id)
                row.update(score_100=score, basis="offline_replay_reconstruction",
                           replay_result=str(path))
                reconstructed.append({
                    "task": task["task"], "layout": case["layout"], "score_100": score,
                    "controls": replay["controls_applied"],
                    "max_link6_position_error_m": replay["max_link6_position_error_m"],
                    "max_joint_error_rad": replay["max_joint_error_rad"],
                    "max_gripper_error": replay["max_gripper_error"],
                    "all_rgb_exact": replay["all_rgb_exact"],
                    "image_mae_255": replay["image_mae_255"],
                })
            if row["score_100"] is not None:
                scores.append(row["score_100"])
            cases.append(row)
        tasks.append({
            "task": task["task"], "name": task["name"],
            "counted_cases": len(scores), "score_sum_100": sum(scores),
            "reconstruction_aggregate_mean_score_100": sum(scores) / len(scores),
            "cases": cases,
        })
    if len(reconstructed) != 9:
        raise ValueError("Expected exactly nine reconstructed cases")
    return {
        "score_kind": "mixed_original_and_offline_replay_reconstruction",
        "exact_original_score_claim": False,
        "original_success_results_changed": False,
        "reconstructed_cases": reconstructed,
        "mean_score_of_nine_replays_100": sum(r["score_100"] for r in reconstructed) / 9,
        "tasks": tasks,
        "counted_cases": sum(t["counted_cases"] for t in tasks),
        "episode_weighted_mean_score_100": (
            sum(t["score_sum_100"] for t in tasks) / sum(t["counted_cases"] for t in tasks)),
        "task_macro_mean_score_100": (
            sum(t["reconstruction_aggregate_mean_score_100"] for t in tasks) / len(tasks)),
        "excluded": ["imitate_sorting_sequence layout4: native unstable",
                     "make_kong layout4: API interruption"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prior-report", type=Path, required=True)
    parser.add_argument("--replay-root", type=Path, required=True)
    args = parser.parse_args()
    report = summarize(json.loads(args.prior_report.read_text()), args.replay_root)
    write_json(args.replay_root / "SCORE_REPORT.json", report)
    lines = [
        "# 九条离线回放重建得分", "",
        "分数来自固定动作的离线回放，不替换原始成功/失败或中断记录。单位 0–100。",
        "全部原布局、配置、资产、GPU 与动作序列经过核对；数值和图像差异见 JSON。",
        "",
        "| 任务 | Layout | 回放得分 | 控制步数 | 最大 link6 位置差（米） |",
        "|---|---:|---:|---:|---:|",
    ]
    for r in report["reconstructed_cases"]:
        lines.append(f"| {r['task']} | {r['layout']} | {r['score_100']:.2f} | "
                     f"{r['controls']} | {r['max_link6_position_error_m']:.8g} |")
    lines += ["", f"九条回放平均分：{report['mean_score_of_nine_replays_100']:.4f}", "",
              "| 十任务合并 | 计入条数 | Mean score（0–100） |",
              "|---|---:|---:|"]
    for t in report["tasks"]:
        lines.append(f"| {t['name']} | {t['counted_cases']} | "
                     f"{t['reconstruction_aggregate_mean_score_100']:.4f} |")
    lines += ["",
              f"按 {report['counted_cases']} 条加权：{report['episode_weighted_mean_score_100']:.4f}",
              f"按十任务等权：{report['task_macro_mean_score_100']:.4f}",
              "",
              "顺序模仿 layout4 按官方无效场景规则剔除；Make Kong layout4 API 中断另列。",
              "协议仍为原先的 9000 秒截止、不限控制次数；不是官方原始步数时限。",
              "原先确认的成功数保持 37/50；不会用回放结果重选成功轨迹。"]
    (args.replay_root / "SCORE_REPORT.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({k: report[k] for k in (
        "mean_score_of_nine_replays_100", "episode_weighted_mean_score_100",
        "task_macro_mean_score_100", "counted_cases")}, indent=2))


if __name__ == "__main__":
    main()
