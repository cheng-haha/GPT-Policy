"""Validate native RoboDojo results used by the ten-task panel."""

import importlib.util
import json
from pathlib import Path
import sys
import threading
import time
from types import SimpleNamespace

import pytest


def test_failed_episode_is_a_completed_scored_case(tmp_path, monkeypatch):
    script = Path(__file__).resolve().parents[1] / "scripts/run_robodojo_panel50.py"
    spec = importlib.util.spec_from_file_location("run_robodojo_panel50", script)
    panel = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(panel)
    monkeypatch.setattr(panel, "ROBODOJO", tmp_path)

    native_path = tmp_path / "_result.json"
    native_path.write_text(json.dumps({
        "eval_time": 2,
        "details": {
            "0": {"layout_id": 0, "success": False, "score": 0.2},
            "1": {"layout_id": 1, "success": True, "score": 1.0},
        },
        "success_rate": 0.5,
        "score": 60.0,
    }))
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(json.dumps({"results": [{
        "task": "organize_table", "status": "PASS", "result_path": str(native_path),
    }]}))
    cases = [{"case_id": f"case_{i}", "layout_id": i} for i in range(2)]
    result = panel.read_completed_group(summary_path, "organize_table", cases)

    assert result["status"] == "complete"
    assert [item["success"] for item in result["cases"]] == [False, True]
    assert [item["score"] for item in result["cases"]] == [0.2, 1.0]


def test_selected_layout_keeps_native_layout_id(tmp_path, monkeypatch):
    robodojo = Path(__file__).resolve().parents[1] / "third_party/RoboDojo"
    monkeypatch.syspath_prepend(str(robodojo))
    from env.seed_manager import seed_manager as module

    monkeypatch.setattr(module, "ASSETS_PATH", str(tmp_path))
    layout_dir = tmp_path / "Eval_Layout/RoboDojo/arx_x5/0"
    layout_dir.mkdir(parents=True)
    for index in range(3):
        (layout_dir / f"organize_table_{index}.json").write_text("{}")
    config = {"num_envs": 1, "task_name": "organize_table", "config_name": "arx_x5", "seed": 0}
    monkeypatch.setenv("ROBODOJO_LAYOUT_ID", "2")
    manager = module.SeedManager(config)
    manager.init_eval()
    assert manager.get_seeds(max_count=1) == [2]
    assert manager.get_seeds(max_count=1) is None

    monkeypatch.setenv("ROBODOJO_LAYOUT_ID", "3")
    with pytest.raises(ValueError, match="unavailable"):
        module.SeedManager(config).init_eval()


def test_panel_schedules_isolated_cases_per_gpu_and_resumes(tmp_path, monkeypatch):
    project = Path(__file__).resolve().parents[1]
    script = project / "scripts/run_robodojo_panel50.py"
    spec = importlib.util.spec_from_file_location("panel50_scheduler", script)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    panel = json.loads((project / "configs/robodojo_panel50.json").read_text())
    robodojo = tmp_path / "RoboDojo"
    robodojo.mkdir()
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}")
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "ROBODOJO", robodojo)
    monkeypatch.setattr(runner, "MANIFEST", manifest)
    monkeypatch.setattr(runner, "validate_panel", lambda: (panel, {}))
    monkeypatch.setattr(runner, "source_fingerprints", lambda: {})

    lock = threading.Lock()
    active_gpus = set()
    seen = []

    def fake_run(command, *, cwd, env, check):
        gpu = int(env["ROBODOJO_GPU_IDS"])
        group_id = command[command.index("--run-id") + 1]
        case_id = group_id.removeprefix("scheduler_test_")
        case = next(item for item in panel["cases"] if item["case_id"] == case_id)
        assert int(env["ROBODOJO_LAYOUT_ID"]) == case["layout_id"]
        assert command[command.index("--eval-num") + 1] == "1"
        assert command[command.index("--only") + 1] == case["runtime_task"]
        with lock:
            assert gpu not in active_gpus
            active_gpus.add(gpu)
            seen.append((case["task"], gpu, case_id))
        time.sleep(0.002)
        result_path = robodojo / "eval_result" / group_id / "_result.json"
        result_path.parent.mkdir(parents=True)
        result_path.write_text(json.dumps({
            "eval_time": 1,
            "details": {"0": {"layout_id": case["layout_id"], "success": False, "score": 0.0}},
            "success_rate": 0.0, "score": 0.0,
        }))
        summary_path = robodojo / "smoke_results" / f"{group_id}.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps({"results": [{
            "task": case["runtime_task"], "status": "PASS", "result_path": str(result_path),
        }]}))
        with lock:
            active_gpus.remove(gpu)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    monkeypatch.setattr(sys, "argv", [str(script), "--gpus", "0,1", "--icl-mode", "none",
                                         "--run-id", "scheduler_test"])
    assert runner.main() == 0
    assert len(seen) == 50
    assert {gpu for _, gpu, _ in seen} == {0, 1}
    assert all(len({gpu for task, gpu, _ in seen if task == name}) == 1 for name in runner.TASKS)
    report = json.loads((tmp_path / "var/runs/robodojo/panel50/scheduler_test.json").read_text())
    assert report["status"] == "complete"
    assert len(report["cases"]) == 50
    assert report["successes"] == 0

    seen.clear()
    monkeypatch.setattr(sys, "argv", [str(script), "--gpus", "0,1", "--icl-mode", "none",
                                         "--run-id", "scheduler_test", "--resume"])
    assert runner.main() == 0
    assert seen == []
