"""Validate native RoboDojo results used by the ten-task panel."""

import importlib.util
import json
from pathlib import Path


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
