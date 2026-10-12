"""Progress file timestamps and running cases must survive real monitor aggregation."""

import json
import runpy
from pathlib import Path


def test_live_progress_age_and_terminal_denominators(tmp_path):
    monitor = runpy.run_path(str(
        Path(__file__).resolve().parents[2] / "scripts/monitor_robodojo_campaign.py"
    ))
    (tmp_path / "freeze.json").write_text('{"config": {"model": "test"}}')
    jobs = [
        {"id": "pending", "status": "pending", "spec": {"task": "task"}},
        {"id": "running", "status": "running", "spec": {"task": "task"}},
        {"id": "failure", "status": "finished", "spec": {"task": "task"},
         "result": {"complete": True, "valid_for_success_rate": True, "native_success": False}},
        {"id": "interrupted", "status": "interrupted", "spec": {"task": "task"},
         "result": {"complete": False, "valid_for_success_rate": False}},
    ]
    (tmp_path / "state.json").write_text(json.dumps({"jobs": jobs}))
    folder = tmp_path / "running"
    folder.mkdir()
    (folder / "progress.json").write_text('{"decisions": 2, "native_step": 5}')
    config, report = monitor["snapshot"](tmp_path)
    assert config["model"] == "test"
    row = report["episodes"][1]
    assert row["progress"]["native_step"] == 5
    assert isinstance(row["progress_age_s"], float) and row["progress_age_s"] >= 0
    assert report["tasks"]["task"] == {
        "planned": 4, "finished": 2, "valid": 1, "successes": 0,
        "native_failures": 1, "invalid_finished": 1,
        "running": 1, "pending": 1, "success_rate_among_valid": 0.0,
        "invalid_finished_reasons": {"missing_result": 1},
        "mean_score_among_valid": None, "scored_valid": 0,
        "confirmed_success_fraction": 0.0, "all_attempts_finished": False,
        "all_cases_valid": False, "final_success_rate": None, "final_mean_score": None,
    }

def test_incomplete_and_missing_native_verdict_cannot_inflate_metrics(tmp_path):
    monitor = runpy.run_path(str(
        Path(__file__).resolve().parents[2] / "scripts/monitor_robodojo_campaign.py"
    ))
    (tmp_path / "freeze.json").write_text('{"config": {}}')
    results = [
        {"complete": True, "valid_for_success_rate": True,
         "native_success": True, "native_score": 1.0, "reason": "native_terminal"},
        {"complete": False, "valid_for_success_rate": True,
         "native_success": True, "native_score": 1.0, "reason": "runtime_error"},
        {"complete": True, "valid_for_success_rate": True,
         "native_success": None, "native_score": 0.5, "reason": "malformed_verdict"},
        {"complete": False, "valid_for_success_rate": False,
         "native_success": None, "native_score": None, "reason": "model_give_up"},
    ]
    jobs = [
        {"id": str(i), "status": "finished", "spec": {"task": "task"}, "result": r}
        for i, r in enumerate(results)
    ]
    (tmp_path / "state.json").write_text(json.dumps({"jobs": jobs}))
    _, report = monitor["snapshot"](tmp_path)
    task = report["tasks"]["task"]
    assert task["valid"] == task["successes"] == task["scored_valid"] == 1
    assert task["native_failures"] == 0
    assert task["invalid_finished"] == 3
    assert task["invalid_finished_reasons"] == {
        "runtime_error": 1, "malformed_verdict": 1, "model_give_up": 1,
    }
    assert task["mean_score_among_valid"] == 1.0
    assert task["confirmed_success_fraction"] == 0.25
    assert task["all_attempts_finished"] is True
    assert task["all_cases_valid"] is False
    assert task["final_success_rate"] is None
    assert task["final_mean_score"] is None


def test_final_metrics_require_full_coverage_and_real_finite_scores(tmp_path):
    monitor = runpy.run_path(str(
        Path(__file__).resolve().parents[2] / "scripts/monitor_robodojo_campaign.py"
    ))
    (tmp_path / "freeze.json").write_text('{"config": {}}')
    jobs = [
        {"id": str(i), "status": "finished", "spec": {"task": "task"},
         "result": {"complete": True, "valid_for_success_rate": True,
                    "native_success": success, "native_score": score}}
        for i, (success, score) in enumerate([(True, 1.0), (False, 0.2)])
    ]
    path = tmp_path / "state.json"
    path.write_text(json.dumps({"jobs": jobs}))
    _, report = monitor["snapshot"](tmp_path)
    task = report["tasks"]["task"]
    assert task["all_attempts_finished"] and task["all_cases_valid"]
    assert task["native_failures"] == 1
    assert task["final_success_rate"] == 0.5
    assert task["final_mean_score"] == 0.6
    for invalid_score in (None, True, float("nan"), float("inf")):
        jobs[1]["result"]["native_score"] = invalid_score
        path.write_text(json.dumps({"jobs": jobs}))
        _, report = monitor["snapshot"](tmp_path)
        task = report["tasks"]["task"]
        assert task["final_success_rate"] == 0.5
        assert task["scored_valid"] == 1
        assert task["mean_score_among_valid"] == 1.0
        assert task["final_mean_score"] is None
