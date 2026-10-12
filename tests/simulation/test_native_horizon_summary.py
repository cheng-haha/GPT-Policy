"""Native horizon statistics must not include achievements after the cutoff."""

import importlib.util
import json
from pathlib import Path

import pytest


@pytest.fixture
def summary():
    script = Path(__file__).resolve().parents[2] / "scripts/summarize_robodojo_native_horizon.py"
    spec = importlib.util.spec_from_file_location("native_horizon_summary_test", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_score_is_at_exact_cutoff_not_peak_or_final(summary, tmp_path):
    path = tmp_path / "scores.jsonl"
    rows = [
        {"native_control_step": step, "native_layout_valid": True,
         "native_complete": step == 4, "score_100": score}
        for step, score in enumerate([0, 40, 10, 50, 100])
    ]
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    score, row = summary.score_at_cutoff(path, 2, 4)
    assert score == 10
    assert row["native_control_step"] == 2


def test_missing_cutoff_evidence_is_not_interpolated(summary, tmp_path):
    path = tmp_path / "scores.jsonl"
    path.write_text(json.dumps({"native_control_step": 0}) + "\n")
    with pytest.raises(ValueError, match="not contiguous"):
        summary.score_at_cutoff(path, 2, 4)


@pytest.mark.parametrize("field,value", [
    ("native_layout_valid", False), ("native_complete", True), ("score_100", None)])
def test_invalid_or_terminal_prefix_cannot_be_scored_as_ordinary_cutoff(
        summary, tmp_path, field, value):
    path = tmp_path / "scores.jsonl"
    rows = [{"native_control_step": i, "native_layout_valid": True,
             "native_complete": False, "score_100": 10} for i in range(3)]
    rows[2][field] = value
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    with pytest.raises(ValueError):
        summary.score_at_cutoff(path, 2, 2)
