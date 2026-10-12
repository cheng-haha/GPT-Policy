import json
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
from configure_robodojo_release import configure
from robodojo_release_evidence import (
    execution_totals,
    native_report,
    portable_context,
    public_executions,
    public_usage,
    usage_totals,
)


def completed_call():
    return {"call": 1, "phase": "decision", "model": "example",
            "status": "completed", "elapsed_s": 1.2, "usage_final": True,
            "usage": {"input_tokens": 100, "cached_input_tokens": 80,
                      "output_tokens": 10, "reasoning_output_tokens": 3,
                      "total_tokens": 110},
            "thread_id": "private", "turn_id": "private", "raw_usage": {"private": 1}}


def test_usage_does_not_double_count_cache_or_reasoning_and_preserves_missing():
    first = public_usage(completed_call())
    failed = {**first, "call": 2, "status": "failed", "usage": None}
    totals = usage_totals([first, failed])
    assert totals["total_tokens"] == 110
    assert totals["uncached_input_tokens"] == 20
    assert totals["reasoning_output_tokens"] == 3
    assert totals["usage_missing_calls"] == 1
    assert totals["token_totals_complete"] is False
    assert "thread_id" not in first and "raw_usage" not in first


def test_inconsistent_token_totals_are_rejected():
    call = public_usage(completed_call())
    call["usage"]["total_tokens"] += 3
    with pytest.raises(ValueError, match="Token total"):
        usage_totals([call])


def test_executed_segment_is_a_nonzero_chunk_not_a_decision_or_waypoint():
    events = [
        {"event": "observation", "private": "unused"},
        {"event": "execution_result", "step": 0, "name": "locate_point", "result": {}},
        {"event": "execution_result", "step": 1, "name": "move_eef_chunk",
         "result": {"start_step": 0, "end_step": 0, "acknowledged_controls": 0}},
        {"event": "execution_result", "step": 2, "name": "move_eef_chunk",
         "result": {"start_step": 0, "end_step": 3, "acknowledged_controls": 3,
                    "rejection": "partial chunk"}},
    ]
    assert execution_totals(public_executions(events)) == {
        "decisions": 3, "control_steps": 3, "executed_action_segments": 1,
        "zero_control_motion_calls": 1, "non_motion_tool_calls": 1,
        "physical_duration_s": .12,
    }


def test_portable_context_preserves_evidence_and_only_changes_locations():
    original = {
        "instruction": "Test", "content": [
            "Keep this text exactly.",
            json.dumps({"source": {"dataset": "/mnt/data/demo/task",
                                    "episode_index": 37},
                        "states": [0.15, -2, 5]}),
            {"image": "/mnt/data/demo/prepared/keyframes/00.jpg", "label": "left", "detail": "high"},
        ],
    }
    output = portable_context(original)
    assert output["content"][0] == original["content"][0]
    assert json.loads(output["content"][1]) == {
        "source": {"dataset": "task", "episode_index": 37}, "states": [0.15, -2, 5]}
    assert output["content"][2] == {"image": "keyframes/00.jpg", "label": "left", "detail": "high"}
    assert original["content"][2]["image"].startswith("/mnt/")


def test_native_panel_rejects_implicit_or_incomplete_selection():
    with pytest.raises(ValueError, match="50 explicit"):
        native_report({"episodes": []})


def test_configurator_verifies_download_before_runtime(tmp_path):
    import hashlib
    protocol = tmp_path / "protocol"
    protocol.mkdir()
    f = tmp_path / "input.json"
    f.write_text('{"instruction":"x","content":["x"]}')
    ctx = {"x": {"input": "input.json", "images": [],
                 "portable_input_sha256": hashlib.sha256(f.read_bytes()).hexdigest()}}
    (protocol / "context_bundle.json").write_text(json.dumps(ctx))
    config = {"prepared_inputs": {"x": "input.json"}, "asset_audits": {},
              "control_step_limit": "native", "live_image_window": None}
    (protocol / "reproduction_config.json").write_text(json.dumps(config))
    out = configure(tmp_path, tmp_path / "RoboDojo", sys.executable, sys.executable, tmp_path / "out")
    assert out["prepared_inputs"]["x"] == str(f)
    f.write_text("changed")
    with pytest.raises(ValueError, match="checksum"):
        configure(tmp_path, tmp_path / "RoboDojo", sys.executable, sys.executable, tmp_path / "out")
