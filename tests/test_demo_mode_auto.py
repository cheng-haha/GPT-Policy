"""Regression coverage for automatic modes shared by CLI and JSON requests."""

import json

import pytest

from gpt_policy.input.manifest import VideoPart
from gpt_policy.input.references import instruction_mode
from gpt_policy.input.request import RunInput, normalize_request, resolve_run_input, save_request


@pytest.mark.parametrize("frame,expected", [
    ({"state": {"left": {"gripper_measured": 0}}}, "video+action"),
    ({"action": {"samples": [{"t_s": 0, "left": {"gripper_command": 0}}]}}, "video+action"),
    ({"state": {}, "action": None}, "video"),
    ({"stage": "Use robot states and action trajectories"}, "video"),
])
@pytest.mark.parametrize("entry", ["demo", "directory", "json", "json-auto"])
def test_auto_uses_data_instead_of_names_and_saves_resolved_mode(tmp_path, frame, expected, entry):
    source = tmp_path / "video+action-demo.json"
    source.write_text(json.dumps({"title": "Video and action trajectories", "keyframes": [frame]}))
    if entry == "directory":
        source.rename(tmp_path / "demo.json")
        run = normalize_request(RunInput("task", "model"), tmp_path)
    elif entry.startswith("json"):
        item = {"video": source.name}
        if entry == "json-auto":
            item["mode"] = "auto"
        request = tmp_path / "input.json"
        request.write_text(json.dumps({"instruction": "task", "content": [item]}))
        run = normalize_request(resolve_run_input(None, request, "model"))
    else:
        run = normalize_request(RunInput("task", "model"), source)
    video = next(p for p in run.content if isinstance(p, VideoPart))
    assert video.mode == expected
    saved = save_request(run, tmp_path / "request.json")
    assert normalize_request(resolve_run_input(None, saved, None)).request() == run.request()


def test_automatic_request_serialization_does_not_turn_into_video_override(tmp_path):
    source = tmp_path / "demo.json"
    source.write_text('{"keyframes": [{"state": {"gripper": 0}}]}')
    run = RunInput("task", "model", (VideoPart(source),))
    saved = save_request(run, tmp_path / "request.json")
    assert normalize_request(resolve_run_input(None, saved, None)).content[0].mode == "video+action"
    explicit = normalize_request(run, mode="video")
    assert explicit.content[0].mode == "video"
    assert normalize_request(explicit, mode="auto").content[0].mode == "video+action"


def test_recording_checks_state_rows_and_raw_video_cannot_inherit_sidecars(tmp_path):
    source = tmp_path / "top.mp4"
    source.touch()
    states = tmp_path / "states.jsonl"
    states.write_text('\n{"state": {}}\n')
    assert normalize_request(RunInput("task", "model"), tmp_path).content[-1].mode == "video"
    states.write_text('{"state": {"gripper_normalized": 0}}\n')
    assert normalize_request(RunInput("task", "model"), tmp_path).content[-1].mode == "video+action"
    assert normalize_request(RunInput("task", "model"), source).content[-1].mode == "video"
    with pytest.raises(ValueError, match="no action data"):
        normalize_request(RunInput("task", "model"), source, "video+action")


def test_invalid_demo_is_not_silently_downgraded(tmp_path):
    source = tmp_path / "demo.json"
    for data in ("broken", '[]', '{"keyframes": [null]}', '{}'):
        source.write_text(data)
        with pytest.raises(ValueError):
            normalize_request(RunInput("task", "model"), source)


@pytest.mark.parametrize("text,expected", [
    ("Use video demonstration keyframes and corresponding robot states.", None),
    ("Use /tmp/video+action/demo.json as a reference.", None),
    ("使用 video 模式", "video"),
    ("Use video mode.", "video"),
    ("mode: video", "video"),
    ("Use video only.", "video"),
    ("Use video + action mode.", "video+action"),
])
def test_generic_video_description_does_not_disable_actions(text, expected):
    assert instruction_mode(text) == expected
