import json
from pathlib import Path

import pytest

from gpt_policy import main
from gpt_policy.input.manifest import ImagePart, VideoPart, load_manifest
from gpt_policy.input.references import instruction_mode
from gpt_policy.input.request import normalize_request, resolve_run_input, RunInput, save_task_request
from test_demonstration import reviewed


@pytest.mark.parametrize("text,expected", [
    ("使用video模式，拧开瓶盖", "video"),
    ("使用 video+action 模式，拧开瓶盖", "video+action"),
    ("模式：VIDEO + ACTION。", "video+action"),
    ("参考 /tmp/video+action/demo.json，使用 video 模式", "video"),
    ("参考 './video + action/demo.json'，拿起水果", None),
    ("参考 'my video demo.json'，拿起水果", None),
    ("参考 https://example.com/video+action.mp4", None),
    ("使用 video_action 模式", None),
])
def test_mode_words_are_explicit_and_do_not_come_from_paths(text, expected):
    assert instruction_mode(text) == expected


def test_conflicting_mode_words_are_rejected():
    with pytest.raises(ValueError, match="Conflicting"):
        instruction_mode("video 模式和 video+action 模式")


@pytest.fixture
def offline_cli(monkeypatch):
    def forbidden(*_, **__):
        raise AssertionError("This input-only test must not call models or open hardware")
    for name in ("preflight_agent", "select_task_request", "CameraSet", "ArxRobot", "_open_yam"):
        monkeypatch.setattr(main, name, forbidden)


@pytest.mark.parametrize("json_mode,prompt_mode,flag_mode,expected", [
    (None, None, None, "video"),
    ("video+action", None, None, "video+action"),
    ("video+action", "video", None, "video"),
    ("video", "video+action", None, "video+action"),
    ("video+action", "video+action", "video", "video"),
    ("video", "video and video+action", "video+action", "video+action"),
])
def test_mode_priority_and_replayable_archive(tmp_path, monkeypatch, offline_cli, capsys,
                                             json_mode, prompt_mode, flag_mode, expected):
    source = reviewed(tmp_path, 2)
    request = tmp_path / "request.json"
    item = {"video": "source/demo.json"}
    if json_mode:
        item["mode"] = json_mode
    request.write_text(json.dumps({"instruction": "拧开瓶盖", "content": ["看左臂持瓶", item, "再拧瓶盖"]}))
    output = tmp_path / "output"
    argv = ["gpt-policy"] + ([f"使用 {prompt_mode} 模式拧开瓶盖"] if prompt_mode else [])
    argv += ["--input-json", str(request), "--prepare-only", str(output)]
    if flag_mode:
        argv += ["--demo-mode", flag_mode]
    monkeypatch.setattr("sys.argv", argv)
    main.main()
    console = " ".join(capsys.readouterr().out.split())
    assert f"Demonstration ready ({expected}): 2 keyframes" in console
    assert "state frames" in console and "action samples" in console
    assert ("video mode omitted available demonstration states/actions" in console) == (expected == "video")
    archives = list(main.REQUEST_DIRECTORY.glob("*.json"))
    assert archives == []  # Explicit JSON is used directly, even with CLI overrides.
    raw = json.loads((output / "request.json").read_text())
    assert raw["content"][1] == {"video": str(source), "mode": expected}
    replay = normalize_request(resolve_run_input(None, output / "request.json", None))
    assert replay.request() == raw
    manifest = load_manifest(output / "input.json")
    assert manifest.content[0].text == "看左臂持瓶"
    assert manifest.content[-1].text == "再拧瓶盖"
    assert len([p for p in manifest.content if isinstance(p, ImagePart)]) == 2
    assert (output / "input-videos/video-000/actions.jsonl").exists() == (expected == "video+action")


@pytest.mark.parametrize("mode", [None, "video+action"])
def test_raw_prompt_reports_actual_numeric_context_without_starting_hardware(
    tmp_path, monkeypatch, offline_cli, capsys, mode,
):
    source = reviewed(tmp_path, 2)
    instruction = (
        f"Use the keyframes and corresponding robot states, end-effector poses, and action trajectories in {source} "
        "as a reference. Unscrew and remove the bottle cap."
    )
    output = tmp_path / "prepared"
    argv = ["gpt-policy", instruction, "--prepare-only", str(output)]
    if mode:
        argv += ["--demo-mode", mode]
    monkeypatch.setattr("sys.argv", argv)
    main.main()

    metadata = json.loads((output / "input-videos/video-000/provenance.json").read_text())
    console = " ".join(capsys.readouterr().out.split())
    if mode is None:
        assert metadata["mode"] == "video"
        assert metadata["numeric_data_omitted"] is True
        assert "0 state frames, 0 action samples" in console
        assert "Use --demo-mode video+action to include them" in console
    else:
        assert metadata["mode"] == "video+action"
        assert metadata["state_keyframes"] == 2
        assert metadata["output_action_samples"] > 0
        assert metadata["numeric_data_omitted"] is False
        assert "WARNING:" not in console


def test_prompt_mode_and_json_mode_have_identical_inputs(tmp_path, monkeypatch, offline_cli):
    source = reviewed(tmp_path, 2)
    instruction = f"参考 {source}，用 video+action 模式拧开瓶盖"
    output = tmp_path / "prompt"
    monkeypatch.setattr("sys.argv", ["gpt-policy", instruction, "--prepare-only", str(output)])
    main.main()
    raw = json.loads((output / "request.json").read_text())
    assert raw["instruction"] == instruction
    assert raw["content"] == [instruction, {"video": str(source), "mode": "video+action"}]
    replay = tmp_path / "replay"
    monkeypatch.setattr("sys.argv", ["gpt-policy", "--input-json", str(output / "request.json"),
                                    "--prepare-only", str(replay)])
    main.main()
    assert json.loads((replay / "request.json").read_text()) == raw
    assert (replay / "input-videos/video-000/actions.jsonl").exists()


def test_plain_prompt_and_instruction_only_json_both_get_standard_archives(tmp_path, monkeypatch, offline_cli):
    request = tmp_path / "plain.json"
    request.write_text('{"instruction": "拿起水果"}')
    for name, arguments in [("prompt", ["拿起水果"]), ("json", ["--input-json", str(request)])]:
        output = tmp_path / name
        monkeypatch.setattr("sys.argv", ["gpt-policy", *arguments, "--prepare-only", str(output)])
        main.main()
        raw = json.loads((output / "request.json").read_text())
        assert raw["instruction"] == "拿起水果"
        assert raw["content"] == ["拿起水果"]
        assert raw == json.loads((output / "input.json").read_text())
    assert len(list(main.REQUEST_DIRECTORY.glob("*.json"))) == 1


def test_task_requests_are_flat_and_reuse_equal_json_without_rewriting(tmp_path):
    run = normalize_request(RunInput("拿起水果", "test"))
    request = save_task_request(run, tmp_path, "pick-up-fruit")
    assert request == tmp_path / "pick-up-fruit.json"
    original = request.stat().st_mtime_ns
    assert save_task_request(run, tmp_path, "pick-up-fruit") == request
    replay = normalize_request(resolve_run_input(None, request, None))
    assert save_task_request(replay, tmp_path, "pick-up-fruit") == request
    assert request.stat().st_mtime_ns == original
    assert list(tmp_path.iterdir()) == [request]


def test_same_name_different_request_preserves_both_and_reuses_variant(tmp_path):
    first = normalize_request(RunInput("拿起水果", "test"))
    second = normalize_request(RunInput("拿起红色水果", "test"))
    original = save_task_request(first, tmp_path, "pick-up-fruit")
    before = original.read_bytes()
    variant = save_task_request(second, tmp_path, "pick-up-fruit")
    assert variant != original
    assert variant.stem.startswith("pick-up-fruit-")
    assert original.read_bytes() == before
    assert save_task_request(second, tmp_path, "pick-up-fruit") == variant
    assert len(list(tmp_path.glob("*.json"))) == 2


def test_equivalent_existing_handwritten_json_is_reused(tmp_path):
    request = tmp_path / "pick-up-fruit.json"
    request.write_text('{"instruction": "拿起水果"}')
    before = request.read_bytes()
    run = normalize_request(RunInput("拿起水果", "test"))
    assert save_task_request(run, tmp_path, "pick-up-fruit") == request
    assert request.read_bytes() == before


def test_multiple_video_blocks_keep_individual_modes_and_json_relative_paths(tmp_path):
    source = reviewed(tmp_path, 2)
    request = tmp_path / "request.json"
    request.write_text(json.dumps({"instruction": "拧开瓶盖", "content": [
        {"video": "source", "mode": "video"},
        {"video": "source/demo.json", "mode": "video+action"},
    ]}))
    run = normalize_request(resolve_run_input(None, request, None))
    assert [p.path for p in run.content] == [source.parent, source]
    assert [p.mode for p in run.content] == ["video", "video+action"]


def test_video_alone_cannot_supply_actions(tmp_path, monkeypatch, offline_cli):
    source = tmp_path / "clip.mp4"
    source.touch()
    monkeypatch.setattr("sys.argv", ["gpt-policy", f"参考 {source}，用 video+action 拧瓶盖"])
    with pytest.raises(ValueError, match="no action data"):
        main.main()
    assert not main.REQUEST_DIRECTORY.exists()


@pytest.mark.parametrize("mode", ["unknown", "", None, [], True])
def test_invalid_json_modes_are_rejected(tmp_path, mode):
    video = tmp_path / "video.mp4"
    video.touch()
    request = tmp_path / "request.json"
    request.write_text(json.dumps({"instruction": "task", "content": [{"video": "video.mp4", "mode": mode}]}))
    with pytest.raises(ValueError, match="mode must be"):
        load_manifest(request)
