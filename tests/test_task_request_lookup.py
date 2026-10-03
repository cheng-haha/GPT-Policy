import json
from dataclasses import replace
from unittest.mock import Mock

import pytest

from gpt_policy import main
from gpt_policy.harness import task_name as naming
from gpt_policy.harness.config import AgentConfig
from gpt_policy.harness.errors import AgentProtocolError
from gpt_policy.input.manifest import VideoPart
from gpt_policy.input.request import RunInput, normalize_request, resolve_run_input


def request(path, instruction="拿起水果", content=None):
    path.write_text(json.dumps({"instruction": instruction, "content": content or [instruction]}))
    return path


def selector(monkeypatch, filename):
    agent = Mock()
    agent.decide.return_value = {"name": "name_recording", "arguments": {
        "task_name": "pick-up-fruit", "request_json": filename,
    }}
    factory = Mock(return_value=agent)
    monkeypatch.setattr(naming, "create_agent", factory)
    return agent, factory


def test_spark_matches_paraphrase_and_preserves_current_instruction(tmp_path, monkeypatch):
    path = request(tmp_path / "pick-up-fruit.json")
    run = normalize_request(RunInput("抓起水果", "robot-model"))
    agent, factory = selector(monkeypatch, path.name)
    name, source, selected = naming.select_task_request(run, tmp_path, AgentConfig())
    assert name == "pick-up-fruit" and source == path
    assert selected.instruction == run.instruction and selected.model == "robot-model"
    assert selected.content[0].text == "拿起水果"
    payload = json.loads(agent.decide.call_args.args[0].observation)
    assert payload["request"] == run.request()
    assert payload["candidates"][0]["file"] == path.name
    assert payload["candidates"][0]["instruction"] == "拿起水果"
    assert agent.decide.call_args.args[0].content is None
    assert factory.call_args.args[0].type == "codex"
    assert factory.call_args.args[0].model == "gpt-5.6-luna"
    assert factory.call_args.args[0].effort == "low"
    agent.decide.assert_called_once()
    agent.close.assert_called_once()
    assert "request_json" not in naming._CONTEXT.output_schema["properties"]["arguments"]["properties"]


def test_no_match_returns_new_name_from_the_same_model_call(tmp_path, monkeypatch):
    request(tmp_path / "other-task.json", "放下水果")
    run = normalize_request(RunInput("拿起水果", "robot-model"))
    agent, _ = selector(monkeypatch, None)
    assert naming.select_task_request(run, tmp_path, AgentConfig()) == ("pick-up-fruit", None, run)
    agent.decide.assert_called_once()
    assert list(tmp_path.glob("*.json")) == [tmp_path / "other-task.json"]


def test_explicit_media_and_modes_filter_incompatible_candidates(tmp_path, monkeypatch):
    source = tmp_path / "demo.json"
    source.write_text('{"keyframes": []}')
    other = tmp_path / "other.json"
    other.write_text('{"keyframes": []}')
    request(tmp_path / "right-request.json", content=[{"video": str(source), "mode": "video+action"}])
    request(tmp_path / "wrong-mode.json", content=[{"video": str(source), "mode": "video"}])
    request(tmp_path / "wrong-path.json", content=[{"video": str(other), "mode": "video+action"}])
    request(tmp_path / "unavailable.json", content=[{"image": "missing.jpg"}])
    run = normalize_request(RunInput("参考示范拿起水果", "robot-model", (VideoPart(source, mode="video+action"),)))
    agent, _ = selector(monkeypatch, "right-request.json")
    naming.select_task_request(run, tmp_path, AgentConfig(), "video+action")
    candidates = json.loads(agent.decide.call_args.args[0].observation)["candidates"]
    assert [c["file"] for c in candidates] == ["right-request.json"]


def test_unknown_selected_filename_is_rejected_before_any_use(tmp_path, monkeypatch):
    agent, _ = selector(monkeypatch, "../outside.json")
    with pytest.raises(AgentProtocolError):
        naming.select_task_request(normalize_request(RunInput("拿起水果", "test")), tmp_path, AgentConfig())
    agent.close.assert_called_once()


@pytest.mark.parametrize("entry", ["flag", "inline", "default", "prepare"])
def test_json_input_never_searches_names_or_saves_another_task(tmp_path, monkeypatch, entry):
    source = request(tmp_path / "pick-up-fruit.json")
    before = source.read_bytes()
    def forbidden(*_, **__):
        raise AssertionError("JSON must bypass task lookup and task-file creation")
    monkeypatch.setattr(main, "select_task_request", forbidden)
    monkeypatch.setattr(main, "save_task_request", forbidden)
    monkeypatch.setattr(main, "settings_path", lambda *_: tmp_path / "config.json")
    monkeypatch.setattr(main, "load_settings", lambda *_: {"runtime": {"input_json": str(source)}})
    monkeypatch.setattr(main, "agent_config", lambda *_: AgentConfig())
    monkeypatch.setattr(main, "preflight_agent", lambda *_: None)
    argv = ["gpt-policy"]
    argv += [f"参考 {source}，抓起水果"] if entry == "inline" else ["--input-json", str(source)] if entry != "default" else []
    if entry == "prepare":
        argv += ["--prepare-only", str(tmp_path / "output")]
    recorder = Mock(side_effect=InterruptedError("stop before hardware"))
    monkeypatch.setattr(main, "RunRecorder", recorder)
    monkeypatch.setattr("sys.argv", argv)
    if entry == "prepare":
        main.main()
        recorder.assert_not_called()
    else:
        with pytest.raises(InterruptedError, match="stop before hardware"):
            main.main()
        assert recorder.call_args.args[1]["request_json"] == str(source)
        assert recorder.call_args.args[1]["task_name"] == source.stem
    assert source.read_bytes() == before
    assert not main.REQUEST_DIRECTORY.exists()


def test_prompt_reuses_selected_json_with_one_lookup_and_no_new_file(tmp_path, monkeypatch):
    main.REQUEST_DIRECTORY.mkdir(exist_ok=True)
    source = request(main.REQUEST_DIRECTORY / "pick-up-fruit.json")
    before = source.read_bytes()
    selected = normalize_request(resolve_run_input(None, source, "robot-model"))
    lookup = Mock(side_effect=lambda run, *_: (source.stem, source, replace(selected, instruction=run.instruction)))
    monkeypatch.setattr(main, "select_task_request", lookup)
    monkeypatch.setattr(main, "settings_path", lambda *_: tmp_path / "config.json")
    monkeypatch.setattr(main, "load_settings", lambda *_: {})
    monkeypatch.setattr(main, "agent_config", lambda *_: AgentConfig(model="robot-model"))
    monkeypatch.setattr(main, "preflight_agent", lambda *_: None)
    recorder = Mock(side_effect=InterruptedError("stop before hardware"))
    monkeypatch.setattr(main, "RunRecorder", recorder)
    monkeypatch.setattr("sys.argv", ["gpt-policy", "抓起水果"])
    with pytest.raises(InterruptedError, match="stop before hardware"):
        main.main()
    assert recorder.call_args.args[1]["instruction"] == "抓起水果"
    assert recorder.call_args.args[1]["request_json"] == str(source)
    lookup.assert_called_once()
    assert list(main.REQUEST_DIRECTORY.iterdir()) == [source]
    assert source.read_bytes() == before
