import json
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import pytest

from gpt_policy import main
from gpt_policy.harness.config import AgentConfig
from gpt_policy.settings import settings_path


@pytest.fixture
def profile(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    agents = tmp_path / "agents"
    agents.mkdir()
    agent = agents / "codex.json"
    agent.write_text('{"model":"test","live_image_window":8}')
    config = tmp_path / "config.json"
    config.write_text(json.dumps({
        "extends": str(settings_path().parent / "examples/yam-local.json"),
        "agent": "codex", "agent_config_dir": ".",
        "camera_backend": "v4l2", "camera_controls": None,
        "runtime": {"record_dir": "run"}, "recording": {"state_hz": 0},
    }))
    return config, agent


@pytest.mark.parametrize("configured,flags,expected", [
    ({}, [], 8),
    ({"live_image_window": None}, [], None),
    ({"live_image_window": 16}, [], 16),
    ({"live_image_window": 8}, ["--no-live-window"], None),
    ({"live_image_window": None}, ["--live-window", "8"], 8),
    ({"live_image_window": 8}, ["--live-window", "3"], 3),
])
def test_check_uses_effective_window_without_opening_hardware_or_writing_profiles(
    profile, monkeypatch, capsys, configured, flags, expected,
):
    config, agent = profile
    agent.write_text(json.dumps({"model": "test", **configured}))
    before = [path.read_bytes() for path in profile]
    monkeypatch.setattr("sys.argv", ["gpt-policy", "--config", str(config), "--check", *flags])
    resources = [Mock() for _ in range(3)]
    for name, resource in zip(("CameraSet", "_open_yam", "create_agent"), resources):
        monkeypatch.setattr(main, name, resource)
    main.main()
    report = json.loads(capsys.readouterr().out)
    assert report["live_image_window"] == expected
    assert report["live_window_enabled"] is (expected is not None)
    assert report["hardware_opened"] is False
    for resource in resources:
        resource.assert_not_called()
    assert [path.read_bytes() for path in profile] == before


@pytest.mark.parametrize("check", [False, True])
@pytest.mark.parametrize("value", ["0", "2", "-1"])
def test_invalid_window_fails_before_hardware(profile, monkeypatch, check, value):
    config, _ = profile
    monkeypatch.setattr("sys.argv", ["gpt-policy", "--config", str(config), "--live-window", value,
                                    *(["--check"] if check else [])])
    resources = [Mock() for _ in range(3)]
    for name, resource in zip(("CameraSet", "_open_yam", "create_agent"), resources):
        monkeypatch.setattr(main, name, resource)
    with pytest.raises(ValueError, match="integer >= 3"):
        main.main()
    for resource in resources:
        resource.assert_not_called()


@pytest.mark.parametrize("flags", [
    ["--live-window", "8", "--no-live-window"],
    ["--live-window", "3.5"],
])
def test_invalid_cli_arguments_are_rejected(monkeypatch, flags):
    monkeypatch.setattr("sys.argv", ["gpt-policy", *flags])
    with pytest.raises(SystemExit) as error:
        main.parse_args()
    assert error.value.code == 2


@pytest.mark.parametrize("flags", [["--no-live-window"], ["--live-window", "8"]])
def test_override_is_rejected_for_non_codex(profile, monkeypatch, flags):
    config, _ = profile
    monkeypatch.setattr(main, "agent_config", lambda *_: AgentConfig(type="claude_code"))
    monkeypatch.setattr("sys.argv", ["gpt-policy", "--config", str(config), "--check", *flags])
    with pytest.raises(ValueError, match="only supported by Codex"):
        main.main()


@pytest.mark.parametrize("flags,expected", [([], 8), (["--no-live-window"], None), (["--live-window", "3"], 3)])
def test_run_passes_effective_window_to_session_and_records_it(profile, tmp_path, monkeypatch, flags, expected):
    config, _ = profile
    before = [path.read_bytes() for path in profile]
    request = tmp_path / "request.json"
    request.write_text('{"content":["test task"]}')
    monkeypatch.setattr("sys.argv", ["gpt-policy", "--config", str(config), "--input-json", str(request), *flags])
    monkeypatch.setattr(main, "preflight_agent", lambda *_: None)
    monkeypatch.setattr(main, "camera_defaults", lambda *_: [])
    monkeypatch.setattr(main, "warm_up_cameras", lambda *_: {})
    monkeypatch.setattr(main, "CameraSet", lambda *_: SimpleNamespace(describe=lambda *_: [], close=Mock()))
    monkeypatch.setattr(main, "RunVideo", lambda *_: SimpleNamespace(details={}, stop=lambda: {}))
    monkeypatch.setattr(main, "_open_yam", lambda *_: SimpleNamespace(dof=6, close=Mock()))
    factory = Mock(return_value=SimpleNamespace(start=Mock(), close=Mock()))
    monkeypatch.setattr(main, "create_agent", factory)
    monkeypatch.setattr(main, "run_loop", lambda *_, **__: "completed")
    monkeypatch.setattr(main, "RunConsole", lambda: MagicMock(task_result=Mock(return_value=None)))
    main.main()
    factory.assert_called_once()
    assert factory.call_args.args[0].live_image_window == expected
    root = tmp_path / "run_unreviewed"
    for filename in ("config.json", "status.json"):
        saved = json.loads((root / filename).read_text())
        assert saved["live_image_window"] == saved["agent"]["live_image_window"] == expected
        assert saved["live_window_enabled"] is (expected is not None)
    events = [json.loads(line) for line in (root / "events.jsonl").read_text().splitlines()]
    started = next(event for event in events if event["event"] == "run_started")
    assert started["live_image_window"] == expected
    assert [path.read_bytes() for path in profile] == before
