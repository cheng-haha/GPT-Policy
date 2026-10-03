"""Examples and deployed profiles must resolve in a standalone checkout."""
import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from gpt_policy import main


@pytest.mark.parametrize("profile", [
    "default.json", "examples/arx-local.json", "examples/yam-local.json",
    "machines/yambox.json", "machines/arx247.json", "machines/arx248.json",
])
@pytest.mark.parametrize("flags,expected", [([], None), (["--no-live-window"], None), (["--live-window", "8"], 8)])
def test_public_profile_check_is_hardware_free(profile, flags, expected, monkeypatch, capsys):
    root = Path(__file__).resolve().parents[1]
    target = (["--machine", Path(profile).stem] if profile.startswith("machines/")
              else ["--config", str(root / "configs" / profile)])
    monkeypatch.setattr("sys.argv", ["gpt-policy", *target, "--check", *flags])
    resources = [Mock(side_effect=AssertionError("configuration check opened a resource")) for _ in range(5)]
    for name, resource in zip(("CameraSet", "_open_yam", "ArxRobot", "BimanualRobot", "create_agent"), resources):
        monkeypatch.setattr(main, name, resource)
    main.main()
    report = json.loads(capsys.readouterr().out)
    assert report["ok"] and not report["hardware_opened"]
    assert report["live_image_window"] == expected
    assert report["live_window_enabled"] is (expected is not None)
    if profile.startswith("machines/"):
        assert report["machine"] == Path(profile).stem
        assert report["top_camera_bases"] == ["left", "right"]
    for resource in resources:
        resource.assert_not_called()


@pytest.mark.parametrize("machine", ["yambox", "arx247", "arx248"])
def test_deployed_profile_input_dependency_is_available(machine):
    from gpt_policy.input import resolve_run_input
    from gpt_policy.settings import load_settings, settings_path
    settings = load_settings(settings_path(machine=machine))
    if path := settings["runtime"]["input_json"]:
        assert Path(path).is_file()
        assert resolve_run_input(None, Path(path), None).instruction
