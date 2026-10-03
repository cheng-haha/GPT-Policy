"""Public checkout examples must resolve without private machine files."""
import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from gpt_policy import main


@pytest.mark.parametrize("profile", ["default.json", "examples/arx-local.json", "examples/yam-local.json"])
@pytest.mark.parametrize("flags,expected", [([], None), (["--no-live-window"], None), (["--live-window", "8"], 8)])
def test_public_profile_check_is_hardware_free(profile, flags, expected, monkeypatch, capsys):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.setattr("sys.argv", ["gpt-policy", "--config", str(root / "configs" / profile), "--check", *flags])
    resources = [Mock(side_effect=AssertionError("configuration check opened a resource")) for _ in range(5)]
    for name, resource in zip(("CameraSet", "_open_yam", "ArxRobot", "BimanualRobot", "create_agent"), resources):
        monkeypatch.setattr(main, name, resource)
    main.main()
    report = json.loads(capsys.readouterr().out)
    assert report["ok"] and not report["hardware_opened"]
    assert report["live_image_window"] == expected
    assert report["live_window_enabled"] is (expected is not None)
    for resource in resources:
        resource.assert_not_called()
