import json
from io import BytesIO
from types import SimpleNamespace
import time
from unittest.mock import Mock

import numpy as np
from PIL import Image
import pytest

from gpt_policy.hardware import camera_warmup as warmup
from gpt_policy.hardware.camera import CapturedImage
from gpt_policy.recording.video import RunVideo


class Cameras:
    def __init__(self, colors, *, encoded=False):
        self.cameras = [SimpleNamespace(name=name, width=16, height=16) for name in colors]
        self.colors = colors
        self.encoded = encoded
        self.now = 0.0
        self.calls = 0

    def capture(self, stop):
        if stop.is_set():
            raise InterruptedError("stopped")
        self.calls += 1
        self.now = round(self.now + 0.1, 6)
        images = {}
        for camera in self.cameras:
            color = self.colors[camera.name](self.now)
            rgb = bytes(color) * 256
            data = b""
            if self.encoded:
                stream = BytesIO()
                Image.frombytes("RGB", (16, 16), rgb).save(stream, "JPEG")
                data = stream.getvalue()
            images[camera.name] = CapturedImage(
                camera.name, data, "image/jpeg", 16, 16, time.time(),
                None if self.encoded else rgb,
            )
        return images


@pytest.mark.parametrize("encoded", [False, True])
def test_waits_for_slowest_camera_and_discards_overexposure_and_blue_cast(monkeypatch, tmp_path, encoded):
    def startup(t, ready):
        if t < 0.5:
            return (255, 255, 255)
        fraction = min(1.0, (t - 0.5) / (ready - 0.5))
        return tuple(round(value + (128 - value) * fraction) for value in (75, 130, 210))

    cameras = Cameras({
        "left": lambda t: startup(t, 1.5),
        "right": lambda t: startup(t, 4.0),
        "top": lambda t: startup(t, 2.0),
    }, encoded=encoded)
    monkeypatch.setattr(warmup, "monotonic", lambda: cameras.now)
    result = warmup.warm_up_cameras(cameras)
    assert 4.7 <= result["elapsed_s"] < 5.2
    assert result["discarded_frames_per_camera"] == cameras.calls
    # Use the same streams for recording and the first model observation.
    video = RunVideo(cameras, tmp_path)
    try:
        for frame in video.snapshot(timeout=1).values():
            np.testing.assert_allclose(warmup._frame_levels(frame)[:3], 128 / 255, atol=0.01)
    finally:
        video.stop()
    for writer in video.writers.values():
        data = writer.path.read_bytes()
        assert writer.sizes
        with Image.open(BytesIO(data[writer.offsets[0]:writer.offsets[0] + writer.sizes[0]])) as first:
            np.testing.assert_allclose(first.getpixel((8, 8)), (128, 128, 128), atol=2)


def test_stable_colored_scene_is_not_forced_to_neutral_white(monkeypatch):
    cameras = Cameras({"left": lambda _: (220, 20, 40)})
    monkeypatch.setattr(warmup, "monotonic", lambda: cameras.now)
    assert warmup.warm_up_cameras(cameras)["elapsed_s"] == 3.0


def test_window_catches_slow_drift_that_adjacent_frames_would_miss(monkeypatch):
    cameras = Cameras({"left": lambda t: (int(60 + min(t, 5) * 20), 140, 150)})
    monkeypatch.setattr(warmup, "monotonic", lambda: cameras.now)
    assert warmup.warm_up_cameras(cameras)["elapsed_s"] >= 5.7


def test_timeout_identifies_unstable_camera(monkeypatch):
    cameras = Cameras({"left": lambda _: (128, 128, 128), "right": lambda t: (0, 0, 255) if int(t * 10) % 2 else (255, 0, 0)})
    monkeypatch.setattr(warmup, "monotonic", lambda: cameras.now)
    with pytest.raises(TimeoutError, match="right"):
        warmup.warm_up_cameras(cameras)
    assert cameras.now == 15.0


def test_deadline_interrupts_capture_even_when_no_frames_arrive():
    def blocked_capture(stop):
        assert stop.wait(1.0)
        raise InterruptedError("capture stopped")
    cameras = SimpleNamespace(cameras=[SimpleNamespace(name="left")], capture=blocked_capture)
    with pytest.raises(TimeoutError, match="left"):
        warmup.warm_up_cameras(cameras, minimum_s=0.02, stable_s=0.01, timeout_s=0.05)


@pytest.mark.parametrize("error", [KeyboardInterrupt, RuntimeError, InterruptedError])
def test_capture_errors_propagate_and_cancel_deadline(monkeypatch, error):
    timers = []
    make_timer = warmup.threading.Timer
    def timer(*args):
        timers.append(make_timer(*args))
        return timers[-1]
    monkeypatch.setattr(warmup.threading, "Timer", timer)
    cameras = SimpleNamespace(cameras=[SimpleNamespace(name="left")], capture=Mock(side_effect=error))
    with pytest.raises(error):
        warmup.warm_up_cameras(cameras)
    assert not timers[0].is_alive()


@pytest.mark.parametrize("machine", ["arx-local", "yam-local"])
@pytest.mark.parametrize("error,suffix", [(TimeoutError, "failed"), (RuntimeError, "failed"),
                                         (KeyboardInterrupt, "interrupted")])
@pytest.mark.parametrize("phase", ["controls", "warmup"])
def test_camera_startup_failure_closes_cameras_without_starting_video_or_robot(tmp_path, monkeypatch, machine, error, suffix, phase):
    import gpt_policy.main as app
    import gpt_policy.hardware.d405_controls as d405
    import gpt_policy.hardware.realsense as realsense
    from gpt_policy.settings import load_settings, settings_path
    settings = load_settings(settings_path().parent / "examples" / f"{machine}.json")
    settings["runtime"]["record_dir"] = str(tmp_path / "run")
    cameras = SimpleNamespace(describe=lambda: [{"name": "left"}], close=Mock(),
                              configure_controls=Mock(return_value=[]))
    monkeypatch.setattr("sys.argv", ["gpt-policy", "拿起水果", "--machine", machine])
    monkeypatch.setattr(app, "load_settings", lambda *_: settings)
    monkeypatch.setattr(app, "preflight_agent", lambda *_: None)
    monkeypatch.setattr(app, "select_task_request", lambda run, *_: ("pick-up-fruit", None, run))
    monkeypatch.setattr(app, "CameraSet", lambda *_: cameras)
    monkeypatch.setattr(realsense, "RealSenseCameraSet", lambda *_: cameras)
    apply_controls = Mock(return_value=[])
    monkeypatch.setattr(d405, "configure_d405_cameras", apply_controls)
    selected_controls = cameras.configure_controls if machine == "yam-local" else apply_controls
    if phase == "controls":
        selected_controls.side_effect = error("controls stopped")
    monkeypatch.setattr(app, "warm_up_cameras", Mock(side_effect=error("warmup stopped")))
    video, robot, agent = Mock(), Mock(), Mock()
    monkeypatch.setattr(app, "RunVideo", video)
    for factory in ("BimanualRobot", "ArxRobot", "_open_yam"):
        monkeypatch.setattr(app, factory, robot)
    monkeypatch.setattr(app, "create_agent", agent)
    if error is not KeyboardInterrupt:
        with pytest.raises(error, match=f"{phase} stopped"):
            app.main()
    else:
        app.main()
    cameras.close.assert_called_once()
    video.assert_not_called()
    robot.assert_not_called()
    agent.assert_not_called()
    root = tmp_path / f"run_{suffix}"
    assert json.loads((root / "status.json").read_text())["outcome"] == suffix
    events = [json.loads(line)["event"] for line in (root / "events.jsonl").read_text().splitlines()]
    assert ("camera_warmup_started" in events) is (phase == "warmup")
    assert "camera_warmup_completed" not in events
    assert selected_controls.call_args.kwargs == {"exposure_us": 12000, "gain": 16}
    assert selected_controls.call_count == 1
    if phase == "warmup":
        assert events.index("camera_controls_applied") < events.index("camera_warmup_started")
    else:
        assert "camera_controls_applied" not in events
        app.warm_up_cameras.assert_not_called()
    if machine != "yam-local":
        cameras.configure_controls.assert_not_called()
    else:
        apply_controls.assert_not_called()
