from types import SimpleNamespace

import pytest

from gpt_policy.hardware import realsense


class Sensor:
    def __init__(self, *, color=True):
        self.color = color
        self.values = {"auto_exposure": 1.0, "exposure_us": 33000.0, "gain": 24.0,
                       "auto_white_balance": 1.0, "white_balance": 4600.0}
        self.writes = []
        self.ignore = None

    def get_stream_profiles(self):
        return [SimpleNamespace(stream_type=lambda: "color" if self.color else "depth")]

    def supports(self, option):
        return option in self.values

    def is_option_read_only(self, option):
        return False

    def get_option_range(self, option):
        low, high, step = {"auto_exposure": (0, 1, 1), "exposure_us": (1, 165000, 1),
                           "gain": (16, 248, 1)}[option]
        return SimpleNamespace(min=low, max=high, step=step)

    def get_option(self, option):
        return self.values[option]

    def set_option(self, option, value):
        self.writes.append((option, value))
        if option != self.ignore:
            self.values[option] = float(value)


@pytest.fixture
def cameras(monkeypatch):
    cameras = realsense.RealSenseCameraSet.__new__(realsense.RealSenseCameraSet)
    cameras.rs = SimpleNamespace(
        option=SimpleNamespace(enable_auto_exposure="auto_exposure", exposure="exposure_us", gain="gain"),
        stream=SimpleNamespace(color="color"),
    )
    cameras.cameras = []
    for name in ("left", "right", "top"):
        # A depth-only sensor also supports exposure; it must not be modified.
        sensors = [Sensor(color=False), Sensor()]
        device = SimpleNamespace(query_sensors=lambda sensors=sensors: sensors)
        profile = SimpleNamespace(get_device=lambda device=device: device)
        pipeline = SimpleNamespace(get_active_profile=lambda profile=profile: profile)
        cameras.cameras.append(SimpleNamespace(name=name, serial=name + "-serial", pipeline=pipeline))
    clock = [0.0]
    monkeypatch.setattr(realsense, "time", SimpleNamespace(
        monotonic=lambda: clock[0], sleep=lambda seconds: clock.__setitem__(0, clock[0] + seconds),
    ))
    return cameras


def sensors(cameras):
    return [c.pipeline.get_active_profile().get_device().query_sensors() for c in cameras.cameras]


def test_each_task_restores_all_rgb_controls_and_preserves_white_balance(cameras):
    expected = {"auto_exposure": 0, "exposure_us": 12000, "gain": 16}
    for run in range(3):
        if run == 2:
            # Replugging or another application changes just the right camera.
            sensors(cameras)[1][1].values.update(auto_exposure=1, exposure_us=33000, gain=32)
        report = cameras.configure_controls(exposure_us=12000, gain=16)
        assert [entry["name"] for entry in report] == ["left", "right", "top"]
        assert report[1]["device"] == "right-serial"
        assert report[1]["before"]["auto_exposure"] == (0 if run == 1 else 1)
        for entry, (depth, rgb) in zip(report, sensors(cameras)):
            assert entry["after"] == expected
            assert not depth.writes
            assert rgb.writes == list(expected.items()) * (run + 1)
            assert rgb.values["auto_white_balance"] == 1
            assert rgb.values["white_balance"] == 4600


@pytest.mark.parametrize("exposure,gain", [(0, 16), (True, 16), (12000.5, 16),
                                         (12000, -1), (12000, True), (200000, 16), (12000, 1)])
def test_invalid_controls_fail_before_any_camera_is_changed(cameras, exposure, gain):
    with pytest.raises(ValueError):
        cameras.configure_controls(exposure_us=exposure, gain=gain)
    assert all(not sensor.writes for pair in sensors(cameras) for sensor in pair)


@pytest.mark.parametrize("problem", ["unsupported", "read_only", "range", "no_rgb"])
def test_invalid_last_sensor_does_not_partially_configure_other_cameras(cameras, problem):
    last = sensors(cameras)[-1][1]
    if problem == "unsupported":
        last.values.pop("gain")
    elif problem == "read_only":
        last.is_option_read_only = lambda option: option == "exposure_us"
    elif problem == "range":
        last.get_option_range = lambda _: SimpleNamespace(min=0, max=10, step=1)
    else:
        last.color = False
    with pytest.raises((ValueError, RuntimeError), match="top"):
        cameras.configure_controls(exposure_us=12000, gain=16)
    assert all(not sensor.writes for pair in sensors(cameras) for sensor in pair)


def test_ignored_exposure_write_aborts_with_camera_and_actual_value(cameras):
    sensors(cameras)[1][1].ignore = "exposure_us"
    with pytest.raises(RuntimeError, match="right: RealSense controls did not apply.*33000"):
        cameras.configure_controls(exposure_us=12000, gain=16)
    assert not sensors(cameras)[-1][1].writes


def test_readback_allows_pending_frame_boundary(cameras):
    right = sensors(cameras)[1][1]
    read = right.get_option
    pending = [2]

    def delayed(option):
        if option == "exposure_us" and right.writes and pending[0]:
            pending[0] -= 1
            return 33000.0
        return read(option)

    right.get_option = delayed
    report = cameras.configure_controls(exposure_us=12000, gain=16)
    assert report[1]["after"]["exposure_us"] == 12000
    assert pending == [0]


def test_empty_camera_set_rejects_controls(cameras):
    cameras.cameras.clear()
    with pytest.raises(ValueError, match="explicitly named"):
        cameras.configure_controls(exposure_us=12000, gain=16)
