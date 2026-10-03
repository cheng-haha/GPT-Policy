import struct
import errno

import pytest

from gpt_policy.hardware import d405_controls as controls
from gpt_policy.settings import load_settings, settings_path


@pytest.fixture
def device(monkeypatch):
    state = {"auto_exposure": 1, "exposure_us": 8000, "gain": 24}
    writes, closed = [], []
    monkeypatch.setattr(controls, "_require_d405", lambda _: None)
    monkeypatch.setattr(controls.os, "open", lambda *_: 11)
    monkeypatch.setattr(controls.os, "close", closed.append)

    def ioctl(fd, request, data, mutate):
        assert fd == 11 and mutate
        if request == controls._XU_QUERY:
            assert data.unit == 3
            key = {3: "exposure_us", 11: "auto_exposure"}[data.selector]
            assert data.size == (4 if data.selector == 3 else 1)
            if data.query == 1:
                value = int.from_bytes(bytes(data.data[:data.size]), "little")
                state[key] = value
                writes.append((key, value))
            else:
                value = {0x81: state[key], 0x82: 1, 0x83: 165000}[data.query]
                for index, byte in enumerate(value.to_bytes(data.size, "little")):
                    data.data[index] = byte
        elif request == 0xC0445624:
            struct.pack_into("<iii", data, 40, 16, 248, 1)
        elif request == 0xC008561B:
            struct.pack_into("<i", data, 4, state["gain"])
        elif request == 0xC008561C:
            state["gain"] = struct.unpack_from("<i", data, 4)[0]
            writes.append(("gain", state["gain"]))
        else:
            raise AssertionError(f"unexpected ioctl: {request:x}")

    monkeypatch.setattr(controls.fcntl, "ioctl", ioctl)
    return state, writes, closed


def test_manual_exposure_applies_in_order_and_records_readback(device):
    state, writes, closed = device
    result = controls.configure_d405_cameras([("top", "/dev/video4")], exposure_us=12000, gain=16)
    assert writes == [("auto_exposure", 0), ("exposure_us", 12000), ("gain", 16)]
    assert result[0]["before"] == {"auto_exposure": 1, "exposure_us": 8000, "gain": 24}
    assert result[0]["after"] == state == {"auto_exposure": 0, "exposure_us": 12000, "gain": 16}
    assert closed == [11]
    writes.clear()
    controls.configure_d405_cameras([("top", "/dev/video4")], exposure_us=12000, gain=16)
    assert not writes


def test_readback_mismatch_fails_and_closes_device(device, monkeypatch):
    original = controls.fcntl.ioctl

    def ignore_exposure_write(fd, request, data, mutate):
        if request == controls._XU_QUERY and data.selector == 3 and data.query == 1:
            return
        return original(fd, request, data, mutate)

    monkeypatch.setattr(controls.fcntl, "ioctl", ignore_exposure_write)
    with pytest.raises(RuntimeError, match="did not apply"):
        controls.configure_d405_cameras([("top", "/dev/video4")], exposure_us=12000, gain=16)
    assert device[2] == [11]


def test_exposure_waits_for_frame_boundary_but_busy_wait_is_bounded(device, monkeypatch):
    original = controls.fcntl.ioctl
    sleeps = []
    monkeypatch.setattr(controls.time, "sleep", sleeps.append)
    busy = 2

    def ioctl(fd, request, data, mutate):
        nonlocal busy
        if request == controls._XU_QUERY and busy:
            busy -= 1
            raise OSError(errno.EBUSY, "pending frame boundary")
        return original(fd, request, data, mutate)

    monkeypatch.setattr(controls.fcntl, "ioctl", ioctl)
    controls.configure_d405_cameras([("top", "/dev/video4")], exposure_us=12000, gain=16)
    assert sleeps == [0.05, 0.05]
    sleeps.clear()
    busy = 100
    with pytest.raises(OSError) as caught:
        controls.configure_d405_cameras([("top", "/dev/video4")], exposure_us=12000, gain=16)
    assert caught.value.errno == errno.EBUSY
    assert len(sleeps) == 20


@pytest.mark.parametrize("exposure,gain", [(200000, 16), (12000, 1)])
def test_out_of_range_controls_do_not_write(device, exposure, gain):
    with pytest.raises(ValueError, match="must be within"):
        controls.configure_d405_cameras([("top", "/dev/video4")], exposure_us=exposure, gain=gain)
    assert not device[1]
    assert device[2] == [11]


def test_all_profiles_explicitly_configure_manual_exposure():
    for machine in ("arx-local", "yam-local"):
        settings = load_settings(settings_path().parent / "examples" / f"{machine}.json")
        assert settings["camera_controls"] == {"exposure_us": 12000, "gain": 16}
