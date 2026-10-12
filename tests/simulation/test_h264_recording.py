"""Exercise the real encoder: color, frame count/rate, completion and failures."""

import json
import shutil
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest

from gpt_policy.recording.h264 import H264Mp4Writer

pytestmark = pytest.mark.skipif(
    not shutil.which("ffmpeg") or not shutil.which("ffprobe"),
    reason="ffmpeg/ffprobe required for H.264 integration",
)


def test_h264_roundtrip(tmp_path):
    target = tmp_path / "camera.mp4"
    writer = H264Mp4Writer(target, 64, 48, 25)
    colors = [(210, 20, 35), (20, 210, 45), (25, 30, 210)]
    for color in colors:
        writer.append(np.full((48, 64, 3), color, dtype=np.uint8))
    assert not target.exists()  # Never advertise an unfinished MP4.
    writer.close()
    writer.close()
    probe = json.loads(
        subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-count_frames",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=codec_name,pix_fmt,width,height,r_frame_rate,nb_read_frames",
                "-of",
                "json",
                str(target),
            ]
        )
    )["streams"][0]
    assert probe == {
        "codec_name": "h264",
        "pix_fmt": "yuv420p",
        "width": 64,
        "height": 48,
        "r_frame_rate": "25/1",
        "nb_read_frames": "3",
    }
    payload = target.read_bytes()
    assert payload.index(b"moov") < payload.index(b"mdat")
    decoded = subprocess.check_output(
        [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-xerror",
            "-i",
            str(target),
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "pipe:1",
        ]
    )
    frames = np.frombuffer(decoded, np.uint8).reshape(3, 48, 64, 3)
    np.testing.assert_allclose(frames.mean(axis=(1, 2)), colors, atol=5)
    with pytest.raises(FileExistsError):
        H264Mp4Writer(target, 64, 48)
    with pytest.raises(RuntimeError, match="closed"):
        writer.append(frames[0])


def test_bad_geometry_and_frames(tmp_path):
    with pytest.raises(ValueError, match="even"):
        H264Mp4Writer(tmp_path / "odd.mp4", 63, 48)
    writer = H264Mp4Writer(tmp_path / "empty.mp4", 64, 48)
    with pytest.raises(ValueError, match="uint8 RGB"):
        writer.append(np.zeros((48, 64, 4), np.uint8))
    with pytest.raises(RuntimeError, match="frames=0"):
        writer.close()
    assert not writer.path.exists()


def test_encoder_failure_never_publishes(tmp_path):
    writer = H264Mp4Writer(tmp_path / "failed.mp4", 64, 48)
    writer.process.terminate()
    writer.process.wait(timeout=10)
    with pytest.raises(RuntimeError, match="ffmpeg exited"):
        writer.append(np.zeros((48, 64, 3), np.uint8))
    with pytest.raises(RuntimeError):
        writer.close()
    assert not writer.path.exists()


def test_missing_ffmpeg_is_explicit(tmp_path, monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda _: None)
    with pytest.raises(RuntimeError, match="requires ffmpeg"):
        H264Mp4Writer(tmp_path / "missing.mp4", 64, 48)


def test_native_finish_closes_all_cameras_on_encoder_failure(tmp_path):
    from gpt_policy.simulation.robodojo.environment import NativeSession

    closed = []

    class Writer:
        def __init__(self, name, fail=False):
            self.name, self.fail = name, fail

        def close(self):
            closed.append(self.name)
            if self.fail:
                raise RuntimeError("encoder error")

    session = NativeSession(SimpleNamespace(), tmp_path, "test", {})
    session.writers = {
        "top": Writer("top", True),
        "left": Writer("left"),
        "right": Writer("right"),
    }
    with pytest.raises(RuntimeError, match="video finalization"):
        session.finish("test")
    assert closed == ["top", "left", "right"]
    assert not (tmp_path / "native_result.json").exists()
