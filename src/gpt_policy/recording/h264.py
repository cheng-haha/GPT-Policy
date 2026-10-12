"""Streaming RGB to H.264 MP4 for browser/VS Code playback (requires ffmpeg)."""

from __future__ import annotations

import os
import shutil
import subprocess
import uuid
from pathlib import Path

import numpy as np


def encoder_options() -> list[str]:
    """Bound CPU use; keep standard 8-bit 4:2:0 and a front-loaded MP4 index."""
    return [
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "18",
        "-threads",
        "2",
        "-filter_threads",
        "1",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
    ]


def publish_video(temporary: Path, destination: Path) -> None:
    """Publish a finished file atomically without replacing any existing result."""
    os.link(temporary, destination)
    temporary.unlink()


class H264Mp4Writer:
    """Encode each RGB frame once; expose the final MP4 only after clean close.

    Per-control PNGs remain the lossless evidence. Interrupted encoding files and
    stderr logs are retained for diagnosis; no MJPEG fallback hides a failure.
    """

    def __init__(self, path: Path, width: int, height: int, fps: int = 25) -> None:
        if any(type(x) is not int or x <= 0 for x in (width, height, fps)):
            raise ValueError("Video dimensions and fps must be positive integers")
        if width % 2 or height % 2:
            raise ValueError("H.264 yuv420p requires even image dimensions")
        executable = shutil.which("ffmpeg")
        if executable is None:
            raise RuntimeError("RoboDojo video recording requires ffmpeg with libx264")
        self.path, self.width, self.height, self.fps = Path(path), width, height, fps
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            raise FileExistsError(self.path)
        token = uuid.uuid4().hex
        self.temporary = self.path.with_name(f".{self.path.stem}.{token}.encoding.mp4")
        self.log_path = self.path.with_name(f"{self.path.stem}.{token}.encoding.log")
        self.frames = 0
        self._closed = False
        self._error: RuntimeError | None = None
        with self.log_path.open("xb") as log:
            self.process = subprocess.Popen(
                [
                    executable,
                    "-nostdin",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-n",
                    "-f",
                    "rawvideo",
                    "-pixel_format",
                    "rgb24",
                    "-video_size",
                    f"{width}x{height}",
                    "-framerate",
                    str(fps),
                    "-i",
                    "pipe:0",
                    *encoder_options(),
                    str(self.temporary),
                ],
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=log,
            )

    def _failure(self, detail: str) -> RuntimeError:
        self._error = RuntimeError(
            f"H.264 encoding failed ({detail}); see {self.log_path}"
        )
        return self._error

    def append(self, rgb: np.ndarray) -> None:
        if self._closed:
            raise RuntimeError("Cannot append to a closed video")
        if self._error is not None:
            raise self._error
        if rgb.dtype != np.uint8 or rgb.shape != (self.height, self.width, 3):
            raise ValueError("Expected uint8 RGB with the configured dimensions")
        if self.process.poll() is not None:
            raise self._failure(f"ffmpeg exited {self.process.returncode}")
        try:
            self.process.stdin.write(np.ascontiguousarray(rgb).tobytes())
            self.process.stdin.flush()
        except (BrokenPipeError, OSError) as error:
            raise self._failure(str(error)) from error
        self.frames += 1

    def close(self) -> None:
        if self._closed:
            if self._error is not None:
                raise self._error
            return
        self._closed = True
        try:
            self.process.stdin.close()
        except (BrokenPipeError, OSError) as error:
            self._failure(str(error))
        try:
            code = self.process.wait(timeout=60)
        except subprocess.TimeoutExpired as error:
            self.process.kill()
            self.process.wait(timeout=10)
            raise self._failure("ffmpeg finalization timed out") from error
        if self._error is not None:
            raise self._error
        if code != 0 or not self.frames:
            raise self._failure(f"ffmpeg exited {code}, frames={self.frames}")
        try:
            publish_video(self.temporary, self.path)
        except OSError as error:
            raise self._failure(f"cannot publish finished video: {error}") from error

    def __enter__(self):
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
