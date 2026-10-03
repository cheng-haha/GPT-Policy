from __future__ import annotations

import io
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image

from gpt_policy.hardware.camera import CapturedImage
from gpt_policy.recording.jpeg import encode_jpeg
from gpt_policy.recording.video import RunVideo


class _Cameras:
    def __init__(self):
        self.cameras = [
            SimpleNamespace(name=name, width=size, height=size)
            for name, size in (("left", 16), ("right", 24), ("top", 32))
        ]
        self.home = False

    def capture(self, stop):
        if stop.wait(0.005):
            raise InterruptedError("stopped")
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
        if self.home:
            colors = [(255, 255, 255)] * 3
        return {
            camera.name: CapturedImage(
                camera.name, b"", "image/png", camera.width, camera.height, time.time(),
                bytes(color) * camera.width * camera.height,
            )
            for camera, color in zip(self.cameras, colors)
        }


class RunVideoTest(unittest.TestCase):
    def test_slow_encoder_does_not_block_fresh_observation(self):
        entered = threading.Event()
        release = threading.Event()

        def slow_encode(rgb, quality):
            entered.set()
            if not release.wait(2.0):
                raise TimeoutError("test encoder was not released")
            return encode_jpeg(rgb, quality)

        with tempfile.TemporaryDirectory() as directory:
            with patch("gpt_policy.recording.video.encode_jpeg", slow_encode):
                video = RunVideo(_Cameras(), Path(directory))
                try:
                    self.assertTrue(entered.wait(1.0))
                    requested_at = time.time()
                    images = video.snapshot(after=requested_at, timeout=0.5)
                    self.assertEqual(set(images), {"left", "right", "top"})
                    self.assertTrue(all(frame.captured_at > requested_at for frame in images.values()))
                    self.assertFalse(release.is_set())
                finally:
                    release.set()
                    video.stop()

    def test_three_separate_videos_include_final_home_frame(self):
        encoded = threading.Event()

        def encode(rgb, quality):
            result = encode_jpeg(rgb, quality)
            encoded.set()
            return result

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            cameras = _Cameras()
            with patch("gpt_policy.recording.video.encode_jpeg", encode):
                video = RunVideo(cameras, path, fps=1)
                try:
                    self.assertTrue(encoded.wait(1.0))
                    cameras.home = True
                finally:
                    details = video.stop()

            self.assertFalse(video._capture_thread.is_alive())
            self.assertFalse(video._record_thread.is_alive())
            self.assertEqual(set(details["videos"]), {"left", "right", "top"})
            self.assertEqual(video.stop(), details)
            self.assertEqual({file.name for file in path.iterdir()}, {"left.mp4", "right.mp4", "top.mp4", "video-frames.jsonl"})
            self.assertGreaterEqual(details["frames"], 2)
            for channel, camera in enumerate(cameras.cameras):
                writer = video.writers[camera.name]
                self.assertTrue(writer.stream.closed)
                self.assertEqual(len(writer.sizes), details["frames"])
                self.assertEqual(details["videos"][camera.name]["path"], f"{camera.name}.mp4")
                data = writer.path.read_bytes()
                self.assertIn(b"moov", data)
                samples = [
                    data[offset:offset + size]
                    for offset, size in zip(writer.offsets, writer.sizes)
                ]
                with Image.open(io.BytesIO(samples[0])) as first:
                    self.assertEqual(first.size, (camera.width, camera.height))
                    pixel = first.getpixel((8, 8))
                    self.assertGreater(pixel[channel], 240)
                    self.assertTrue(all(value < 15 for index, value in enumerate(pixel) if index != channel))
                with Image.open(io.BytesIO(samples[-1])) as last:
                    self.assertEqual(last.getpixel((8, 8)), (255, 255, 255))

            with self.assertRaisesRegex(RuntimeError, "已停止"):
                video.snapshot(after=time.time())

    def test_encoding_time_is_included_in_frame_period(self):
        now = 0.0
        waits = []

        class Stop:
            def is_set(self):
                return len(waits) == 2

            def wait(self, duration):
                nonlocal now
                waits.append(duration)
                now += duration

        def write(_images):
            nonlocal now
            now += 0.04

        video = RunVideo.__new__(RunVideo)
        video.fps = 10
        video._stop = Stop()
        video._record_error = None
        video.snapshot = lambda: {}
        video._write_frames = write
        with patch("gpt_policy.recording.video.time.monotonic", lambda: now):
            video._record()
        self.assertIsNone(video._record_error)
        self.assertEqual(len(waits), 2)
        for wait in waits:
            self.assertAlmostEqual(wait, 0.06)


if __name__ == "__main__":
    unittest.main()
