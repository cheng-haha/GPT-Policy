from __future__ import annotations

import unittest
from threading import Event
from unittest.mock import patch

from gpt_policy.hardware import camera as camera_module
from gpt_policy.hardware.camera import IncompleteFrameError, V4L2Camera


class CameraFrameTest(unittest.TestCase):
    def make_camera(self) -> V4L2Camera:
        camera = V4L2Camera.__new__(V4L2Camera)
        camera.fd = 10
        camera.name = "test"
        camera.width = 2
        camera.height = 1
        camera.pixelformat = camera_module._YUYV
        camera.bytesperline = 4
        camera._maps = [bytearray([128, 128, 128, 128])]
        camera.dropped_frames = 0
        camera._discard_queued_frames = lambda: None
        return camera

    def test_short_uncompressed_payload_has_clear_error(self) -> None:
        camera = self.make_camera()
        with self.assertRaises(IncompleteFrameError):
            camera._to_rgb(b"\x80\x80")

    def test_capture_drops_short_frame_and_retries(self) -> None:
        camera = self.make_camera()
        dequeues = 0

        def ioctl(_fd, request, buffer, _mutate=True):
            nonlocal dequeues
            if request == camera_module._DQBUF:
                dequeues += 1
                buffer.index = 0
                buffer.bytesused = 2 if dequeues == 1 else 4
                buffer.flags = 0
            return 0

        with (
            patch.object(camera_module.select, "select", return_value=([10], [], [])),
            patch.object(camera_module.fcntl, "ioctl", side_effect=ioctl),
        ):
            image = camera.capture()

        self.assertEqual(dequeues, 2)
        self.assertEqual(camera.dropped_frames, 1)
        self.assertEqual((image.width, image.height), (2, 1))

    def test_capture_can_stop_when_camera_has_no_frame(self) -> None:
        camera = self.make_camera()
        stop = Event()

        def select(_read, _write, _error, timeout):
            self.assertLessEqual(timeout, 0.2)
            stop.set()
            return [], [], []

        with patch.object(camera_module.select, "select", side_effect=select):
            with self.assertRaises(InterruptedError):
                camera.capture(stop=stop)


if __name__ == "__main__":
    unittest.main()
