"""RGB capture from named RealSense serials; one owner per camera."""

from __future__ import annotations

import math
import time
from types import SimpleNamespace

import numpy as np

from .camera import CapturedImage
from ..recording.jpeg import encode_jpeg


class RealSenseCameraSet:
    def __init__(self, specs, width=640, height=480, fps=30):
        import pyrealsense2 as rs

        self.rs = rs
        self.cameras = []
        try:
            for name, serial in specs:
                pipeline = rs.pipeline()
                config = rs.config()
                config.enable_device(serial)
                config.enable_stream(rs.stream.color, width, height, rs.format.rgb8, fps)
                pipeline.start(config)
                self.cameras.append(SimpleNamespace(
                    name=name, serial=serial, width=width, height=height, pipeline=pipeline,
                ))
        except BaseException:
            self.close()
            raise

    def configure_controls(self, *, exposure_us: int, gain: int) -> list[dict]:
        """Reapply manual RGB exposure on each task before warmup.

        Camera power cycles and other applications can change these controls.
        Use the existing streams, preserve white balance, and reject a failed
        readback before recording or robot initialization.
        """
        for name, value in (("exposure_us", exposure_us), ("gain", gain)):
            if type(value) is not int or value <= 0:
                raise ValueError(f"camera_controls.{name} must be a positive integer")
        if not self.cameras:
            raise ValueError("camera_controls requires explicitly named RealSense devices")
        rs = self.rs
        options = {
            "auto_exposure": rs.option.enable_auto_exposure,
            "exposure_us": rs.option.exposure,
            "gain": rs.option.gain,
        }
        target = {"auto_exposure": 0, "exposure_us": exposure_us, "gain": gain}
        selected = []
        # Validate every RGB sensor before changing any camera. On D405 this
        # is the Stereo Module, rather than a separate RGB Camera sensor.
        for camera in self.cameras:
            device = camera.pipeline.get_active_profile().get_device()
            sensors = [sensor for sensor in device.query_sensors()
                       if any(p.stream_type() == rs.stream.color
                              for p in sensor.get_stream_profiles())]
            if len(sensors) != 1:
                raise RuntimeError(f"{camera.name}: expected one RealSense RGB sensor")
            sensor = sensors[0]
            for key, option in options.items():
                if not sensor.supports(option) or sensor.is_option_read_only(option):
                    raise ValueError(f"{camera.name}: {key} is not writable")
                limits = sensor.get_option_range(option)
                value = target[key]
                steps = (value - limits.min) / limits.step if limits.step > 0 else 0
                if not limits.min <= value <= limits.max or not math.isclose(
                    steps, round(steps), abs_tol=1e-6,
                ):
                    raise ValueError(
                        f"{camera.name}: {key} must be within "
                        f"{limits.min}..{limits.max}, step {limits.step}"
                    )
            selected.append((camera, sensor))
        results = []
        for camera, sensor in selected:
            before = {key: sensor.get_option(option) for key, option in options.items()}
            # Always write all three values, even when they already match.
            for key, option in options.items():
                sensor.set_option(option, target[key])
            deadline = time.monotonic() + 1.0
            while True:
                after = {key: sensor.get_option(option) for key, option in options.items()}
                if after == target:
                    break
                if time.monotonic() >= deadline:
                    raise RuntimeError(
                        f"{camera.name}: RealSense controls did not apply: "
                        f"expected {target}, got {after}"
                    )
                time.sleep(0.05)
            results.append({"name": camera.name, "device": camera.serial,
                            "before": before, "after": after})
        return results

    def capture(self, stop=None):
        images = {}
        for camera in self.cameras:
            deadline = time.monotonic() + 3.0
            while True:
                if stop is not None and stop.is_set():
                    raise InterruptedError("RealSense capture stopped")
                ok, frames = camera.pipeline.try_wait_for_frames(200)
                if ok and frames.get_color_frame():
                    break
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"No RGB frame from {camera.name} ({camera.serial})")
            frame = frames.get_color_frame()
            rgb = np.asanyarray(frame.get_data()).copy()
            images[camera.name] = CapturedImage(
                camera.name, encode_jpeg(rgb, 85), "image/jpeg", camera.width, camera.height,
                time.time(), rgb.tobytes(), frame.get_timestamp() / 1000,
                "realsense_" + str(frame.get_frame_timestamp_domain()),
            )
        return images

    def describe(self, images=None):
        return [{
            "name": c.name, "serial": c.serial, "device": c.serial, "format": "RGB8",
            "width": c.width, "height": c.height,
            **({"captured_at": str(images[c.name].captured_at)} if images and c.name in images else {}),
        } for c in self.cameras]

    def close(self):
        errors = []
        for camera in self.cameras:
            try:
                camera.pipeline.stop()
            except Exception as exc:
                errors.append(str(exc))
        self.cameras.clear()
        if errors:
            raise RuntimeError(f"RealSense cleanup: {errors}")
