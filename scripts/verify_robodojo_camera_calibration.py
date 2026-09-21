#!/usr/bin/env python3
"""Check live Isaac Sim camera projection against GPT-Policy back-projection.

Run with the RoboDojo simulator's Python. No policy, model call, robot motion,
or task-specific object state is required. Both cameras use known landmarks
to check intrinsics, sensor poses, base transforms, and optical-axis signs.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "src"))
    sys.path.insert(0, str(root / "third_party" / "RoboDojo"))

    from isaaclab.app import AppLauncher

    app = AppLauncher(headless=True, enable_cameras=True, device=f"cuda:{args.gpu}").app
    try:
        import numpy as np
        from env.camera_manager.camera_manager import CameraManager
        from env_cfg.camera.template import D435, GEMINI_345LG
        from isaacsim.sensors.camera import Camera

        from gpt_policy.robodojo import RoboDojoCalibration

        world_from_base = np.array([[0, -1, 0, -0.3], [1, 0, 0, -0.45], [0, 0, 1, 0.765], [0, 0, 0, 1]])
        landmarks = np.array([[0.0, 0.0, 0.78], [0.04, -0.06, 0.78], [-0.03, 0.02, 0.80]])
        maximum_error = 0.0
        for name, settings, position, angle in (
            ("head", GEMINI_345LG, [0.0, -0.41, 1.308], 30.0),
            ("wrist", D435, [0.03, -0.17, 1.01], 40.0),
        ):
            camera = Camera(prim_path=f"/World/{name}", resolution=settings["resolution"])
            camera.set_lens_distortion_model("pinhole")
            camera.set_focal_length(settings["focal_length"])
            camera.set_horizontal_aperture(settings["horizontal_aperture"], False)
            camera.set_vertical_aperture(settings["horizontal_aperture"] * settings["resolution"][1] / settings["resolution"][0], False)
            half_angle = np.deg2rad(angle) / 2
            camera.set_world_pose(np.array(position), np.array([np.cos(half_angle), np.sin(half_angle), 0, 0]), camera_axes="usd")
            manager = CameraManager.__new__(CameraManager)
            manager.cameras = [[camera]]
            intrinsics = manager.get_camera_intrinsics(0)
            extrinsics = manager.get_camera_extrinsics(0, 0)
            calibration = RoboDojoCalibration.from_observation(
                {"world_from_base": {"left": world_from_base.tolist()}},
                {"vision": {name: {"intrinsic_matrix": intrinsics, "extrinsic_matrix": extrinsics}}},
            )
            pixels = camera.get_image_coords_from_world_points(landmarks)
            expected_fy = settings["resolution"][0] * settings["focal_length"] / settings["horizontal_aperture"]
            np.testing.assert_allclose(intrinsics[1, 1], expected_fy, rtol=1e-6)
            for landmark, pixel in zip(landmarks, pixels):
                ray = calibration.camera_ray_in_base(name, pixel, "left")
                target = np.linalg.solve(world_from_base, [*landmark, 1.0])[:3]
                displacement = target - ray["ray_origin_base_xyz"]
                depth = float(np.dot(displacement, ray["ray_direction_base_xyz"]))
                error = float(np.linalg.norm(np.cross(displacement, ray["ray_direction_base_xyz"])))
                maximum_error = max(maximum_error, error)
                assert depth > 0, (name, pixel, depth)
                assert error < 1e-6, (name, pixel, error)
            print(f"{name}: fx={intrinsics[0, 0]:.6f}, fy={intrinsics[1, 1]:.6f}; 3 forward rays match SDK-projected landmarks", flush=True)
        print(f"PASS: live camera projection/back-projection, maximum ray error = {maximum_error:.3e} m", flush=True)
        return 0
    except BaseException as error:
        logging.getLogger(__name__).exception("Camera calibration verification failed")
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(130 if isinstance(error, KeyboardInterrupt) else 1)
    finally:
        app.close()


if __name__ == "__main__":
    raise SystemExit(main())
