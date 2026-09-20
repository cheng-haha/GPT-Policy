"""Coordinate and camera calibration for the RoboDojo bridge.

RoboDojo reports camera poses as camera-to-world and end-effector quaternions
as ``wxyz``. GPT-Policy commands ``xyzw`` TCP poses in an arm base frame. This
module is the single place where those conventions are converted.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


def _matrix(value: Any, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (4, 4) or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a finite 4x4 matrix")
    return result


def _intrinsics(value: Any, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (3, 3) or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a finite 3x3 matrix")
    if abs(float(np.linalg.det(result))) < 1e-12:
        raise ValueError(f"{name} must be invertible")
    return result


def quat_wxyz_to_matrix(quaternion: Any) -> np.ndarray:
    q = np.asarray(quaternion, dtype=np.float64).reshape(4)
    if not np.isfinite(q).all() or np.linalg.norm(q) < 1e-12:
        raise ValueError("quaternion must be finite and non-zero")
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def matrix_to_quat_xyzw(matrix: Any) -> list[float]:
    """Convert a proper rotation matrix to GPT-Policy's ``xyzw`` order."""
    r = np.asarray(matrix, dtype=np.float64)
    if r.shape != (3, 3) or not np.isfinite(r).all():
        raise ValueError("rotation must be a finite 3x3 matrix")
    # Stable Shoemake-style conversion, followed by normalization.
    trace = float(np.trace(r))
    if trace > 0:
        s = 2 * np.sqrt(trace + 1.0)
        w, x, y, z = 0.25 * s, (r[2, 1] - r[1, 2]) / s, (r[0, 2] - r[2, 0]) / s, (r[1, 0] - r[0, 1]) / s
    elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
        s = 2 * np.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2])
        w, x, y, z = (r[2, 1] - r[1, 2]) / s, 0.25 * s, (r[0, 1] + r[1, 0]) / s, (r[0, 2] + r[2, 0]) / s
    elif r[1, 1] > r[2, 2]:
        s = 2 * np.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2])
        w, x, y, z = (r[0, 2] - r[2, 0]) / s, (r[0, 1] + r[1, 0]) / s, 0.25 * s, (r[1, 2] + r[2, 1]) / s
    else:
        s = 2 * np.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1])
        w, x, y, z = (r[1, 0] - r[0, 1]) / s, (r[0, 2] + r[2, 0]) / s, (r[1, 2] + r[2, 1]) / s, 0.25 * s
    q = np.asarray([x, y, z, w], dtype=np.float64)
    q /= np.linalg.norm(q)
    return q.tolist()


@dataclass(frozen=True)
class RoboDojoCalibration:
    """Static calibration plus optional current camera-to-world poses."""

    world_from_base: Mapping[str, np.ndarray]
    tcp_from_link6: Mapping[str, np.ndarray]
    intrinsics: Mapping[str, np.ndarray]
    camera_extrinsics_world: Mapping[str, np.ndarray]

    @classmethod
    def from_manifest(cls, manifest: Mapping[str, Any]) -> "RoboDojoCalibration":
        return cls(
            world_from_base={k: _matrix(v, f"world_from_base[{k}]") for k, v in manifest.get("world_from_base", {}).items()},
            tcp_from_link6={k: _matrix(v, f"tcp_from_link6[{k}]") for k, v in manifest.get("tcp_from_link6", {}).items()},
            intrinsics={k: _intrinsics(v, f"intrinsics[{k}]") for k, v in manifest.get("intrinsics", {}).items()},
            camera_extrinsics_world={k: _matrix(v, f"camera_extrinsics_world[{k}]") for k, v in manifest.get("camera_extrinsics_world", {}).items()},
        )

    @classmethod
    def from_observation(
        cls, manifest: Mapping[str, Any], observation: Mapping[str, Any]
    ) -> "RoboDojoCalibration":
        """Complete a static manifest with matrices emitted by RoboDojo.

        Enable ``observation.vision.intrinsic_matrix`` and
        ``extrinsic_matrix`` in the RoboDojo env config. The first frame then
        supplies the exact active camera matrices; this avoids duplicating
        camera geometry in a hand-written calibration file.
        """
        vision = observation.get("vision", {})
        intrinsics = dict(manifest.get("intrinsics", {}))
        extrinsics = dict(manifest.get("camera_extrinsics_world", {}))
        for name, data in vision.items():
            if not isinstance(data, Mapping):
                continue
            if data.get("intrinsic_matrix") is not None:
                intrinsics[name] = data["intrinsic_matrix"]
            if data.get("extrinsic_matrix") is not None:
                extrinsics[name] = data["extrinsic_matrix"]
        merged = dict(manifest)
        merged["intrinsics"] = intrinsics
        merged["camera_extrinsics_world"] = extrinsics
        return cls.from_manifest(merged)

    def base_from_world(self, arm: str) -> np.ndarray:
        return np.linalg.inv(self.world_from_base[arm])

    def base_from_camera(self, camera: str, arm: str, camera_to_world: Any | None = None) -> np.ndarray:
        camera_world = _matrix(camera_to_world, f"camera_extrinsics_world[{camera}]") if camera_to_world is not None else self.camera_extrinsics_world[camera]
        return self.base_from_world(arm) @ camera_world

    def camera_ray_in_base(self, camera: str, pixel_xy: Any, arm: str, camera_to_world: Any | None = None) -> dict[str, Any]:
        k = self.intrinsics[camera]
        pixel = np.asarray([float(pixel_xy[0]), float(pixel_xy[1]), 1.0])
        direction_camera = np.linalg.inv(k) @ pixel
        direction_camera /= np.linalg.norm(direction_camera)
        transform = self.base_from_camera(camera, arm, camera_to_world)
        direction_base = transform[:3, :3] @ direction_camera
        direction_base /= np.linalg.norm(direction_base)
        return {
            "frame": f"{arm}_base_link",
            "ray_origin_base_xyz": transform[:3, 3].tolist(),
            "ray_direction_base_xyz": direction_base.tolist(),
            "camera": camera,
            "pixel_xy": [float(pixel_xy[0]), float(pixel_xy[1])],
        }

    def context_manifest(self) -> dict[str, Any]:
        """Small, serializable semantic manifest suitable for model context."""
        return {
            "backend": "robodojo",
            "pose_frame": "each arm's base_link",
            "position_unit": "metres",
            "quaternion_order": "xyzw in GPT-Policy; RoboDojo source is wxyz",
            "tcp_definition": "calibrated fingertip TCP; +z points to fingertips and +y is the gripper opening axis",
            "camera_frames": sorted(self.intrinsics),
            "camera_extrinsic_convention": "camera-to-world from RoboDojo, converted by host to base-from-camera",
            "intrinsics_available": sorted(self.intrinsics),
            "extrinsics_available": sorted(self.camera_extrinsics_world),
        }
