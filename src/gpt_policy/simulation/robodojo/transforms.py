"""Explicit site/frame conversions; simulation never uses ARX hardware calibration."""

import numpy as np
from scipy.spatial.transform import Rotation


def transform(position, quaternion_wxyz):
    q = np.asarray(quaternion_wxyz, float)
    t = np.eye(4)
    t[:3, :3] = Rotation.from_quat(q[[1, 2, 3, 0]]).as_matrix()
    t[:3, 3] = position
    return t


def pose(t):
    q = Rotation.from_matrix(t[:3, :3]).as_quat()
    return {"position": t[:3, 3].tolist(), "quaternion_wxyz": q[[3, 0, 1, 2]].tolist()}


def xyzw_pose(t):
    return [*t[:3, 3].tolist(), *Rotation.from_matrix(t[:3, :3]).as_quat().tolist()]


def pose_matrix(xyzw):
    value = np.asarray(xyzw, float)
    if (
        value.shape != (7,)
        or not np.isfinite(value).all()
        or abs(np.linalg.norm(value[3:]) - 1) > 1e-4
    ):
        raise ValueError("Expected finite xyz + unit xyzw")
    return transform(value[:3], value[[6, 3, 4, 5]])


def to_tool(environment_link6, environment_base, link6_tool):
    return np.linalg.inv(environment_base) @ environment_link6 @ link6_tool


def to_link6(base_tool, environment_base, link6_tool):
    return environment_base @ base_tool @ np.linalg.inv(link6_tool)
