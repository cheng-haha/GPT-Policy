"""Small rigid-pose helpers shared by planning and perception."""

from __future__ import annotations

import numpy as np


def rotation_6d_to_quaternion(rotation: object) -> np.ndarray:
    """Decode two matrix *columns* into unit xyzw, without changing frames.

    Recorded rotations must already be orthonormal to float/rounding tolerance.
    Gram-Schmidt removes small numeric drift; it must not hide corrupt data.
    """
    values = np.asarray(rotation, dtype=np.float64)
    if values.shape != (6,) or not np.isfinite(values).all():
        raise ValueError("rotation6d must be a finite six-vector")
    columns = values.reshape(2, 3).T
    if not np.allclose(columns.T @ columns, np.eye(2), atol=1e-3, rtol=0):
        raise ValueError("rotation6d columns must be unit and orthogonal")
    a = columns[:, 0] / np.linalg.norm(columns[:, 0])
    b = columns[:, 1] - a * np.dot(a, columns[:, 1])
    b /= np.linalg.norm(b)
    matrix = np.column_stack((a, b, np.cross(a, b)))
    # The largest eigenvector of this symmetric matrix is quaternion xyzw.
    # Unlike trace-only division, this remains stable at rotations of pi.
    m = matrix
    k = np.array([
        [m[0,0]-m[1,1]-m[2,2], m[0,1]+m[1,0], m[0,2]+m[2,0], m[2,1]-m[1,2]],
        [m[0,1]+m[1,0], m[1,1]-m[0,0]-m[2,2], m[1,2]+m[2,1], m[0,2]-m[2,0]],
        [m[0,2]+m[2,0], m[1,2]+m[2,1], m[2,2]-m[0,0]-m[1,1], m[1,0]-m[0,1]],
        [m[2,1]-m[1,2], m[0,2]-m[2,0], m[1,0]-m[0,1], np.trace(m)],
    ])
    quaternion = np.linalg.eigh(k)[1][:, -1]
    quaternion /= np.linalg.norm(quaternion)
    if quaternion[3] < 0:
        quaternion = -quaternion
    return quaternion


def quaternion_to_rpy(quaternion: object) -> np.ndarray:
    q = np.asarray(quaternion, dtype=np.float64)
    if q.shape != (4,) or not np.isfinite(q).all():
        raise ValueError("pose quaternion must be a finite four-vector")
    norm = np.linalg.norm(q)
    if norm < 1e-9:
        raise ValueError("pose quaternion cannot be zero")
    x, y, z, w = q / norm
    # Independent roll/yaw atan2 formulas become 0/0 at pitch +/- pi/2.
    # Resolve the coupled angle from the full matrix, preserving the rotation.
    return matrix_to_rpy(np.array([
        [1 - 2 * (y*y + z*z), 2 * (x*y - w*z), 2 * (x*z + w*y)],
        [2 * (x*y + w*z), 1 - 2 * (x*x + z*z), 2 * (y*z - w*x)],
        [2 * (x*z - w*y), 2 * (y*z + w*x), 1 - 2 * (x*x + y*y)],
    ]))


def rpy_to_quaternion(rpy: object) -> list[float]:
    roll, pitch, yaw = np.asarray(rpy, dtype=np.float64) / 2.0
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    return [
        float(sr * cp * cy - cr * sp * sy),
        float(cr * sp * cy + sr * cp * sy),
        float(cr * cp * sy - sr * sp * cy),
        float(cr * cp * cy + sr * sp * sy),
    ]


def rpy_to_matrix(rpy: object) -> np.ndarray:
    roll, pitch, yaw = np.asarray(rpy, dtype=np.float64)
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ])


def matrix_to_rpy(rotation: object) -> np.ndarray:
    matrix = np.asarray(rotation, dtype=np.float64)
    horizontal = np.hypot(matrix[0, 0], matrix[1, 0])
    pitch = np.arctan2(-matrix[2, 0], horizontal)
    if horizontal > 1e-8:
        roll = np.arctan2(matrix[2, 1], matrix[2, 2])
        yaw = np.arctan2(matrix[1, 0], matrix[0, 0])
    else:
        roll = np.arctan2(-matrix[1, 2], matrix[1, 1])
        yaw = 0.0
    return np.array([roll, pitch, yaw])


def pose_to_matrix(pose: object) -> np.ndarray:
    values = np.asarray(pose, dtype=np.float64)
    if values.shape != (6,) or not np.isfinite(values).all():
        raise ValueError("pose must be a finite six-vector")
    transform = np.eye(4)
    transform[:3, :3] = rpy_to_matrix(values[3:])
    transform[:3, 3] = values[:3]
    return transform


def matrix_to_pose(transform: object) -> np.ndarray:
    matrix = np.asarray(transform, dtype=np.float64)
    return np.r_[matrix[:3, 3], matrix_to_rpy(matrix[:3, :3])]


def interpolate_pose(start: object, end: object, fraction: float) -> np.ndarray:
    """Linearly blend translation and slerp orientation."""
    first, second = np.asarray(start, dtype=np.float64), np.asarray(end, dtype=np.float64)
    q0 = np.asarray(rpy_to_quaternion(first[3:]))
    q1 = np.asarray(rpy_to_quaternion(second[3:]))
    dot = float(np.dot(q0, q1))
    if dot < 0:
        q1, dot = -q1, -dot
    if dot > 0.9995:
        quaternion = q0 + fraction * (q1 - q0)
    else:
        angle = np.arccos(np.clip(dot, -1.0, 1.0))
        quaternion = (
            np.sin((1 - fraction) * angle) * q0
            + np.sin(fraction * angle) * q1
        ) / np.sin(angle)
    quaternion /= np.linalg.norm(quaternion)
    return np.r_[first[:3] + fraction * (second[:3] - first[:3]), quaternion_to_rpy(quaternion)]


def rotation_distance(first_rpy: object, second_rpy: object) -> float:
    relative = rpy_to_matrix(first_rpy) @ rpy_to_matrix(second_rpy).T
    cosine = np.clip((np.trace(relative) - 1) / 2, -1.0, 1.0)
    return float(np.arccos(cosine))


def rotation_vector(rotation: object) -> np.ndarray:
    """Return the SO(3) logarithm of a rotation matrix."""
    matrix = np.asarray(rotation, dtype=np.float64)
    cosine = float(np.clip((np.trace(matrix) - 1.0) / 2.0, -1.0, 1.0))
    angle = float(np.arccos(cosine))
    skew = np.array(
        [
            matrix[2, 1] - matrix[1, 2],
            matrix[0, 2] - matrix[2, 0],
            matrix[1, 0] - matrix[0, 1],
        ],
        dtype=np.float64,
    )
    if angle < 1e-7:
        return 0.5 * skew
    if np.pi - angle < 1e-5:
        # This branch is not used by the numerical Jacobian (whose rotations
        # are tiny), but keeps pose-error diagnostics well defined near pi.
        diagonal = np.maximum((np.diag(matrix) + 1.0) / 2.0, 0.0)
        axis = np.sqrt(diagonal)
        largest = int(np.argmax(axis))
        if axis[largest] > 1e-7:
            if largest == 0:
                axis[1] = (matrix[0, 1] + matrix[1, 0]) / (4.0 * axis[0])
                axis[2] = (matrix[0, 2] + matrix[2, 0]) / (4.0 * axis[0])
            elif largest == 1:
                axis[0] = (matrix[0, 1] + matrix[1, 0]) / (4.0 * axis[1])
                axis[2] = (matrix[1, 2] + matrix[2, 1]) / (4.0 * axis[1])
            else:
                axis[0] = (matrix[0, 2] + matrix[2, 0]) / (4.0 * axis[2])
                axis[1] = (matrix[1, 2] + matrix[2, 1]) / (4.0 * axis[2])
        norm = np.linalg.norm(axis)
        return angle * axis / norm if norm > 1e-9 else np.zeros(3)
    return angle * skew / (2.0 * np.sin(angle))
