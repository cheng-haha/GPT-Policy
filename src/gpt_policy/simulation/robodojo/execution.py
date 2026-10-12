"""Opt-in native IK and robot-state feedback; no object-state access."""

import math

import numpy as np

from .control import MAX_POSITION_DELTA, MAX_ROTATION_DELTA, measured_tracking


class IKRejected(ValueError):
    """No physical action was submitted for this proposal."""

    def __init__(self, message, **details):
        super().__init__(message)
        self.details = details


def execution_config(config):
    backend = config.get("ik_backend", "bounded_dls")
    if backend not in ("bounded_dls", "native"):
        raise ValueError("ik_backend must be bounded_dls or native")
    increment = config.get("max_joint_increment_rad", 0.05)
    if (type(increment) not in (int, float) or not math.isfinite(increment)
            or not 0 < increment <= 0.15):
        raise ValueError("max_joint_increment_rad must be finite in (0, 0.15]")
    feedback = config.get("chunk_feedback", False)
    if type(feedback) is not bool:
        raise ValueError("chunk_feedback must be boolean")
    distribute = config.get("distribute_closed_gripper_moves", False)
    if type(distribute) is not bool:
        raise ValueError("distribute_closed_gripper_moves must be boolean")
    return {"ik_backend": backend, "max_joint_increment_rad": float(increment),
            "chunk_feedback": feedback, "distribute_closed_gripper_moves": distribute}


def continuous_joints(candidate, current, limits, increment, previous=None):
    """Choose legal 2*pi equivalents, without clipping or changing the IK pose."""
    candidate, current, limits = (np.asarray(x, float) for x in (candidate, current, limits))
    if (candidate.shape != current.shape or current.ndim != 1
            or limits.shape != (len(current), 2)
            or not all(np.isfinite(x).all() for x in (candidate, current, limits))
            or np.any(limits[:, 0] > limits[:, 1])):
        raise IKRejected("Malformed native IK joint solution or limits")
    low = np.maximum(limits[:, 0], current - increment)
    high = np.minimum(limits[:, 1], current + increment)
    if previous is not None:
        previous = np.asarray(previous, float)
        if previous.shape != current.shape or not np.isfinite(previous).all():
            raise IKRejected("Invalid previous acknowledged joint target")
        low = np.maximum(low, previous - increment)
        high = np.minimum(high, previous + increment)
    # Tolerance only covers floating-point boundary error, not extra motion.
    first = np.ceil((low - candidate - 1e-7) / (2 * np.pi))
    last = np.floor((high - candidate + 1e-7) / (2 * np.pi))
    invalid = (first > last) | (low > high + 1e-7)
    if np.any(invalid):
        # Diagnostic equivalents only: never clip, submit or retry this proposal.
        nearest = candidate + np.round((current - candidate) / (2 * np.pi)) * 2 * np.pi
        violations = [
            {
                "joint_index": int(i),
                "candidate_equivalent_rad": float(nearest[i]),
                "measured_joint_rad": float(current[i]),
                "delta_from_measured_rad": float(nearest[i] - current[i]),
                "previous_ack_target_rad": float(previous[i]) if previous is not None else None,
                "delta_from_previous_ack_rad": float(nearest[i] - previous[i]) if previous is not None else None,
                "joint_limits_rad": limits[i].tolist(),
                "allowed_interval_rad": [float(low[i]), float(high[i])],
            }
            for i in np.flatnonzero(invalid)
        ]
        raise IKRejected(
            f"Native IK exceeds {increment:g} rad joint continuity or joint limits; "
            "request a smaller Cartesian/orientation change from the returned robot state",
            code="joint_continuity_or_limits",
            max_joint_increment_rad=float(increment),
            violating_joints=violations,
        )
    turns = np.clip(np.round((current - candidate) / (2 * np.pi)), first, last)
    return (candidate + turns * 2 * np.pi).astype(np.float32)


class TrackingGuard:
    """Stop an unexecuted suffix after two ACKs outside existing path bounds."""

    def __init__(self):
        self.bad_samples = 0

    def update(self, point, observation, action):
        tracking = measured_tracking(point, observation)
        joint_errors = {
            arm: float(np.max(np.abs(
                np.asarray(observation["arms"][arm]["joint_pos"])
                - np.asarray(action[offset:offset + 6])
            ))) for arm, offset in (("left", 0), ("right", 7))
        }
        bad = any(
            row["target_position_error_m"] > MAX_POSITION_DELTA
            or row["target_rotation_error_rad"] > MAX_ROTATION_DELTA
            or joint_errors[arm] > 0.15
            for arm, row in tracking.items()
        )
        self.bad_samples = self.bad_samples + 1 if bad else 0
        return self.bad_samples >= 2
