"""Validate the entire requested path before any physical submission."""

import numpy as np
from scipy.spatial.transform import Rotation

from .transforms import pose_matrix

ARMS = ("left", "right")
MAX_POSITION_DELTA = 0.05
MAX_ROTATION_DELTA = 0.35


def validate_path(arguments, observation, *, early_arrival=False):
    if arguments.get("observation_id") != observation["observation_id"]:
        raise ValueError("Stale observation/path identity")
    if not isinstance(arguments.get("note"), str) or not arguments["note"].strip():
        raise ValueError("Visible evidence and action purpose required")
    waypoints = arguments.get("waypoints")
    if not isinstance(waypoints, list) or not 1 <= len(waypoints) <= 64:
        raise ValueError("Expected 1..64 waypoints")
    previous = {arm: observation["arms"][arm]["link6_pose_xyzquat"] for arm in ARMS}
    for point in waypoints:
        keys = set(point)
        required = {"left", "right", "steps"}
        if keys != required and not (early_arrival and keys == required | {"mode"}):
            raise ValueError("Explicit left/right/steps required; mode requires early_arrival")
        if "mode" in point and point["mode"] not in ("move", "hold"):
            raise ValueError("Waypoint mode must be move or hold")
        if type(point["steps"]) is not int or not 1 <= point["steps"] <= 5:
            raise ValueError("Each waypoint permits 1..5 native controls")
        for arm in ARMS:
            target = point[arm]
            if set(target) != {"pose_xyzquat", "gripper"}:
                raise ValueError("Explicit pose_xyzquat/gripper required")
            t = pose_matrix(target["pose_xyzquat"])
            old = pose_matrix(previous[arm])
            if np.linalg.norm(t[:3, 3] - old[:3, 3]) > MAX_POSITION_DELTA + 1e-9:
                raise ValueError("EEF segment exceeds 5 cm")
            if (
                Rotation.from_matrix(t[:3, :3] @ old[:3, :3].T).magnitude()
                > MAX_ROTATION_DELTA + 1e-9
            ):
                raise ValueError("EEF segment exceeds 0.35 rad")
            grip = target["gripper"]
            if (
                isinstance(grip, bool)
                or not isinstance(grip, (float, int))
                or not np.isfinite(grip)
                or not 0 <= grip <= 1
            ):
                raise ValueError("Gripper must be finite normalized opening [0,1]")
            previous[arm] = target["pose_xyzquat"]
    return waypoints


def native_targets(point):
    return {
        arm: {
            "position": point[arm]["pose_xyzquat"][:3],
            "quaternion_wxyz": [point[arm]["pose_xyzquat"][i] for i in (6, 3, 4, 5)],
            "gripper_closed": point[arm]["gripper"] < 0.5,
            "gripper_opening": point[arm]["gripper"],
        }
        for arm in ARMS
    }


def measured_tracking(point, observation):
    """Post-ACK robot-only feedback; IK proposals are not arrival evidence."""
    result = {}
    for arm in ARMS:
        target = pose_matrix(point[arm]["pose_xyzquat"])
        measured = pose_matrix(observation["arms"][arm]["link6_pose_xyzquat"])
        result[arm] = {
            "target_position_error_m": float(
                np.linalg.norm(target[:3, 3] - measured[:3, 3])
            ),
            "target_rotation_error_rad": float(
                Rotation.from_matrix(target[:3, :3] @ measured[:3, :3].T).magnitude()
            ),
            "requested_gripper": point[arm]["gripper"],
            "control_target_gripper": observation["arms"][arm]["gripper"],
            # Older observations have only the target proxy: never label it measured.
            "measured_gripper": observation["arms"][arm].get("measured_gripper"),
            "measured_after_ack": observation["step_id"],
        }
    return result
