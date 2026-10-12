"""Optional target timing within an existing waypoint budget; no object-state input."""

import copy

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from .control import ARMS


def waypoint_targets(point, observation, previous_grippers, *, enabled=False):
    """Return one commanded pose per existing control, preserving the exact endpoint."""
    arms = []
    if (enabled and point.get("mode") == "move" and point["steps"] > 1
            and previous_grippers is not None
            and all(point[a]["gripper"] == previous_grippers[a] for a in ARMS)):
        arms = [a for a in ARMS if point[a]["gripper"] < 0.5]
    if not arms:
        return [point] * point["steps"], arms
    starts = {a: np.asarray(observation["arms"][a]["link6_pose_xyzquat"], float) for a in arms}
    ends = {a: np.asarray(point[a]["pose_xyzquat"], float) for a in arms}
    rotations = {a: Slerp([0, 1], Rotation.from_quat([starts[a][3:], ends[a][3:]])) for a in arms}
    schedule = []
    for fraction in np.linspace(0, 1, point["steps"] + 1)[1:]:
        target = copy.deepcopy(point)
        for arm in arms:
            target[arm]["pose_xyzquat"] = [
                *((1 - fraction) * starts[arm][:3] + fraction * ends[arm][:3]).tolist(),
                *rotations[arm](fraction).as_quat().tolist(),
            ]
        schedule.append(target)
    schedule[-1] = copy.deepcopy(point)
    return schedule, arms
