"""Opt-in waypoint completion from consecutive measured robot ACKs only."""

import numpy as np
from scipy.spatial.transform import Rotation

from .control import ARMS, measured_tracking
from .transforms import pose_matrix

ARRIVAL_LIMITS = {
    "position_m": 0.003,
    "rotation_rad": 0.03,
    "linear_speed_m_s": 0.025,
    "angular_speed_rad_s": 0.25,
    "joint_speed_rad_s": 0.125,
    "gripper_speed_s": 0.1,
    "consecutive_samples": 2,
}


def early_arrival_enabled(config):
    enabled = config.get("early_arrival", False)
    if type(enabled) is not bool:
        raise ValueError("early_arrival must be a boolean")
    return enabled


def stop_on_arrival_enabled(config):
    """Executor-only ablation; leaves the move/hold prompt and feedback schema intact."""
    enabled = config.get("stop_on_arrival", True)
    if type(enabled) is not bool:
        raise ValueError("stop_on_arrival must be a boolean")
    return enabled


def pass_through_enabled(config):
    enabled = config.get("pass_through", False)
    if type(enabled) is not bool:
        raise ValueError("pass_through must be a boolean")
    if enabled and not (early_arrival_enabled(config) and stop_on_arrival_enabled(config)):
        raise ValueError("pass_through requires early_arrival and stop_on_arrival")
    return enabled


class ArrivalCheck:
    """A new check per waypoint; no stability carried across target changes."""

    def __init__(self, point, observation, frequency):
        if not np.isfinite(frequency) or frequency <= 0:
            raise ValueError("Positive native control frequency required")
        self.point, self.previous, self.frequency = point, observation, frequency
        self.stable_samples = 0
        self.feedback = {}

    @property
    def pose_arrived(self):
        """Both arms reached the existing pose tolerance after an actual ACK."""
        return bool(self.feedback) and all(
            self.feedback[arm]["target_position_error_m"] <= ARRIVAL_LIMITS["position_m"]
            and self.feedback[arm]["target_rotation_error_rad"] <= ARRIVAL_LIMITS["rotation_rad"]
            for arm in ARMS
        )

    def update(self, observation):
        if observation["step_id"] != self.previous["step_id"] + 1:
            raise ValueError("Arrival requires consecutive native ACK observations")
        tracking = measured_tracking(self.point, observation)
        qualified = True
        for arm in ARMS:
            old, now = self.previous["arms"][arm], observation["arms"][arm]
            old_pose = pose_matrix(old["link6_pose_xyzquat"])
            now_pose = pose_matrix(now["link6_pose_xyzquat"])
            speed = {
                "linear_speed_m_s": float(np.linalg.norm(
                    now_pose[:3, 3] - old_pose[:3, 3]) * self.frequency),
                "angular_speed_rad_s": float(Rotation.from_matrix(
                    now_pose[:3, :3] @ old_pose[:3, :3].T).magnitude() * self.frequency),
                "joint_speed_rad_s": float(np.max(np.abs(
                    np.asarray(now["joint_pos"]) - old["joint_pos"])) * self.frequency),
                # Deliberately preserve the existing target-proxy gate in this
                # feedback-only change; physical aperture is a separate observation.
                "gripper_speed_s": abs(now["gripper"] - old["gripper"]) * self.frequency,
            }
            error = tracking[arm]
            settled = (
                error["target_position_error_m"] <= ARRIVAL_LIMITS["position_m"]
                and error["target_rotation_error_rad"] <= ARRIVAL_LIMITS["rotation_rad"]
                and all(np.isfinite(v) and v <= ARRIVAL_LIMITS[k] for k, v in speed.items())
            )
            self.feedback[arm] = {
                **error, **speed, "settled": bool(settled),
                "gripper_speed_source": "native_control_target_proxy",
            }
            qualified &= settled
        self.stable_samples = self.stable_samples + 1 if qualified else 0
        self.previous = observation
        return self.stable_samples >= ARRIVAL_LIMITS["consecutive_samples"]
