"""Normalize RoboDojo frames to the observation/action contract of GPT-Policy."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

import numpy as np

from .calibration import RoboDojoCalibration, matrix_to_quat_xyzw, quat_wxyz_to_matrix


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def _pose_wxyz_to_xyzw(pose: Any, base_from_world: np.ndarray | None = None) -> list[float]:
    values = np.asarray(pose, dtype=np.float64).reshape(7)
    transform = np.eye(4)
    transform[:3, :3] = quat_wxyz_to_matrix(values[3:])
    transform[:3, 3] = values[:3]
    if base_from_world is not None:
        transform = base_from_world @ transform
    result = transform[:3, 3].tolist()
    result.extend(matrix_to_quat_xyzw(transform[:3, :3]))
    return result


class RoboDojoAdapter:
    """Bridge one RoboDojo environment frame to GPT-Policy.

    The adapter accepts the dictionary returned by ``ObsManager.get_obs`` for
    one environment. It never drops simulator data: the original frame is
    retained under ``simulator_raw`` for recordings and diagnostics, while the
    normalized state follows the same keys consumed by the real hardware loop.
    """

    def __init__(
        self,
        calibration: RoboDojoCalibration,
        arms: tuple[str, ...] = ("left", "right"),
        robot_model: str = "X5",
    ) -> None:
        if not arms:
            raise ValueError("at least one arm is required")
        self.calibration = calibration
        self.arms = arms
        self.robot_model = robot_model

    def state(self, frame: Mapping[str, Any]) -> dict[str, Any]:
        source = frame.get("state", {})
        arm_states: dict[str, dict[str, Any]] = {}
        for arm in self.arms:
            prefix = f"{arm}_" if len(self.arms) > 1 else ""
            joint_key = f"{prefix}arm_joint_state"
            gripper_key = f"{prefix}ee_joint_state"
            pose_key = f"{prefix}ee_pose"
            pose = source.get(pose_key)
            if pose is None:
                raise ValueError(f"RoboDojo frame is missing state.{pose_key}")
            joints = _jsonable(source.get(joint_key, []))
            grip = _jsonable(source.get(gripper_key, [0.0]))
            base_from_world = None
            if arm in self.calibration.world_from_base:
                base_from_world = self.calibration.base_from_world(arm)
            arm_states[arm] = {
                "joint_positions_rad": joints,
                "joint_velocities_rad_s": _jsonable(source.get(f"{prefix}arm_joint_velocity", [])),
                "joint_torques_nm": _jsonable(source.get(f"{prefix}arm_joint_torque", [])),
                "tcp_xyzrpy": None,
                "tcp_xyzquat": _pose_wxyz_to_xyzw(pose, base_from_world),
                # ObsManager exposes ee_joint_state in the normalized action
                # convention, not metres. Keep metres explicitly unavailable
                # instead of relabelling a normalized value as hardware width.
                "gripper_position_m": None,
                "gripper_normalized": self._normalized_gripper(grip),
                "gripper_command_normalized": _jsonable(frame.get("action", {}).get(gripper_key)),
                "gripper_velocity_m_s": _jsonable(source.get(f"{prefix}ee_joint_velocity", [])),
                "gripper_torque_nm": _jsonable(source.get(f"{prefix}ee_joint_torque", [])),
                "gravity_compensation": None,
                "source_pose_frame": "RoboDojo relative world/environment frame",
                "source_quaternion_order": "wxyz",
            }
        raw_without_vision = deepcopy(dict(frame))
        raw_without_vision.pop("vision", None)
        result: dict[str, Any] = {
            "arms": arm_states,
            "backend": "robodojo",
            "robot_model": self.robot_model,
            "interface": "robodojo",
            "interfaces": {arm: "robodojo" for arm in self.arms},
            "calibration": self.calibration.context_manifest(),
            "simulator_raw": _jsonable(raw_without_vision),
        }
        if len(self.arms) == 1:
            result.update(arm_states[self.arms[0]])
        return result

    @staticmethod
    def _normalized_gripper(value: Any) -> float | None:
        arr = np.asarray(value, dtype=np.float64).reshape(-1)
        if not len(arr) or not np.isfinite(arr[0]):
            return None
        # RoboDojo's gripper action is normalized by the env client. For a raw
        # joint observation we preserve the raw value and only expose a
        # normalized value when the caller has supplied one explicitly.
        return float(arr[0])

    def cameras(self, frame: Mapping[str, Any]) -> list[dict[str, Any]]:
        result = []
        for name, data in (frame.get("vision") or {}).items():
            if not isinstance(data, Mapping):
                continue
            item = {"name": name, "device": "robodojo"}
            item.update(_jsonable(dict(data)))
            if name in self.calibration.intrinsics and "intrinsic_matrix" not in item:
                item["intrinsic_matrix"] = self.calibration.intrinsics[name].tolist()
            if name in self.calibration.camera_extrinsics_world and "extrinsic_matrix" not in item:
                item["extrinsic_matrix"] = self.calibration.camera_extrinsics_world[name].tolist()
            result.append(item)
        return result

    def observation_metadata(self, frame: Mapping[str, Any], env_step: int) -> dict[str, Any]:
        return {
            "backend": "robodojo",
            "env_step": int(env_step),
            "task_instruction": frame.get("instruction"),
            "success": frame.get("success"),
            "end_flag": frame.get("end_flag"),
            "calibration": self.calibration.context_manifest(),
            "raw_observation_fields": sorted(frame.keys()),
        }

    def action(self, action: Mapping[str, Any]) -> dict[str, Any]:
        """Convert GPT-Policy actions to RoboDojo field names and ``wxyz``."""
        output: dict[str, Any] = {}
        for arm in self.arms:
            prefix = f"{arm}_" if len(self.arms) > 1 else ""
            target = action.get("target")
            source_pose = target.get(arm) if isinstance(target, Mapping) and len(self.arms) > 1 else target
            if source_pose is None and "poses" in action:
                points = action["poses"]
                if points:
                    point = points[-1]
                    source_pose = point.get(arm) if isinstance(point, Mapping) else point
            if isinstance(source_pose, Mapping):
                source_pose = source_pose.get("pose_xyzquat", source_pose.get("tcp_pose_xyzquat"))
            if source_pose is not None:
                values = np.asarray(source_pose, dtype=np.float64).reshape(7)
                q_xyzw = values[3:] / np.linalg.norm(values[3:])
                transform = np.eye(4)
                transform[:3, :3] = quat_wxyz_to_matrix([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])
                transform[:3, 3] = values[:3]
                if arm in self.calibration.world_from_base:
                    transform = self.calibration.world_from_base[arm] @ transform
                q = matrix_to_quat_xyzw(transform[:3, :3])
                output[f"{prefix}ee_pose"] = np.r_[transform[:3, 3], q[3], q[:3]].tolist()
            grip_key = f"{prefix}ee_joint_state"
            if grip_key in action:
                output[grip_key] = _jsonable(action[grip_key])
            elif "positions" in action and isinstance(action["positions"], Mapping) and arm in action["positions"]:
                # In the bimanual tool schema, null explicitly means hold
                # this side's current gripper reference. Do not coerce it to
                # float(None); the model layer fills the measured state for
                # omitted targets before the action reaches RoboDojo.
                position = action["positions"][arm]
                if position is not None:
                    output[grip_key] = [float(position)]
            elif len(self.arms) == 1 and "gripper" in action:
                position = action["gripper"]
                if position is not None:
                    output[grip_key] = [float(position)]
        if not output:
            raise ValueError("GPT-Policy action contains no RoboDojo pose or gripper target")
        return output
