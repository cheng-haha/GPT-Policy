"""Measured-state, bounded DLS control for RoboDojo's dual X5 arms.

The numerical FK/Jacobian and limits follow GPT-as-Policy's published
``robodojo_server/kinematics.py`` (commit 8f3d362, MIT). This adapter keeps
GPT-Policy's EEF action interface and submits native RoboDojo joint actions.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation


def _transform(pose):
    value = np.asarray(pose, dtype=float).reshape(7)
    if not np.isfinite(value).all() or abs(np.linalg.norm(value[3:]) - 1) > 1e-3:
        raise ValueError("DLS requires a finite xyz/wxyz pose with a unit quaternion")
    result = np.eye(4)
    result[:3, :3] = Rotation.from_quat(value[[4, 5, 6, 3]]).as_matrix()
    result[:3, 3] = value[:3]
    return result


def _angle(a, b):
    return float(Rotation.from_matrix(a[:3, :3] @ b[:3, :3].T).magnitude())


class _ArmFK:
    def __init__(self, urdf, joint_names, base, tip):
        edges = {item.find("child").get("link"): item
                 for item in ET.parse(urdf).getroot().findall("joint")}
        self.names = list(joint_names)
        self.chain = []
        while tip != base:
            joint = edges[tip]
            origin = joint.find("origin")
            xyz = np.fromstring(origin.get("xyz", "0 0 0"), sep=" ") if origin is not None else np.zeros(3)
            rpy = np.fromstring(origin.get("rpy", "0 0 0"), sep=" ") if origin is not None else np.zeros(3)
            matrix = np.eye(4)
            matrix[:3, :3], matrix[:3, 3] = Rotation.from_euler("xyz", rpy).as_matrix(), xyz
            axis = joint.find("axis")
            axis = np.fromstring(axis.get("xyz"), sep=" ") if axis is not None else np.array([1., 0., 0.])
            self.chain.insert(0, (joint.get("name"), joint.get("type"), matrix, axis))
            tip = joint.find("parent").get("link")
        moving = [name for name, kind, _, _ in self.chain if kind != "fixed"]
        if set(moving) != set(self.names):
            raise ValueError(f"DLS URDF joint mismatch: {moving} != {self.names}")

    def matrix(self, q):
        values = dict(zip(self.names, np.asarray(q, dtype=float)))
        if len(values) != len(self.names) or not np.isfinite(list(values.values())).all():
            raise ValueError("Invalid measured DLS arm joints")
        result = np.eye(4)
        for name, kind, origin, axis in self.chain:
            motion = np.eye(4)
            if kind in ("revolute", "continuous"):
                motion[:3, :3] = Rotation.from_rotvec(axis * values[name]).as_matrix()
            elif kind != "fixed":
                raise ValueError(f"Unsupported URDF joint type {kind}")
            result = result @ origin @ motion
        return result

    def bounded_target(self, q, limits, root, goal):
        now = root @ self.matrix(q)
        translation = goal[:3, 3] - now[:3, 3]
        rotation = Rotation.from_matrix(goal[:3, :3] @ now[:3, :3].T).as_rotvec()
        def bounded(vector, cap):
            return vector * min(1., cap / max(np.linalg.norm(vector), 1e-12))
        error = np.r_[bounded(translation, .02), bounded(rotation, .1)]
        jacobian = np.zeros((6, len(q)))
        for index in range(len(q)):
            shifted = q.copy()
            shifted[index] += 1e-5
            next_pose = root @ self.matrix(shifted)
            jacobian[:3, index] = (next_pose[:3, 3] - now[:3, 3]) / 1e-5
            jacobian[3:, index] = Rotation.from_matrix(
                next_pose[:3, :3] @ now[:3, :3].T).as_rotvec() / 1e-5
        delta = jacobian.T @ np.linalg.solve(
            jacobian @ jacobian.T + .05**2 * np.eye(6), error)
        low = np.maximum(limits[:, 0], q - .05)
        high = np.minimum(limits[:, 1], q + .05)
        if np.any(low > high):
            raise ValueError("Measured joint outside DLS bounded interval")
        return np.clip(q + delta, low, high).astype(np.float32)


class DualX5DLS:
    """Convert one dual-arm native EE target to up to five native joint actions."""

    def __init__(self, env):
        self.manager = env.robot_manager
        self.robots = {robot.arm_name.split("_")[0]: robot
                       for robot in self.manager.robot_list if robot.type == "target"}
        if set(self.robots) != {"left", "right"}:
            raise ValueError("Published DLS controller requires left/right X5 target arms")
        self.fk = {arm: _ArmFK(robot.urdf_path, robot.arm_joints_name,
                              robot.base_link, robot.ee_link_name)
                   for arm, robot in self.robots.items()}
        self.verify_fk()

    def _root(self, arm):
        robot = self.robots[arm]
        return _transform(self.manager.get_link_pose(robot, robot.base_link, is_relative=True)[0])

    def _measured(self, arm):
        robot = self.robots[arm]
        q = np.asarray(self.manager.get_joint(robot)[0], dtype=float)
        actual = _transform(self.manager.get_real_endpose(robot)[0])
        return q, actual

    def verify_fk(self):
        report = {}
        for arm in ("left", "right"):
            q, actual = self._measured(arm)
            predicted = self._root(arm) @ self.fk[arm].matrix(q)
            position_error = float(np.linalg.norm(predicted[:3, 3] - actual[:3, 3]))
            angle_error = _angle(predicted, actual)
            report[arm] = {"position_error_m": position_error,
                           "rotation_error_rad": angle_error}
            if position_error >= .002 or angle_error >= .01:
                raise ValueError(f"DLS FK does not match measured {arm} link6: {report[arm]}")
        return report

    def _joint_action(self, targets, grippers):
        action = {}
        for arm in ("left", "right"):
            robot = self.robots[arm]
            q, _ = self._measured(arm)
            index = self.manager.robot_list.index(robot)
            limits = self.manager.robot_key[index].data.soft_joint_pos_limits[
                0, robot.arm_joint_indices].cpu().numpy()
            next_q = self.fk[arm].bounded_target(q, limits, self._root(arm), targets[arm])
            action[self.manager.process_name(robot.arm_name)] = next_q.tolist()
            action[f"{arm}_ee_joint_state"] = [grippers[arm]]
        return action

    def execute(self, env, action, steps=5):
        if not 1 <= steps <= 5:
            raise ValueError("DLS control steps must be in [1, 5]")
        targets, grippers = {}, {}
        motion = False
        for arm in ("left", "right"):
            target = _transform(action[f"{arm}_ee_pose"])
            _, actual = self._measured(arm)
            distance = float(np.linalg.norm(target[:3, 3] - actual[:3, 3]))
            angle = _angle(target, actual)
            if distance > .050001 or angle > .350001:
                raise ValueError(f"{arm} DLS target exceeds 5 cm / 0.35 rad: {distance:.4f} m, {angle:.4f} rad")
            targets[arm] = target
            motion |= distance > .001 or angle > .005
            opening = float(np.asarray(action[f"{arm}_ee_joint_state"]).reshape(-1)[0])
            if not np.isfinite(opening) or not 0 <= opening <= 1:
                raise ValueError(f"Invalid {arm} gripper opening")
            grippers[arm] = opening
        self.verify_fk()
        executed = 0
        for _ in range(steps if motion else 1):
            env.take_action(self._joint_action(targets, grippers))
            executed += 1
            if env.is_episode_end():
                break
        feedback = {"controller": "robot_only_numerical_jacobian_dls",
                    "native_control_steps": executed, "arms": {}}
        for arm in ("left", "right"):
            _, actual = self._measured(arm)
            feedback["arms"][arm] = {
                "translation_error_m": float(np.linalg.norm(targets[arm][:3, 3] - actual[:3, 3])),
                "rotation_error_rad": _angle(targets[arm], actual),
            }
        return feedback
