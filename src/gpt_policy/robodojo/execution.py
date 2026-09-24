"""Completion semantics for discrete RoboDojo pose commands."""
from __future__ import annotations

from copy import deepcopy

import numpy as np


def _mimic_coefficients(robot):
    mimic = robot.gripper_move["mimic"]
    if mimic and isinstance(mimic[0], (list, tuple)) and len(mimic[0]) >= 3:
        mimic = mimic[0]
    return float(mimic[1]), float(mimic[2])


def interpolate_control(env, control_info, env_idx):
    """Build a rest-to-rest joint trajectory sized by motion, not camera FPS."""
    cfg = env.eval_cfg["action_execution"]
    speed = float(cfg.get("joint_speed_rad_s", 1.0))
    acceleration = float(cfg.get("joint_acceleration_rad_s2", 4.0))
    if speed <= 0 or acceleration <= 0:
        raise ValueError("action execution speed and acceleration must be positive")
    starts = {}
    grippers = {}
    duration = float(cfg.get("min_duration_s", 0.2))
    for robot in env.robot_manager.robot_list:
        if robot.type != "target":
            continue
        key = env.robot_manager.process_name(robot.arm_name)
        if key in control_info:
            start = np.asarray(env.robot_manager.get_joint(robot, [env_idx])[env_idx])
            delta = np.asarray(control_info[key]["position"]) - start
            starts[key] = (start, delta)
            # Cubic smoothstep: max s'=1.5, max |s''|=6.
            distance = float(np.max(np.abs(delta)))
            duration = max(duration, 1.5 * distance / speed, np.sqrt(6 * distance / acceleration))
        gripper_name = getattr(robot, "gripper_name", None)
        grip_key = env.robot_manager.process_name(gripper_name) if gripper_name else None
        if grip_key in control_info and getattr(robot, "ee_type", None) == "gripper":
            current = env.robot_manager.get_end_effector_real_val(robot, [env_idx])[env_idx]
            target = np.asarray(control_info[grip_key]["position"], dtype=np.float64).reshape(-1)
            if current is not None and target.size:
                start = float(np.asarray(current, dtype=np.float64).reshape(-1)[0])
                end = float(target[0])
                grippers[grip_key] = (robot, start, end)
    count = max(1, int(np.ceil(duration / env.dt)))
    duration = count * env.dt
    controls = []
    for index in range(1, count + 1):
        fraction = index / count
        blend = fraction * fraction * (3 - 2 * fraction)
        velocity = 6 * fraction * (1 - fraction) / duration
        item = deepcopy(control_info)
        for key, (start, delta) in starts.items():
            item[key] = {"position": (start + blend * delta).tolist(),
                         "velocity": (velocity * delta).tolist()}
        for key, (robot, start, end) in grippers.items():
            scale = robot.gripper_scale
            position = float(np.clip(start + blend * (end - start), scale[0], scale[1]))
            multiplier, offset = _mimic_coefficients(robot)
            item[key] = {
                "position": [position, position * multiplier + offset],
                "velocity": [0.0, 0.0],
            }
        controls.append(item)
    return controls


def settle_control(env, targets):
    """Keep the final targets applied until measured joints have settled.

    If the timeout elapses, return the unsettled state to the policy instead of
    ending the episode. Only simulation steps advance here; observation capture
    follows completion or timeout.
    """
    cfg = env.eval_cfg["action_execution"]
    tolerance = float(cfg.get("joint_tolerance_rad", 0.001))
    velocity_tolerance = float(cfg.get("joint_velocity_tolerance_rad_s", 0.01))
    gripper_tolerance = float(cfg.get("gripper_tolerance", tolerance))
    gripper_velocity_tolerance = float(
        cfg.get("gripper_velocity_tolerance", velocity_tolerance)
    )
    timeout_s = float(cfg.get("settle_timeout_s", 6.0))
    if timeout_s <= 0:
        raise ValueError("settle_timeout_s must be positive")
    steps = max(1, int(np.ceil(timeout_s / env.dt)))
    stable = {idx: 0 for idx in targets}
    result = {}
    for step in range(steps + 1):
        for idx, control in targets.items():
            error = speed = 0.0
            gripper_error = gripper_speed = 0.0
            for robot, entity in zip(env.robot_manager.robot_list, env.robot_manager.robot_key):
                key = env.robot_manager.process_name(robot.arm_name)
                if robot.type != "target":
                    continue
                if key in control:
                    measured = env.robot_manager.get_joint(robot, [idx])[idx]
                    error = max(error, float(np.max(np.abs(np.asarray(control[key]["position"]) - measured))))
                    velocities = entity.data.joint_vel[idx, robot.arm_joint_indices].detach().cpu().numpy()
                    speed = max(speed, float(np.max(np.abs(velocities))))
                gripper_name = getattr(robot, "gripper_name", None)
                grip_key = env.robot_manager.process_name(gripper_name) if gripper_name else None
                if grip_key in control and getattr(robot, "ee_type", None) == "gripper":
                    measured_gripper = env.robot_manager.get_end_effector_real_val(
                        robot, [idx]
                    )[idx]
                    target_gripper = np.asarray(
                        control[grip_key]["position"], dtype=np.float64
                    ).reshape(-1)
                    measured_gripper = np.asarray(measured_gripper, dtype=np.float64).reshape(-1)
                    if target_gripper.size and measured_gripper.size:
                        count = min(target_gripper.size, measured_gripper.size)
                        gripper_error = max(
                            gripper_error,
                            float(np.max(np.abs(target_gripper[:count] - measured_gripper[:count]))),
                        )
                    gripper_velocities = entity.data.joint_vel[
                        idx, robot.gripper_joint_indices
                    ].detach().cpu().numpy()
                    if gripper_velocities.size:
                        gripper_speed = max(
                            gripper_speed, float(np.max(np.abs(gripper_velocities)))
                        )
            is_stable = (
                error <= tolerance
                and speed <= velocity_tolerance
                and gripper_error <= gripper_tolerance
                and gripper_speed <= gripper_velocity_tolerance
            )
            stable[idx] = stable[idx] + 1 if is_stable else 0
            result[idx] = {
                "settled": stable[idx] >= 3,
                "joint_error_rad": error,
                "joint_velocity_rad_s": speed,
                "gripper_error": gripper_error,
                "gripper_velocity": gripper_speed,
                "settle_time_s": step * env.dt,
            }
        if all(item["settled"] for item in result.values()):
            return result
        if step == steps:
            print(f"[action_execution] waiting_for_settle: {result}")
            return result
        env.robot_manager.control_manager.push(list(targets), [[deepcopy(control)] for control in targets.values()])
        env.step(env_idx_list=list(targets))
        if env.physx_monitor_enabled:
            env._check_physx_broken_envs()
            env._check_endpose_finite(list(targets))
