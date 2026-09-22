"""Completion semantics for discrete RoboDojo pose commands."""
from __future__ import annotations

from copy import deepcopy

import numpy as np


def interpolate_control(env, control_info, env_idx):
    """Build a rest-to-rest joint trajectory sized by motion, not camera FPS."""
    cfg = env.eval_cfg["action_execution"]
    speed = float(cfg.get("joint_speed_rad_s", 1.0))
    acceleration = float(cfg.get("joint_acceleration_rad_s2", 4.0))
    if speed <= 0 or acceleration <= 0:
        raise ValueError("action execution speed and acceleration must be positive")
    starts = {}
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
        controls.append(item)
    return controls


def settle_control(env, targets):
    """Keep the final targets applied until measured joints have settled.

    The timeout is reported as incomplete execution; it never marks arrival.
    Only simulation steps advance here; observation capture follows completion.
    """
    cfg = env.eval_cfg["action_execution"]
    tolerance = float(cfg.get("joint_tolerance_rad", 0.001))
    velocity_tolerance = float(cfg.get("joint_velocity_tolerance_rad_s", 0.01))
    steps = int(np.ceil(float(cfg.get("settle_timeout_s", 2.0)) / env.dt))
    stable = {idx: 0 for idx in targets}
    result = {}
    for step in range(steps + 1):
        for idx, control in targets.items():
            error = speed = 0.0
            for robot, entity in zip(env.robot_manager.robot_list, env.robot_manager.robot_key):
                key = env.robot_manager.process_name(robot.arm_name)
                if robot.type != "target" or key not in control:
                    continue
                measured = env.robot_manager.get_joint(robot, [idx])[idx]
                error = max(error, float(np.max(np.abs(np.asarray(control[key]["position"]) - measured))))
                velocities = entity.data.joint_vel[idx, robot.arm_joint_indices].detach().cpu().numpy()
                speed = max(speed, float(np.max(np.abs(velocities))))
            stable[idx] = stable[idx] + 1 if error <= tolerance and speed <= velocity_tolerance else 0
            result[idx] = {"settled": stable[idx] >= 3, "joint_error_rad": error,
                           "joint_velocity_rad_s": speed, "settle_time_s": step * env.dt}
        if all(item["settled"] for item in result.values()) or step == steps:
            return result
        env.robot_manager.control_manager.push(list(targets), [[deepcopy(control)] for control in targets.values()])
        env.step(env_idx_list=list(targets))
        if env.physx_monitor_enabled:
            env._check_physx_broken_envs()
            env._check_endpose_finite(list(targets))
