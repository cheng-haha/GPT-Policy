"""Read robot finger joints without changing control targets or physics."""

import numpy as np


def gripper_measurement(manager, robot):
    """Normalized first-finger position, not a grasp or force measurement."""
    values = np.asarray(
        manager.get_end_effector_real_val(robot, env_idx_list=[0])[0], dtype=float
    ).reshape(-1)
    bounds = np.asarray(robot.gripper_scale, dtype=float)
    if not values.size or not np.isfinite(values).all():
        raise ValueError("Physical gripper feedback unavailable or nonfinite")
    if bounds.shape != (2,) or not np.isfinite(bounds).all() or bounds[1] <= bounds[0]:
        raise ValueError("Invalid physical gripper range")
    sign = robot.gripper_move.get("sign", 1)
    if sign not in (-1, 1):
        raise ValueError("Invalid physical gripper direction")
    normalized = ((values[0] - bounds[0]) if sign == 1 else (bounds[1] - values[0])) / (
        bounds[1] - bounds[0]
    )
    return {
        "measured_gripper": float(np.clip(normalized, 0.0, 1.0)),
        "measured_gripper_unclipped": float(normalized),
        "gripper_joint_positions": values.tolist(),
    }
