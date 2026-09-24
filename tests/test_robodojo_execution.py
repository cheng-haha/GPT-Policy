"""Regression checks for bounded RoboDojo action completion."""

from types import SimpleNamespace

import numpy as np
import pytest

from gpt_policy.robodojo.deploy import _execute_native_actions
from gpt_policy.robodojo.execution import interpolate_control, settle_control
from third_party.RoboDojo.env.robot_manager.control_manager import MetaControl


class _Velocity:
    def __getitem__(self, key):
        return self

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return np.zeros(1)


def _env(timeout, settle_after_steps=None):
    env = SimpleNamespace(dt=0.1, steps=0)
    robot = SimpleNamespace(type="target", arm_name="left_arm", arm_joint_indices=[0])

    def get_joint(robot, indices):
        if settle_after_steps is not None and env.steps >= settle_after_steps:
            return {0: np.ones(1)}
        return {0: np.zeros(1)}

    manager = SimpleNamespace(
        robot_list=[robot],
        robot_key=[SimpleNamespace(data=SimpleNamespace(joint_vel=_Velocity()))],
        process_name=lambda name: name,
        get_joint=get_joint,
        control_manager=SimpleNamespace(push=lambda *args: None),
    )
    env.eval_cfg = {"action_execution": {"settle_timeout_s": timeout}}
    env.robot_manager = manager
    env.physx_monitor_enabled = False

    def step(*, env_idx_list):
        env.steps += 1

    env.step = step
    return env


def test_interpolate_control_maps_gripper_to_real_joint_scale():
    robot = SimpleNamespace(
        type="target",
        ee_type="gripper",
        arm_name="left_arm",
        gripper_name="left_ee",
        gripper_scale=[-0.01, 0.044],
        gripper_move={"mimic": [[], 1.0, 0.0]},
    )
    manager = SimpleNamespace(
        robot_list=[robot],
        process_name=lambda name: f"{name}_joint_state",
        get_joint=lambda robot, indices: {0: np.zeros(1)},
        get_end_effector_real_val=lambda robot, env_idx_list: {0: [0.044, 0.044]},
    )
    env = SimpleNamespace(
        dt=0.1,
        eval_cfg={"action_execution": {"min_duration_s": 0.2}},
        robot_manager=manager,
    )

    controls = interpolate_control(env, {"left_ee_joint_state": {"position": [-0.01, -0.01]}}, 0)

    assert controls[0]["left_ee_joint_state"]["position"] == pytest.approx([0.017, 0.017])
    assert controls[-1]["left_ee_joint_state"]["position"] == pytest.approx([-0.01, -0.01])


def test_metacontrol_does_not_apply_hidden_gripper_slew_limit():
    robot = SimpleNamespace(
        ee_type="gripper",
        gripper_name="left_ee",
        gripper_move={"mimic": [[], 1.0, 0.0]},
    )
    manager = SimpleNamespace(
        get_robot_obs_name=lambda: ["left_ee_joint_state"],
        restore_name=lambda name: name,
        get_robot_by_gripper_name=lambda name: robot,
    )
    control = MetaControl({"left_ee_joint_state": {"position": [1.0, 1.0]}})
    result = control.get_action(manager, 0)
    assert result["left_ee_joint_state"]["position"] == pytest.approx([1.0, 1.0])


def test_settle_control_returns_unsettled_state_after_timeout():
    env = _env(0.3, settle_after_steps=4)
    result = settle_control(env, {0: {"left_arm": {"position": [1.0]}}})
    assert result[0]["settled"] is False
    assert result[0]["joint_error_rad"] == 1.0
    assert result[0]["settle_time_s"] == pytest.approx(0.3)
    assert env.steps == 3


def test_settle_control_waits_for_gripper_position_and_velocity():
    env = SimpleNamespace(dt=0.1, steps=0)
    robot = SimpleNamespace(
        type="target", arm_name="left_arm", arm_joint_indices=[0],
        ee_type="gripper", gripper_name="left_ee", gripper_joint_indices=[1],
    )

    class _JointVel:
        def __getitem__(self, key):
            return self

        def detach(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return np.zeros(1)

    entity = SimpleNamespace(data=SimpleNamespace(joint_vel=_JointVel()))
    manager = SimpleNamespace(
        robot_list=[robot],
        robot_key=[entity],
        process_name=lambda name: name,
        get_joint=lambda robot, indices: {0: np.ones(1)},
        get_end_effector_real_val=lambda robot, indices: {0: np.zeros(1)},
        control_manager=SimpleNamespace(push=lambda *args: None),
    )
    env.eval_cfg = {"action_execution": {"settle_timeout_s": 0.3}}
    env.robot_manager = manager
    env.physx_monitor_enabled = False
    env.step = lambda *, env_idx_list: setattr(env, "steps", env.steps + 1)

    result = settle_control(
        env,
        {0: {
            "left_arm": {"position": [1.0]},
            "left_ee": {"position": [1.0, 1.0]},
        }},
    )
    assert result[0]["settled"] is False
    assert result[0]["gripper_error"] == pytest.approx(1.0)
    assert result[0]["gripper_velocity"] == pytest.approx(0.0)


def test_settle_control_returns_when_settled_before_timeout():
    env = _env(0.3, settle_after_steps=1)
    result = settle_control(env, {0: {"left_arm": {"position": [1.0]}}})
    assert result[0]["settled"] is True


def test_nonpositive_timeout_is_rejected():
    with pytest.raises(ValueError, match="settle_timeout_s"):
        settle_control(_env(0), {0: {"left_arm": {"position": [1.0]}}})


def test_native_chunk_settles_only_after_final_waypoint():
    waits = []
    cfg = {"wait_until_settled": True, "settle_after_trajectory": True}
    env = SimpleNamespace(
        eval_cfg={"action_execution": cfg},
        _last_ik_feedback=[{"execution": {"settled": True}}],
        is_episode_end=lambda: False,
    )

    def take_action(action):
        waits.append((action, cfg["wait_until_settled"]))

    env.take_action = take_action
    assert _execute_native_actions(env, ["first", "second", "final"]) is None
    assert waits == [("first", False), ("second", False), ("final", True)]
    assert cfg["wait_until_settled"] is True


def test_native_unsettled_feedback_no_longer_blocks_post_action_observation():
    cfg = {"wait_until_settled": True, "settle_after_trajectory": True}
    env = SimpleNamespace(
        eval_cfg={"action_execution": cfg},
        _last_ik_feedback=[{"execution": {
            "settled": False,
            "joint_error_rad": 0.02,
            "joint_velocity_rad_s": 0.04,
            "settle_time_s": 6.0,
        }}],
        is_episode_end=lambda: False,
        take_action=lambda action: None,
    )
    result = _execute_native_actions(env, ["target"])
    assert result is None
