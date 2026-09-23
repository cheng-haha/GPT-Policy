"""Regression checks for bounded RoboDojo action completion."""

from types import SimpleNamespace

import numpy as np
import pytest

from gpt_policy.robodojo.execution import settle_control


class _Velocity:
    def __getitem__(self, key):
        return self

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return np.zeros(1)


def _env(timeout):
    robot = SimpleNamespace(type="target", arm_name="left_arm", arm_joint_indices=[0])
    manager = SimpleNamespace(
        robot_list=[robot],
        robot_key=[SimpleNamespace(data=SimpleNamespace(joint_vel=_Velocity()))],
        process_name=lambda name: name,
        get_joint=lambda robot, indices: {0: np.zeros(1)},
        control_manager=SimpleNamespace(push=lambda *args: None),
    )
    env = SimpleNamespace(
        dt=0.1,
        eval_cfg={"action_execution": {"settle_timeout_s": timeout}},
        robot_manager=manager,
        physx_monitor_enabled=False,
        steps=0,
    )

    def step(*, env_idx_list):
        env.steps += 1

    env.step = step
    return env


def test_unreachable_target_returns_after_timeout():
    env = _env(0.3)
    result = settle_control(env, {0: {"left_arm": {"position": [1.0]}}})
    assert result[0]["settled"] is False
    assert result[0]["joint_error_rad"] == 1.0
    assert env.steps == 3


def test_nonpositive_timeout_is_rejected():
    with pytest.raises(ValueError, match="settle_timeout_s"):
        settle_control(_env(0), {0: {"left_arm": {"position": [1.0]}}})
