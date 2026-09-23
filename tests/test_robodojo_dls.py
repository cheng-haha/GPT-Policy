"""Controller boundary checks; run with RoboDojo's simulator Python."""

import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

pytest.importorskip("scipy", reason="DLS executes in the RoboDojo simulator environment")

from gpt_policy.robodojo.adapter import RoboDojoAdapter
from gpt_policy.robodojo.calibration import RoboDojoCalibration
from gpt_policy.robodojo.dls import _ArmFK, _transform
from gpt_policy.robodojo.deploy import eval_one_episode
from gpt_policy.robodojo.model import GPTPolicyModel


def test_dls_joint_limit_and_forward_kinematics():
    urdf = '''<robot name="test">
      <link name="base"/><link name="link1"/><link name="tip"/>
      <joint name="j1" type="revolute"><parent link="base"/><child link="link1"/>
        <axis xyz="0 0 1"/><limit lower="-1" upper="1" effort="1" velocity="1"/></joint>
      <joint name="fixed" type="fixed"><parent link="link1"/><child link="tip"/>
        <origin xyz="1 0 0"/></joint></robot>'''
    with TemporaryDirectory() as directory:
        path = Path(directory) / "arm.urdf"
        path.write_text(urdf)
        arm = _ArmFK(path, ["j1"], "base", "tip")
        q = np.array([0.], dtype=float)
        goal = _transform([1., .04, 0., np.cos(.2 / 2), 0., 0., np.sin(.2 / 2)])
        result = arm.bounded_target(q, np.array([[-1., 1.]]), np.eye(4), goal)
        assert result.shape == (1,)
        assert 0 < result[0] <= .05 + 1e-7
        assert np.allclose(arm.matrix(q)[:3, 3], [1., 0., 0.])


def test_dls_mode_enforces_published_target_bound():
    calibration = RoboDojoCalibration.from_manifest({
        "world_from_base": {"left": np.eye(4).tolist(), "right": np.eye(4).tolist()}})
    model = GPTPolicyModel.__new__(GPTPolicyModel)
    model.arms = ("left", "right")
    model.adapter = RoboDojoAdapter(calibration, model.arms)
    model._reachability_bounds = {"x": [-.05, .65], "y": [-.55, .55], "z": [None, .45]}
    model._max_reachability_step_m = .30
    model._max_reachability_rotation_rad = .35
    model.latest_frame = {"state": {"left_ee_pose": [.1, 0., .1, 1., 0., 0., 0.]}}
    target = [{"left_ee_pose": [.16, 0., .1, 1., 0., 0., 0.]}]
    model._control_mode = "dls"
    assert model._validate_reachability(target)["reason"] == "unreachable"


def test_deploy_routes_dls_without_calling_native_ee():
    class Env:
        task_name = "organize_table"
        success = [False]
        end_flag = [False]
        take_action_cnt = [0]

        def is_episode_end(self):
            return self.end_flag[0]

        def get_obs(self):
            return {"state": {}}

        def take_action(self, _action):
            raise AssertionError("DLS mode must submit joint actions via its controller")

    class Client:
        def call(self, func_name, **_kwargs):
            if func_name == "is_episode_done":
                return False
            if func_name == "get_action":
                return [{"left_ee_pose": [0.] * 7}]
            return None

    env = Env()

    class Controller:
        def __init__(self, passed_env):
            assert passed_env is env

        def execute(self, passed_env, _action):
            passed_env.take_action_cnt[0] += 3
            passed_env.end_flag[0] = True
            return {"controller": "dls", "native_control_steps": 3}

    with patch.dict(os.environ, {"ROBODOJO_CONTROL_MODE": "dls"}), \
         patch("gpt_policy.robodojo.dls.DualX5DLS", Controller):
        eval_one_episode(env, Client())
    assert env._gpt_policy_control_feedback["native_control_steps"] == 3
    assert env.take_action_cnt == [3]
