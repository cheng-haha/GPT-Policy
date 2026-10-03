from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from gpt_policy.hardware.bimanual import BimanualRobot
from gpt_policy.hardware.robot import ArxRobot
from gpt_policy.hardware.yam import YamRobot
from gpt_policy.motion.coordination import TrajectoryIKError
from gpt_policy.tools import ToolExecutor, load_tool_catalog


def plan():
    return {"joint_positions_rad": np.zeros((2, 6)), "relative_times_s": [.1, .2],
            "result": {"planned_duration_s": .2, "motion_waypoints": 2,
                       "_trace": {"model_tcp_points_xyzrpy": [[0] * 6, [.1] * 6],
                                  "segments": [{"duration_s": .2, "samples": 2}]}}}


@pytest.mark.parametrize("kind", [ArxRobot, YamRobot])
def test_single_arm_check_uses_planner_without_dispatch_or_target_changes(kind):
    robot = kind.__new__(kind)
    robot.plan_eef_trajectory = Mock(return_value=plan())
    robot.send_eef_trajectory = Mock(side_effect=AssertionError("check moved robot"))
    robot.state = Mock(side_effect=AssertionError("check should return planned evidence only"))
    arguments = {"poses": [{"pose_xyzquat": [.3, .1, .1, 0, 1, 0, 0]}], "note": "Compare lower approach"}
    result = ToolExecutor(load_tool_catalog(), ("left",), robot, None).execute("check_path", arguments, {}, {})
    robot.plan_eef_trajectory.assert_called_once_with(arguments["poses"], arguments["note"])
    assert result["path_check"]["accepted"]
    assert not result["path_check"]["executed"]
    assert not result["path_check"]["collision_checked"]


@pytest.mark.parametrize("reject", [False, True])
def test_bimanual_check_never_dispatches_even_when_peer_rejects(reject):
    arms = {side: SimpleNamespace(plan_eef_trajectory=Mock(return_value=deepcopy(plan())))
            for side in ("left", "right")}
    if reject:
        arms["right"].plan_eef_trajectory.side_effect = TrajectoryIKError(
            segment=1, sample=23, status=-1, status_name="MINK_NOT_CONVERGED",
            translation_error_m=.0024, rotation_error_rad=.0015)
    robot = BimanualRobot.from_arms(arms, {"left": "can0", "right": "can1"}, {})
    robot._send_plans = Mock(side_effect=AssertionError("check moved robot"))
    args = {"poses": [{"left": {"pose_xyzquat": [0] * 7}, "right": {"pose_xyzquat": [0] * 7}}], "note": "check"}
    if reject:
        with pytest.raises(TrajectoryIKError) as exc:
            robot.execute("check_path", args)
        assert exc.value.details["arm"] == "right"
    else:
        result = robot.execute("check_path", args)
        assert set(result["path_check"]["arms"]) == {"left", "right"}
        assert result["path_check"]["replanned_before_execution"]
    robot._send_plans.assert_not_called()
