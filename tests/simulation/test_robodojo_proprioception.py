"""Physical aperture provenance and unchanged target-proxy arrival behavior."""

from types import SimpleNamespace

import numpy as np
import pytest
from test_robodojo_arrival import arguments, observation, session

from gpt_policy.simulation.robodojo.control import measured_tracking
from gpt_policy.simulation.robodojo.proprioception import gripper_measurement


def read_grip(values, *, sign=1, scale=(0., .044)):
    calls = []
    def get(robot, env_idx_list):
        calls.append(env_idx_list)
        return {0: np.asarray(values)}
    manager = SimpleNamespace(get_end_effector_real_val=get)
    robot = SimpleNamespace(gripper_scale=scale, gripper_move={"sign": sign})
    result = gripper_measurement(manager, robot)
    assert calls == [[0]]
    return result


@pytest.mark.parametrize("raw,sign,opening", [(0., 1, 0.), (.044, 1, 1.),
                                             (.011, 1, .25), (.011, -1, .75)])
def test_actual_joints_are_normalized_in_command_direction(raw, sign, opening):
    result = read_grip([raw, -raw], sign=sign)
    assert result["measured_gripper"] == pytest.approx(opening)
    assert result["measured_gripper_unclipped"] == pytest.approx(opening)
    assert result["gripper_joint_positions"] == [raw, -raw]


def test_raw_overshoot_is_preserved_beside_clipped_opening():
    result = read_grip([.048, -.048])
    assert result["measured_gripper"] == 1.
    assert result["measured_gripper_unclipped"] > 1.
    assert result["gripper_joint_positions"][0] == .048


@pytest.mark.parametrize("values,kwargs", [
    ([], {}), ([float("nan")], {}), ([0], {"scale": (0, 0)}),
    ([0], {"scale": (0, float("inf"))}), ([0], {"sign": 0}),
])
def test_bad_measurement_fails_without_fallback_to_target(values, kwargs):
    with pytest.raises(ValueError):
        read_grip(values, **kwargs)


def test_tracking_distinguishes_request_target_and_measured_opening():
    obs = observation()
    for arm in obs["arms"].values():
        arm.update(gripper=.2, measured_gripper=.45)
    point = arguments(obs)["waypoints"][0]
    row = measured_tracking(point, obs)["left"]
    assert row["requested_gripper"] == 1.
    assert row["control_target_gripper"] == .2
    assert row["measured_gripper"] == .45
    del obs["arms"]["left"]["measured_gripper"]
    assert measured_tracking(point, obs)["left"]["measured_gripper"] is None


def test_physical_aperture_does_not_silently_change_arrival_gate(tmp_path):
    def physical_motion(obs):
        for arm in obs["arms"].values():
            arm["measured_gripper"] = .2 * obs["step_id"]
    current = session(tmp_path / "physical", alter=physical_motion)
    legacy = session(tmp_path / "legacy")
    a = current.move(arguments(current.obs))
    b = legacy.move(arguments(legacy.obs))
    assert a["acknowledged_controls"] == b["acknowledged_controls"] == 2
    assert a["waypoint_results"] == b["waypoint_results"]
    assert a["tracking"]["left"]["measured_gripper"] == .4
