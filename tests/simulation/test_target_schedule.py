"""Target scheduling must keep budgets, endpoint arrival and all refusal guards."""

import copy
import json

import numpy as np
import pytest
from test_robodojo_arrival import arguments, observation, session

from gpt_policy.simulation.robodojo.execution import IKRejected, execution_config
from gpt_policy.simulation.robodojo.target_schedule import waypoint_targets


def closed_move():
    obs = observation()
    point = arguments(obs, steps=3)["waypoints"][0]
    point["left"]["gripper"] = 0.
    point["left"]["pose_xyzquat"][0] = .03
    return obs, point, {"left": 0., "right": 1.}


def test_schedule_only_distributes_closed_arm_and_keeps_endpoint_and_budget():
    obs, point, previous = closed_move()
    original = copy.deepcopy((obs, point))
    targets, arms = waypoint_targets(point, obs, previous, enabled=True)
    assert arms == ["left"] and len(targets) == point["steps"]
    assert [t["left"]["pose_xyzquat"][0] for t in targets] == pytest.approx([.01, .02, .03])
    assert targets[-1] == point
    assert all(t["right"] == point["right"] for t in targets)
    assert (obs, point) == original


@pytest.mark.parametrize("kind", ["disabled", "hold", "open", "changed", "initial", "one_control"])
def test_ineligible_motion_retains_original_target(kind):
    obs, point, previous = closed_move()
    if kind == "hold":
        point["mode"] = "hold"
    if kind == "open":
        point["left"]["gripper"] = previous["left"] = 1.
    if kind == "changed":
        previous["left"] = 1.
    if kind == "initial":
        previous = None
    if kind == "one_control":
        point["steps"] = 1
    targets, arms = waypoint_targets(point, obs, previous, enabled=kind != "disabled")
    assert arms == []
    assert targets == [point] * point["steps"]


@pytest.mark.parametrize("value", [None, 1, "true"])
def test_config_requires_boolean(value):
    with pytest.raises(ValueError, match="distribute_closed_gripper_moves"):
        execution_config({"distribute_closed_gripper_moves": value})


def following_session(tmp_path, *, reject_at=None, drift=False, **kwargs):
    s = session(tmp_path, **kwargs)
    s.execution["distribute_closed_gripper_moves"] = True
    s.last_gripper_command = {"left": 0., "right": 1.}
    calls = []
    def target(values):
        calls.append(copy.deepcopy(values))
        if len(calls) == reject_at:
            raise IKRejected("fixture refusal", code="test")
        return {"action": np.zeros(14), "diagnostics": {}}
    def observe():
        s.obs = observation(s.step_id)
        for arm in ("left", "right"):
            target = calls[-1][arm]
            q = target["quaternion_wxyz"]
            s.obs["arms"][arm]["link6_pose_xyzquat"] = [
                *target["position"], q[1], q[2], q[3], q[0]]
            s.obs["arms"][arm]["gripper"] = target["gripper_opening"]
        if drift:
            s.obs["arms"]["left"]["link6_pose_xyzquat"][0] = -.05
        return s.obs
    s.kinematics.target = target
    s.observe = observe
    args = arguments(s.obs, steps=3)
    args["waypoints"] = [closed_move()[1]]
    return s, args, calls


def test_intermediate_arrival_cannot_skip_full_goal(tmp_path):
    s, args, calls = following_session(tmp_path)
    s.pass_through = True
    following = copy.deepcopy(args["waypoints"][0])
    following["left"]["pose_xyzquat"][0] = .06
    args["waypoints"].append(following)
    result = s.move(args)
    assert result["acknowledged_controls"] == 6
    assert [c["left"]["position"][0] for c in calls] == pytest.approx([.01,.02,.03,.04,.05,.06])
    assert all(w["executed_controls"] == 3 for w in result["waypoint_results"])
    rows = [json.loads(l) for l in (tmp_path / "actions.jsonl").read_text().splitlines()]
    assert rows[0]["requested"]["left"]["pose_xyzquat"][0] == .03
    assert rows[0]["applied_target"]["left"]["pose_xyzquat"][0] == pytest.approx(.01)
    assert rows[-1]["applied_target"] == rows[-1]["requested"]


def test_new_ik_refusal_preserves_accepted_prefix_and_discards_suffix(tmp_path):
    s, args, calls = following_session(tmp_path, reject_at=2)
    result = s.move(args)
    assert result["acknowledged_controls"] == 1
    assert len(calls) == 2 and result["rejection"] == "fixture refusal"


def test_distributed_target_is_rechecked_against_fresh_measured_pose(tmp_path):
    s, args, calls = following_session(tmp_path, drift=True)
    result = s.move(args)
    assert result["acknowledged_controls"] == 1 and len(calls) == 1
    assert "5 cm" in result["rejection"]


@pytest.mark.parametrize("stop", ["native", "wall"])
def test_scheduling_does_not_run_past_native_or_wall_limit(tmp_path, stop):
    s, args, calls = following_session(
        tmp_path, native_end=1 if stop == "native" else None,
        wall_end=1 if stop == "wall" else None)
    result = s.move(args)
    assert result["acknowledged_controls"] == 1 and len(calls) == 1
    assert result["waypoint_results"][0]["saved_controls"] == 0
