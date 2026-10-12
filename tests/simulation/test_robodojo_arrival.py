"""Completion changes control count only with explicit intent and measured evidence."""

import copy
import json
from types import SimpleNamespace

import jsonschema
import numpy as np
import pytest

from gpt_policy.simulation.robodojo.arrival import (
    ArrivalCheck,
    early_arrival_enabled,
    stop_on_arrival_enabled,
)
from gpt_policy.simulation.robodojo.budget import EpisodeBudget
from gpt_policy.simulation.robodojo.control import ARMS, validate_path
from gpt_policy.simulation.robodojo.environment import NativeSession
from gpt_policy.simulation.robodojo.tools import agent_context, catalog


def observation(step=0):
    return {
        "observation_id": f"e:{step}", "step_id": step,
        "arms": {arm: {"link6_pose_xyzquat": [0, 0, 1, 0, 0, 0, 1],
                       "joint_pos": [0.] * 6, "gripper": 1.} for arm in ARMS},
    }


def arguments(obs, mode="move", steps=5):
    point = {arm: {"pose_xyzquat": list(obs["arms"][arm]["link6_pose_xyzquat"]),
                   "gripper": 1.} for arm in ARMS}
    point["steps"] = steps
    if mode is not None:
        point["mode"] = mode
    return {"observation_id": obs["observation_id"], "note": "visible free motion",
            "waypoints": [point]}


def session(tmp_path, *, enabled=True, alter=None, native_end=None, wall_end=None, stop_on_arrival=True):
    env = SimpleNamespace(step_lim=1100, end_flag=[False], take_action_cnt=[0],
                          unstable_envs=set(), obs_manager=SimpleNamespace(collect_freq=25))
    s = NativeSession(env, tmp_path, "classify_objects", {}, control={
        "early_arrival": enabled, "stop_on_arrival": stop_on_arrival,
    })
    s.budget = EpisodeBudget(env, {})
    if wall_end:
        s.budget.expired = lambda: s.step_id >= wall_end
    s.obs = observation()
    s.last_gripper_command = {a: 1. for a in ARMS}
    # Even perfect IK diagnostics may never be used as arrival evidence.
    s.kinematics = SimpleNamespace(target=lambda _: {
        "action": np.zeros(14), "diagnostics": {"position_error": 0}})
    def take_action(command):
        env.take_action_cnt[0] += 1
        if native_end and env.take_action_cnt[0] >= native_end:
            env.end_flag[0] = True
    def observe():
        s.obs = observation(s.step_id)
        if alter:
            alter(s.obs)
        return s.obs
    env.take_action = take_action
    s.observe = observe
    s.finish = lambda reason: setattr(s, "reason", reason)
    return s


def test_move_needs_two_post_ack_samples_and_logs_exact_savings(tmp_path):
    s = session(tmp_path)
    receipt = s.move(arguments(s.obs))
    assert receipt["acknowledged_controls"] == s.env.take_action_cnt[0] == 2
    row = json.loads((tmp_path / "waypoint-execution.jsonl").read_text())
    assert row["requested_controls"] == 5 and row["executed_controls"] == 2
    assert row["saved_controls"] == 3 and row["stop_reason"] == "stable_arrival"
    assert row["stable_samples"] == 2
    assert all(x["measured_after_ack"] == 2 for x in row["arrival"].values())
    assert len((tmp_path / "actions.jsonl").read_text().splitlines()) == 2
    assert not s.env.end_flag[0]  # Arrival never overrides native task success.


@pytest.mark.parametrize("kind", ["left_position", "right_position", "rotation", "joint", "gripper", "moving"])
def test_both_arms_must_be_measured_arrived_and_stable(tmp_path, kind):
    def alter(obs):
        arm = obs["arms"]["right" if kind == "right_position" else "left"]
        if kind.endswith("position"):
            arm["link6_pose_xyzquat"][0] = .01
        elif kind == "rotation":
            from scipy.spatial.transform import Rotation
            arm["link6_pose_xyzquat"][3:] = Rotation.from_rotvec([0, 0, .04]).as_quat().tolist()
        elif kind == "joint":
            arm["joint_pos"][0] = .01 * obs["step_id"]
        elif kind == "gripper":
            arm["gripper"] = 1 - .01 * obs["step_id"]
        else:
            # Always within 3 mm, but oscillating faster than the speed threshold.
            arm["link6_pose_xyzquat"][0] = .002 * (-1) ** obs["step_id"]
    s = session(tmp_path, alter=alter)
    result = s.move(arguments(s.obs))
    assert result["acknowledged_controls"] == 5
    assert result["waypoint_results"][0]["saved_controls"] == 0


@pytest.mark.parametrize("kind", ["hold", "legacy", "disabled", "changed_gripper", "first_command"])
def test_fixed_holds_and_gripper_changes_are_never_shortened(tmp_path, kind):
    s = session(tmp_path, enabled=kind != "disabled")
    args = arguments(s.obs, mode=None if kind in ("legacy", "disabled") else "hold" if kind == "hold" else "move")
    if kind == "changed_gripper":
        args["waypoints"][0]["left"]["gripper"] = 0.
    if kind == "first_command":
        s.last_gripper_command = None
    result = s.move(args)
    assert result["acknowledged_controls"] == 5
    if kind == "disabled":
        assert "waypoint_results" not in result
        assert not (tmp_path / "waypoint-execution.jsonl").exists()
    else:
        assert result["waypoint_results"][0]["saved_controls"] == 0


def test_stability_resets_after_excursion():
    obs = observation()
    check = ArrivalCheck(arguments(obs)["waypoints"][0], obs, 25)
    for step, x, expected in [(1, 0, False), (2, .01, False), (3, 0, False),
                               (4, 0, False), (5, 0, True)]:
        current = observation(step)
        current["arms"]["left"]["link6_pose_xyzquat"][0] = x
        assert check.update(current) == expected


def test_refuses_non_ack_sample():
    obs = observation()
    check = ArrivalCheck(arguments(obs)["waypoints"][0], obs, 25)
    with pytest.raises(ValueError, match="consecutive"):
        check.update(obs)


@pytest.mark.parametrize("stop", ["native", "wall"])
def test_terminal_and_wall_stop_before_arrival_never_count_as_savings(tmp_path, stop):
    s = session(tmp_path, native_end=1 if stop == "native" else None,
                wall_end=1 if stop == "wall" else None)
    args = arguments(s.obs)
    args["waypoints"] *= 2
    result = s.move(args)
    assert result["acknowledged_controls"] == 1
    row = result["waypoint_results"][0]
    assert row["saved_controls"] == 0
    assert row["stop_reason"] == ("native_terminal" if stop == "native" else "wall_time_budget")
    if stop == "native":
        assert s.reason == "native_terminal"


def test_streak_not_carried_into_next_waypoint(tmp_path):
    s = session(tmp_path)
    args = arguments(s.obs)
    args["waypoints"].append(copy.deepcopy(args["waypoints"][0]))
    receipt = s.move(args)
    assert receipt["acknowledged_controls"] == 4
    assert [r["executed_controls"] for r in receipt["waypoint_results"]] == [2, 2]


def test_mode_opt_in_and_full_path_validation(tmp_path):
    s = session(tmp_path)
    args = arguments(s.obs)
    with pytest.raises(ValueError):
        validate_path(args, s.obs)
    validate_path(args, s.obs, early_arrival=True)
    args["waypoints"].append(copy.deepcopy(args["waypoints"][0]))
    args["waypoints"][1]["left"]["pose_xyzquat"][0] += .051
    with pytest.raises(ValueError, match="5 cm"):
        s.move(args)
    assert s.env.take_action_cnt[0] == 0


def test_fresh_measured_path_limit_still_checked(tmp_path):
    def alter(obs):
        for arm in ARMS:
            obs["arms"][arm]["link6_pose_xyzquat"][0] = -.02
    s = session(tmp_path, alter=alter)
    args = arguments(s.obs)
    args["waypoints"][0]["steps"] = 1
    args["waypoints"].append(copy.deepcopy(args["waypoints"][0]))
    args["waypoints"][1]["left"]["pose_xyzquat"][0] = .04
    result = s.move(args)
    assert result["acknowledged_controls"] == 1
    assert "5 cm" in result["rejection"]


@pytest.mark.parametrize("value", [None, 1, "true", {}, []])
def test_bad_config_rejected(value):
    with pytest.raises(ValueError, match="boolean"):
        early_arrival_enabled({"early_arrival": value})


def test_schema_and_prompt_are_opt_in():
    def schema(enabled):
        return next(x["function"]["parameters"] for x in
                    catalog(early_arrival=enabled).function_schemas(6, ARMS)
                    if x["function"]["name"] == "move_eef_chunk")
    old, new = schema(False), schema(True)
    args = arguments(observation())
    jsonschema.validate(args, new)
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(args, old)
    args["waypoints"][0].pop("mode")
    jsonschema.validate(args, old)
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(args, new)
    assert agent_context().instructions == agent_context(early_arrival=False).instructions
    assert 'mode="hold"' in agent_context(early_arrival=True).instructions


def test_executor_ablation_keeps_arrival_measurements_and_feedback_shape(tmp_path):
    enabled = session(tmp_path / "enabled")
    disabled = session(tmp_path / "disabled", stop_on_arrival=False)
    short = enabled.move(arguments(enabled.obs))
    full = disabled.move(arguments(disabled.obs))
    assert short["acknowledged_controls"] == 2
    assert full["acknowledged_controls"] == 5
    assert set(short) == set(full)
    assert set(short["waypoint_results"][0]) == set(full["waypoint_results"][0])
    row = full["waypoint_results"][0]
    assert row["mode"] == "move"
    assert row["stable_samples"] == 5  # Continue measuring; only the break is disabled.
    assert row["saved_controls"] == 0 and row["stop_reason"] == "requested_controls"
    ledger = json.loads((tmp_path / "disabled/waypoint-execution.jsonl").read_text())
    assert all(a["settled"] for a in ledger["arrival"].values())
    assert all(a["measured_after_ack"] == 5 for a in ledger["arrival"].values())


@pytest.mark.parametrize("stop", ["native", "wall"])
def test_executor_ablation_preserves_terminal_and_wall_limits(tmp_path, stop):
    s = session(tmp_path, stop_on_arrival=False,
                native_end=3 if stop == "native" else None,
                wall_end=3 if stop == "wall" else None)
    result = s.move(arguments(s.obs))
    assert result["acknowledged_controls"] == 3
    row = result["waypoint_results"][0]
    assert row["saved_controls"] == 0
    assert row["stop_reason"] == ("native_terminal" if stop == "native" else "wall_time_budget")


@pytest.mark.parametrize("value", [None, 1, "false", {}, []])
def test_stop_on_arrival_rejects_non_boolean(value):
    with pytest.raises(ValueError, match="boolean"):
        stop_on_arrival_enabled({"stop_on_arrival": value})


def test_executor_ablation_defaults_preserve_existing_behavior():
    assert stop_on_arrival_enabled({})
    assert stop_on_arrival_enabled({"stop_on_arrival": True})
    assert not stop_on_arrival_enabled({"stop_on_arrival": False})
