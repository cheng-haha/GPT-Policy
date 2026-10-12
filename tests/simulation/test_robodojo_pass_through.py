"""Free intermediate targets advance only after a measured, validated native ACK."""

import copy

import pytest
from scipy.spatial.transform import Rotation
from test_robodojo_arrival import arguments, session

from gpt_policy.simulation.robodojo.arrival import pass_through_enabled
from gpt_policy.simulation.robodojo.execution import IKRejected
from gpt_policy.simulation.robodojo.tools import agent_context


def moving(obs):
    # Inside the pose tolerance, but faster than the old settled-arrival limit.
    for arm in ("left", "right"):
        obs["arms"][arm]["link6_pose_xyzquat"][0] = .002 * (-1) ** obs["step_id"]


def setup_path(tmp_path, *, alter=moving, **kwargs):
    s = session(tmp_path, alter=alter, **kwargs)
    s.pass_through = True
    args = arguments(s.obs)
    args["waypoints"].append(copy.deepcopy(args["waypoints"][0]))
    return s, args


def test_intermediate_advances_while_endpoint_keeps_original_rule(tmp_path):
    s, args = setup_path(tmp_path)
    result = s.move(args)
    first, last = result["waypoint_results"]
    assert result["acknowledged_controls"] == s.env.take_action_cnt[0] == 6
    assert first["executed_controls"] == 1 and first["saved_controls"] == 4
    assert first["stop_reason"] == "pass_through_arrival"
    assert first["stable_samples"] == 0
    assert last["executed_controls"] == 5 and last["saved_controls"] == 0
    assert len((tmp_path / "actions.jsonl").read_text().splitlines()) == 6
    assert not s.env.end_flag[0]


@pytest.mark.parametrize("kind", ["left_position", "right_position", "left_rotation", "right_rotation"])
def test_both_measured_poses_must_arrive(tmp_path, kind):
    def outside(obs):
        arm = obs["arms"][kind.split("_")[0]]
        if kind.endswith("position"):
            arm["link6_pose_xyzquat"][0] = .0031
        else:
            arm["link6_pose_xyzquat"][3:] = Rotation.from_rotvec([0, 0, .031]).as_quat().tolist()
    s, args = setup_path(tmp_path, alter=outside)
    result = s.move(args)
    assert result["acknowledged_controls"] == 10
    assert all(r["saved_controls"] == 0 for r in result["waypoint_results"])


@pytest.mark.parametrize("kind", [
    "current_hold", "next_hold", "initial_gripper", "changed_gripper",
    "next_left_gripper", "next_right_gripper", "single_point", "disabled",
])
def test_endpoints_contact_boundaries_and_opt_out_are_preserved(tmp_path, kind):
    s, args = setup_path(tmp_path)
    if kind == "current_hold":
        args["waypoints"][0]["mode"] = "hold"
    elif kind == "next_hold":
        args["waypoints"][1]["mode"] = "hold"
    elif kind == "initial_gripper":
        s.last_gripper_command = None
    elif kind == "changed_gripper":
        for point in args["waypoints"]:
            point["left"]["gripper"] = 0
    elif kind.startswith("next_"):
        args["waypoints"][1][kind.split("_")[1]]["gripper"] = 0
    elif kind == "single_point":
        args["waypoints"].pop()
    else:
        s.pass_through = False
    first = s.move(args)["waypoint_results"][0]
    assert first["executed_controls"] == 5
    assert first["saved_controls"] == 0


def test_full_path_validation_still_precedes_any_motion(tmp_path):
    s, args = setup_path(tmp_path)
    args["waypoints"][1]["left"]["pose_xyzquat"][0] = .051
    with pytest.raises(ValueError, match="5 cm"):
        s.move(args)
    assert s.env.take_action_cnt[0] == 0


def test_fresh_path_validation_after_early_advance_still_rejects(tmp_path):
    s, args = setup_path(tmp_path)
    args["waypoints"][1]["left"]["pose_xyzquat"][0] = .05
    result = s.move(args)
    # First ACK is -2 mm: the next 50 mm target is now 52 mm away.
    assert result["acknowledged_controls"] == 1
    assert result["waypoint_results"][0]["stop_reason"] == "pass_through_arrival"
    assert "5 cm" in result["rejection"]


def test_ik_rejection_does_not_advance_or_count_savings(tmp_path):
    s, args = setup_path(tmp_path)
    def reject(_):
        raise IKRejected("bounded IK rejection")
    s.kinematics.target = reject
    result = s.move(args)
    assert result["acknowledged_controls"] == 0
    assert result["waypoint_results"][0]["saved_controls"] == 0
    assert result["waypoint_results"][0]["stop_reason"] == "ik_rejected"


@pytest.mark.parametrize("stop", ["native", "wall"])
def test_terminal_and_budget_take_priority_over_pass_through(tmp_path, stop):
    s, args = setup_path(tmp_path, native_end=1 if stop == "native" else None,
                         wall_end=1 if stop == "wall" else None)
    result = s.move(args)
    assert result["acknowledged_controls"] == 1
    row = result["waypoint_results"][0]
    assert row["saved_controls"] == 0
    assert row["stop_reason"] == ("native_terminal" if stop == "native" else "wall_time_budget")


def test_tracking_guard_still_interrupts_before_advancing(tmp_path):
    def drift(obs):
        moving(obs)
        obs["arms"]["right"]["joint_pos"][0] = .16
    s, args = setup_path(tmp_path, alter=drift)
    s.execution["chunk_feedback"] = True
    args["waypoints"].append(copy.deepcopy(args["waypoints"][0]))
    result = s.move(args)
    assert result["acknowledged_controls"] == 2
    assert result["waypoint_results"][0]["stop_reason"] == "pass_through_arrival"
    assert result["waypoint_results"][1]["stop_reason"] == "tracking_deviation"
    assert result["waypoint_results"][1]["saved_controls"] == 0


@pytest.mark.parametrize("value", [None, 1, "true", {}, []])
def test_pass_through_flag_is_explicit_boolean(value):
    with pytest.raises(ValueError, match="boolean"):
        pass_through_enabled({"pass_through": value})


@pytest.mark.parametrize("overrides", [{"early_arrival": False}, {"stop_on_arrival": False}])
def test_pass_through_rejects_conflicting_ablation_flags(overrides):
    cfg = {"pass_through": True, "early_arrival": True, "stop_on_arrival": True, **overrides}
    with pytest.raises(ValueError, match="requires"):
        pass_through_enabled(cfg)


def test_configuration_defaults_and_model_contract():
    assert not pass_through_enabled({})
    assert pass_through_enabled({"pass_through": True, "early_arrival": True})
    old = agent_context(early_arrival=True)
    new = agent_context(early_arrival=True, pass_through=True)
    assert old.instructions == agent_context(early_arrival=True, pass_through=False).instructions
    assert old.output_schema == new.output_schema
    assert "pass_through_arrival" not in old.instructions
    assert "pass_through_arrival" in new.instructions
    assert "hold or gripper changes retain" in new.instructions
