"""Native IK continuity, atomic rejection, and chunk interruption contracts."""

import copy
from types import SimpleNamespace

import numpy as np
import pytest
from test_robodojo_arrival import arguments, observation, session

from gpt_policy.simulation.robodojo.execution import (
    IKRejected,
    TrackingGuard,
    continuous_joints,
    execution_config,
)
from gpt_policy.simulation.robodojo.kinematics import DualKinematics
from gpt_policy.simulation.robodojo.tools import agent_context


@pytest.mark.parametrize("value", [True, None, "0.15", 0, -0.1, 0.151, float("nan"), float("inf")])
def test_increment_rejects_unauthorized_or_invalid_bounds(value):
    with pytest.raises(ValueError):
        execution_config({"max_joint_increment_rad": value})


def test_defaults_preserve_previous_execution():
    assert execution_config({}) == {
        "ik_backend": "bounded_dls", "max_joint_increment_rad": .05, "chunk_feedback": False,
        "distribute_closed_gripper_moves": False,
    }
    assert agent_context().instructions == agent_context(chunk_feedback=False).instructions
    assert agent_context().output_schema == agent_context(chunk_feedback=True).output_schema
    assert "10-15" in agent_context(chunk_feedback=True).instructions


def test_periodic_solution_continuity_and_joint_limits():
    limits = [[-7., 7.], [-1., 1.]]
    q = continuous_joints([2*np.pi + .15, -.15], [0., 0.], limits, .15)
    assert q == pytest.approx([.15, -.15])
    for candidate in ([.1501, 0], [0, 2], [float("nan"), 0]):
        with pytest.raises(IKRejected):
            continuous_joints(candidate, [0, 0], limits, .15)
    # A current-state-safe target can still jump from the last commanded target.
    with pytest.raises(IKRejected):
        continuous_joints([.1], [0.], [[-1, 1]], .15, previous=[-.1])
    with pytest.raises(IKRejected):
        continuous_joints([1.05], [1.], [[-1, 1]], .15)


def native_kinematics(right_result=None):
    k = DualKinematics.__new__(DualKinematics)
    k.control = execution_config({"ik_backend": "native", "max_joint_increment_rad": .15})
    k.previous = {}
    robots = [SimpleNamespace(arm_joints_name=[f"j{i}" for i in range(6)],
                              arm_joint_indices=list(range(6))) for _ in range(2)]
    k.robots = dict(zip(("left", "right"), robots))
    class Tensor:
        def __getitem__(self, key): return self
        def cpu(self): return self
        def numpy(self): return np.array([[-1., 1.]] * 6)
    def solve(pose, index, robot, trans):
        assert trans == "world" and index == 0
        if robot is robots[1] and right_result is not None: return right_result
        return {"status": "Success", "joint_names": list(reversed(robot.arm_joints_name)),
                "joint_value": [0, 0, 0, 0, 0, .1]}
    k.manager = SimpleNamespace(robot_list=robots, get_joint=lambda r: [np.zeros(6)],
        robot_key=[SimpleNamespace(data=SimpleNamespace(soft_joint_pos_limits=Tensor()))]*2,
        solve_ik=solve)
    def matrix(q):
        t = np.eye(4); t[0, 3] = q[0]; return t
    k.fk = {arm: SimpleNamespace(matrix=matrix) for arm in k.robots}
    k.root = lambda arm: np.eye(4)
    k.check = lambda: {"passed": True}
    targets = {arm: {"position": [.1, 0, 0], "quaternion_wxyz": [1, 0, 0, 0],
                     "gripper_closed": False} for arm in k.robots}
    return k, targets


def test_native_maps_names_and_commits_only_after_ack():
    k, targets = native_kinematics()
    proposal = k.target(targets)
    assert np.array(proposal["action"])[[0, 7]] == pytest.approx([.1, .1])
    assert k.previous == {}
    k.acknowledged(proposal["action"])
    assert k.previous["left"][0] == pytest.approx(.1)


@pytest.mark.parametrize("result", [
    {"status": "Fail"},
    {"status": "Success", "joint_names": ["j0"], "joint_value": [0]},
])
def test_one_arm_failure_never_commits_other_arm_or_gripper(result):
    k, targets = native_kinematics(result)
    with pytest.raises(IKRejected):
        k.target(targets)
    assert k.previous == {}


def test_independent_fk_check_refuses_site_mismatch():
    k, targets = native_kinematics()
    targets["right"]["position"][0] = .11
    with pytest.raises(IKRejected, match="FK"):
        k.target(targets)


def test_expected_ik_rejection_keeps_episode_and_can_retry(tmp_path):
    s = session(tmp_path)
    args = arguments(s.obs)
    args["waypoints"] *= 3
    def reject(_): raise IKRejected("smaller target required")
    s.kinematics.target = reject
    receipt = s.move(args)
    assert receipt["acknowledged_controls"] == s.env.take_action_cnt[0] == 0
    assert "smaller target" in receipt["rejection"]
    assert not s.poisoned and not s.finished
    assert not (tmp_path / "actions.jsonl").exists()
    s.kinematics.target = lambda _: {"action": np.zeros(14), "diagnostics": {}}
    assert s.move(arguments(s.obs))["acknowledged_controls"] == 2


def test_partial_ik_rejection_retains_only_acknowledged_prefix(tmp_path):
    s = session(tmp_path)
    s.execution["ik_backend"] = "native"
    calls = []
    def target(_):
        if calls: raise IKRejected("no solution")
        return {"action": np.zeros(14), "diagnostics": {}}
    s.kinematics.target = target
    s.kinematics.acknowledged = lambda action: calls.append(list(action))
    args = arguments(s.obs)
    args["waypoints"] *= 3
    receipt = s.move(args)
    assert receipt["acknowledged_controls"] == len(calls) == 1
    assert receipt["end_step"] == s.obs["step_id"] == 1
    assert receipt["rejection"] == "no solution"


def test_tracking_guard_stops_suffix_and_counts_every_executed_step(tmp_path):
    def drift(obs): obs["arms"]["right"]["joint_pos"][0] = .16
    s = session(tmp_path, alter=drift)
    s.execution["chunk_feedback"] = True
    args = arguments(s.obs, mode="hold")
    args["waypoints"] *= 3
    receipt = s.move(args)
    assert receipt["acknowledged_controls"] == s.env.take_action_cnt[0] == 2
    assert "Tracking" in receipt["rejection"]
    assert receipt["waypoint_results"][0]["stop_reason"] == "tracking_deviation"
    assert not s.finished and not s.poisoned


def test_tracking_streak_resets_after_good_ack():
    guard = TrackingGuard()
    point = arguments(observation())["waypoints"][0]
    for bad, expected in [(True, False), (False, False), (True, False), (True, True)]:
        obs = observation()
        obs["arms"]["right"]["joint_pos"][0] = .2 if bad else 0
        assert guard.update(point, obs, np.zeros(14)) is expected


def test_long_chunk_respects_native_end_without_padding(tmp_path):
    s = session(tmp_path, native_end=7)
    s.execution["chunk_feedback"] = True
    args = arguments(s.obs, mode="hold")
    args["waypoints"] = [copy.deepcopy(args["waypoints"][0]) for _ in range(3)]
    result = s.move(args)
    assert result["acknowledged_controls"] == s.env.take_action_cnt[0] == 7
    assert [x["executed_controls"] for x in result["waypoint_results"]] == [5, 2]


def test_continuity_feedback_names_joint_and_both_baselines():
    k, targets = native_kinematics()
    k.previous["right"] = np.array([-.1, 0, 0, 0, 0, 0])
    with pytest.raises(IKRejected) as caught:
        k.target(targets)
    details = caught.value.details
    assert details["arm"] == "right"
    assert details["code"] == "joint_continuity_or_limits"
    assert details["max_joint_increment_rad"] == .15
    joint, = details["violating_joints"]
    assert joint["joint_name"] == "j0" and joint["joint_index"] == 0
    assert joint["delta_from_measured_rad"] == pytest.approx(.1)
    assert joint["delta_from_previous_ack_rad"] == pytest.approx(.2)
    assert joint["allowed_interval_rad"] == pytest.approx([-.15, .05])
    assert "left" not in k.previous  # Never commit the other arm's valid proposal.


def test_rejection_details_count_prefix_without_replaying_it(tmp_path):
    import json
    s = session(tmp_path)
    def target(_):
        if s.step_id:
            raise IKRejected("smaller target", code="joint_continuity_or_limits",
                             arm="left", violating_joints=[{"joint_name": "joint2"}])
        return {"action": np.zeros(14), "diagnostics": {}}
    s.kinematics.target = target
    receipt = s.move(arguments(s.obs))
    details = receipt["rejection_details"]
    assert details["waypoint_index"] == 0
    assert details["acknowledged_controls_before_rejection"] == 1
    assert details["waypoint_acknowledged_controls"] == 1
    assert details["rejected_proposal_controls"] == 0
    assert details["unexecuted_suffix_discarded"]
    assert receipt["acknowledged_controls"] == s.env.take_action_cnt[0] == 1
    assert not s.finished and not s.poisoned
    ledger = json.loads((tmp_path / "ik-rejections.jsonl").read_text())
    assert ledger["arm"] == "left" and ledger["step"] == 1
