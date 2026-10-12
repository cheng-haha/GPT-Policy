"""The transport probe must preserve endpoint, budget and action guards."""

import copy
from types import SimpleNamespace

import numpy as np
import pytest

from gpt_policy.simulation.robodojo.control import validate_path
from gpt_policy.simulation.robodojo.transport_probe import (
    replay_prefix_control,
    transport_points,
)


@pytest.fixture
def fixture():
    observation = {"observation_id": "probe:856", "arms": {
        a: {"link6_pose_xyzquat": [0, 0, 1, 0, 0, 0, 1]} for a in ("left", "right")}}
    target = {"mode": "move", "steps": 3, **{
        a: {"pose_xyzquat": [0.03, 0, 1, 0, 0, 0, 1], "gripper": 0}
        for a in ("left", "right")}}
    return observation, target

def test_target_distribution_preserves_budget_endpoint_other_arm_and_inputs(fixture):
    observation, target = fixture
    original = copy.deepcopy((observation, target))
    points = transport_points(observation, target, "distributed", {"left": 0, "right": 0})
    assert sum(p["steps"] for p in points) == target["steps"]
    assert [p["left"]["pose_xyzquat"][0] for p in points] == pytest.approx([.01, .02, .03])
    assert points[-1]["left"] == target["left"]
    assert all(p["right"] == target["right"] for p in points)
    validate_path({"observation_id": "probe:856", "note": "probe", "waypoints": points},
                  observation, early_arrival=True)
    assert (observation, target) == original

def test_direct_retains_original_semantics(fixture):
    observation, target = fixture
    assert transport_points(observation, target, "direct", {"left": 0, "right": 0}) == [target]

@pytest.mark.parametrize("change", ["gripper", "hold", "variant"])
def test_invalid_interventions_are_refused(fixture, change):
    observation, target = fixture
    variant = "distributed"
    if change == "gripper":
        target["left"]["gripper"] = 1
    if change == "hold":
        target["mode"] = "hold"
    if change == "variant":
        variant = "unknown"
    with pytest.raises(ValueError):
        transport_points(observation, target, variant, {"left": 0, "right": 0})

def test_replay_updates_continuity_only_after_ack(tmp_path, monkeypatch):
    from gpt_policy.simulation.robodojo import transport_probe as module
    monkeypatch.setattr(module, "native_stop_reason", lambda env: None)
    acked = []
    env = SimpleNamespace(take_action_cnt=[0])
    def take(command):
        env.take_action_cnt[0] += 1
    env.take_action = take
    session = SimpleNamespace(
        env=env, budget=SimpleNamespace(expired=lambda: False), output=tmp_path,
        kinematics=SimpleNamespace(acknowledged=lambda action: acked.append(action)),
        step_id=0, observe=lambda: None)
    row = {"command": [0.] * 14, "requested": {"left": {"gripper": 0}, "right": {"gripper": 0}}}
    replay_prefix_control(session, row)
    assert session.step_id == 1
    assert session.last_gripper_command == {"left": 0, "right": 0}
    np.testing.assert_array_equal(acked[0], row["command"])
    env.take_action = lambda command: None
    with pytest.raises(ValueError, match="ACK"):
        replay_prefix_control(session, row)
    assert len(acked) == 1
