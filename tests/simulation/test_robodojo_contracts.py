"""Safety and episode-accounting contracts; no Isaac or model required."""

import copy
import socket
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from gpt_policy.simulation.robodojo.control import validate_path
from gpt_policy.simulation.robodojo.evaluation import (
    outcome,
    register_native_evaluation,
)
from gpt_policy.simulation.robodojo.protocol import (
    RPCClient,
    receive_packet,
    send_packet,
)
from gpt_policy.simulation.robodojo.transforms import pose_matrix, to_link6, to_tool


def packet():
    pose = [0, 0, 1, 0, 0, 0, 1]
    obs = {
        "observation_id": "episode:0",
        "arms": {arm: {"link6_pose_xyzquat": pose} for arm in ("left", "right")},
    }
    args = {
        "observation_id": "episode:0",
        "note": "Hold after observation.",
        "waypoints": [
            dict(
                steps=1,
                **{
                    arm: {"pose_xyzquat": pose.copy(), "gripper": 1.0}
                    for arm in ("left", "right")
                },
            )
        ],
    }
    return obs, args


@pytest.mark.parametrize(
    "fault",
    [
        "stale",
        "translation",
        "rotation",
        "nan",
        "quaternion",
        "gripper",
        "steps",
        "boolean_steps",
        "arm",
        "empty",
        "note",
    ],
)
def test_invalid_actions_rejected(fault):
    obs, args = packet()
    point = args["waypoints"][0]
    if fault == "stale":
        args["observation_id"] = "episode:1"
    if fault == "translation":
        point["left"]["pose_xyzquat"][0] = 0.050001
    if fault == "rotation":
        point["left"]["pose_xyzquat"][3:] = (
            Rotation.from_rotvec([0.351, 0, 0]).as_quat().tolist()
        )
    if fault == "nan":
        point["left"]["pose_xyzquat"][0] = float("nan")
    if fault == "quaternion":
        point["left"]["pose_xyzquat"][6] = 0.99
    if fault == "gripper":
        point["right"]["gripper"] = 1.01
    if fault == "steps":
        point["steps"] = 6
    if fault == "boolean_steps":
        point["steps"] = True
    if fault == "arm":
        del point["right"]
    if fault == "empty":
        args["waypoints"] = []
    if fault == "note":
        args["note"] = ""
    with pytest.raises((ValueError, KeyError)):
        validate_path(args, obs)


def test_all_waypoints_validated_before_execution():
    obs, args = packet()
    args["waypoints"].append(copy.deepcopy(args["waypoints"][0]))
    args["waypoints"][1]["left"]["pose_xyzquat"][0] = 0.06
    with pytest.raises(ValueError, match="5 cm"):
        validate_path(args, obs)


def test_exact_limits_preserved():
    obs, args = packet()
    p = args["waypoints"][0]["left"]["pose_xyzquat"]
    p[0] = 0.05
    p[3:] = Rotation.from_rotvec([0.35, 0, 0]).as_quat().tolist()
    assert len(validate_path(args, obs)) == 1


def env(complete=False, success=False, invalid=False):
    return SimpleNamespace(
        end_flag=[complete],
        success=[success],
        step_lim=1100,
        unstable_envs={0} if invalid else set(),
    )


@pytest.mark.parametrize(
    "reason",
    ["decision_budget", "quota_error", "disconnect", "model_done", "server_close"],
)
def test_incomplete_never_gets_success_or_score(reason):
    result = outcome(env(), 10, reason)
    assert result["native_success"] is None and result["native_score"] is None
    assert not result["valid_for_success_rate"]


def test_complete_native_failure_is_reportable():
    result = outcome(env(True, False), 1100, "native_terminal")
    assert result["complete"] and result["valid_for_success_rate"]
    assert result["native_success"] is False


@pytest.mark.parametrize("invalid,poisoned", [(True, False), (False, True)])
def test_invalid_or_uncertain_is_ineligible(invalid, poisoned):
    result = outcome(env(True, True, invalid), 10, "terminal", poisoned)
    assert result["native_success"] is None and not result["valid_for_success_rate"]


def test_refuses_empty_native_checks():
    e = env()
    e.num_envs = 1
    e.run_reward = lambda: None
    e.reward_manager = SimpleNamespace(
        check_list=[[]], final_check_list=[[]], trigger_check_list=[[]]
    )
    with pytest.raises(ValueError, match="empty"):
        register_native_evaluation(e)


def test_registration_and_support_arm_setup():
    e = env()
    e.num_envs = 1
    e.run_reward = lambda: None
    e.reward_manager = SimpleNamespace(
        check_list=[["native"]], final_check_list=[[]], trigger_check_list=[[]]
    )
    e.interact = True
    calls = []
    e.get_running_env_idx_list = lambda: [0]
    e.query_support_arm_traj = lambda **kw: calls.append(kw)
    assert register_native_evaluation(e)["registered"]
    assert calls == [{"env_idx": 0}]


def test_site_frame_roundtrip_uses_real_transform():
    base = pose_matrix([-0.3, -0.45, 0.765, 0, 0, 2**-0.5, 2**-0.5])
    link = pose_matrix([0.1, 0.2, 1.0, 0, 0, 0, 1])
    tool = np.eye(4)
    tool[:3, 3] = [0.145, 0, 0]
    tcp = to_tool(link, base, tool)
    assert np.allclose(to_link6(tcp, base, tool), link)
    assert not np.allclose(tcp, link)


def test_rpc_preserves_rgb_bytes():
    a, b = socket.socketpair()
    rgb = np.array([[[255, 0, 0], [0, 0, 255]]], dtype=np.uint8)
    try:
        send_packet(a, {"rgb": rgb})
        assert np.array_equal(receive_packet(b)["rgb"], rgb)
    finally:
        a.close()
        b.close()


def test_timeout_does_not_retry_or_reuse_connection():
    a, b = socket.socketpair()
    client = RPCClient.__new__(RPCClient)
    client.sock = a
    try:
        with (
            patch("gpt_policy.simulation.robodojo.protocol.send_packet") as send,
            patch(
                "gpt_policy.simulation.robodojo.protocol.receive_packet",
                side_effect=TimeoutError,
            ),
        ):
            with pytest.raises(TimeoutError):
                client.request("move", arguments={})
            assert send.call_count == 1 and client.sock is None
            with pytest.raises(RuntimeError, match="closed"):
                client.request("move", arguments={})
            assert send.call_count == 1
    finally:
        b.close()


def test_tracking_feedback_uses_measured_post_ack_pose_and_gripper():
    from scipy.spatial.transform import Rotation

    from gpt_policy.simulation.robodojo.control import measured_tracking

    point = {
        arm: {"pose_xyzquat": [0, 0, 1, 0, 0, 0, 1], "gripper": 0}
        for arm in ("left", "right")
    }
    current = {
        "step_id": 7,
        "arms": {
            arm: {
                "link6_pose_xyzquat": [
                    0.003,
                    0.004,
                    1,
                    *Rotation.from_rotvec([0, 0, 0.2]).as_quat(),
                ],
                "gripper": 0.2,
                "measured_gripper": 0.4,
            }
            for arm in ("left", "right")
        },
    }
    feedback = measured_tracking(point, current)
    for arm in feedback.values():
        assert arm["target_position_error_m"] == pytest.approx(0.005)
        assert arm["target_rotation_error_rad"] == pytest.approx(0.2)
        assert arm["requested_gripper"] == 0 and arm["measured_gripper"] == 0.4
        assert arm["control_target_gripper"] == 0.2
        assert arm["measured_after_ack"] == 7
        assert "success" not in arm
