"""Motion progress uses measured snapshots, without inventing a success threshold."""

from types import SimpleNamespace

import numpy as np
import pytest

from gpt_policy.hardware.feedback import motion_progress
from gpt_policy.hardware.motion_control import MotionControl
from gpt_policy.hardware.robot import ArxRobot
from gpt_policy.hardware.yam import YamRobot


@pytest.mark.parametrize("target,actual", [
    ([.2, -.1, .3], [.2, -.1, .3]),  # Hold, including a chunk returning to its start.
    ([.21, -.12, .33], [.21, -.12, .33]),  # Reached final target.
    ([.21, -.12, .33], [.2, -.10004, .30008]),  # Small measured response.
])
def test_progress_is_net_translation_and_preserves_signed_remaining_error(target, actual):
    start = {"tcp_xyzrpy": [.2, -.1, .3, 0, 0, 0],
             "timestamp_s": 100., "timestamp_source": "test_snapshot_time"}
    target, actual = np.r_[target, [0, 0, 0]], np.r_[actual, [0, 0, 0]]
    result = motion_progress(start, target, actual, 102.)
    np.testing.assert_allclose(result["requested_delta_xyz_m"], target[:3] - [.2, -.1, .3])
    np.testing.assert_allclose(result["achieved_delta_xyz_m"], actual[:3] - [.2, -.1, .3])
    np.testing.assert_allclose(result["remaining_delta_xyz_m"], target[:3] - actual[:3])
    np.testing.assert_allclose(np.array(result["requested_delta_xyz_m"]) - result["achieved_delta_xyz_m"],
                               result["remaining_delta_xyz_m"], atol=1e-16)
    assert result["start_timestamp_s"] == 100.
    assert result["end_timestamp_s"] == 102.
    assert result["timestamp_source"] == "test_snapshot_time"
    assert set(result) == {"requested_delta_xyz_m", "achieved_delta_xyz_m", "remaining_delta_xyz_m",
                           "start_timestamp_s", "end_timestamp_s", "timestamp_source"}


def test_recorded_insert_tcp_values_reveal_submillimetre_response():
    # Saved step 24/25 measured TCP values from the 20260913-042901 plug run.
    # These historical consecutive results test arithmetic only: live execution
    # captures its own start, rather than reusing a preceding result or target.
    origin = [.2962520011556088, .36337295721953555, .08398501453951185]
    target = np.array([.298, .36, .089])
    actual = np.array([.29619490330290776, .3632989271726949, .08397572685217589])
    result = motion_progress({"tcp_xyzrpy": origin, "timestamp_s": 1.,
                              "timestamp_source": "test_snapshot_time"}, target, actual, 2.)
    assert np.linalg.norm(result["requested_delta_xyz_m"]) > .006
    assert np.linalg.norm(result["achieved_delta_xyz_m"]) == pytest.approx(.00009395144359905749)
    assert np.linalg.norm(result["remaining_delta_xyz_m"]) == pytest.approx(.006275716312943451)
    # The command-to-command 9 mm z change is not the measured-start request.
    assert result["requested_delta_xyz_m"][2] == pytest.approx(.005014985460488147)


def _frames():
    return SimpleNamespace(sdk_to_tcp=lambda pose: np.asarray(pose).copy())


def test_yam_start_is_first_execution_feedback_not_plan_or_previous_target():
    arm = object.__new__(YamRobot)
    arm.hz = 100.
    arm.max_lateness = .25
    arm.motion = MotionControl()
    arm._stop = SimpleNamespace(wait=lambda _: False)
    arm.clock = lambda: 0.
    arm.frames = _frames()
    submitted = []

    def forward_kinematics(joints):
        assert len(submitted) == 2  # Diagnostics never delay either command.
        return joints.copy()

    arm.ik = SimpleNamespace(forward_kinematics=forward_kinematics)
    arm._target = np.full(6, .7)  # Prior command must not become the origin.
    origin = np.array([.2, -.1, .3, 0, 0, 0, .4])
    samples = iter([
        {"q": origin.copy(), "timestamp": 123.},
        {"q": origin + .001, "timestamp": 123.01},
    ])
    arm._feedback = lambda: next(samples)
    arm.driver = SimpleNamespace(command_joint_pos=lambda q: submitted.append(q.copy()))
    target = np.array([.201, -.1, .3, 0, 0, 0])
    plan = {"start_joint_positions_rad": np.full(6, .6), "joint_positions_rad": target[None],
            "relative_times_s": [.01], "start_gripper_normalized": .4,
            "result": {"_trace": {"tcp_samples_xyzrpy": [target.tolist()]}}}
    arm._stream(plan, 0.)
    start = plan["result"]["_trace"]["execution_start"]
    np.testing.assert_array_equal(start["joint_positions_rad"], origin[:6])
    np.testing.assert_array_equal(start["tcp_xyzrpy"], origin[:6])
    assert start["timestamp_s"] == 123.
    # Instrumentation leaves the existing planned commands and held gripper intact.
    np.testing.assert_array_equal(submitted[0], np.r_[np.full(6, .6), .4])
    np.testing.assert_array_equal(submitted[-1], np.r_[target, .4])
    arm._last_settle_report = {"settled": True}
    measured = {"tcp_xyzrpy": (origin[:6] + [0.00001, 0, 0, 0, 0, 0]).tolist(),
                "joint_positions_rad": origin[:6].tolist(), "timestamp_s": 123.31}
    feedback = arm._execution_feedback(plan, measured)
    np.testing.assert_allclose(feedback["motion_progress"]["requested_delta_xyz_m"], [.001, 0, 0])
    np.testing.assert_allclose(feedback["motion_progress"]["achieved_delta_xyz_m"], [.00001, 0, 0])
    assert feedback["motion_progress"]["end_timestamp_s"] == 123.31
    assert feedback["settle"] == {"settled": True}


class _JointState:
    def __init__(self, positions, timestamp=0.):
        self.positions = np.array(positions, dtype=float)
        self.timestamp = timestamp

    def pos(self):
        return self.positions


def test_arx_progress_pairs_fk_with_the_joint_snapshot_timestamp():
    arm = object.__new__(ArxRobot)
    arm.motion = MotionControl()
    arm.frames = _frames()
    fk_calls = []

    def forward_kinematics(joints):
        fk_calls.append(joints.copy())
        return joints.copy()

    arm.solver = SimpleNamespace(forward_kinematics=forward_kinematics)
    arm._check_feedback = lambda: None
    origin = np.array([.2, -.1, .3, 0, 0, 0])
    sample = _JointState(origin, 5.)
    submitted = []
    arm.controller = SimpleNamespace(
        get_joint_state=lambda: sample, get_timestamp=lambda: 5.1,
        set_joint_traj=lambda commands: submitted.extend(commands),
    )
    target = origin + [.001, -.002, .003, 0, 0, 0]
    command = _JointState(target)
    plan = {"commands": [command], "relative_times_s": [.5],
            "start_joint_positions_rad": np.full(6, .6),
            "result": {"_trace": {"tcp_samples_xyzrpy": [target.tolist()]}}}
    arm.send_eef_trajectory(plan, wait=False)
    assert fk_calls == []
    start = plan["result"]["_trace"]["execution_start"]
    sample.positions[:] = 9.  # SDK ownership must not mutate the recorded origin.
    np.testing.assert_array_equal(start["joint_positions_rad"], origin)
    assert "tcp_xyzrpy" not in start  # No additional FK work in the submission path.
    assert start["timestamp_s"] == 5.
    assert submitted == [command]
    assert command.timestamp == 5.6
    arm._last_settle_report = {"settled": True}
    actual = origin + [.00001, -.00002, .00003, 0, 0, 0]
    measured = {"joint_positions_rad": actual.tolist(), "timestamp_s": 5.9,
                # An independently acquired EEF snapshot must not supply the
                # pose paired with the earlier joint snapshot's timestamp.
                "tcp_xyzrpy": target.tolist(), "gripper_normalized": .4, "gripper_torque_nm": 0.}
    feedback = arm._execution_feedback(plan, measured)
    np.testing.assert_array_equal(start["tcp_xyzrpy"], origin)
    np.testing.assert_allclose(feedback["motion_progress"]["requested_delta_xyz_m"], [.001, -.002, .003])
    np.testing.assert_allclose(feedback["motion_progress"]["achieved_delta_xyz_m"], [.00001, -.00002, .00003])
    np.testing.assert_allclose(feedback["measured_tcp_xyzrpy"], actual)
    assert feedback["motion_progress"]["end_timestamp_s"] == 5.9
    assert feedback["motion_progress"]["timestamp_source"] == "arx_joint_state_controller_time"
