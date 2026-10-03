from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from gpt_policy.hardware.gripper import GripperStreamer
from gpt_policy.hardware.motion_control import MotionControl
from gpt_policy.hardware.robot import ArxRobot


class _State:
    def __init__(
        self, gripper_pos: float, gripper_torque: float = 0.0, dof: int = 6
    ) -> None:
        self.gripper_pos = gripper_pos
        self.gripper_vel = 9.0
        self.gripper_torque = gripper_torque
        self._pos = np.arange(dof, dtype=float)
        self._vel = np.full(dof, 7.0)
        self._torque = np.full(dof, 6.0)

    def pos(self):
        return self._pos

    def vel(self):
        return self._vel

    def torque(self):
        return self._torque


class _Command(_State):
    def __init__(self, dof: int = 6) -> None:
        super().__init__(0.0, dof)
        self.timestamp = 99.0


class _Controller:
    def __init__(
        self,
        start: float,
        blocked_at: float | None = None,
        response: float = 1.0,
        feedback_torque: float = 0.0,
    ) -> None:
        self.position = start
        self.blocked_at = blocked_at
        self.response = response
        self.feedback_torque = feedback_torque
        self.commands: list[_Command] = []

    def get_joint_state(self):
        return _State(self.position, self.feedback_torque)

    def set_joint_cmd(self, command):
        self.commands.append(command)
        if self.blocked_at is None:
            self.position += self.response * (command.gripper_pos - self.position)
        else:
            self.position = max(self.blocked_at, command.gripper_pos)
            self.feedback_torque = 0.8 if command.gripper_pos < self.blocked_at else 0.0


class GripperStreamerTest(unittest.TestCase):
    width = 0.088

    @staticmethod
    def command():
        return _Command()

    @patch("gpt_policy.hardware.gripper.time.sleep", lambda _: None)
    def test_free_motion_reaches_requested_opening(self):
        controller = _Controller(self.width)
        report = GripperStreamer(
            controller, self.command, self.width, -3.4, 3.0, 100.0, 1.5
        ).move(0.48)
        self.assertAlmostEqual(controller.position, 0.48 * self.width)
        self.assertAlmostEqual(report["active_command_normalized"], 0.48)
        self.assertTrue(
            all(command.gripper_torque == 0.0 for command in controller.commands)
        )

    @patch("gpt_policy.hardware.gripper.time.sleep", lambda _: None)
    def test_delayed_feedback_does_not_stop_reference_progress(self):
        controller = _Controller(self.width, response=0.1)
        report = GripperStreamer(
            controller, self.command, self.width, -3.4, 3.0, 100.0, 1.5
        ).move(0.48)
        self.assertAlmostEqual(report["active_command_normalized"], 0.48)
        self.assertLess(controller.position, self.width)

    @patch("gpt_policy.hardware.gripper.time.sleep", lambda _: None)
    def test_blockage_does_not_accumulate_requested_position_error(self):
        blocked_at = 0.0625
        controller = _Controller(self.width, blocked_at)
        streamer = GripperStreamer(
            controller, self.command, self.width, -3.4, 3.0, 100.0, 1.5
        )
        report = streamer.move(0.48)
        expected = (blocked_at - streamer.step_m) / self.width
        self.assertAlmostEqual(controller.position, blocked_at)
        self.assertAlmostEqual(report["active_command_normalized"], expected)
        self.assertGreater(report["active_command_normalized"], 0.48)
        self.assertTrue(report["sdk_contact"])

    def test_feedback_torque_is_not_replayed_as_command_torque(self):
        arm = ArxRobot.__new__(ArxRobot)
        arm.motion = MotionControl()
        arm.sdk = SimpleNamespace(JointState=_Command)
        arm.config = SimpleNamespace(joint_dof=6)
        arm.controller = _Controller(self.width, feedback_torque=8.0)
        command = arm._command_copy()
        np.testing.assert_array_equal(command.vel(), np.zeros(6))
        np.testing.assert_array_equal(command.torque(), np.zeros(6))
        self.assertEqual(command.gripper_vel, 0.0)
        self.assertEqual(command.gripper_torque, 0.0)


if __name__ == "__main__":
    unittest.main()
