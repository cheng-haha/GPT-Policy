from __future__ import annotations

import unittest
from copy import deepcopy
from threading import Barrier
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from gpt_policy.geometry.frames import FrameCalibration
from gpt_policy.hardware.bimanual import BimanualRobot
from gpt_policy.hardware.motion_control import MotionControl
from gpt_policy.hardware.robot import ArxRobot, _load_sdk
from gpt_policy.motion.coordination import (
    TrajectoryIKError,
    synchronize_bimanual_plan_times,
)
from gpt_policy.motion.ik import ContinuousIK
from gpt_policy.motion.planner import copy_sdk_vector
from gpt_policy.settings import load_settings
from gpt_policy.harness.protocol import output_schema, tool_schemas
from gpt_policy.motion.trajectory import MotionLimits


class _JointState:
    def __init__(self, dof: int) -> None:
        self._position = np.zeros(dof)
        self._velocity = np.zeros(dof)
        self._torque = np.zeros(dof)
        self.gripper_pos = 0.0
        self.gripper_vel = 0.0
        self.gripper_torque = 0.0
        self.timestamp = 0.0

    def pos(self):
        return self._position

    def vel(self):
        return self._velocity

    def torque(self):
        return self._torque


class _Controller:
    def get_joint_state(self):
        return _JointState(6)


class _Gain:
    def __init__(self, kp, kd, gripper_kp, gripper_kd):
        self._kp = np.asarray(kp)
        self._kd = np.asarray(kd)
        self.gripper_kp = gripper_kp
        self.gripper_kd = gripper_kd

    def kp(self):
        return self._kp

    def __mul__(self, scale):
        return _Gain(
            self._kp * scale, self._kd * scale,
            self.gripper_kp * scale, self.gripper_kd * scale,
        )

    def __add__(self, other):
        return _Gain(
            self._kp + other._kp, self._kd + other._kd,
            self.gripper_kp + other.gripper_kp,
            self.gripper_kd + other.gripper_kd,
        )


class _IdentityFrames:
    @staticmethod
    def sdk_to_tcp(pose):
        return np.asarray(pose)

    @staticmethod
    def tcp_to_sdk(pose):
        return np.asarray(pose)


class _IdentitySolver:
    @staticmethod
    def multi_trial_ik(pose, _seed, _trials):
        return 0, np.asarray(pose)

    @staticmethod
    def forward_kinematics(joints):
        return np.asarray(joints)


class _PoisonedOwner:
    """Mimic pybind's reference view becoming invalid with its parent object."""

    def __init__(self, values) -> None:
        self.values = np.asarray(values, dtype=np.float64).copy()

    def pos(self):
        return self.values

    def __del__(self):
        self.values[:] = 999.0


def _attach_ik(arm: ArxRobot) -> None:
    size = int(arm.config.joint_dof)
    lower = getattr(arm.config, "joint_pos_min", np.full(size, -10.0))
    upper = getattr(arm.config, "joint_pos_max", np.full(size, 10.0))
    arm.ik = ContinuousIK(
        arm.solver,
        lower,
        upper,
        arm.ik_refine_iterations,
        arm.ik_translation_tolerance_m,
        arm.ik_rotation_tolerance_rad,
        arm.ik_execution_translation_tolerance_m,
        arm.ik_execution_rotation_tolerance_rad,
    )


class RobotTrajectoryContractTest(unittest.TestCase):
    def test_single_arm_planning_failure_never_submits_partial_trajectory(self) -> None:
        arm = ArxRobot.__new__(ArxRobot)
        arm.motion = MotionControl()
        submitted = []

        def fail_before_send(_requested, _note):
            raise TrajectoryIKError(
                segment=1,
                sample=4,
                status=-9,
                status_name="E_EXCEED_JOINT_LIMIT",
                translation_error_m=0.01,
                rotation_error_rad=0.02,
            )

        arm.plan_eef_trajectory = fail_before_send
        arm.send_eef_trajectory = lambda plan: submitted.append(plan)
        with self.assertRaises(TrajectoryIKError):
            arm._move_eef_trajectory(
                "move_to",
                {"target": {"pose_xyzquat": [0.0] * 7}, "note": "test"},
            )
        self.assertEqual(submitted, [])

    def test_small_settle_error_is_diagnostic_below_fault_limit(self) -> None:
        class UnsettledController:
            @staticmethod
            def get_timestamp():
                return 1.0

            @staticmethod
            def set_joint_cmd(_command):
                return None

            @staticmethod
            def get_joint_state():
                state = _JointState(6)
                state.pos()[:] = 0.02
                state.vel()[:] = 1.0
                return state

            @staticmethod
            def get_joint_cmd():
                return _JointState(6)

        arm = ArxRobot.__new__(ArxRobot)
        arm.motion = MotionControl()
        arm.controller = UnsettledController()
        arm.config = SimpleNamespace(joint_dof=6, joint_vel_max=np.full(6, 5), joint_torque_max=np.full(6, 30))
        arm.tracking_error_limit_rad = .12
        arm._reference_end_s = 0.0
        arm._arm_target = np.zeros(6)
        arm._target_command = object
        arm.settle_timeout_s = 0.001
        arm.settle_samples = 2
        arm.settle_position_tolerance_rad = 0.01
        arm.settle_velocity_tolerance_rad_s = 0.01
        arm.trajectory_hz = 100.0

        arm.wait_reference_complete()

        self.assertFalse(arm._last_settle_report["settled"])

    def test_sdk_reference_is_copied_before_parent_destruction(self) -> None:
        copied = copy_sdk_vector(_PoisonedOwner(np.arange(6.0)), "pos")
        np.testing.assert_array_equal(copied, np.arange(6.0))

    def test_model_trajectory_schema_has_no_waypoint_maximum(self) -> None:
        outer_poses = next(
            schema["properties"]["poses"]
            for schema in output_schema(6)["properties"]["arguments"]["anyOf"]
            if "poses" in schema["properties"]
        )
        move_chunk = next(
            tool
            for tool in tool_schemas(6)
            if tool["function"]["name"] == "move_eef_chunk"
        )
        tool_poses = move_chunk["function"]["parameters"]["properties"]["poses"]
        self.assertNotIn("maxItems", outer_poses)
        self.assertNotIn("maxItems", tool_poses)
        self.assertEqual(outer_poses["minItems"], 1)
        self.assertEqual(tool_poses["minItems"], 1)

    def test_near_boundary_ik_result_is_accepted_only_inside_execution_tolerance(
        self,
    ) -> None:
        class NearBoundarySolver:
            @staticmethod
            def multi_trial_ik(pose, _seed, _trials):
                solution = np.asarray(pose).copy()
                solution[0] -= 0.001
                return -9, solution

            @staticmethod
            def forward_kinematics(joints):
                return np.asarray(joints)

            @staticmethod
            def get_ik_status_name(_status):
                return "E_EXCEED_JOINT_LIMIT"

        ik = ContinuousIK(
            NearBoundarySolver(),
            np.full(6, -10.0),
            np.full(6, 10.0),
            0,
            1e-4,
            5e-4,
            0.002,
            np.radians(1.0),
        )
        target = np.zeros(6)

        solved = ik.solve(
            target,
            np.zeros(6),
            segment=1,
            sample=1,
        )
        self.assertAlmostEqual(solved[0], -0.001)

        ik.execution_translation_tolerance_m = 0.0005
        with self.assertRaises(TrajectoryIKError):
            ik.solve(
                target,
                np.zeros(6),
                segment=1,
                sample=1,
            )

    def test_planner_keeps_model_endpoint_and_only_adds_a_hold(self) -> None:
        arm = ArxRobot.__new__(ArxRobot)
        arm.motion = MotionControl()
        arm.sdk = SimpleNamespace(JointState=_JointState)
        arm.frames = _IdentityFrames()
        arm.controller = _Controller()
        arm.solver = _IdentitySolver()
        arm.config = SimpleNamespace(joint_dof=6, gripper_width=0.088)
        arm.trajectory_hz = 100.0
        arm.motion_limits = MotionLimits(
            output_hz=100.0,
            cartesian_step_m=0.005,
            cartesian_step_rad=0.035,
            tcp_velocity_m_s=0.08,
            tcp_angular_velocity_rad_s=0.5,
            joint_velocity_rad_s=np.full(6, 1.0),
            joint_acceleration_rad_s2=np.full(6, 2.0),
            joint_jerk_rad_s3=np.full(6, 12.0),
        )
        arm.endpoint_hold_s = 0.12
        arm._gripper_target = 0.5
        arm.ik_refine_iterations = 0
        arm.ik_translation_tolerance_m = 1e-4
        arm.ik_rotation_tolerance_rad = 5e-4
        arm.ik_execution_translation_tolerance_m = 0.002
        arm.ik_execution_rotation_tolerance_rad = np.radians(1.0)
        _attach_ik(arm)

        endpoint = np.array([0.08, -0.04, 0.02, 0.1, -0.2, 0.3])
        plan = arm.plan_eef_trajectory(
            [{"pose_xyzrpy": endpoint.tolist()}], "contract test"
        )
        result = plan["result"]
        motion_count = result["motion_waypoints"]

        np.testing.assert_array_equal(
            result["_trace"]["model_tcp_points_xyzrpy"][-1], endpoint
        )
        np.testing.assert_array_equal(
            plan["commands"][motion_count - 1].pos(), endpoint
        )
        for command in plan["commands"][motion_count:]:
            np.testing.assert_array_equal(command.pos(), endpoint)
        segment = result["_trace"]["segments"][-1]
        self.assertEqual(segment["endpoint_fk_translation_error_m"], 0.0)
        self.assertAlmostEqual(segment["endpoint_fk_rotation_error_rad"], 0.0, places=14)
        self.assertTrue(np.all(np.diff(plan["relative_times_s"]) > 0.0))

    def test_reported_first_sample_ik_failure_regression(self) -> None:
        try:
            sdk = _load_sdk()
        except RuntimeError as exc:
            self.skipTest(str(exc))
        config = sdk.RobotConfigFactory.get_instance().get_config("X5")
        solver = sdk.Arx5Solver(
            config.urdf_path,
            config.joint_dof,
            config.joint_pos_min,
            config.joint_pos_max,
            config.base_link_name,
            config.eef_link_name,
            config.gravity_vector,
        )
        joint_position = np.array(
            [
                -0.039101600646972656,
                -0.00133514404296875,
                -0.00095367431640625,
                -0.050927162170410156,
                -0.00324249267578125,
                0.01430511474609375,
            ]
        )
        class RecordedController:
            @staticmethod
            def get_joint_state():
                return _PoisonedOwner(joint_position)

        arm = ArxRobot.__new__(ArxRobot)
        arm.motion = MotionControl()
        arm.sdk = sdk
        arm.frames = FrameCalibration(load_settings())
        arm.controller = RecordedController()
        arm.solver = solver
        arm.config = config
        arm.trajectory_hz = 100.0
        arm.motion_limits = MotionLimits(
            output_hz=100.0,
            cartesian_step_m=0.005,
            cartesian_step_rad=0.035,
            tcp_velocity_m_s=0.08,
            tcp_angular_velocity_rad_s=0.5,
            joint_velocity_rad_s=np.asarray(config.joint_vel_max) * 0.25,
            joint_acceleration_rad_s2=np.full(6, 2.0),
            joint_jerk_rad_s3=np.full(6, 12.0),
        )
        arm.endpoint_hold_s = 0.12
        arm._gripper_target = 0.5
        arm.ik_refine_iterations = 12
        arm.ik_translation_tolerance_m = 1e-4
        arm.ik_rotation_tolerance_rad = 5e-4
        arm.ik_execution_translation_tolerance_m = 0.002
        arm.ik_execution_rotation_tolerance_rad = np.radians(1.0)
        _attach_ik(arm)

        target = [
            0.24977,
            0.03361,
            0.14368,
            0.0182291,
            0.7245203,
            -0.0073726,
            0.6889729,
        ]
        plan = arm.plan_eef_trajectory(
            [{"pose_xyzquat": target}], "regression for dangling SDK state view"
        )
        np.testing.assert_allclose(
            plan["result"]["_trace"]["model_tcp_points_xyzrpy"][-1][:3],
            target[:3],
            atol=0.0,
        )

    def test_recorded_joint_limit_failure_is_typed_and_keeps_target(self) -> None:
        try:
            sdk = _load_sdk()
        except RuntimeError as exc:
            self.skipTest(str(exc))
        config = sdk.RobotConfigFactory.get_instance().get_config("X5")
        solver = sdk.Arx5Solver(
            config.urdf_path,
            config.joint_dof,
            config.joint_pos_min,
            config.joint_pos_max,
            config.base_link_name,
            config.eef_link_name,
            config.gravity_vector,
        )
        recorded_joint_position = np.array(
            [
                -0.09098148345947266,
                2.1269168853759766,
                1.9911117553710938,
                -1.5096893310546875,
                -0.00209808349609375,
                -0.09632301330566406,
            ]
        )

        class RecordedController:
            @staticmethod
            def get_joint_state():
                return _PoisonedOwner(recorded_joint_position)

        arm = ArxRobot.__new__(ArxRobot)
        arm.motion = MotionControl()
        arm.sdk = sdk
        arm.frames = FrameCalibration(load_settings())
        arm.controller = RecordedController()
        arm.solver = solver
        arm.config = config
        arm.trajectory_hz = 100.0
        arm.motion_limits = MotionLimits(
            output_hz=100.0,
            cartesian_step_m=0.005,
            cartesian_step_rad=0.035,
            tcp_velocity_m_s=0.08,
            tcp_angular_velocity_rad_s=0.5,
            joint_velocity_rad_s=np.asarray(config.joint_vel_max) * 0.25,
            joint_acceleration_rad_s2=np.full(6, 2.0),
            joint_jerk_rad_s3=np.full(6, 12.0),
        )
        arm.endpoint_hold_s = 0.12
        arm._gripper_target = 0.55
        arm.ik_refine_iterations = 12
        arm.ik_translation_tolerance_m = 1e-4
        arm.ik_rotation_tolerance_rad = 5e-4
        arm.ik_execution_translation_tolerance_m = 0.002
        arm.ik_execution_rotation_tolerance_rad = np.radians(1.0)
        _attach_ik(arm)
        requested = [0.38, -0.27, 0.12, 0.0, 1.0, 0.0, 0.0]

        with self.assertRaises(TrajectoryIKError) as raised:
            arm.plan_eef_trajectory(
                [{"pose_xyzquat": requested}],
                "20260910-231411-730 step 15 regression",
            )

        details = raised.exception.details
        self.assertEqual(details["reason"], "ik_joint_limit")
        self.assertEqual(details["segment"], 1)
        # Strict FK verification can detect the unreachable boundary before
        # KDL changes its own, looser status to E_EXCEED_JOINT_LIMIT.
        self.assertGreaterEqual(details["sample"], 150)
        self.assertGreater(details["best_translation_error_m"], 0.0)

    def test_bimanual_segment_timing_slows_only_and_keeps_samples(self) -> None:
        def plan(times, durations):
            return {
                "commands": [object()] * 6,
                "relative_times_s": list(times),
                "result": {
                    "motion_waypoints": 4,
                    "planned_duration_s": times[-1],
                    "_trace": {
                        "relative_times_s": list(times),
                        "tcp_samples_xyzrpy": [[index] for index in range(6)],
                        "joint_waypoints_rad": [[index] for index in range(6)],
                        "segments": [
                            {"segment": 1, "samples": 2, "duration_s": durations[0]},
                            {"segment": 2, "samples": 2, "duration_s": durations[1]},
                        ],
                    },
                },
            }

        left = plan([0.5, 1.0, 2.0, 3.0, 3.1, 3.2], [1.0, 2.0])
        right = plan([1.0, 2.0, 2.25, 2.5, 2.6, 2.7], [2.0, 0.5])
        original_tcp = {
            "left": deepcopy(left["result"]["_trace"]["tcp_samples_xyzrpy"]),
            "right": deepcopy(right["result"]["_trace"]["tcp_samples_xyzrpy"]),
        }

        common = synchronize_bimanual_plan_times({"left": left, "right": right})

        self.assertEqual(common, [2.0, 2.0])
        np.testing.assert_allclose(left["relative_times_s"], [1, 2, 3, 4, 4.1, 4.2])
        np.testing.assert_allclose(right["relative_times_s"], [1, 2, 3, 4, 4.1, 4.2])
        self.assertEqual(left["result"]["_trace"]["tcp_samples_xyzrpy"], original_tcp["left"])
        self.assertEqual(right["result"]["_trace"]["tcp_samples_xyzrpy"], original_tcp["right"])

    def test_bimanual_ik_error_identifies_arm(self) -> None:
        class FailingArm:
            @staticmethod
            def plan_eef_trajectory(_requested, _note):
                raise TrajectoryIKError(
                    segment=2,
                    sample=7,
                    status=-9,
                    status_name="E_EXCEED_JOINT_LIMIT",
                    translation_error_m=0.01,
                    rotation_error_rad=0.02,
                )

        robot = BimanualRobot.__new__(BimanualRobot)
        robot.arms = {"left": FailingArm(), "right": FailingArm()}
        with self.assertRaises(TrajectoryIKError) as raised:
            robot._execute_bimanual_eef(
                "move_to",
                {"target": {"left": {"pose_xyzquat": [0] * 7}, "right": None}, "note": "test"},
            )
        self.assertEqual(raised.exception.details["arm"], "left")

    def test_coordinated_send_inserts_future_start_hold(self) -> None:
        class TrajectoryController:
            submitted = None

            @staticmethod
            def get_timestamp():
                return 1.0

            def set_joint_traj(self, commands):
                self.submitted = commands

            def get_joint_state(self):
                state = _JointState(6)
                state.pos()[:] = np.arange(6.0)
                return state

            get_joint_cmd = get_joint_state

        arm = ArxRobot.__new__(ArxRobot)
        arm.motion = MotionControl()
        arm.sdk = SimpleNamespace(JointState=_JointState)
        arm.config = SimpleNamespace(joint_dof=6, gripper_width=0.088)
        arm.controller = TrajectoryController()
        arm.config.joint_vel_max = np.full(6, 5)
        arm.config.joint_torque_max = np.full(6, 30)
        arm.tracking_error_limit_rad = .12
        arm._arm_target = None
        commands = [_JointState(6), _JointState(6)]
        commands[0].pos()[:] = 1.0
        commands[1].pos()[:] = 2.0
        plan = {
            "commands": commands,
            "relative_times_s": [0.5, 1.0],
            "start_joint_positions_rad": np.arange(6.0),
            "start_gripper_normalized": 0.75,
        }

        arm.send_eef_trajectory(plan, wait=False, start_time=5.0)

        submitted = arm.controller.submitted
        self.assertEqual(len(submitted), 3)
        np.testing.assert_array_equal(submitted[0].pos(), np.arange(6.0))
        self.assertEqual(submitted[0].timestamp, 5.0)
        self.assertEqual(submitted[1].timestamp, 5.5)
        self.assertEqual(submitted[2].timestamp, 6.0)
        self.assertEqual(arm._reference_end_s, 6.0)

    def test_bimanual_motion_sends_and_settles_in_parallel(self) -> None:
        send_barrier = Barrier(2, timeout=1.0)
        wait_barrier = Barrier(2, timeout=1.0)
        starts = {}

        def make_plan(duration):
            return {
                "commands": [object()] * 4,
                "relative_times_s": [duration / 2, duration, duration + 0.1, duration + 0.2],
                "result": {
                    "motion_waypoints": 2,
                    "planned_duration_s": duration + 0.2,
                    "_trace": {
                        "relative_times_s": [duration / 2, duration, duration + 0.1, duration + 0.2],
                        "segments": [{"segment": 1, "samples": 2, "duration_s": duration}],
                    },
                },
            }

        class MotionArm:
            def __init__(self, side, duration):
                self.side = side
                self.plan = make_plan(duration)
                self.clock = lambda: 10.0

            def plan_eef_trajectory(self, _requested, _note):
                return self.plan

            def send_eef_trajectory(self, _plan, _wait, start_time):
                starts[self.side] = start_time
                send_barrier.wait()

            @staticmethod
            def wait_reference_complete():
                wait_barrier.wait()

            @staticmethod
            def state():
                return {}

            @staticmethod
            def _execution_feedback(_plan, _state):
                return {}

        robot = BimanualRobot.__new__(BimanualRobot)
        robot.arms = {
            "left": MotionArm("left", 1.0),
            "right": MotionArm("right", 2.0),
        }
        robot.interfaces = {"left": "can1", "right": "can3"}
        robot.start_delay_s = 0.1
        result = robot._execute_bimanual_eef(
            "move_to",
            {
                "target": {
                    "left": {"pose_xyzquat": [0] * 7},
                    "right": {"pose_xyzquat": [0] * 7},
                },
                "note": "parallel test",
            },
        )

        self.assertEqual(starts, {"left": 10.1, "right": 10.1})
        self.assertEqual(result["trajectory"]["common_segment_durations_s"], [2.0])
        np.testing.assert_allclose(
            robot.arms["left"].plan["relative_times_s"], [1, 2, 2.1, 2.2]
        )
        np.testing.assert_allclose(
            robot.arms["right"].plan["relative_times_s"], [1, 2, 2.1, 2.2]
        )

    def test_return_home_slows_both_legs_and_waits_for_zero_pose(self) -> None:
        class HomeController:
            def __init__(self, position, gripper):
                self.now = 12.5
                self.measured = _JointState(6)
                self.measured.pos()[0] = position
                self.measured.gripper_pos = gripper
                self.commands = []
                self.gains = []
                self.gain = _Gain(
                    np.full(6, 150.0), np.ones(6), 2.0, 0.1
                )

            def get_timestamp(self):
                return self.now

            def sleep(self, duration):
                self.now += duration

            def get_joint_state(self):
                for _sent_at, command in self.commands:
                    if command.timestamp <= self.now:
                        self.measured = deepcopy(command)
                return self.measured

            get_joint_cmd = get_joint_state

            def set_joint_cmd(self, command):
                copied = deepcopy(command)
                if copied.timestamp == 0.0:
                    copied.timestamp = self.now
                self.commands.append((self.now, copied))

            def get_gain(self):
                return self.gain

            def set_gain(self, gain):
                self.gain = gain
                self.gains.append(gain)

            @staticmethod
            def get_controller_config():
                return SimpleNamespace(
                    default_kp=np.full(6, 150.0), default_kd=np.ones(6),
                    default_gripper_kp=2.0, default_gripper_kd=0.1,
                )

        # Cover long arm travel, closed gripper, minimum duration, custom
        # scaling. Re-arming damping is covered by the motion fault tests.
        for position, gripper, scale, approach_s in (
            (2.5, 0.088, 3.0, 7.5),
            (0.0, 0.0, 3.0, 6.0),
            (0.0, 0.088, 3.0, 1.5),
            (-1.0, 0.088, 5.0, 5.0),
        ):
            with self.subTest(position=position, gripper=gripper, scale=scale):
                arm = ArxRobot.__new__(ArxRobot)
                arm.motion = MotionControl()
                arm.sdk = SimpleNamespace(JointState=_JointState, Gain=_Gain)
                arm.config = SimpleNamespace(joint_dof=6, gripper_width=0.088)
                arm.config.joint_vel_max = np.full(6, 5)
                arm.config.joint_torque_max = np.full(6, 30)
                arm.tracking_error_limit_rad = .12
                controller = HomeController(position, gripper)
                arm.controller = controller
                arm.home_time_scale = scale
                arm.settle_timeout_s = 3.0
                arm.settle_samples = 2
                arm.settle_position_tolerance_rad = 0.03
                arm.settle_velocity_tolerance_rad_s = 0.05
                arm.trajectory_hz = 30.0
                arm.state = lambda: {
                    "joint_positions_rad": controller.get_joint_state().pos().tolist(),
                }

                with (
                    patch("gpt_policy.hardware.robot.time.sleep", controller.sleep),
                    patch("gpt_policy.hardware.robot.time.monotonic", controller.get_timestamp),
                ):
                    result = arm.return_home()

                self.assertEqual(len(controller.commands), 5)
                _sent_at, approach = controller.commands[1]
                final_sent_at, final = controller.commands[3]
                hold_sent_at, hold = controller.commands[4]
                np.testing.assert_array_equal(approach.pos(), [0, 0, 0.03, 0, 0, 0])
                self.assertAlmostEqual(approach.timestamp, 12.5 + approach_s)
                self.assertGreaterEqual(final_sent_at, approach.timestamp)
                self.assertAlmostEqual(final.timestamp - final_sent_at, 0.5 * scale)
                self.assertGreaterEqual(hold_sent_at, final.timestamp)
                np.testing.assert_array_equal(final.pos(), np.zeros(6))
                np.testing.assert_array_equal(hold.vel(), np.zeros(6))
                self.assertEqual(approach.gripper_pos, 0.088)
                self.assertEqual(final.gripper_pos, 0.088)
                np.testing.assert_array_equal(arm._arm_target, np.zeros(6))
                self.assertEqual(arm._gripper_target, 1.0)
                self.assertEqual(result["home"]["time_scale"], scale)
                self.assertEqual(result["home"]["segment_durations_s"], [approach_s, 0.5 * scale])
                self.assertEqual(result["home"]["max_joint_residual_rad"], 0.0)
                self.assertTrue(result["home"]["settle"]["settled"])
                self.assertEqual(controller.gains, [])

    def test_bimanual_return_home_runs_both_arms_concurrently(self) -> None:
        barrier = Barrier(2, timeout=1.0)

        class HomeArm:
            def __init__(self, side):
                self.side = side

            def return_home(self):
                barrier.wait()
                return {"side": self.side}

        robot = BimanualRobot.__new__(BimanualRobot)
        robot.arms = {side: HomeArm(side) for side in ("left", "right")}
        robot.interfaces = {"left": "can1", "right": "can3"}

        result = robot.return_home()

        self.assertTrue(result["concurrent"])
        self.assertEqual(set(result["arms"]), {"left", "right"})


if __name__ == "__main__":
    unittest.main()
