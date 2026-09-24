import unittest

import numpy as np

from gpt_policy.robodojo import RoboDojoAdapter, RoboDojoCalibration
from gpt_policy.robodojo.model import GPTPolicyModel, _canonical_tool_name


class RoboDojoAdapterTest(unittest.TestCase):
    def setUp(self):
        self.calibration = RoboDojoCalibration.from_manifest({
            "world_from_base": {
                "left": np.eye(4).tolist(),
                "right": np.eye(4).tolist(),
            },
            "intrinsics": {"cam_head": [[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]]},
            "camera_extrinsics_world": {"cam_head": np.eye(4).tolist()},
        })
        self.adapter = RoboDojoAdapter(self.calibration, ("left", "right"))

    def test_converts_robo_wxyz_to_policy_xyzw_without_dropping_raw_frame(self):
        frame = {
            "state": {
                "left_arm_joint_state": [0, 1, 2, 3, 4, 5],
                "left_ee_joint_state": [0.2],
                "left_ee_pose": [0.1, 0.2, 0.3, 1, 0, 0, 0],
                "right_arm_joint_state": [5, 4, 3, 2, 1, 0],
                "right_ee_joint_state": [0.8],
                "right_ee_pose": [0.4, 0.5, 0.6, 1, 0, 0, 0],
            },
            "vision": {"cam_head": {"color": np.zeros((2, 2, 3), dtype=np.uint8)}},
        }
        state = self.adapter.state(frame)
        self.assertEqual(state["arms"]["left"]["tcp_xyzquat"], [0.1, 0.2, 0.3, 0, 0, 0, 1])
        self.assertIn("simulator_raw", state)
        cameras = self.adapter.cameras(frame)
        self.assertEqual(cameras[0]["intrinsic_matrix"], [[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]])

    def test_action_converts_policy_xyzw_to_robo_wxyz(self):
        action = self.adapter.action({"target": {"left": [0.1, 0.2, 0.3, 0, 0, 0, 1], "right": None}})
        self.assertEqual(action["left_ee_pose"], [0.1, 0.2, 0.3, 1, 0, 0, 0])

    def test_bimanual_set_gripper_sends_selected_side_instead_of_hold(self):
        model = GPTPolicyModel.__new__(GPTPolicyModel)
        model.latest_frame = {"state": {
            "left_ee_pose": [0.1, 0.0, 0.2, 1.0, 0.0, 0.0, 0.0],
            "left_ee_joint_state": [0.8],
            "right_ee_pose": [0.2, 0.0, 0.2, 1.0, 0.0, 0.0, 0.0],
            "right_ee_joint_state": [0.7],
        }}
        arm_state = {
            "joint_positions_rad": [0.0] * 6,
            "joint_velocities_rad_s": [0.0] * 6,
            "joint_torques_nm": [0.0] * 6,
            "tcp_xyzrpy": [0.0] * 6,
            "tcp_xyzquat": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            "gripper_position_m": 0.0,
            "gripper_normalized": 1.0,
            "gripper_command_normalized": [1.0],
            "gripper_velocity_m_s": 0.0,
            "gripper_torque_nm": 0.0,
        }
        model.latest_state = {"arms": {"left": arm_state, "right": arm_state}}
        model.latest_images = {}
        model._icl_images = {}
        model._icl_summary = None
        model.model_cfg = {}
        model.previous = None
        model._started = True
        model.adapter = self.adapter
        model.arms = ("left", "right")
        model.step = 0
        model._terminal_reason = None
        model.agent = type("Agent", (), {
            "decide": lambda self, turn: {
                "name": "set_gripper",
                "arguments": {"positions": {"left": 0.0, "right": None}, "note": "close left"},
            }
        })()
        model._trace_event = lambda *_args, **_kwargs: None
        model._validate_reachability = lambda actions: None

        action = model.get_action()[0]

        self.assertEqual(action["left_ee_joint_state"], [0.0])
        self.assertEqual(action["right_ee_joint_state"], [0.7])

    def test_fixed_low_side_push_path_round_trips_through_world_tcp(self):
        angle = np.deg2rad(90.0)
        world_from_base = np.array([
            [np.cos(angle), -np.sin(angle), 0.0, 0.30],
            [np.sin(angle), np.cos(angle), 0.0, -0.45],
            [0.0, 0.0, 1.0, 0.765],
            [0.0, 0.0, 0.0, 1.0],
        ])
        calibration = RoboDojoCalibration.from_manifest({
            "world_from_base": {"left": world_from_base.tolist(), "right": np.eye(4).tolist()},
        })
        adapter = RoboDojoAdapter(calibration, ("left", "right"))
        # A safe diagnostic trajectory: constant low z, horizontal side push.
        path = [
            [0.10, 0.20, 0.11, 0.0, 0.0, 0.0, 1.0],
            [0.16, 0.20, 0.11, 0.0, 0.0, 0.0, 1.0],
            [0.22, 0.20, 0.11, 0.0, 0.0, 0.0, 1.0],
        ]
        for target in path:
            world_action = adapter.action({"target": {"left": target, "right": None}})
            frame = {"state": {
                "left_arm_joint_state": [0.0] * 6,
                "left_ee_joint_state": [1.0],
                "left_ee_pose": world_action["left_ee_pose"],
                "right_arm_joint_state": [0.0] * 6,
                "right_ee_joint_state": [1.0],
                "right_ee_pose": [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            }}
            measured = adapter.state(frame)["arms"]["left"]["tcp_xyzquat"]
            np.testing.assert_allclose(measured, target, atol=1e-9)

    def test_camera_ray_is_in_base_frame(self):
        ray = self.calibration.camera_ray_in_base("cam_head", [50, 50], "left")
        self.assertEqual(ray["frame"], "left_base_link")
        np.testing.assert_allclose(ray["ray_direction_base_xyz"], [0, 0, -1])

    def test_terminal_tool_aliases_are_normalized(self):
        self.assertEqual(_canonical_tool_name("done"), "terminal.done")
        self.assertEqual(_canonical_tool_name("give_up"), "terminal.give_up")
        self.assertEqual(_canonical_tool_name("terminal.done"), "terminal.done")

    def test_execution_feedback_reports_world_tcp_error(self):
        model = GPTPolicyModel.__new__(GPTPolicyModel)
        model.adapter = self.adapter
        model.arms = ("left", "right")
        model._last_commanded_world = {
            "left": [0.10, 0.20, 0.11, 1.0, 0.0, 0.0, 0.0],
            "right": [0.30, 0.20, 0.11, 1.0, 0.0, 0.0, 0.0],
        }
        feedback = model._execution_feedback({"state": {
            "left_ee_pose": [0.11, 0.20, 0.11, 1.0, 0.0, 0.0, 0.0],
            "right_ee_pose": [0.36, 0.20, 0.11, 1.0, 0.0, 0.0, 0.0],
        }})
        self.assertFalse(feedback["all_within_tolerance"])
        self.assertTrue(feedback["arms"]["left"]["within_tolerance"])
        self.assertAlmostEqual(feedback["arms"]["right"]["translation_error_m"], 0.06)

    def test_reachability_rejects_out_of_workspace_target(self):
        model = GPTPolicyModel.__new__(GPTPolicyModel)
        model.arms = ("left", "right")
        model.adapter = self.adapter
        model._reachability_bounds = {"x": [-0.05, 0.65], "y": [-0.55, 0.55], "z": [0.04, 0.45]}
        model.latest_frame = {"state": {
            "left_ee_pose": [0.10, 0.0, 0.16, 1.0, 0.0, 0.0, 0.0],
        }}
        result = model._validate_reachability([{
            "left_ee_pose": [-0.30, 0.0, 0.16, 1.0, 0.0, 0.0, 0.0],
        }])
        self.assertEqual(result["reason"], "unreachable")
        self.assertFalse(result["executed"])
        self.assertIn("workspace", result["error"])

    def test_reachability_allows_large_single_step_inside_workspace(self):
        model = GPTPolicyModel.__new__(GPTPolicyModel)
        model.arms = ("left", "right")
        model.adapter = self.adapter
        model._reachability_bounds = {"x": [-0.05, 0.65], "y": [-0.55, 0.55], "z": [0.04, 0.45]}
        model.latest_frame = {"state": {
            "left_ee_pose": [0.10, 0.0, 0.16, 1.0, 0.0, 0.0, 0.0],
        }}
        result = model._validate_reachability([{
            "left_ee_pose": [0.45, 0.0, 0.16, 1.0, 0.0, 0.0, 0.0],
        }])
        self.assertIsNone(result)

    def test_reachability_allows_low_table_clearance_when_z_lower_bound_is_disabled(self):
        model = GPTPolicyModel.__new__(GPTPolicyModel)
        model.arms = ("left", "right")
        model.adapter = self.adapter
        model._reachability_bounds = {"x": [-0.05, 0.65], "y": [-0.55, 0.55], "z": [None, 0.45]}
        model.latest_frame = {"state": {
            "right_ee_pose": [0.30, 0.10, 0.16, 1.0, 0.0, 0.0, 0.0],
        }}
        result = model._validate_reachability([{
            "right_ee_pose": [0.397, 0.1006, 0.027, 1.0, 0.0, 0.0, 0.0],
        }])
        self.assertIsNone(result)

    def test_locate_point_requires_valid_triangulation_quality(self):
        model = GPTPolicyModel.__new__(GPTPolicyModel)
        model.arms = ("left",)
        model.latest_frame = {"vision": {"cam_left_wrist": {}}}
        model._observation_history = {0: {"vision": {"cam_left_wrist": {}}}}
        # Two wrist views with a clear baseline intersect at (0, 0, 1).
        ray_a = (np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0]))
        ray_b = (np.array([0.2, 0.0, 0.0]), np.array([-0.2, 0.0, 1.0]) / np.linalg.norm([0.2, 0.0, 1.0]))
        rays = iter([ray_a, ray_b, ray_a, ray_b])
        model._frame_ray = lambda *_args: next(rays)
        result = model._locate_point({
            "camera": "cam_left_wrist", "pixel_xy": [10, 10],
            "reference_step": 0, "reference_pixel_xy": [11, 10],
        })
        self.assertTrue(result["metric_position_available"])
        self.assertIn("triangulation_by_arm", result)


if __name__ == "__main__":
    unittest.main()
