import unittest
from types import SimpleNamespace
from unittest.mock import patch

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

    def test_bimanual_gripper_close_reaches_submitted_action(self):
        model = GPTPolicyModel.__new__(GPTPolicyModel)
        model.arms = ("left", "right")
        model.adapter = self.adapter
        model.latest_frame = {"state": {
            "left_ee_pose": [0, 0, 0, 1, 0, 0, 0],
            "right_ee_pose": [0, 0, 0, 1, 0, 0, 0],
            "left_ee_joint_state": [1.0],
            "right_ee_joint_state": [1.0],
        }}
        model.latest_state = {}
        model.latest_images = {}
        model._icl_images = {}
        model._terminal_reason = None
        model._started = True
        model.step = 0
        model.model_cfg = {}
        model.previous = None
        events = []
        model._trace_event = lambda event, payload: events.append((event, payload))

        for positions, expected in [
            ({"left": 0, "right": None}, (0.0, 1.0)),
            ({"left": None, "right": 0.05}, (1.0, 0.05)),
            ({"left": None, "right": None}, (1.0, 1.0)),
        ]:
            with self.subTest(positions=positions):
                model.agent = SimpleNamespace(decide=lambda _turn: {
                    "name": "set_gripper", "arguments": {"positions": positions, "note": "test"},
                })
                with patch("gpt_policy.robodojo.model.observation", return_value="test"):
                    action = model.get_action()[0]
                self.assertEqual(action["left_ee_joint_state"], [expected[0]])
                self.assertEqual(action["right_ee_joint_state"], [expected[1]])
                self.assertEqual(events[-1][1]["actions"][0], action)

    def test_execution_feedback_reports_world_tcp_error(self):
        model = GPTPolicyModel.__new__(GPTPolicyModel)
        model.adapter = self.adapter
        model.arms = ("left", "right")
        model.step = 0
        model._observation_history = {}
        model._max_tracking_error_m = 0.03
        model._max_tracking_error_rad = 0.20
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

    def test_ik_rejected_arm_is_not_counted_as_tracking_failure(self):
        model = GPTPolicyModel.__new__(GPTPolicyModel)
        model.adapter = self.adapter
        model.arms = ("left", "right")
        model.step = 0
        model._observation_history = {}
        model._max_tracking_error_m = 0.03
        model._max_tracking_error_rad = 0.20
        model._last_commanded_world = {
            "left": [0.10, 0.20, 0.11, 1.0, 0.0, 0.0, 0.0],
            "right": [0.30, 0.20, 0.11, 1.0, 0.0, 0.0, 0.0],
        }
        feedback = model._execution_feedback(
            {"state": {
                "left_ee_pose": [0.10, 0.20, 0.11, 1.0, 0.0, 0.0, 0.0],
                "right_ee_pose": [0.30, 0.20, 0.11, 1.0, 0.0, 0.0, 0.0],
            }},
            {"status": "ik_failed", "arms": {
                "left": {"status": "Success"},
                "right": {"status": "ik_failed", "reason": "no solution"},
            }},
        )
        self.assertEqual(feedback["not_executed_arms"], ["right"])
        self.assertFalse(feedback["arms"]["right"]["tracking_checked"])
        self.assertTrue(feedback["all_within_tolerance"])

    def test_incomplete_native_move_is_not_reported_as_unexecuted_or_stalled(self):
        model = GPTPolicyModel.__new__(GPTPolicyModel)
        model.adapter = self.adapter
        model.arms = ("left", "right")
        model.step = 1
        model._max_tracking_error_m = 0.03
        model._max_tracking_error_rad = 0.20
        model._observation_history = {0: {"state": {
            "left_ee_pose": [0.0, 0.0, 0.10, 1.0, 0.0, 0.0, 0.0],
        }}}
        model._last_commanded_world = {
            "left": [0.05, 0.0, 0.10, 1.0, 0.0, 0.0, 0.0],
        }
        feedback = model._execution_feedback(
            {"state": {"left_ee_pose": [0.015, 0.0, 0.10, 1.0, 0.0, 0.0, 0.0]}},
            {"status": "incomplete", "arms": {"left": {"status": "Success", "tracking_status": "incomplete"}}},
        )
        self.assertEqual(feedback["not_executed_arms"], [])
        self.assertEqual(feedback["arms"]["left"]["status"], "incomplete")
        self.assertFalse(feedback["arms"]["left"]["stalled"])
        self.assertAlmostEqual(feedback["arms"]["left"]["translation_progress_m"], 0.015)

    def test_nonmoving_arm_is_reported_as_stalled(self):
        model = GPTPolicyModel.__new__(GPTPolicyModel)
        model.adapter = self.adapter
        model.arms = ("left", "right")
        model.step = 1
        model._max_tracking_error_m = 0.03
        model._max_tracking_error_rad = 0.20
        model._observation_history = {0: {"state": {
            "left_ee_pose": [0.0, 0.0, 0.10, 1.0, 0.0, 0.0, 0.0],
        }}}
        model._last_commanded_world = {
            "left": [0.05, 0.0, 0.10, 1.0, 0.0, 0.0, 0.0],
        }
        feedback = model._execution_feedback({"state": {
            "left_ee_pose": [0.0, 0.0, 0.10, 1.0, 0.0, 0.0, 0.0],
        }})
        self.assertTrue(feedback["arms"]["left"]["stalled"])

    def test_reachability_rejects_out_of_workspace_target(self):
        model = GPTPolicyModel.__new__(GPTPolicyModel)
        model.arms = ("left", "right")
        model.adapter = self.adapter
        model._reachability_bounds = {"x": [-0.05, 0.65], "y": [-0.55, 0.55], "z": [0.04, 0.45]}
        model._max_reachability_rotation_rad = 0.35
        model.latest_frame = {"state": {
            "left_ee_pose": [0.10, 0.0, 0.16, 1.0, 0.0, 0.0, 0.0],
        }}
        result = model._validate_reachability([{
            "left_ee_pose": [-0.30, 0.0, 0.16, 1.0, 0.0, 0.0, 0.0],
        }])
        self.assertEqual(result["reason"], "workspace_limit")
        self.assertFalse(result["executed"])
        self.assertIn("workspace", result["error"])

    def test_reachability_allows_large_single_translation_within_workspace(self):
        model = GPTPolicyModel.__new__(GPTPolicyModel)
        model.arms = ("left", "right")
        model.adapter = self.adapter
        model._reachability_bounds = {"x": [-0.05, 0.65], "y": [-0.55, 0.55], "z": [0.04, 0.45]}
        model._max_reachability_rotation_rad = 0.35
        model.latest_frame = {"state": {
            "left_ee_pose": [0.10, 0.0, 0.16, 1.0, 0.0, 0.0, 0.0],
        }}
        result = model._validate_reachability([{
            "left_ee_pose": [0.45, 0.0, 0.16, 1.0, 0.0, 0.0, 0.0],
        }])
        self.assertIsNone(result)

    def test_reachability_enforces_published_eef_action_bounds(self):
        model = GPTPolicyModel.__new__(GPTPolicyModel)
        model.arms = ("left", "right")
        model.adapter = self.adapter
        model._reachability_bounds = {"x": [-0.05, 0.65], "y": [-0.55, 0.55], "z": [None, 0.45]}
        model._max_reachability_rotation_rad = 0.35
        model.latest_frame = {"state": {
            "left_ee_pose": [0.10, 0.0, 0.11, 1.0, 0.0, 0.0, 0.0],
        }}
        far = model._validate_reachability([{
            "left_ee_pose": [0.16, 0.0, 0.11, 1.0, 0.0, 0.0, 0.0],
        }])
        self.assertIsNone(far)
        angle = 0.4
        rotated = model._validate_reachability([{
            "left_ee_pose": [0.10, 0.0, 0.11, np.cos(angle / 2), 0.0, 0.0, np.sin(angle / 2)],
        }])
        self.assertEqual(rotated["reason"], "unreachable")
        self.assertAlmostEqual(rotated["max_rotation_step_rad"], 0.35)

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
