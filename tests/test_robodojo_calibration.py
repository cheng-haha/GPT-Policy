import json
import unittest
from pathlib import Path

import numpy as np

from gpt_policy.robodojo import RoboDojoAdapter, RoboDojoCalibration
from gpt_policy.robodojo.model import GPTPolicyModel


class RoboDojoCalibrationTest(unittest.TestCase):
    def setUp(self):
        self.intrinsics = np.array([[400.0, 0.0, 320.0], [0.0, 360.0, 240.0], [0.0, 0.0, 1.0]])
        self.manifest = {
            "world_from_base": {"left": np.eye(4).tolist()},
            "intrinsics": {"camera": self.intrinsics.tolist()},
            "camera_extrinsics_world": {"camera": np.eye(4).tolist()},
        }

    def test_usd_pixel_axes_and_forward_direction(self):
        calibration = RoboDojoCalibration.from_manifest(self.manifest)
        for pixel, direction in (
            ([320, 240], [0, 0, -1]),
            ([720, 240], [1, 0, -1]),
            ([320, 600], [0, -1, -1]),
        ):
            with self.subTest(pixel=pixel):
                ray = calibration.camera_ray_in_base("camera", pixel, "left")
                np.testing.assert_allclose(ray["ray_direction_base_xyz"], direction / np.linalg.norm(direction))

    def test_explicit_opencv_pose_is_not_flipped_again(self):
        calibration = RoboDojoCalibration.from_manifest(self.manifest | {"camera_extrinsic_axes": "opencv"})
        ray = calibration.camera_ray_in_base("camera", [320, 600], "left")
        np.testing.assert_allclose(ray["ray_direction_base_xyz"], np.array([0, 1, 1]) / np.sqrt(2))

    def test_projection_round_trip_with_rotated_camera_and_base(self):
        world_from_base = np.array([[0, -1, 0, -0.3], [1, 0, 0, -0.45], [0, 0, 1, 0.765], [0, 0, 0, 1]])
        world_from_camera = np.array([[1, 0, 0, 0.2], [0, 0.8, -0.6, -0.1], [0, 0.6, 0.8, 1.2], [0, 0, 0, 1]])
        calibration = RoboDojoCalibration.from_manifest(self.manifest | {
            "world_from_base": {"left": world_from_base.tolist()},
            "camera_extrinsics_world": {"camera": world_from_camera.tolist()},
        })
        for point_camera in ([0.1, 0.05, -0.4, 1], [-0.08, -0.04, -0.6, 1]):
            with self.subTest(point=point_camera):
                point_usd = np.array(point_camera)
                pixel = self.intrinsics @ np.array([point_usd[0], -point_usd[1], -point_usd[2]])
                point_base = np.linalg.solve(world_from_base, world_from_camera @ point_usd)[:3]
                ray = calibration.camera_ray_in_base("camera", pixel[:2] / pixel[2], "left")
                expected = point_base - ray["ray_origin_base_xyz"]
                np.testing.assert_allclose(ray["ray_direction_base_xyz"], expected / np.linalg.norm(expected), atol=1e-12)

    def test_observation_replaces_stale_matrices_without_mutation(self):
        camera_pose = np.eye(4)
        camera_pose[:3, 3] = [0.1, 0.2, 0.3]
        intrinsics = self.intrinsics.copy()
        intrinsics[0, 2] = 300
        frame = {"vision": {"camera": {"intrinsic_matrix": intrinsics, "extrinsic_matrix": camera_pose}}}
        calibration = RoboDojoCalibration.from_observation(self.manifest, frame)
        ray = calibration.camera_ray_in_base("camera", [300, 240], "left")
        np.testing.assert_allclose(ray["ray_origin_base_xyz"], [0.1, 0.2, 0.3])
        np.testing.assert_allclose(ray["ray_direction_base_xyz"], [0, 0, -1])
        np.testing.assert_array_equal(self.manifest["camera_extrinsics_world"]["camera"], np.eye(4))
        np.testing.assert_array_equal(frame["vision"]["camera"]["extrinsic_matrix"], camera_pose)

    def test_context_distinguishes_raw_usd_and_optical_transforms(self):
        calibration = RoboDojoCalibration.from_manifest(self.manifest)
        frame = {"vision": {"camera": {"color": np.zeros((4, 4, 3), dtype=np.uint8)}}}
        camera = RoboDojoAdapter(calibration, ("left",)).cameras(frame)[0]
        self.assertNotIn("color", camera)
        self.assertEqual(camera["extrinsic_axes"], "usd")
        np.testing.assert_array_equal(camera["extrinsic_matrix"], np.eye(4))
        np.testing.assert_array_equal(camera["base_from_optical_camera"]["left"], np.diag([1, -1, -1, 1]))

    def test_live_base_pose_overrides_nominal_manifest(self):
        measured_base = np.eye(4)
        measured_base[:3, 3] = [0.1, -0.2, 0.3]
        calibration = RoboDojoCalibration.from_observation(self.manifest, {"calibration": {"world_from_base": {"left": measured_base.tolist()}}})
        np.testing.assert_array_equal(calibration.world_from_base["left"], measured_base)
        np.testing.assert_array_equal(self.manifest["world_from_base"]["left"], np.eye(4))

    def test_environment_offset_is_applied_once_to_state_and_action(self):
        origin = np.eye(4)
        origin[:3, 3] = [7.0, -7.0, 0.0]
        base = origin.copy()
        base[0, 3] += 0.1
        calibration = RoboDojoCalibration.from_observation(self.manifest, {"calibration": {"world_from_base": {"left": base.tolist()}, "world_from_environment": origin.tolist()}})
        adapter = RoboDojoAdapter(calibration, ("left",))
        source = [0.3, 0.0, 0.2, 1.0, 0.0, 0.0, 0.0]
        tcp = adapter.state({"state": {"ee_pose": source}})["tcp_xyzquat"]
        np.testing.assert_allclose(tcp[:3], [0.2, 0.0, 0.2], atol=1e-12)
        np.testing.assert_allclose(adapter.action({"target": tcp})["ee_pose"], source, atol=1e-12)

    def test_x5_tcp_offset_and_tool_axes_under_rotations(self):
        manifest = self.manifest | {"tcp_from_link6": {"left": [[0, 0, -1, 0], [0, 1, 0, 0], [1, 0, 0, -0.145], [0, 0, 0, 1]]}}
        calibration = RoboDojoCalibration.from_manifest(manifest)
        adapter = RoboDojoAdapter(calibration, ("left",))
        from gpt_policy.robodojo.calibration import quat_wxyz_to_matrix

        generator = np.random.default_rng(42)
        for quaternion in generator.normal(size=(20, 4)):
            quaternion /= np.linalg.norm(quaternion)
            source = [0.2, -0.1, 0.3, *quaternion]
            tcp = adapter.state({"state": {"ee_pose": source}})["tcp_xyzquat"]
            rotation = quat_wxyz_to_matrix(quaternion)
            np.testing.assert_allclose(tcp[:3], np.array(source[:3]) + rotation @ [0.145, 0, 0], atol=1e-12)
            tcp_rotation = quat_wxyz_to_matrix([tcp[6], *tcp[3:6]])
            np.testing.assert_allclose(tcp_rotation[:, 2], rotation[:, 0], atol=1e-12)
            action = adapter.action({"target": tcp})["ee_pose"]
            np.testing.assert_allclose(action[:3], source[:3], atol=1e-12)
            np.testing.assert_allclose(quat_wxyz_to_matrix(action[3:]), rotation, atol=1e-12)

    def test_franka_tcp_uses_hand_z_offset(self):
        calibration = RoboDojoCalibration.from_manifest(self.manifest | {"tcp_from_link6": {"left": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, -0.102], [0, 0, 0, 1]]}})
        transform = calibration.world_from_tcp("left", [0, 0, 0.5, 0, 1, 0, 0])
        np.testing.assert_allclose(transform[:3, 3], [0, 0, 0.398])

    def test_feedback_measures_tcp_not_just_stationary_source_link(self):
        model = GPTPolicyModel.__new__(GPTPolicyModel)
        model.arms = ("left",)
        calibration = RoboDojoCalibration.from_manifest(self.manifest | {"tcp_from_link6": {"left": [[0, 0, -1, 0], [0, 1, 0, 0], [1, 0, 0, -0.145], [0, 0, 0, 1]]}})
        model.adapter = RoboDojoAdapter(calibration, model.arms)
        model._last_commanded_world = {"left": [0, 0, 0.3, 1, 0, 0, 0]}
        result = model._execution_feedback({"state": {"ee_pose": [0, 0, 0.3, np.sqrt(0.5), 0, 0, np.sqrt(0.5)]}})
        self.assertAlmostEqual(result["arms"]["left"]["translation_error_m"], 0.145 * np.sqrt(2))


class RoboDojoCameraReplayTest(unittest.TestCase):
    def setUp(self):
        self.fixture = json.loads((Path(__file__).parent / "fixtures" / "robodojo_push_t_camera_replay.json").read_text())
        self.model = GPTPolicyModel.__new__(GPTPolicyModel)
        self.model.arms = ("left", "right")
        self.model._calibration_manifest = self.fixture["manifest"]
        self.model._observation_history = {int(step): frame for step, frame in self.fixture["frames"].items()}

    def test_recorded_head_and_wrist_center_rays_point_down(self):
        calibration = RoboDojoCalibration.from_observation(self.fixture["manifest"], self.fixture["frames"]["0"])
        for camera in ("cam_head", "cam_left_wrist", "cam_right_wrist"):
            with self.subTest(camera=camera):
                ray = calibration.camera_ray_in_base(camera, [320, 240], "right")
                self.assertLess(ray["ray_direction_base_xyz"][2], -0.49)

    def test_recorded_three_baselines_have_positive_depth_and_submillimetre_residual(self):
        corner_points = []
        for pair in self.fixture["pairs"]:
            with self.subTest(step=pair["current_step"]):
                self.model.latest_frame = self.fixture["frames"][str(pair["current_step"])]
                result = self.model._locate_point(pair["arguments"])
                self.assertTrue(result["metric_position_available"])
                for candidate in result["triangulation_by_arm"].values():
                    self.assertTrue(candidate["triangulation_valid"])
                    self.assertTrue(all(depth > 0 for depth in candidate["ray_depths_m"]))
                    self.assertLess(candidate["triangulation_residual_m"], 0.0003)
                    self.assertAlmostEqual(candidate["triangulation_candidate_base_xyz"][2], 0.014, delta=0.002)
                if pair["current_step"] in (9, 10):
                    corner_points.append(result["triangulation_by_arm"]["right"]["triangulation_candidate_base_xyz"])
        self.assertLess(np.linalg.norm(np.array(corner_points[0]) - corner_points[1]), 0.001)

    def test_same_fixed_head_view_does_not_create_metric_depth(self):
        self.model.latest_frame = self.fixture["frames"]["0"]
        result = self.model._locate_point({"camera": "top", "pixel_xy": [326, 254], "reference_step": 0, "reference_pixel_xy": [326, 254]})
        self.assertFalse(result["metric_position_available"])


if __name__ == "__main__":
    unittest.main()
