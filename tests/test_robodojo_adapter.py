import unittest

import numpy as np

from gpt_policy.robodojo import RoboDojoAdapter, RoboDojoCalibration


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

    def test_camera_ray_is_in_base_frame(self):
        ray = self.calibration.camera_ray_in_base("cam_head", [50, 50], "left")
        self.assertEqual(ray["frame"], "left_base_link")
        np.testing.assert_allclose(ray["ray_direction_base_xyz"], [0, 0, 1])


if __name__ == "__main__":
    unittest.main()
