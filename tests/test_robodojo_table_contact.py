import json
from pathlib import Path
import unittest

from gpt_policy.robodojo import RoboDojoAdapter, RoboDojoCalibration
from gpt_policy.robodojo.model import GPTPolicyModel


ROOT = Path(__file__).resolve().parents[1]


class TableContactTest(unittest.TestCase):
    def setUp(self):
        model = GPTPolicyModel.__new__(GPTPolicyModel)
        model.robot_model = "X5"
        model.arms = ("left", "right")
        model._reachability_bounds = {"x": [-0.05, 0.65], "y": [-0.55, 0.55], "z": [None, 0.45]}
        model._max_reachability_rotation_rad = 0.35
        manifest = json.loads((ROOT / "configs/examples/robodojo_calibration.json").read_text())
        model.adapter = RoboDojoAdapter(RoboDojoCalibration.from_manifest(manifest))
        initial = model.adapter.action({"target": {"left": {"pose_xyzquat": [0.14, -0.052, 0.09, 0, 1, 0, 0]}}})
        model.latest_frame = {"task_name": "fold_clothes", "state": initial}
        self.model = model

    def target(self, z, quaternion=(0, 1, 0, 0)):
        return self.model.adapter.action({"target": {"left": {
            "pose_xyzquat": [0.148, -0.052, z, *quaternion]}}})

    def test_failed_run_25mm_request_is_allowed(self):
        self.assertIsNone(self.model._validate_reachability([self.target(0.025)]))
        self.assertIsNone(self.model._validate_reachability([self.target(0.015)]))

    def test_no_policy_floor_even_below_base(self):
        self.assertIsNone(self.model._validate_reachability([self.target(-0.01)]))

    def test_height_ceiling_still_rejects_and_serializes_null_floor(self):
        result = self.model._validate_reachability([self.target(0.50)])
        self.assertEqual(result["reason"], "workspace_limit")
        self.assertIsNone(json.loads(json.dumps(result))["workspace_bounds"]["z"][0])

    def test_other_tasks_also_have_no_default_floor(self):
        self.model.latest_frame["task_name"] = "push_T"
        self.assertIsNone(self.model._validate_reachability([self.target(0.025)]))

    def test_every_chunk_waypoint_is_checked(self):
        result = self.model._validate_reachability([self.target(0.025), self.target(0.50)])
        self.assertEqual(result["waypoint_index"], 1)


if __name__ == "__main__":
    unittest.main()
