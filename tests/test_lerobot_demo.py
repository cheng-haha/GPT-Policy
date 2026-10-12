"""Offline integrity tests: episode boundaries, derived labels, real image wire."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

from gpt_policy.input.action_sampling import compress_samples
from gpt_policy.input.lerobot_demo import (
    CAMERAS, LeRobotEpisodeExtractor, checked_frame_audit, endpose_sequence, numeric_samples, prepare, validate_rows,
)
from gpt_policy.geometry.poses import rotation_6d_to_quaternion, rpy_to_matrix, quaternion_to_rpy
from gpt_policy.input.video import FrameSelection, SelectedFrame


def fixture(root):
    info = {"codebase_version": "v3.0", "robot_type": "arx_x5", "fps": 25,
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        "features": {key: {"shape": [4, 6, 3]} for key in CAMERAS.values()},
        "starvla_metadata": {"action_mode": "absolute_next_step", "action": {"frame_offset": 1},
            "rotation_6d_convention": "first_two_rotation_matrix_columns_column_major",
            "coordinate_system": {"type": "world"}}}
    meta = {"episode_index": 1, "length": 4, "dataset_from_index": 748, "dataset_to_index": 752,
        "tasks": ["Sort objects."], "data/chunk_index": 0, "data/file_index": 0}
    files = ["data/chunk-000/file-000.parquet", "meta/tasks.parquet", "meta/episodes/chunk-000/file-000.parquet"]
    for key in CAMERAS.values():
        meta.update({f"videos/{key}/chunk_index": 0, f"videos/{key}/file_index": 0,
            f"videos/{key}/from_timestamp": 29.92, f"videos/{key}/to_timestamp": 30.08})
        files.append(f"videos/{key}/chunk-000/file-000.mp4")
    for file in files:
        path = root/file
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fixture")
    (root/"meta/info.json").write_text(json.dumps(info))
    (root/"meta/source_stage2.json").write_text("{}")
    rows = []
    endposes = [[float(i), 0., 0., 1., 0., 0., 0., 1., 0., 0.] * 2 for i in range(4)]
    for i in range(4):
        rows.append({"episode_index": 1, "frame_index": i, "index": 748+i, "timestamp": i/25,
            "observation.state": [0.] * 14, "action": [0.] * 14,
            "observation.states.eef": endposes[i],
            "actions.eef": endposes[min(i+1,3)].copy()})
    return info, meta, rows


class FakeDataset:
    def __init__(self, repo_id, root, episodes, **kwargs):
        info, meta, self.hf_dataset = fixture(root)
        self.meta = SimpleNamespace(info=info, episodes=[meta])
        assert episodes == [1]

    def __getitem__(self, index):
        return {**self.hf_dataset[index], **{k: np.full((3,4,6), index/4) for k in CAMERAS.values()}}


class FakeSelector:
    config = SimpleNamespace(model="test-model")

    def cache_identity(self):
        return {"selector": "offline-fixture"}

    def review_identity(self):
        return "offline-fixture"

    def select(self, instruction, video, extraction):
        return FrameSelection(tuple(SelectedFrame(f.index, "fixture evidence") for f in extraction.candidates),
                              "fixture summary", {})

    def review(self, instruction, video, frames, selection, limit):
        return selection


class TestLeRobotDemo(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)/"dataset"
        self.info, self.meta, self.rows = fixture(self.root)

    def tearDown(self):
        self.temp.cleanup()

    def test_nonzero_episode_and_video_offset(self):
        extractor = LeRobotEpisodeExtractor(self.root, 1, dataset_factory=FakeDataset)
        frames = [{"video_frame_index": 0}, {"video_frame_index": 3}]
        extractor.attach(frames)
        self.assertEqual(frames[0]["alignment"]["dataset_index"], 748)
        self.assertAlmostEqual(frames[-1]["alignment"]["video_timestamps_s"]["top"], 30.04)
        self.assertNotIn("action", frames[-1])
        self.assertEqual(len(frames[0]["action"]["samples"]), 4)

    def test_reject_foreign_episode_missing_frame_and_repeated_timestamp(self):
        for field, value in [("episode_index", 0), ("frame_index", 9), ("timestamp", 0)]:
            rows = copy.deepcopy(self.rows)
            rows[1][field] = value
            with self.assertRaises(ValueError):
                validate_rows(rows, 1, self.meta)

    def test_next_eef_is_checked_but_never_labeled_command(self):
        samples = numeric_samples(self.rows)
        self.assertNotIn("eef_target_xyz_xyzw", samples[0]["left"])
        self.assertEqual(samples[0]["left"]["endpose_next_state_xyz_xyzw_gripper"],
                         [1., 0., 0., 0., 0., 0., 1., 0.])
        self.rows[1]["actions.eef"][0] = 999.
        with self.assertRaisesRegex(ValueError, "next-state"):
            numeric_samples(self.rows)

    def test_rotation_columns_roundtrip_and_pi(self):
        rng = np.random.default_rng(12)
        angles = [np.zeros(3), [np.pi, 0, 0], [0, np.pi, 0], [0, 0, np.pi],
                  [0, np.pi/2, 0], *rng.uniform(-np.pi, np.pi, size=(100, 3))]
        for rpy in angles:
            matrix = rpy_to_matrix(rpy)
            q = rotation_6d_to_quaternion(matrix[:, :2].T.reshape(6))
            np.testing.assert_allclose(rpy_to_matrix(quaternion_to_rpy(q)), matrix, atol=1e-10)
            self.assertAlmostEqual(np.linalg.norm(q), 1.)
        # A +90-degree yaw maps tool x to base y, fixing the column/order convention.
        np.testing.assert_allclose(rotation_6d_to_quaternion([0, 1, 0, -1, 0, 0]),
                                   [0, 0, np.sqrt(.5), np.sqrt(.5)], atol=1e-12)

    def test_reject_corrupt_rotation_and_allow_float_drift(self):
        for value in ([0]*6, [1,0,0,1,0,0], [2,0,0,0,1,0], [np.nan]*6, [1]*5):
            with self.assertRaises(ValueError):
                rotation_6d_to_quaternion(value)
        np.testing.assert_allclose(rotation_6d_to_quaternion([1+1e-6,0,0,1e-6,1,0]),
                                   [0,0,0,1], atol=1e-12)

    def test_continuous_signs_and_next_state_have_no_extra_shift(self):
        for row, yaw in zip(self.rows, np.deg2rad([170, 179, 181, 190])):
            rotation = rpy_to_matrix([0,0,yaw])[:, :2].T.reshape(6).tolist()
            row["observation.states.eef"][3:9] = rotation
            row["observation.states.eef"][13:19] = rotation
        for i, row in enumerate(self.rows):
            row["actions.eef"] = self.rows[min(i+1,3)]["observation.states.eef"].copy()
        original = copy.deepcopy(self.rows)
        converted = endpose_sequence(self.rows)
        self.assertTrue(np.all(np.sum(converted[1:,:,3:7]*converted[:-1,:,3:7], axis=-1) > 0))
        samples = numeric_samples(self.rows)
        for i in range(4):
            for arm, a in (("left", 0), ("right", 1)):
                np.testing.assert_array_equal(samples[i][arm]["endpose_next_state_xyz_xyzw_gripper"],
                                              converted[min(i+1,3),a])
        self.assertEqual(self.rows, original)

    def test_reject_nonfinite_or_wrong_shape_next_state(self):
        for value in ([0]*19, [np.nan]*20):
            rows = copy.deepcopy(self.rows)
            rows[0]["actions.eef"] = value
            with self.assertRaises(ValueError):
                numeric_samples(rows)

    def test_frame_audit_must_match_episode_and_source_hashes(self):
        extractor = LeRobotEpisodeExtractor(self.root, 1, dataset_factory=FakeDataset)
        audit = {"dataset": str(self.root), "episode": 1, "frames": 4,
                 "matches_native_link6_in_environment_frame": True,
                 "files": {str((self.root/k).resolve()): v for k,v in extractor.source_hashes.items()}}
        path = Path(self.temp.name)/"audit.json"
        path.write_text(json.dumps(audit))
        self.assertEqual(checked_frame_audit(path, extractor), audit)
        for field, value in [("episode", 0), ("frames", 3), ("files", {}),
                             ("matches_native_link6_in_environment_frame", False)]:
            path.write_text(json.dumps({**audit, field: value}))
            with self.assertRaises(ValueError):
                checked_frame_audit(path, extractor)

    def test_audited_source_frame_reaches_context_without_tcp_relabel(self):
        from unittest.mock import patch
        extractor = LeRobotEpisodeExtractor(self.root, 1, dataset_factory=FakeDataset)
        audit = {"dataset": str(self.root), "episode": 1, "frames": 4,
                 "matches_native_link6_in_environment_frame": True,
                 "files": {str((self.root/k).resolve()): v for k,v in extractor.source_hashes.items()}}
        path = Path(self.temp.name)/"audit.json"
        path.write_text(json.dumps(audit))
        output = Path(self.temp.name)/"audited"
        with patch("gpt_policy.input.lerobot_demo.LeRobotEpisodeExtractor", return_value=extractor):
            prepare(self.root, 1, output, FakeSelector(), frame_audit=path)
        header = json.loads(json.loads((output/"prepared/input.json").read_text())["content"][1])
        self.assertEqual(header["coordinate_frame"]["verified_source"]["site"], "link6")
        self.assertEqual(header["coordinate_frame"]["verified_source"]["frame"], "environment_origin")
        self.assertIn("TCP/base mapping still required", header["coordinate_frame"]["mapping_to_current_robot"])
        self.assertEqual(json.loads((output/"frame_audit.json").read_text()), audit)

    def test_reference_gripper_transition_survives_compression(self):
        rows = [{"t_s": i/25, "left": {"gripper_reference": 0. if i < 8 else 1.}}
                for i in range(20)]
        sampled, _ = compress_samples(rows)
        self.assertIn(rows[7], sampled)
        self.assertIn(rows[8], sampled)

    def test_prepared_bundle_loads_actual_images_portably(self):
        from unittest.mock import patch
        factory = LeRobotEpisodeExtractor
        output = Path(self.temp.name)/"output"
        with patch("gpt_policy.input.lerobot_demo.LeRobotEpisodeExtractor",
                   side_effect=lambda root, episode, config: factory(root, episode, config, FakeDataset)):
            result = prepare(self.root, 1, output, FakeSelector())
        self.assertTrue(result["ok"])
        self.assertEqual(result["images"], 6)
        demo = json.loads((output/"reviewed.json").read_text())
        self.assertEqual(demo["keyframes"][0]["t_s"], 0.)
        self.assertEqual(demo["keyframes"][-1]["t_s"], .12)
        for frame in demo["keyframes"]:
            for path in frame["images"].values():
                self.assertTrue((output/path).is_file())
        self.assertIn("unverified", demo["outcome"])
        from gpt_policy.input.video_cache import _sha256_file
        provenance = json.loads((output/"prepared/provenance.json").read_text())
        self.assertEqual(provenance["sha256"], _sha256_file(output/"source_reviewed.json"))
        raw = [json.loads(line) for line in (output/"raw_endposes.jsonl").read_text().splitlines()]
        self.assertEqual(raw[0]["observation.states.eef"], self.rows[0]["observation.states.eef"])
        self.assertEqual(len(demo["keyframes"][0]["state"]["left"]["endpose_xyz_xyzw_gripper"]), 8)
        compiled = (output/"prepared/input.json").read_text()
        self.assertNotIn("xyz_rot6d_gripper", compiled)
        self.assertIn("no frame or reference-point transform", compiled)

    def test_model_smoke_uses_native_tool_response_envelope(self):
        from unittest.mock import patch
        from gpt_policy.input.lerobot_demo import model_smoke
        path = Path(self.temp.name)/"input.json"
        path.write_text(json.dumps({"instruction": "Sort", "content": ["historical reference"]}))
        class Client:
            def __init__(self, *args): pass
            def start_thread(self, instructions): pass
            def close(self): pass
            def decide(self, text, schema, content):
                assert schema["required"] == ["name", "arguments"]
                assert schema["properties"]["name"]["const"] == "inspect_context"
                return {"name": "inspect_context", "arguments": {"task": "Sort"}}
        config = SimpleNamespace(model="test", executable="unused", effort="medium")
        with patch("gpt_policy.harness.codex.CodexAppServer", Client):
            self.assertEqual(model_smoke(path, config), {"task": "Sort"})

    def test_confirmed_expert_outcome_reaches_compiled_context(self):
        from unittest.mock import patch
        factory = LeRobotEpisodeExtractor
        output = Path(self.temp.name)/"expert"
        with patch("gpt_policy.input.lerobot_demo.LeRobotEpisodeExtractor",
                   side_effect=lambda root, episode, config: factory(root, episode, config, FakeDataset)):
            prepare(self.root, 1, output, FakeSelector(), confirmed_expert=True)
        compiled = json.loads((output/"prepared/input.json").read_text())
        header = json.loads(compiled["content"][1])
        self.assertEqual(header["demonstrator"], "expert")
        self.assertTrue(header["outcome"]["success"])
        self.assertEqual(header["outcome"]["source"], "dataset_owner_confirmation")
        self.assertEqual(header["outcome"]["scope"], "historical_expert_demonstration")
        images = [p for p in compiled["content"] if isinstance(p, dict) and "image" in p]
        self.assertEqual(len(images), 6)  # Two timestamps, all three views retained.


if __name__ == "__main__":
    unittest.main()
