"""Next-state labels must not acquire an extra shift when restoring sparse context."""
import copy
import json
import sys
import unittest
from pathlib import Path

from gpt_policy.input.demonstration import _frame_context

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from restore_make_kong_action_references import (
    COLUMNS,
    decode_rows,
    objects,
    overview,
    restore,
)


def fixture():
    source = {"dataset": "make_kong", "episode_index": 0}
    measured = [{"t_s": i * .04, "source_frame_index": i,
                 **{arm: {"joint_measured_rad": [i * .1] * 6,
                          "eef_measured_xyz_xyzw": [i * .01, 0., .5, 0., 0., 0., 1.],
                          "gripper_measured": 1. - i * .1} for arm in ("left", "right")}}
                for i in range(6)]
    raw = []
    for i, row in enumerate(measured):
        sample = {"t_s": row["t_s"], "source_frame_index": i}
        for arm in ("left", "right"):
            current, nxt = row[arm], measured[min(i + 1, 5)][arm]
            sample[arm] = {"joint_measured_rad": current["joint_measured_rad"],
                           "joint_reference_rad": nxt["joint_measured_rad"],
                           "gripper_measured": current["gripper_measured"],
                           "gripper_reference": nxt["gripper_measured"],
                           "endpose_next_state_xyz_xyzw_gripper": nxt["eef_measured_xyz_xyzw"] + [nxt["gripper_measured"]]}
        raw.append(sample)
    columns = [["t_s"], ["source_frame_index"]] + [[arm, field] for arm in ("left", "right")
               for field in ("joint_measured_rad", "eef_measured_xyz_xyzw", "gripper_measured")]
    header = {"source": source, "summary": "Expert facts. Execution notes: keep this wording.",
              "coordinate_frame": {"action_semantics": "measured"},
              "numeric_reference_semantics": "measured_states",
              "action_sample_encoding": {"columns": columns, "description": "measured"}}
    frame = {"t_s": 0., "video_frame_index": 0, "state": {"left": measured[0]["left"]},
             "stage": "Approach", "action": {"kind": "historical_measured_state_sequence",
             "next_keyframe": 1, "reference_sampling": {"method": "reviewed"},
             "compression": {"method": "periodic_and_gripper_events_with_reviewed_anchors",
                             "input_samples": 6, "output_samples": 3},
             "samples": [measured[i] for i in (0, 3, 5)]}}
    base = {"content": ["Unchanged instruction", json.dumps(header),
                        _frame_context(frame, tuple(map(tuple, columns))),
                        {"image": "keyframes/first.jpg"}, json.dumps({"video_frame_index": 5, "stage": "Done"})]}
    original_header = {**header, "action_sample_encoding": {"columns": COLUMNS, "description": "shared encoding"},
                       "coordinate_frame": {"action_semantics": "original next-state reference"}}
    original = {"content": [json.dumps(original_header)]}
    raw_demo = {"source": source, "keyframes": [{"action": {"samples": raw}}]}
    geometry = {"source": source, "keyframes": [{"action": {"samples": measured}}]}
    return base, original, raw_demo, geometry


class RestoreReferencesTest(unittest.TestCase):
    def test_sparse_rows_use_next_source_frame_and_terminal_clamp(self):
        base, original, raw, geometry = fixture()
        snapshot = copy.deepcopy((base, original, raw, geometry))
        result, full, evidence = restore(base, original, raw, geometry)
        header = overview(result)[1]
        action = objects(result)[2]["action"]
        rows = decode_rows(action, header["action_sample_encoding"]["columns"])
        self.assertEqual([r[("source_frame_index",)] for r in rows], [0, 3, 5])
        self.assertEqual([r[("left", "endpose_next_state_xyz_xyzw_gripper")][0] for r in rows], [.01, .04, .05])
        self.assertEqual([r[("left", "joint_reference_rad")][0] for r in rows], [.1, .4, .5])
        self.assertEqual([r[("left", "joint_measured_rad")][0] for r in rows], [0., .3, .5])
        self.assertEqual([r[("left", "gripper_reference")] for r in rows], [.9, .6, .5])
        self.assertEqual(header["action_sample_encoding"]["columns"], COLUMNS)
        self.assertNotIn("numeric_reference_semantics", header)
        self.assertEqual(set(action), {"kind", "next_keyframe", "compression", "sample_rows"})
        self.assertEqual(header["summary"], overview(base)[1]["summary"])
        self.assertEqual(objects(result)[2]["state"], objects(base)[2]["state"])
        self.assertEqual(result["content"][3:], base["content"][3:])
        self.assertEqual(result["content"][0], base["content"][0])
        self.assertEqual(len(full[0]["action"]["samples"]), 6)
        self.assertEqual(evidence["source_frames_verified"], 6)
        self.assertEqual((base, original, raw, geometry), snapshot)

    def test_rejects_already_double_shifted_labels(self):
        args = fixture()
        args[2]["keyframes"][0]["action"]["samples"][0]["left"]["joint_reference_rad"] = [.2] * 6
        with self.assertRaisesRegex(ValueError, "Unverified next-state"):
            restore(*args)

    def test_rejects_wrong_episode(self):
        args = copy.deepcopy(fixture())
        args[3]["source"] = {"dataset": "make_kong", "episode_index": 1}
        with self.assertRaisesRegex(ValueError, "identity mismatch"):
            restore(*args)

    def test_rejects_conflicting_overlapping_source_frames(self):
        args = fixture()
        conflicting = copy.deepcopy(args[2]["keyframes"][0])
        conflicting["action"]["samples"][0]["t_s"] = .001
        args[2]["keyframes"].append(conflicting)
        with self.assertRaisesRegex(ValueError, "Conflicting source"):
            restore(*args)


if __name__ == "__main__":
    unittest.main()
