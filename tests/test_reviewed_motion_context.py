"""Reviewed context preserves source motion information without changing default sampling."""
import copy
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from reorganize_robodojo_context import geometry_anchors

from gpt_policy.input.action_sampling import compress_samples
from gpt_policy.input.demonstration import prepare_demonstration
from gpt_policy.input.request import RunInput


def samples(points, yaw=None, grip=None):
    yaw = yaw or [0.] * len(points)
    grip = grip or [1.] * len(points)
    return [{"t_s": i*.04, "source_frame_index": i,
             **{arm: {"eef_measured_xyz_xyzw": [*p, *Rotation.from_euler("z", a).as_quat()],
                      "gripper_measured": g, "joint_measured_rad": [0.] * 6}
                for arm in ("left", "right")}}
            for i, (p, a, g) in enumerate(zip(points, yaw, grip))]


def test_short_lift_and_descent_survive_compilation(tmp_path):
    rows = samples([[0, 0, 0], [0, 0, .1], [.1, 0, .1], [.1, 0, 0]])
    original = copy.deepcopy(rows)
    keep = geometry_anchors(rows)
    assert keep == [0, 1, 2, 3]
    default, _ = compress_samples(rows)
    assert len(default) == 2
    sampled, _ = compress_samples(rows, preserve_indices=keep)
    assert sampled == rows == original
    Image.new("RGB", (4, 4)).save(tmp_path/"frame.png")
    source = {
        "schema_version": 1, "title": "fixture", "numeric_reference_semantics": "measured_states",
        "keyframes": [{"t_s": 0, "images": {"top": "frame.png"}, "action": {
            "kind": "historical_measured_state_sequence", "samples": rows, "preserve_sample_indices": keep,
        }}],
    }
    (tmp_path/"demo.json").write_text(json.dumps(source))
    prepare_demonstration(RunInput("fixture", "test"), tmp_path/"demo.json",
                          "video+action", tmp_path/"prepared")
    compiled = json.loads((tmp_path/"prepared/demo.json").read_text())
    assert compiled["keyframes"][0]["action"]["samples"] == rows
    content = json.loads((tmp_path/"prepared/input.json").read_text())["content"]
    header = json.loads(content[1])
    assert header["numeric_reference_semantics"] == "measured_states"
    assert "not issued controls" in header["action_sample_encoding"]["description"]
    assert all("command" not in path[-1] and "reference" not in path[-1]
               for path in header["action_sample_encoding"]["columns"])


def test_orientation_turn_and_feedback_transition_are_retained():
    rows = samples([[0, 0, 0]]*5, yaw=[0, 0, 1., 0, 0], grip=[1, 1, 1, .3, .3])
    keep = geometry_anchors(rows)
    assert 2 in keep and 3 in keep
    # Quaternion sign changes encode no physical turn.
    straight = samples([[i*.01, 0, 0] for i in range(5)])
    straight[2]["left"]["eef_measured_xyz_xyzw"][3:] = [0, 0, 0, -1]
    assert geometry_anchors(straight) == [0, 4]


@pytest.mark.parametrize("indices", [[-1], [2], [True], [0.5]])
def test_invalid_reviewed_anchor_rejected(indices):
    with pytest.raises(ValueError, match="Preserved"):
        compress_samples(samples([[0, 0, 0]]*2), preserve_indices=indices)


def test_empty_override_leaves_default_output_identical():
    rows = samples([[i*.01, 0, 0] for i in range(30)])
    assert compress_samples(rows) == compress_samples(rows, preserve_indices=[])
    assert "preserved_indices" not in compress_samples(rows)[1]


@pytest.mark.parametrize("value", [0, -1, np.nan, np.inf])
def test_bad_geometry_tolerance_rejected(value):
    with pytest.raises(ValueError, match="tolerances"):
        geometry_anchors(samples([[0, 0, 0]]*2), position_m=value)
