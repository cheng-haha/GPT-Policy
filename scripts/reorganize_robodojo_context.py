#!/usr/bin/env python3
"""Reorganize a reviewed expert episode offline, retaining measured motion landmarks."""
import argparse
import copy
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
from prepare_robodojo_contexts import digest, validate
from scipy.spatial.transform import Rotation, Slerp

from gpt_policy.input.action_sampling import gripper_events
from gpt_policy.input.demonstration import prepare_demonstration
from gpt_policy.input.lerobot_demo import endpose_sequence, save
from gpt_policy.input.request import RunInput


def geometry_anchors(samples, position_m=.005, rotation_rad=np.pi/36):
    """Retain measured-state turns; tolerances describe source reconstruction, not control."""
    if not np.isfinite([position_m, rotation_rad]).all() or min(position_m, rotation_rad) <= 0:
        raise ValueError("Geometry tolerances must be finite and positive")
    if len(samples) < 2:
        return list(range(len(samples)))
    times = np.array([s["t_s"] for s in samples])
    if not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError("Measured samples require strictly increasing timestamps")
    poses = {}
    for arm in ("left", "right"):
        values = np.array([s[arm]["eef_measured_xyz_xyzw"] for s in samples], dtype=float)
        if values.shape != (len(samples), 7) or not np.isfinite(values).all():
            raise ValueError("Expected finite measured xyz/xyzw poses")
        if not np.allclose(np.linalg.norm(values[:, 3:], axis=1), 1., atol=1e-6):
            raise ValueError("Expected unit measured quaternions")
        poses[arm] = values
    keep = {0, len(samples)-1} | gripper_events(samples, ("gripper_measured",), .1)
    pending = list(zip(sorted(keep), sorted(keep)[1:]))
    while pending:
        lo, hi = pending.pop()
        if hi-lo < 2:
            continue
        t = times[lo+1:hi]
        fraction = ((t-times[lo])/(times[hi]-times[lo]))[:, None]
        errors = np.zeros(len(t))
        for values in poses.values():
            predicted = values[lo, :3] + fraction*(values[hi, :3]-values[lo, :3])
            position = np.linalg.norm(values[lo+1:hi, :3]-predicted, axis=1)/position_m
            rotations = Slerp(times[[lo, hi]], Rotation.from_quat(values[[lo, hi], 3:]))(t)
            angle = (rotations.inv()*Rotation.from_quat(values[lo+1:hi, 3:])).magnitude()/rotation_rad
            errors = np.maximum(errors, np.maximum(position, angle))
        pivot = int(np.argmax(errors))
        if errors[pivot] > 1:
            mid = lo+1+pivot
            keep.add(mid)
            pending.extend(((lo, mid), (mid, hi)))
    return sorted(keep)


def reorganize(original, output, spec_path):
    original, output = Path(original).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    spec = json.loads(Path(spec_path).read_text())
    source = json.loads((original/"source.json").read_text())
    demo = json.loads((original/"source_reviewed.json").read_text())
    old = json.loads((original/"prepared/demo.json").read_text())
    raw = [json.loads(line) for line in (original/"raw_endposes.jsonl").read_text().splitlines()]
    if spec["episode"] != source["episode"]["episode_index"] or len(raw) != source["episode"]["length"]:
        raise ValueError("Episode identity/length mismatch")
    choices = spec["keyframes"]
    indices = [f["frame"] for f in choices]
    if not 2 <= len(indices) <= 16 or indices != sorted(set(indices)) or indices[0] != 0 or indices[-1] != len(raw)-1:
        raise ValueError("Choose at most 16 chronological frames including both episode endpoints")
    raw_samples = {}
    for frame in demo["keyframes"]:
        for sample in frame.get("action", {}).get("samples", []):
            i = sample["source_frame_index"]
            if i in raw_samples and sample != raw_samples[i]:
                raise ValueError("Conflicting overlapping source samples")
            raw_samples[i] = sample
    if sorted(raw_samples) != list(range(len(raw))):
        raise ValueError("Expected all original measured frames")
    poses = endpose_sequence(raw)
    samples = []
    for i, row in enumerate(raw):
        if row["frame_index"] != i:
            raise ValueError("Noncontiguous raw frame index")
        sample = {"t_s": float(row["timestamp"]), "source_frame_index": i}
        for a, arm in enumerate(("left", "right")):
            prior = raw_samples[i][arm]
            if i+1 < len(raw):
                nxt = raw_samples[i+1][arm]
                if prior["joint_reference_rad"] != nxt["joint_measured_rad"] or prior["gripper_reference"] != nxt["gripper_measured"]:
                    raise ValueError("Dataset references are not verified next measured states")
            np.testing.assert_allclose(prior["endpose_next_state_xyz_xyzw_gripper"],
                                       poses[min(i+1, len(raw)-1), a], atol=1e-10, rtol=0)
            if abs(poses[i, a, 7]-prior["gripper_measured"]) > 1e-6:
                raise ValueError("Same-frame gripper fields disagree")
            sample[arm] = {
                "joint_measured_rad": prior["joint_measured_rad"],
                "eef_measured_xyz_xyzw": poses[i, a, :7].tolist(),
                "gripper_measured": prior["gripper_measured"],
            }
        samples.append(sample)
    # Verify source identities before producing any new package.
    watched = [original/f for f in ("source.json", "source_reviewed.json", "raw_endposes.jsonl",
                                    "frame_audit.json", "prepared/input.json")]
    original_hashes = {str(p): digest(p) for p in watched}
    for v in source["videos"].values():
        if digest(Path(v["path"])) != v["sha256"]:
            raise ValueError("Source video hash changed")
    old_frames = {f["video_frame_index"]: f for f in old["keyframes"]}
    for f in old["keyframes"]:
        for value in f["images"].values():
            p = original/"prepared"/value
            original_hashes[str(p)] = digest(p)
    output.mkdir(parents=True)
    shutil.copy2(spec_path, output/"selection.json")
    for name in ("raw_endposes.jsonl", "frame_audit.json"):
        shutil.copy2(original/name, output/name)
    missing = [i for i in indices if i not in old_frames]
    decoded = {}
    for view, v in source["videos"].items():
        if not missing:
            continue
        directory = output/"evidence"/view
        directory.mkdir(parents=True)
        select = "select=" + "+".join(f"eq(n\\,{i})" for i in missing)
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-threads", "2",
                        "-filter_threads", "1", "-ss", str(v["from_timestamp"]), "-i", v["path"],
                        "-vf", select, "-fps_mode", "vfr", "-frames:v", str(len(missing)),
                        "-q:v", "2", str(directory/"%03d.jpg")], check=True)
        for j, i in enumerate(missing, 1):
            decoded[i, view] = str(directory/f"{j:03d}.jpg")
    frames = []
    for k, choice in enumerate(choices):
        i = choice["frame"]
        frame = copy.deepcopy(old_frames.get(i, {}))
        for field in ("stage", "observation", "roles", "result"):
            if field in choice:
                frame[field] = choice[field]
            if field not in frame:
                raise ValueError(f"Missing reviewed visual annotation: {i}/{field}")
        t = samples[i]["t_s"]
        frame.update(t_s=t, video_frame_index=i)
        frame.pop("image_ids", None)
        frame["images"] = {
            view: str(original/"prepared"/old_frames[i]["images"][view]) if i in old_frames else decoded[i, view]
            for view in ("top", "left_wrist", "right_wrist")
        }
        frame["state"] = {
            arm: {"joint_measured_rad": samples[i][arm]["joint_measured_rad"],
                  "gripper_measured": samples[i][arm]["gripper_measured"],
                  "endpose_xyz_xyzw_gripper": poses[i, a].tolist()}
            for a, arm in enumerate(("left", "right"))
        }
        frame["alignment"] = {
            "method": "LeRobot episode timestamp and per-camera video offset",
            "episode_index": spec["episode"], "source_frame_index": i,
            "dataset_index": source["episode"]["dataset_from_index"]+i, "timestamp_s": t,
            "video_timestamps_s": {view: v["from_timestamp"]+t for view, v in source["videos"].items()},
            "decode_tolerance_s": .001, "hardware_synchronized": False,
        }
        frame.pop("action", None)
        if k+1 < len(indices):
            segment = samples[i:indices[k+1]+1]
            anchors = geometry_anchors(segment, spec["position_tolerance_m"], spec["rotation_tolerance_rad"])
            frame["action"] = {
                "kind": "historical_measured_state_sequence", "next_keyframe": k+1,
                "samples": segment, "preserve_sample_indices": anchors,
                "reference_sampling": {
                    "method": "measured_pose_reconstruction_and_gripper_feedback",
                    "position_tolerance_m": spec["position_tolerance_m"],
                    "rotation_tolerance_rad": spec["rotation_tolerance_rad"],
                    "scope": "offline source interpolation only; not execution accuracy or control targets",
                },
            }
        frames.append(frame)
    demo["keyframes"] = frames
    demo["summary"] = spec["summary"]
    demo["numeric_reference_semantics"] = "measured_states"
    demo["coordinate_frame"]["action_semantics"] = (
        "All sample rows now contain same-frame measured joints, link6 poses and gripper openings at t_s. "
        "Original dataset next-state training labels are preserved in the original package, not relabeled as commands. "
        "Observed gripper openings may be contact-limited; original commanded openings are unknown."
    )
    demo["coordinate_frame"]["mapping_to_current_robot"] = (
        "Source link6/environment_origin verified by full-episode FK. "
        "Current object alignment still needs live visual verification."
    )
    demo["source"]["reorganization"] = {
        "original_package": str(original), "original_input_sha256": original_hashes[str(original/"prepared/input.json")],
        "same_episode": True, "selection": "reviewed expert-only motion landmarks",
        "numeric_semantics": "same-frame measurements, no invented issued controls",
    }
    save(output/"source_reviewed.json", demo)
    _, counts = prepare_demonstration(RunInput(demo["title"], "gpt-6-astra"), output/"source_reviewed.json",
                                      "video+action", output/"prepared")
    source["previous_selection"] = source.pop("selection", None)
    source["selection"] = {"method": "offline expert-only visual review", "spec": spec,
                           "spec_sha256": digest(output/"selection.json")}
    source["counts"] = counts
    source["reorganization"] = demo["source"]["reorganization"]
    save(output/"source.json", source)
    portable = json.loads((output/"prepared/demo.json").read_text())
    for f in portable["keyframes"]:
        f["images"] = {view: "prepared/"+value for view, value in f["images"].items()}
    save(output/"reviewed.json", portable)
    verification = validate(output)
    for p, expected in original_hashes.items():
        if digest(Path(p)) != expected:
            raise ValueError("Original context changed")
    verification.update(original_files_unchanged=True, same_episode=True, model_calls=0,
                        numeric_reference_semantics="measured_states", source_hashes=original_hashes)
    save(output/"context_verification.json", verification)
    save(output/"prepared_inputs.override.json", {Path(demo["source"]["dataset"]).name: str(output/"prepared/input.json")})
    print(json.dumps(verification, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--spec", type=Path, required=True)
    args = parser.parse_args()
    reorganize(args.original, args.output, args.spec)
