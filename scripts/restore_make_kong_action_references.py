"""Restore dataset next-state references at the exact adaptive_v3 sample positions."""
import argparse
import copy
import hashlib
import json
import shutil
from pathlib import Path

from gpt_policy.input.demonstration import _frame_context

V3_SHA256 = "bf6fa1d72c1bc576cec32b72c678336f26c2201a3c4533009f6dc0b90756cacf"
ORIGINAL_SHA256 = "760497e69cd8744fb8ccd940a5f55dce8a25c3992d7cdcc9a715ddcad1ddfc99"
KIND = "dataset_next_step_joint_reference"
FIELDS = ("joint_measured_rad", "joint_reference_rad", "gripper_measured",
          "gripper_reference", "endpose_next_state_xyz_xyzw_gripper")
COLUMNS = [["t_s"], ["source_frame_index"]] + [[arm, key] for arm in ("left", "right") for key in FIELDS]


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def objects(manifest):
    return {i: json.loads(x) for i, x in enumerate(manifest["content"])
            if isinstance(x, str) and x.startswith("{")}


def overview(manifest):
    return next((i, x) for i, x in objects(manifest).items() if "action_sample_encoding" in x)


def samples(demo):
    result = {}
    for frame in demo["keyframes"]:
        for sample in frame.get("action", {}).get("samples", []):
            index = sample["source_frame_index"]
            if index in result and result[index] != sample:
                raise ValueError("Conflicting source samples")
            result[index] = sample
    if not result or sorted(result) != list(range(len(result))):
        raise ValueError("Expected contiguous complete source samples")
    return result


def decode_rows(action, columns):
    previous = None
    result = []
    for row in action["sample_rows"]:
        if len(row) != len(columns) or (previous is None and "=" in row):
            raise ValueError("Invalid encoded sample row")
        values = [previous[i] if value == "=" else value for i, value in enumerate(row)]
        result.append(dict(zip(map(tuple, columns), values)))
        previous = values
    return result


def restore(base, original, raw_demo, geometry_demo):
    """Pure conversion: preserve v3 selection/text; never shift already shifted labels."""
    after = copy.deepcopy(base)
    header_index, header = overview(base)
    _, old_header = overview(original)
    if old_header["action_sample_encoding"]["columns"] != COLUMNS:
        raise ValueError("Original action schema differs from the shared dataset schema")
    if header.get("numeric_reference_semantics") != "measured_states":
        raise ValueError("Expected the measured-state v3 input")
    for field in ("dataset", "episode_index"):
        if len({d["source"][field] for d in (header, old_header, raw_demo, geometry_demo)}) != 1:
            raise ValueError("Source episode identity mismatch")
    raw, measured = samples(raw_demo), samples(geometry_demo)
    if raw.keys() != measured.keys():
        raise ValueError("Source frame coverage mismatch")
    for i, row in raw.items():
        if row["t_s"] != measured[i]["t_s"]:
            raise ValueError("Source timestamps disagree")
        for arm in ("left", "right"):
            current, nxt = measured[i][arm], measured[min(i + 1, len(raw) - 1)][arm]
            expected = {"joint_measured_rad": current["joint_measured_rad"],
                        "gripper_measured": current["gripper_measured"],
                        "joint_reference_rad": nxt["joint_measured_rad"],
                        "gripper_reference": nxt["gripper_measured"],
                        "endpose_next_state_xyz_xyzw_gripper": nxt["eef_measured_xyz_xyzw"] + [nxt["gripper_measured"]]}
            if row[arm] != expected:
                raise ValueError(f"Unverified next-state reference at frame {i}/{arm}")
    new_header = copy.deepcopy(header)
    new_header.pop("numeric_reference_semantics")
    new_header["coordinate_frame"]["action_semantics"] = old_header["coordinate_frame"]["action_semantics"]
    if "reorganization" in new_header["source"]:
        new_header["source"]["reorganization"]["numeric_semantics"] = (
            "Current measured joints/gripper and original dataset next-state references; "
            "original issued commands remain unknown."
        )
    new_header["action_sample_encoding"] = copy.deepcopy(old_header["action_sample_encoding"])
    new_header["action_sample_encoding"]["description"] = new_header["action_sample_encoding"]["description"].replace(
        "Actions use endpoints, 1 Hz samples and gripper-command transitions, not a full trajectory.",
        "Actions preserve the base input's endpoints, periodic samples and reviewed geometry/gripper-feedback anchors."
    )
    new_header["action_sample_encoding"]["description"] += (
        " This derivative retains the exact adaptive_v3 source-frame selection, including its "
        "reviewed geometry and measured-gripper anchors; it does not resample reference events. "
        "Reference fields already describe the next original source frame, not the next selected row. "
        "The terminal reference is clamped to the final source state."
    )
    after["content"][header_index] = json.dumps(new_header, ensure_ascii=False, separators=(",", ":"))
    raw_actions, selection, prior_sampling = [], {}, {}
    frames = [(i, f) for i, f in objects(base).items() if "video_frame_index" in f]
    for k, (index, frame) in enumerate(frames):
        if "action" not in frame:
            continue
        action = frame["action"]
        if action["kind"] != "historical_measured_state_sequence":
            raise ValueError("Unexpected base action kind")
        rows = decode_rows(action, header["action_sample_encoding"]["columns"])
        indices = [row[("source_frame_index",)] for row in rows]
        lo, hi = frame["video_frame_index"], frames[k + 1][1]["video_frame_index"]
        if indices != sorted(set(indices)) or indices[0] != lo or indices[-1] != hi:
            raise ValueError("Selected frames must retain ordered segment endpoints")
        for row, source_index in zip(rows, indices):
            if row[("t_s",)] != round(raw[source_index]["t_s"], 6):
                raise ValueError("Selected timestamp mismatch")
            for arm in ("left", "right"):
                for field, value in measured[source_index][arm].items():
                    rounded = [round(x, 3) for x in value] if isinstance(value, list) else round(value, 3)
                    if row[(arm, field)] != rounded:
                        raise ValueError("Base measured rows do not match the source")
        replacement = {"kind": KIND, "next_keyframe": action["next_keyframe"],
                       "compression": copy.deepcopy(action["compression"]),
                       "samples": [raw[i] for i in indices]}
        after["content"][index] = _frame_context({**frame, "action": replacement}, tuple(map(tuple, COLUMNS)))
        raw_actions.append({"keyframe": k, "action": {"kind": KIND, "next_keyframe": action["next_keyframe"],
                            "samples": [raw[i] for i in range(lo, hi + 1)]}})
        selection[k] = indices
        prior_sampling[k] = action.get("reference_sampling")
    return after, raw_actions, {"source_frames_verified": len(raw), "selected_source_frames": selection,
                                "original_reference_sampling": prior_sampling}


def build(base, original, geometry, output, comparison_config):
    base, original, geometry, output = (Path(p).resolve() for p in (base, original, geometry, output))
    paths = [base / "prepared/input.json", original / "prepared/input.json",
             original / "source_reviewed.json", geometry / "source_reviewed.json"]
    before_hashes = {str(p): digest(p) for p in paths}
    if before_hashes[str(paths[0])] != V3_SHA256 or before_hashes[str(paths[1])] != ORIGINAL_SHA256:
        raise ValueError("Expected the audited v3 and original Make Kong inputs")
    after, raw_actions, evidence = restore(*(read(p) for p in paths))
    mappings = read(comparison_config)["prepared_inputs"]
    compared = {}
    for task, path in mappings.items():
        if task == "make_kong":
            continue
        manifest = read(path)
        if overview(manifest)[1]["action_sample_encoding"]["columns"] != COLUMNS:
            raise ValueError(f"Different action columns in {task}")
        if any(f["action"]["kind"] != KIND for f in objects(manifest).values() if "action" in f):
            raise ValueError(f"Different action kind in {task}")
        compared[task] = {"path": path, "sha256": digest(path)}
    if len({x["path"] for x in compared.values()}) != 9:
        raise ValueError("Expected all nine comparison contexts")
    images = [x for x in after["content"] if isinstance(x, dict)]
    selected_count = sum(map(len, evidence["selected_source_frames"].values()))
    if len(images) != 48 or len(evidence["selected_source_frames"]) != 15 or selected_count != 175:
        raise ValueError("Expected 16 keyframes, 48 images and 175 selected rows")
    # Never overwrite any existing package, including a partial earlier build.
    output.mkdir(parents=True, exist_ok=False)
    prepared = output / "prepared"
    prepared.mkdir()
    image_hashes = {}
    for image in images:
        rel = Path(image["image"])
        if rel.is_absolute() or ".." in rel.parts:
            raise ValueError("Expected portable image paths")
        src, dst = base / "prepared" / rel, prepared / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        image_hashes[str(rel)] = digest(src)
        if digest(dst) != image_hashes[str(rel)]:
            raise ValueError("Copied image hash mismatch")
    save(prepared / "input.json", after)
    (prepared / "actions.jsonl").write_text("".join(json.dumps(x, allow_nan=False) + "\n" for x in raw_actions))
    from gpt_policy.input.lerobot_demo import smoke

    smoke_result = smoke(prepared / "input.json")
    save(output / "smoke.json", smoke_result)
    if any(digest(p) != value for p, value in before_hashes.items()):
        raise ValueError("Source files changed during conversion")
    if any(digest(x["path"]) != x["sha256"] for x in compared.values()):
        raise ValueError("Comparison context changed during conversion")
    report = {"base_package": str(base), "original_package": str(original), "geometry_package": str(geometry),
              "source_hashes": before_hashes, "input_sha256": digest(prepared / "input.json"),
              "builder_sha256": digest(__file__), "action_kind": KIND, "columns": COLUMNS,
              "keyframes": 16, "images": image_hashes, "selected_rows": selected_count,
              "other_task_contexts": compared, "source_files_unchanged": True,
              "input_smoke": smoke_result, "model_calls": 0, "simulation_calls": 0, **evidence}
    save(output / "context_verification.json", report)
    save(output / "prepared_inputs.override.json", {"make_kong": str(prepared / "input.json")})
    save(output / "ten_tasks.prepared_inputs.json", {**mappings, "make_kong": str(prepared / "input.json")})
    (output / "README.md").write_text(
        "# Make Kong adaptive_v4: shared next-state action schema\n\n"
        "Derived from adaptive_v3. Its 16 keyframes, 48 images, 175 selected source-frame positions, "
        "states, factual text and Execution notes are retained. Actions restore the original "
        "12-column dataset next-step reference schema used by the other nine tasks. Measured "
        "fields describe t; reference fields describe the next original source frame (terminal "
        "clamped), not the next sampled row or known issued commands. Geometric selection is "
        "preserved without rerunning compression. Full-precision interval data are in "
        "prepared/actions.jsonl; provenance and comparisons are in context_verification.json.\n\n"
        "Use prepared/input.json and prepared_inputs.override.json for future runs. "
        "ten_tasks.prepared_inputs.json contains all existing task/variant mappings with only "
        "Make Kong changed. No historical input, evaluation config or result was overwritten. "
        "No model or simulation was run; this change has no new success-rate claim.\n"
    )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("base", "original", "geometry", "output", "comparison-config"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    result = build(args.base, args.original, args.geometry, args.output, args.comparison_config)
    print(json.dumps({k: result[k] for k in ("input_sha256", "keyframes", "selected_rows", "source_frames_verified")}, indent=2))
