"""Prepare one local RoboDojo LeRobot v3 episode with GPT-Policy's compiler.

LeRobot owns episode filtering and video offsets. No policy or simulator runs here.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import inspect
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

from ..geometry.poses import rotation_6d_to_quaternion
from .action_sampling import gripper_events
from .demonstration import prepare_demonstration
from .manifest import ImagePart, VideoPart, load_manifest
from .request import RunInput
from .video import CandidateFrame, FfmpegVideoExtractor, VideoExtraction, VideoMetadata, VideoProcessingConfig
from .video_cache import VideoProcessingCache, _sha256_file

CAMERAS = {
    "top": "observation.images.cam_high",
    "left_wrist": "observation.images.cam_left_wrist",
    "right_wrist": "observation.images.cam_right_wrist",
}


def expert_success():
    """Historical outcome confirmed by the dataset owner, not a new rollout."""
    return {"success": True, "source": "dataset_owner_confirmation",
            "scope": "historical_expert_demonstration"}


def plain(value):
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, dict):
        return {k: plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(v) for v in value]
    return value


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def validate_rows(rows, episode, metadata):
    """A shard can contain many episodes; reject even one foreign/missing row."""
    if len(rows) != metadata["length"] or not rows:
        raise ValueError("Episode length does not match metadata")
    start = int(metadata["dataset_from_index"])
    if int(metadata["dataset_to_index"]) != start + len(rows):
        raise ValueError("Invalid episode index bounds")
    for i, row in enumerate(rows):
        if (row["episode_index"] != episode or row["frame_index"] != i
                or row["index"] != start + i):
            raise ValueError("Episode/frame/global index mismatch")
    times = [float(r["timestamp"]) for r in rows]
    if any(not math.isfinite(t) for t in times) or any(b <= a for a, b in zip(times, times[1:])):
        raise ValueError("Episode timestamps must strictly increase")
    return times


def endpose_sequence(rows):
    """20D source -> two 8D xyz/xyzw/gripper poses, retaining source sites/frames.

    Align quaternion signs over the full episode before selecting keyframes or
    deriving next-state references. q and -q represent the same orientation.
    """
    poses = []
    for row in rows:
        eef = np.asarray(row["observation.states.eef"], dtype=float)
        if eef.shape != (20,) or not np.isfinite(eef).all():
            raise ValueError("Expected finite 20D endpose")
        arms = []
        for arm, offset in enumerate((0, 10)):
            value = eef[offset:offset+10]
            if not 0 <= value[9] <= 1:
                raise ValueError("Endpose gripper outside [0, 1]")
            q = rotation_6d_to_quaternion(value[3:9])
            if poses and np.dot(poses[-1][arm][3:7], q) < 0:
                q = -q
            arms.append(np.r_[value[:3], q, value[9]])
        poses.append(arms)
    return np.asarray(poses)


def numeric_samples(rows, *, endposes=None):
    if endposes is None:
        endposes = endpose_sequence(rows)
    result = []
    for i, row in enumerate(rows):
        state = np.asarray(row["observation.state"], dtype=float)
        action = np.asarray(row["action"], dtype=float)
        eef = np.asarray(row["observation.states.eef"], dtype=float)
        next_eef = np.asarray(row["actions.eef"], dtype=float)
        if state.shape != (14,) or action.shape != (14,) or eef.shape != (20,) or next_eef.shape != (20,):
            raise ValueError("Expected RoboDojo 14D joints and 20D endpose")
        if not all(np.isfinite(x).all() for x in (state, action, eef, next_eef)):
            raise ValueError("Non-finite numeric reference")
        expected = rows[min(i + 1, len(rows) - 1)]["observation.states.eef"]
        if not np.allclose(next_eef, expected, atol=2e-5, rtol=0):
            raise ValueError("actions.eef does not match declared next-state derivation")
        sample = {"t_s": float(row["timestamp"]), "source_frame_index": int(row["frame_index"])}
        for a, (arm, j) in enumerate((("left", 0), ("right", 7))):
            for opening in (state[j + 6], action[j + 6]):
                if not 0 <= opening <= 1:
                    raise ValueError("Gripper reference outside [0, 1]")
            # Dataset training targets are references, not proof of issued commands.
            sample[arm] = {
                "joint_measured_rad": state[j:j+6].tolist(),
                "joint_reference_rad": action[j:j+6].tolist(),
                "gripper_measured": float(state[j+6]),
                "gripper_reference": float(action[j+6]),
                "endpose_next_state_xyz_xyzw_gripper": endposes[min(i+1, len(rows)-1), a].tolist(),
            }
        result.append(sample)
    return result


class LeRobotEpisodeExtractor(FfmpegVideoExtractor):
    """Decode bounded candidates through LeRobotDataset, preserving local time."""
    def __init__(self, root, episode, config=None, dataset_factory=None):
        super().__init__(config)
        self.root = Path(root).resolve()
        self.episode = episode
        if dataset_factory is None:
            from lerobot.datasets.lerobot_dataset import LeRobotDataset
            dataset_factory = LeRobotDataset
        # Offline environment variables prevent fallback network downloads.
        self.dataset = dataset_factory(
            f"local/{self.root.name}", root=self.root, episodes=[episode],
            download_videos=False, video_backend="pyav", tolerance_s=.001,
        )
        self.info = self.dataset.meta.info
        if self.info.get("codebase_version") != "v3.0":
            raise ValueError("Expected LeRobot v3.0")
        semantics = self.info.get("starvla_metadata", {})
        if (semantics.get("action_mode") != "absolute_next_step"
                or semantics.get("rotation_6d_convention") != "first_two_rotation_matrix_columns_column_major"
                or semantics.get("action", {}).get("frame_offset") != 1):
            raise ValueError("Unsupported action or rotation convention")
        matches = [plain(r) for r in self.dataset.meta.episodes if int(r["episode_index"]) == episode]
        if len(matches) != 1:
            raise ValueError("Episode metadata must be unique")
        self.meta = matches[0]
        self.dataset[0]  # Load numeric table and verify that real video decoding works.
        self.rows = [plain(row) for row in self.dataset.hf_dataset]
        self.times = validate_rows(self.rows, episode, self.meta)
        self.endposes = endpose_sequence(self.rows)
        self.samples = numeric_samples(self.rows, endposes=self.endposes)
        self.tasks = list(self.meta["tasks"])
        if len(self.tasks) != 1:
            raise ValueError("Select an episode with exactly one task")
        self.instruction = self.tasks[0]
        self.event_times = tuple(self.times[i] for i in sorted(gripper_events(
            self.samples, fields=("gripper_reference", "gripper_measured"))))
        self.end_time_s = self.times[-1]
        self.media = {}
        for camera, key in CAMERAS.items():
            if key not in self.info["features"]:
                raise ValueError(f"Missing camera {key}")
            prefix = f"videos/{key}/"
            path = self.root / self.info["video_path"].format(video_key=key,
                chunk_index=self.meta[prefix+"chunk_index"], file_index=self.meta[prefix+"file_index"])
            self.media[camera] = {"path": str(path), "sha256": _sha256_file(path),
                "from_timestamp": self.meta[prefix+"from_timestamp"],
                "to_timestamp": self.meta[prefix+"to_timestamp"], "episode_index": episode}
        self.source = Path(self.media["top"]["path"])
        data = self.root / self.info["data_path"].format(
            chunk_index=self.meta["data/chunk_index"], file_index=self.meta["data/file_index"])
        self.source_hashes = {str(p.relative_to(self.root)): _sha256_file(p) for p in
            [self.root/"meta/info.json", self.root/"meta/source_stage2.json", data,
             self.root/"meta/tasks.parquet", *sorted((self.root/"meta/episodes").glob("chunk-*/*.parquet"))]}
        self.source_stats = {str(self.root/k): (self.root/k).stat() for k in self.source_hashes}
        self.source_stats.update({m["path"]: Path(m["path"]).stat() for m in self.media.values()})
        module_path = Path(inspect.getfile(dataset_factory))
        self.loader = {"class": dataset_factory.__name__, "module": str(module_path),
                       "sha256": _sha256_file(module_path)}

    def view_identity(self):
        return {**self.media, "numeric_source": {"sha256": self.source_hashes}}

    def extract_candidates(self, source, destination):
        info = self.info["features"][CAMERAS["top"]]
        h, w, _ = info["shape"]
        metadata = VideoMetadata(self.times[-1] - self.times[0] + 1/self.info["fps"],
            w, h, self.info["fps"], "lerobot-decoded-rgb", source.stat().st_size)
        times = list(self._timestamps(metadata)) + [self.times[0], self.times[-1]]
        for t in self.event_times:
            times.extend([max(self.times[0], t-.3), t, min(self.times[-1], t+.3)])
        indices = sorted({int(np.argmin(np.abs(np.asarray(self.times) - t))) for t in times})
        frames = []
        for i, n in enumerate(indices):
            row = self.dataset[n]
            if int(row["episode_index"]) != self.episode or int(row["frame_index"]) != n:
                raise ValueError("Decoded image escaped selected episode")
            views = {}
            for camera, key in CAMERAS.items():
                pixels = np.asarray(row[key], dtype=float)
                if pixels.shape != (3, h, w) or not np.isfinite(pixels).all() or pixels.min() < 0 or pixels.max() > 1:
                    raise ValueError("Expected LeRobot RGB float tensor in [0,1]")
                path = destination / "candidates" / camera / f"{n:06d}.jpg"
                path.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(np.rint(pixels.transpose(1,2,0)*255).astype("uint8")).save(path, quality=95)
                views[camera] = CandidateFrame(i, self.times[n], path, n, capture_delta_s=0.)
            top = views.pop("top")
            frames.append(replace(top, views=views))
        print(f"Decoded {len(frames)} candidate times, {len(frames)*len(CAMERAS)} images", flush=True)
        return VideoExtraction(source, metadata, tuple(frames))

    def export_keyframes(self, extraction, selected, destination):
        # Candidates retain source resolution; do not decode/recompress them again.
        return tuple(selected)

    def verify_sources(self):
        for name, before in self.source_stats.items():
            after = Path(name).stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise RuntimeError(f"Source changed during preparation: {name}")

    def attach(self, frames):
        for i, frame in enumerate(frames):
            n = frame["video_frame_index"]
            row = self.rows[n]
            sample = self.samples[n]
            frame["state"] = {arm: {
                "joint_measured_rad": sample[arm]["joint_measured_rad"],
                "gripper_measured": sample[arm]["gripper_measured"],
                "endpose_xyz_xyzw_gripper": self.endposes[n, a].tolist(),
            } for a, arm in enumerate(("left", "right"))}
            frame["alignment"] = {"method": "LeRobot episode timestamp and per-camera video offset",
                "episode_index": self.episode, "source_frame_index": n,
                "dataset_index": row["index"], "timestamp_s": self.times[n],
                "video_timestamps_s": {c: v["from_timestamp"] + self.times[n] for c,v in self.media.items()},
                "decode_tolerance_s": .001, "hardware_synchronized": False}
            if i + 1 < len(frames):
                end = frames[i+1]["video_frame_index"]
                frame["action"] = {"kind": "dataset_next_step_joint_reference",
                    "next_keyframe": i+1, "samples": self.samples[n:end+1]}


def checked_frame_audit(path, extractor):
    """Bind an offline FK finding to this exact source episode, not a robot name."""
    audit = json.loads(Path(path).read_text())
    if (Path(audit["dataset"]).resolve() != extractor.root or audit["episode"] != extractor.episode
            or audit["frames"] != len(extractor.rows)
            or audit.get("matches_native_link6_in_environment_frame") is not True):
        raise ValueError("Frame audit does not verify this dataset episode")
    for relative, digest in extractor.source_hashes.items():
        if relative == "meta/info.json" or relative.startswith("data/"):
            if audit.get("files", {}).get(str((extractor.root/relative).resolve())) != digest:
                raise ValueError("Frame audit source hash mismatch")
    return audit


def prepare(root, episode, output, selector, config=None, *, confirmed_expert=False, frame_audit=None,
            instruction_override=None, relabel_reason=None):
    # Relabel only the demonstrated task text; preserve all source rows and their identity.
    if (instruction_override is not None or relabel_reason is not None) and not all(
        isinstance(v, str) and v.strip() for v in (instruction_override, relabel_reason)
    ):
        raise ValueError("Instruction relabel requires a nonempty instruction and provenance reason")
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    extractor = LeRobotEpisodeExtractor(root, episode, config)
    audit = checked_frame_audit(frame_audit, extractor) if frame_audit else None
    instruction = instruction_override or extractor.instruction
    relabel = None if instruction_override is None else {
        "original_instruction": extractor.instruction,
        "replacement_instruction": instruction,
        "reason": relabel_reason,
        "source": "user_authorized_visual_relabel",
        "native_target_success_verified": False,
    }
    cache = VideoProcessingCache(output.parent / ".lerobot-context-cache")
    result = cache.resolve(instruction, VideoPart(extractor.source,
        f"{Path(root).name}, episode {episode}"), extractor, selector)
    frames = []
    for frame, choice in zip(result.keyframes, result.selection.selected):
        views = {"top": frame, **frame.views}
        frames.append({"t_s": frame.timestamp_s, "video_frame_index": frame.frame_index,
            "stage": choice.stage, "observation": choice.reason,
            "roles": {"left": choice.left, "right": choice.right}, "result": choice.result,
            "images": {c: str(f.path) for c,f in views.items()}})
    extractor.attach(frames)
    extractor.verify_sources()
    demo = {"schema_version": 1, "title": instruction,
        "source": {"format": "lerobot_v3", "dataset": str(Path(root).resolve()), "episode_index": episode},
        "source_robot": {"model": extractor.info["robot_type"], "site": "endpose, not verified control TCP"},
        "demonstrator": "expert" if confirmed_expert else "dataset_episode",
        "coverage": "full episode, selected visual evidence",
        "outcome": expert_success() if confirmed_expert else "unverified: no native success field in source metadata",
        "coordinate_frame": {
            "source": extractor.info["starvla_metadata"]["coordinate_system"],
            "joint_unit": "radian", "gripper": "0 closed, 1 open",
            "eef_format": "[x,y,z,qx,qy,qz,qw,gripper]; unit xyzw; continuous quaternion signs per arm",
            "rotation_conversion": "source rot6d matrix columns -> xyzw only; no frame or reference-point transform",
            "eef_position_unit": "not declared in source; do not assume a verified metric TCP mapping",
            "action_semantics": "joint_reference_rad/gripper_reference are dataset absolute_next_step targets, "
                "already offset one frame; not proof of original issued commands. actions.eef is shifted "
                "measured endpose, not an EEF command; labeled endpose_next_state in intervals.",
            "mapping_to_current_robot": "unverified; historical numeric references only"},
        "summary": result.selection.summary, "keyframes": frames}
    if relabel is not None:
        demo["source"]["instruction_relabel"] = relabel
    if relabel is not None and confirmed_expert:
        demo["outcome"]["instruction_scope"] = "original_dataset_task"
    if audit:
        demo["source_robot"]["site"] = "link6 (offline FK verified for this episode)"
        demo["coordinate_frame"].update({
            "verified_source": {"site": "link6", "frame": "environment_origin", "position_unit": "m",
                "evidence": "all-frame robot-only FK audit of this source episode",
                "note": "shared environment frame, not per-arm base_link; source metadata label is generic"},
            "eef_position_unit": "m (verified against native X5 URDF FK for this episode)",
            "mapping_to_current_robot": "source link6/environment frame verified; current TCP/base mapping still required"})
        save(output/"frame_audit.json", audit)
    reviewed = output / "source_reviewed.json"
    save(reviewed, demo)
    raw = output / "raw_endposes.jsonl"
    with raw.open("w") as stream:
        for row in extractor.rows:
            stream.write(json.dumps({k: row[k] for k in ("frame_index", "timestamp",
                "observation.states.eef", "actions.eef")}, allow_nan=False) + "\n")
    run_input, counts = prepare_demonstration(RunInput(instruction, selector.config.model or "gpt-6-astra"),
        reviewed, "video+action", output / "prepared")
    save(output/"source.json", {"episode": extractor.meta, "instruction_relabel": relabel, "info_sha256": extractor.source_hashes["meta/info.json"],
        "source_hashes": extractor.source_hashes, "videos": extractor.media, "loader": extractor.loader,
        "raw_endposes": {"path": raw.name, "sha256": _sha256_file(raw), "format": "dual xyz_rot6d_gripper"},
        "rotation_conversion": {"source": "first_two_rotation_matrix_columns_column_major",
            "output": "xyzw", "frame_or_site_transform_applied": False,
            "sign_continuity": "per arm, full episode, before next-state shift and keyframe sampling"},
        "frame_audit": {"path": "frame_audit.json", "sha256": _sha256_file(output/"frame_audit.json")}
            if audit else None,
        "selection_cache_key": result.cache_key, "selection": result.selection.wire,
        "counts": counts})
    # Reviewed source and prepared input both remain portable within this package.
    portable = json.loads((output/"prepared/demo.json").read_text())
    for frame in portable["keyframes"]:
        frame["images"] = {c: "prepared/"+p for c,p in frame["images"].items()}
    save(output / "reviewed.json", portable)
    audit = smoke(output/"prepared/input.json")
    save(output/"smoke.json", audit)
    return audit


def smoke(input_path):
    from ..harness.input_content import to_app_server_items, validate_input_size
    manifest = load_manifest(Path(input_path))
    images = [p for p in manifest.content if isinstance(p, ImagePart)]
    for part in images:
        with Image.open(part.path) as im:
            im.verify()
    items = to_app_server_items(manifest.content)
    image_items = [i for i in items if i["type"] == "image"]
    if not 0 < len(images) <= 48 or len(image_items) != len(images):
        raise ValueError("Invalid image payload count")
    if not all(i["url"].startswith("data:image/") for i in image_items):
        raise ValueError("Images were not encoded as actual bytes")
    return {"ok": True, "scope": "prepared input load, image decode, native app-server serialization; no physics",
        "images": len(images), "text_characters": validate_input_size(manifest.content),
        "wire_image_count": len(image_items), "input": str(input_path)}


def model_smoke(input_path, config, *, client_factory=None):
    """One real read-only vision turn over the final compiled context."""
    from ..harness.codex import CodexAppServer
    manifest = load_manifest(Path(input_path))
    fields = ("task", "initial_scene", "final_scene", "state_action_distinction", "limitations")
    schema = {"type": "object", "additionalProperties": False, "required": ["name", "arguments"],
        "properties": {"name": {"type": "string", "const": "inspect_context"},
            "arguments": {"type": "object", "additionalProperties": False, "required": list(fields),
                          "properties": {key: {"type": "string"} for key in fields}}}}
    factory = client_factory or CodexAppServer
    client = factory(config.model or "gpt-6-astra", config.executable, config.effort or "medium")
    try:
        client.start_thread("Inspect a historical robot demonstration, without executing any commands. "
            "Describe visible evidence from the first and last images and distinguish measured state "
            "from reference actions. No robot, file or shell tools. Return only the requested JSON.")
        result = client.decide("Smoke-test this compiled historical context. Report the task, initial "
            "and final visible scenes, state/action semantics, and limitations. Do not claim native "
            "task success from visual evidence or episode length.", schema, content=manifest.content)
        if result.get("name") != "inspect_context":
            raise ValueError("Unexpected context smoke response")
        return result["arguments"]
    finally:
        client.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--episode", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--codex-bin", default="codex")
    parser.add_argument("--model", default="gpt-6-astra")
    parser.add_argument("--effort", default="medium")
    parser.add_argument("--instruction-override", help="Reviewed replacement task text for this demonstration")
    parser.add_argument("--relabel-reason", help="Required provenance reason for an instruction override")
    parser.add_argument("--model-smoke", action="store_true")
    parser.add_argument("--expert-success", action="store_true",
                        help="Dataset owner confirms this is a successful expert demonstration")
    parser.add_argument("--frame-audit", type=Path,
                        help="Optional offline FK report bound to this exact dataset episode and source hashes")
    args = parser.parse_args()
    from ..harness.config import AgentConfig
    from ..harness.video_selector import CodexVideoSelector
    config = VideoProcessingConfig()
    selector = CodexVideoSelector(AgentConfig(model=args.model, executable=args.codex_bin, effort=args.effort), config)
    print(json.dumps(prepare(args.dataset, args.episode, args.output, selector, config,
                             confirmed_expert=args.expert_success, frame_audit=args.frame_audit,
                             instruction_override=args.instruction_override,
                             relabel_reason=args.relabel_reason), indent=2), flush=True)
    if args.model_smoke:
        result = model_smoke(args.output/"prepared/input.json", selector.config)
        save(args.output/"model_smoke.json", result)
        print("MODEL_CONTEXT_SMOKE_OK", flush=True)


if __name__ == "__main__":
    main()
