#!/usr/bin/env python3
"""Inspect completed native recordings without recomputing or overriding outcomes."""

import argparse
import json
import subprocess
from pathlib import Path

CAMERAS = ("cam_high", "cam_left_wrist", "cam_right_wrist")


def read_json(path):
    return json.loads(path.read_text())


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def recorded_image_path(run, stored, original_run=None):
    """Resolve archived image references before consulting the reused current path."""
    path = Path(stored)
    if original_run is not None:
        try:
            return Path(run) / path.relative_to(original_run)
        except ValueError:
            pass
    return path


def review(run, destination):
    run = Path(run).resolve()
    sim = run / "sim"
    original_run = None
    relocation = run.parent / "replacement_manifest.json"
    if relocation.is_file():
        manifest = read_json(relocation)
        if (Path(manifest["archive"]).resolve() == run.parent
                and any(e["original_id"] == run.name for e in manifest["cases"])):
            original_run = Path(manifest["current"]) / run.name
    if not (run / "result.json").is_file() or not (sim / "cleanup.json").is_file():
        raise ValueError("Only completed, cleaned-up attempts can be reviewed")
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    authoritative = read_json(run / "result.json")
    native = read_json(sim / "native_result.json")
    cleanup = read_json(sim / "cleanup.json")
    frames = rows(sim / "video-frames.jsonl")
    actions = rows(sim / "actions.jsonl")
    steps = native["native_control_steps"]
    expected_frames = list(range(steps + 1))
    checks = {
        "actions_one_per_ack": [a["step"] for a in actions]
        == list(range(1, steps + 1)),
        "frames_one_per_ack_plus_initial": [f["native_control_step"] for f in frames]
        == expected_frames,
        "frame_indices_contiguous": [f["frame_index"] for f in frames]
        == expected_frames,
        "child_exited_cleanly": cleanup.get("process_exited") is True
        and cleanup.get("exit_code") == 0,
    }
    if "native_success" in authoritative:
        checks["native_result_matches_authority"] = all(
            authoritative.get(k) == native.get(k)
            for k in (
                "complete",
                "native_success",
                "native_score",
                "native_control_steps",
            )
        )
    missing = []
    for frame in frames:
        for camera in CAMERAS:
            path = recorded_image_path(run, frame["image_paths"].get(camera, ""), original_run)
            if not path.is_file():
                missing.append({"step": frame["native_control_step"], "camera": camera})
    checks["three_png_views_per_frame"] = not missing
    videos = {}
    # Decode samples from the actual MP4s, not from their associated PNGs.
    sample_indices = sorted({0, steps // 4, steps // 2, steps * 3 // 4, steps})
    for camera in CAMERAS:
        video = sim / (camera + ".mp4")
        probe = json.loads(
            subprocess.check_output(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-count_frames",
                    "-select_streams",
                    "v:0",
                    "-show_entries",
                    "stream=width,height,r_frame_rate,nb_read_frames,duration",
                    "-of",
                    "json",
                    str(video),
                ],
                text=True,
                timeout=120,
            )
        )["streams"][0]
        videos[camera] = probe
        checks[camera + "_decoded_frame_count"] = (
            int(probe["nb_read_frames"]) == steps + 1
        )
        selection = "+".join(f"eq(n\\,{index})" for index in sample_indices)
        subprocess.run(
            [
                "ffmpeg",
                "-nostdin",
                "-v",
                "error",
                "-i",
                str(video),
                "-vf",
                "select=" + selection,
                "-vsync",
                "0",
                "-frames:v",
                str(len(sample_indices)),
                str(destination / f"{camera}_sample_%03d.png"),
            ],
            check=True,
            timeout=120,
        )
        for ordinal, index in enumerate(sample_indices, 1):
            (destination / f"{camera}_sample_{ordinal:03d}.png").rename(
                destination / f"{camera}_{index:06d}.png"
            )
    terminal_path = sim / "terminal_conditions.json"
    terminal = (
        read_json(terminal_path)
        if terminal_path.is_file()
        else {
            "audit_available": False,
            "reason": "not_recorded_by_this_runtime",
        }
    )
    if native.get("complete") and native.get("task") == "classify_objects":
        checks["classify_terminal_audit_available"] = terminal.get(
            "audit_available"
        ) is not False and all(
            k in terminal for k in ("containment", "height", "robot_home")
        )
    report = {
        "run": str(run),
        "archived_source_prefix": str(original_run) if original_run else None,
        "authoritative_result": authoritative,
        "native_result": native,
        "engineering_integrity_checks": checks,
        "integrity_passed": all(checks.values()),
        "videos": videos,
        "sample_indices": sample_indices,
        "missing_frames": missing,
        "terminal_conditions": terminal,
        "interpretation": "Integrity checks are not policy success. Outcome remains the authoritative result; visual samples require human/agent review.",
    }
    (destination / "evidence.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = review(args.run, args.output or args.run / "integrity_review")
    print(
        json.dumps(
            {
                "run": report["run"],
                "integrity_passed": report["integrity_passed"],
                "checks": report["engineering_integrity_checks"],
                "sample_indices": report["sample_indices"],
            },
            indent=2,
        )
    )
    if not report["integrity_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
