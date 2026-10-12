#!/usr/bin/env python3
"""Make H.264 playback copies for finished episodes; preserve original evidence."""

import argparse
import hashlib
import json
import subprocess
import uuid
from pathlib import Path

from gpt_policy.recording.h264 import encoder_options, publish_video

CAMERAS = ("cam_high", "cam_left_wrist", "cam_right_wrist")


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def probe(path):
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-count_frames",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name,pix_fmt,width,height,r_frame_rate,nb_read_frames",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=120,
    )
    if result.stderr.strip():
        raise RuntimeError(result.stderr)
    return json.loads(result.stdout)["streams"][0]


def export(run):
    sim = run / "sim"
    if not (run / "result.json").is_file() or not (sim / "cleanup.json").is_file():
        raise ValueError(f"Only finished, cleaned-up episodes can be exported: {run}")
    output = sim / "preview_h264"
    output.mkdir(exist_ok=True)
    evidence = []
    for camera in CAMERAS:
        source = sim / f"{camera}.mp4"
        target = output / source.name
        manifest = output / f"{camera}.json"
        source_hash = digest(source)
        if target.exists() or manifest.exists():
            if not (target.exists() and manifest.exists()):
                raise FileExistsError(f"Incomplete prior export: {target}")
            prior = json.loads(manifest.read_text())
            if prior["source_sha256"] != source_hash or prior[
                "output_sha256"
            ] != digest(target):
                raise ValueError(f"Existing preview provenance mismatch: {target}")
            evidence.append(prior)
            continue
        original = probe(source)
        temporary = output / f".{camera}.{uuid.uuid4().hex}.encoding.mp4"
        subprocess.run(
            [
                "ffmpeg",
                "-nostdin",
                "-hide_banner",
                "-v",
                "error",
                "-xerror",
                "-n",
                "-threads",
                "2",
                "-i",
                str(source),
                "-map",
                "0:v:0",
                *encoder_options(),
                str(temporary),
            ],
            check=True,
            timeout=180,
        )
        encoded = probe(temporary)
        if any(
            encoded[k] != original[k]
            for k in ("width", "height", "r_frame_rate", "nb_read_frames")
        ):
            raise ValueError(f"Preview frame geometry/count/rate changed: {source}")
        if encoded["codec_name"] != "h264" or encoded["pix_fmt"] != "yuv420p":
            raise ValueError(f"Unexpected preview codec: {encoded}")
        if digest(source) != source_hash:
            raise ValueError(f"Source changed during export: {source}")
        publish_video(temporary, target)
        record = {
            "source": str(source),
            "source_sha256": source_hash,
            "output": str(target),
            "output_sha256": digest(target),
            "source_probe": original,
            "output_probe": encoded,
            "purpose": "lossy playback derivative; original video and PNG evidence preserved",
        }
        with manifest.open("x") as stream:
            json.dump(record, stream, indent=2)
            stream.write("\n")
        evidence.append(record)
        print(json.dumps({"video": str(target), **encoded}), flush=True)
    return evidence


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path)
    args = parser.parse_args()
    for run in args.runs:
        export(run.resolve())
