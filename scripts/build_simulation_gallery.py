#!/usr/bin/env python3
"""Build a static, file://-compatible gallery from the allowlisted public release.

Never opens raw model sessions or replays the simulator. Three videos are joined
frame-for-frame. The execution display contains numerical public action records,
not reconstructed model reasoning. All 50 selected outcomes are retained.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

TASKS = {
    "classify_objects": "Classify objects",
    "classify_objects_by_language": "Classify by language",
    "organize_table": "Organize table",
    "imitate_sorting_sequence": "Imitate sorting sequence",
    "arrange_largest_number": "Arrange largest number",
    "pack_objects_into_box": "Pack objects into box",
    "build_tower": "Build tower",
    "make_kong": "Make Kong",
    "fold_clothes": "Fold clothes",
    "put_bottles_into_dustbin": "Put bottles into dustbin",
}
VIEWS = ("left_wrist", "overhead", "right_wrist")


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def dump(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")) + "\n")


def digest(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def probe(path):
    data = json.loads(subprocess.check_output([
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
        "stream=width,height,r_frame_rate,nb_frames,pix_fmt,codec_name", "-of", "json", str(path)
    ]))
    return data["streams"][0]


def make_trace(folder, result, supplement=None):
    actions = read_jsonl(folder / "actions.jsonl")
    segments = read_jsonl(folder / "execution_segments.jsonl")
    scores = read_jsonl(folder / "native_score_trace.jsonl")
    controls = result["control_steps"]
    assert [a["control_step"] for a in actions] == list(range(1, controls + 1))
    assert [s["native_control_step"] for s in scores] == list(range(controls + 1))
    assert scores[-1]["score_100"] == result["score_100"]
    assert scores[-1]["native_complete"]
    calls, step = [], 0
    for segment in segments:
        calls.append({"decision": segment["decision_index"] + 1, "tool": segment["tool"],
                      "step": step, "controls": segment["acknowledged_controls"]})
        step += segment["acknowledged_controls"]
    assert step == controls
    if supplement is not None:
        assert supplement["source_hashes"]["execution_segments"] == digest(folder / "execution_segments.jsonl")
        assert len(supplement["calls"]) == len(calls)
        for call, recorded in zip(calls, supplement["calls"]):
            assert all(call[k] == recorded[k] for k in ("decision", "tool", "step"))
            call.update({k: recorded[k] for k in ("note", "points", "localization") if k in recorded})
    executed = [s for s in segments if s["acknowledged_controls"] > 0]
    assert sum(s["acknowledged_controls"] for s in executed) == controls
    trace, previous, segment_index = [], None, 0
    for action in actions:
        step = action["control_step"]
        while step > executed[segment_index]["end_step"]:
            segment_index += 1
        segment = executed[segment_index]
        assert segment["start_step"] < step <= segment["end_step"]
        targets = {}
        for arm, value in action.get("requested_targets", {}).items():
            assert arm in ("left", "right")
            targets[arm] = {key: value[key] for key in ("pose_xyzquat", "gripper") if key in value}
        record = {
            "step": step, "decision": segment["decision_index"] + 1,
            "tool": segment["tool"], "targets": targets,
        }
        identity = (record["decision"], record["tool"], targets)
        if identity != previous:
            trace.append(record)
            previous = identity
    score_changes = []
    for sample in scores:
        value = sample["score_100"]
        if not score_changes or value != score_changes[-1][1]:
            score_changes.append([sample["native_control_step"], value])
    return {"actions": trace, "scores": score_changes, "calls": calls}


def build_case(result_path, source, output, supplements):
    result = json.loads(result_path.read_text())
    assert result["task"] in TASKS
    assert result["native_complete"] and result["native_valid_for_success_rate"]
    relative = result_path.parent.relative_to(source)
    destination = output / relative
    destination.mkdir(parents=True, exist_ok=True)
    inputs = [source / result["videos"][view] for view in VIEWS]
    specs = [probe(p) for p in inputs]
    for spec in specs:
        assert (spec["width"], spec["height"]) == (640, 480)
        assert Fraction(spec["r_frame_rate"]) == 25
        assert int(spec["nb_frames"]) == result["control_steps"] + 1
    video = destination / "rollout.mp4"
    hashes = {str(p.relative_to(source)): digest(p) for p in inputs}
    settings = {"view_order": VIEWS, "fps": 25, "crf": 24, "preset": "fast", "width": 1920, "height": 480}
    provenance = destination / "source.json"
    stamp = {"inputs": hashes, "encoding": {**settings, "view_order": list(VIEWS)}}
    reuse = video.exists() and provenance.exists() and json.loads(provenance.read_text()) == stamp
    if not reuse:
        cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-filter_complex_threads", "1"]
        for path in inputs:
            cmd += ["-threads", "1", "-i", str(path)]
        cmd += ["-filter_complex", "[0:v][1:v][2:v]hstack=inputs=3[v]", "-map", "[v]",
                "-an", "-c:v", "libx264", "-threads", "2", "-preset", "fast", "-crf", "24",
                "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(video)]
        subprocess.run(cmd, check=True)
        dump(provenance, stamp)
    spec = probe(video)
    assert int(spec["nb_frames"]) == result["control_steps"] + 1
    assert (spec["width"], spec["height"], spec["codec_name"], spec["pix_fmt"]) == (1920, 480, "h264", "yuv420p")
    poster = destination / "poster.jpg"
    if not poster.exists() or not reuse:
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(video),
                        "-frames:v", "1", "-q:v", "3", str(poster)], check=True)
    trace = make_trace(result_path.parent, result, supplements.get(result["episode_id"]))
    trace_path = destination / "trace.js"
    trace_path.write_text("window.gptPolicySimulationTraces ??= {};\nwindow.gptPolicySimulationTraces["
                          + json.dumps(result["episode_id"]) + "]="
                          + json.dumps(trace, separators=(",", ":")) + ";\n")
    # Evidence provenance uses only public relative paths, never host paths.
    for name in ("result.json", "actions.jsonl", "execution_segments.jsonl", "native_score_trace.jsonl"):
        path = result_path.parent / name
        hashes[str(path.relative_to(source))] = digest(path)
    case = {key: result[key] for key in ("episode_id", "layout", "variant", "seed", "score_100", "success", "decisions", "control_steps")}
    case.update({"video": str(video.relative_to(output)), "poster": str(poster.relative_to(output)),
                 "trace": str(trace_path.relative_to(output)), "frames": int(spec["nb_frames"]),
                 "duration": int(spec["nb_frames"]) / 25, "source_hashes": hashes})
    print(f"READY {result['episode_id']} ({video.stat().st_size / 1e6:.1f} MB)", flush=True)
    return result["task"], result["task_name"], case


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--tool-calls", type=Path, help="Allowlisted tool-note supplement; defaults to output/tool-calls.json if present")
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    assert source != output and source not in output.parents, "Write outside the public release"
    results = sorted(source.glob("tasks/*/seed_*/*/result.json"))
    assert len(results) == 50, f"Expected all 50 selected cases, got {len(results)}"
    output.mkdir(parents=True, exist_ok=True)
    supplement_path = args.tool_calls or output / "tool-calls.json"
    if args.tool_calls:
        assert supplement_path.is_file(), f"Missing tool-call supplement: {supplement_path}"
    supplements = json.loads(supplement_path.read_text())["episodes"] if supplement_path.exists() else {}
    if supplements:
        assert len(supplements) == 50
        assert set(supplements) == {json.loads(p.read_text())["episode_id"] for p in results}
    tasks = {key: {"id": key, "en": name, "cases": []} for key, name in TASKS.items()}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for task_id, chinese, case in pool.map(lambda p: build_case(p, source, output, supplements), results):
            tasks[task_id]["zh"] = chinese
            tasks[task_id]["cases"].append(case)
    for task in tasks.values():
        task["cases"].sort(key=lambda c: (c["variant"] != "standard", c["layout"]))
        assert len(task["cases"]) == 5
    cases = [case for t in tasks.values() for case in t["cases"]]
    assert len({c["episode_id"] for c in cases}) == 50
    assert sum(c["success"] for c in cases) == 36
    assert abs(sum(c["score_100"] for c in cases) / 50 - 80.2) < 1e-8
    manifest = {"release": "gpt-policy-robodojo-10tasks-20261011", "fps": 25,
                "view_order": VIEWS, "tasks": list(tasks.values())}
    dump(output / "manifest.json", manifest)
    (output / "manifest.js").write_text("window.gptPolicySimulation=" + json.dumps(manifest, ensure_ascii=False, separators=(",", ":")) + ";\n")
    for name in ("EVALUATION.md", "LICENSE.txt", "THIRD_PARTY.md"):
        original = source / name
        if name == "EVALUATION.md" and not original.exists():
            original = source / "protocol" / name
        shutil.copy2(original, output / name)
    print(f"COMPLETE: 50 episodes, 36 successes, mean score 80.20; {sum((output / c['video']).stat().st_size for c in cases) / 1e6:.1f} MB", flush=True)


if __name__ == "__main__":
    main()
