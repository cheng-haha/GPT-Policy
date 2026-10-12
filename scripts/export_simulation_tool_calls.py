#!/usr/bin/env python3
"""Read selected run records and emit only tool notes and localization geometry.

Run over SSH with this script on stdin; stdout is the portable JSON supplement.
No transcripts, observations, provider sessions or reasoning fields are exported.
The current selection manifest, not directory order, selects recovery attempts.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", type=Path)
    parser.add_argument("release", type=Path)
    args = parser.parse_args()
    selected = read(args.runs / "FINAL_SUMMARY.json")["episodes"]
    public = {read(p)["episode_id"]: p.parent for p in args.release.glob("tasks/*/seed_*/*/result.json")}
    assert len(selected) == len(public) == 50
    episodes = {}
    totals = Counter()
    for item in selected:
        case_id = item["case"]["case_id"]
        source = Path(item["result"]).resolve()
        assert args.runs.resolve() in source.parents
        assert digest(source) == item["result_sha256"]
        folder = public[case_id]
        segments_path = folder / "execution_segments.jsonl"
        segments = [json.loads(line) for line in segments_path.read_text().splitlines()]
        events_path = source.parent / "policy/events.jsonl"
        calls = []
        by_index = {}
        with events_path.open() as stream:
            for line in stream:
                row = json.loads(line)
                if row.get("event") == "model_decision":
                    arguments = row["arguments"]
                    call = {"decision": row["step"] + 1, "tool": row["name"],
                            "step": row["native_step"], "note": arguments.get("note", "")}
                    assert isinstance(call["note"], str)
                    if call["tool"] == "locate_point":
                        call["points"] = {key: {k: arguments[key][k] for k in ("step", "camera", "pixel_xy")}
                                          for key in ("first", "second") if isinstance(arguments.get(key), dict)}
                    assert row["step"] not in by_index
                    by_index[row["step"]] = call
                    calls.append(call)
                elif row.get("event") == "execution_result" and row.get("name") == "locate_point":
                    call = by_index[row["step"]]
                    assert call["tool"] == "locate_point"
                    result = row["result"]
                    call["localization"] = {key: result[key] for key in (
                        "frame", "metric_position_available", "position", "residual_m", "rejection_reasons") if key in result}
        assert len(calls) == len(segments), case_id
        step = 0
        for call, segment in zip(calls, segments):
            assert (call["decision"], call["tool"]) == (segment["decision_index"] + 1, segment["tool"]), case_id
            assert call["step"] == step, (case_id, call["decision"], call["step"], step)
            step += segment["acknowledged_controls"]
        assert step == read(folder / "result.json")["control_steps"]
        totals.update(call["tool"] for call in calls)
        episodes[case_id] = {"calls": calls, "source_hashes": {
            "selected_result": digest(source), "tool_events": digest(events_path),
            "execution_segments": digest(segments_path)}}
    print(json.dumps({"schema_version": 1, "episodes": episodes}, ensure_ascii=False, separators=(",", ":")))
    print(f"Verified {len(episodes)} episodes: {dict(totals)}", file=sys.stderr)


if __name__ == "__main__":
    main()
