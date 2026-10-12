#!/usr/bin/env python3
"""Derive the Make Kong transfer-context variant without resampling its demonstration."""
import argparse
import copy
import hashlib
import json
import shutil
from pathlib import Path

SOURCE_SHA256 = "604af0ae493432fdcc492ba0749bfa613c7a013c4bbdb5af7ac6542dcded53e3"
OLD = ("For imitation, preserve the demonstrated contact side, object-to-gripper orientation, "
       "push/pull direction, arm roles and stage order.")
NEW = ("Preserve task-required order and object-relative contact geometry. Unless the task "
       "requires otherwise, demonstrated arm assignments and world poses are examples; "
       "choose the arm and targets from the current scene.")
PLACEMENT = ("Finish transport above the currently available placement space, inspect the held "
             "tile's orientation, then descend in a separate short action.")

PUSH_GUIDANCE = 'Only when deliberately pushing an object with an empty, closed gripper, start with 2–5 mm `hold` increments in short chunks and inspect fresh images before advancing. Keep the gripper and pushed object clear of neighbors; withdraw and reassess if a neighbor moves or the target jams. Stop once the intended effect is visible.'
SUMMARY_NOTES = "\n\nExecution notes: " + PUSH_GUIDANCE + "\nFor placement: " + PLACEMENT


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def adapt(original, output):
    original, output = Path(original).resolve(), Path(output).resolve()
    source = original / "prepared/input.json"
    if digest(source) != SOURCE_SHA256:
        raise ValueError("Expected the reviewed episode 0 geometry context")
    before = json.loads(source.read_text())
    after = copy.deepcopy(before)
    if after["content"][0].count(OLD) != 1:
        raise ValueError("Expected exactly one arm-role preservation sentence")
    after["content"][0] = after["content"][0].replace(OLD, NEW)
    overview = json.loads(after["content"][1])
    if not isinstance(overview.get("summary"), str) or SUMMARY_NOTES in overview["summary"]:
        raise ValueError("Expected the unmodified demonstration summary")
    overview["summary"] += SUMMARY_NOTES
    after["content"][1] = json.dumps(overview, ensure_ascii=False, separators=(",", ":"))
    images = [x for x in before["content"] if isinstance(x, dict)]
    if len(images) != 48:
        raise ValueError("Expected all 16 three-view keyframes")
    # Exclusive creation protects every earlier package and experiment.
    prepared = output / "prepared"
    prepared.mkdir(parents=True, exist_ok=False)
    evidence = []
    for image in images:
        relative = Path(image["image"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Expected portable image paths")
        src, dst = source.parent / relative, prepared / relative
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        assert digest(src) == digest(dst)
        evidence.append({"image": str(relative), "sha256": digest(src)})
    (prepared / "input.json").write_text(json.dumps(after, ensure_ascii=False, indent=2) + "\n")
    restored = copy.deepcopy(after)
    restored["content"][0] = before["content"][0]
    restored["content"][1] = before["content"][1]
    assert restored == before
    rows = sum(len(json.loads(x).get("action", {}).get("sample_rows", []))
               for x in before["content"] if isinstance(x, str) and x.startswith("{"))
    assert rows == 175 and digest(source) == SOURCE_SHA256
    provenance = {
        "source_package": str(original), "source_input_sha256": SOURCE_SHA256,
        "input_sha256": digest(prepared / "input.json"),
        "changed_content_indices": [0, 1],
        "header_replacement": {"before": OLD, "after": NEW},
        "placement_transfer_guidance": PLACEMENT,
        "push_guidance": PUSH_GUIDANCE,
        "guidance_location": "content[1].summary: appended Execution notes",
        "push_guidance_source": "Authored shared prompt at c22e567, relocated unchanged; developed during prior Make Kong pilots, not inferred solely from this expert episode",
        "historical_factual_text_preserved": True, "numeric_rows_unchanged": rows,
        "images": evidence, "original_unchanged": True, "model_calls": 0,
    }
    (output / "context_verification.json").write_text(json.dumps(provenance, indent=2) + "\n")
    (output / "prepared_inputs.override.json").write_text(
        json.dumps({"make_kong": str(prepared / "input.json")}, indent=2) + "\n")
    (output / "README.md").write_text(
        "# Make Kong: adaptive transfer context\n\n"
        "Derived compiled input, not a newly selected expert episode. Only the arm-role "
        "transfer sentence and Execution notes appended to the existing summary differ. "
        "Pushing and placement notes share the same summary field used by other tasks. "
        "Frame descriptions use only stage, observation, roles and result; no "
        "execution_guidance or transfer_guidance fields are added. The numeric-reference "
        "semantics marker is preserved for the compiler. The pushing wording is relocated "
        "unchanged from the shared prompt at c22e567 and was developed during prior Make "
        "Kong pilots, not inferred solely from this expert episode. The placement wording "
        "comes from adaptive_v1's pre-descent note. Both are authored guidance, not "
        "recorded expert actions. All original "
        "historical annotations, 175 measured-state rows and 48 image bytes are preserved. "
        "The image copies make this input portable. Full expert provenance remains in "
        + str(original) + "; it is not relabeled as a new expert or evaluation success.\n")
    print(json.dumps({k: v for k, v in provenance.items() if k != "images"}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    adapt(args.original, args.output)
