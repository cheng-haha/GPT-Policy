#!/usr/bin/env python3
"""Relabel reviewed RoboDojo context text without reselecting or altering evidence."""
import argparse
import json
import shutil
from pathlib import Path

from prepare_robodojo_contexts import digest, validate

from gpt_policy.input.demonstration import prepare_demonstration
from gpt_policy.input.lerobot_demo import save, smoke
from gpt_policy.input.request import RunInput


def relabel(original, output, instruction, summary, reason):
    original, output = Path(original).resolve(), Path(output).resolve()
    if not all(isinstance(v, str) and v.strip() for v in (instruction, summary, reason)):
        raise ValueError("Instruction, summary and relabel reason must be nonempty")
    if output.exists():
        raise FileExistsError(output)
    source = json.loads((original / "source.json").read_text())
    demo = json.loads((original / "source_reviewed.json").read_text())
    old_prepared = json.loads((original / "prepared/demo.json").read_text())
    if demo["source"].get("instruction_relabel"):
        raise ValueError("This operation requires an original, unrelabelled source")
    revision = {
        "original_instruction": demo["title"],
        "replacement_instruction": instruction,
        "reason": reason,
        "source": "user_authorized_visual_relabel",
        "native_target_success_verified": False,
    }
    old_summary = demo["summary"]
    demo["title"], demo["summary"] = instruction, summary
    demo["source"]["instruction_relabel"] = revision
    demo["outcome"]["instruction_scope"] = "original_dataset_task"
    # Use existing package image bytes, independent of the selector's cache.
    for frame, old in zip(demo["keyframes"], old_prepared["keyframes"], strict=True):
        for view, image in old["images"].items():
            frame["images"][view] = str(original / "prepared" / image)
    output.mkdir(parents=True)
    for name in ("raw_endposes.jsonl", "frame_audit.json"):
        shutil.copy2(original / name, output / name)
    save(output / "source_reviewed.json", demo)
    _, counts = prepare_demonstration(
        RunInput(instruction, "gpt-6-astra"), output / "source_reviewed.json",
        "video+action", output / "prepared",
    )
    source["instruction_relabel"] = revision
    source["text_revision"] = {
        "original_package": str(original),
        "original_input_sha256": digest(original / "prepared/input.json"),
        "original_summary": old_summary,
        "replacement_summary": summary,
        "selection_reused_without_model_call": True,
    }
    # Original selector output stays intact as historical provenance.
    source["counts"] = counts
    save(output / "source.json", source)
    portable = json.loads((output / "prepared/demo.json").read_text())
    for old, new in zip(old_prepared["keyframes"], portable["keyframes"], strict=True):
        for key in ("t_s", "video_frame_index", "state", "action", "alignment", "stage", "observation", "roles", "result"):
            if old.get(key) != new.get(key):
                raise ValueError(f"Unexpected evidence change: {key}")
        for view, image in new["images"].items():
            if digest(output / "prepared" / image) != digest(original / "prepared" / old["images"][view]):
                raise ValueError("Image bytes changed")
        new["images"] = {v: "prepared/" + p for v, p in new["images"].items()}
    save(output / "reviewed.json", portable)
    if counts["output_action_samples"] != json.loads((original / "source.json").read_text())["counts"]["output_action_samples"]:
        raise ValueError("Action reference count changed")
    save(output / "smoke.json", smoke(output / "prepared/input.json"))
    result = validate(output)
    result.update(original_images_and_numeric_references_unchanged=True,
                  model_reselection=False, original_package=str(original))
    save(output / "context_verification.json", result)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    args = parser.parse_args()
    relabel(**json.loads(args.spec.read_text()))
