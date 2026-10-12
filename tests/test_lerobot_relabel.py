"""Instruction relabels preserve source identity, geometry and expert provenance."""

import json
from pathlib import Path

import pytest
from test_lerobot_demo import FakeDataset, FakeSelector, fixture

from gpt_policy.input import lerobot_demo

INSTRUCTION = "Put watches left, cars middle, and figurines right."
REASON = "User-authorized relabel matching the observed expert trajectory."


def test_relabel_reaches_selection_and_context_without_changing_source(monkeypatch, tmp_path):
    root = tmp_path / "dataset"
    fixture(root)
    factory = lerobot_demo.LeRobotEpisodeExtractor
    monkeypatch.setattr(
        lerobot_demo,
        "LeRobotEpisodeExtractor",
        lambda root, episode, config: factory(root, episode, config, FakeDataset),
    )

    class Selector(FakeSelector):
        def __init__(self):
            self.instructions = []

        def select(self, instruction, video, extraction):
            self.instructions.append(instruction)
            return super().select(instruction, video, extraction)

        def review(self, instruction, video, frames, selection, limit):
            self.instructions.append(instruction)
            return selection

    original, relabelled = tmp_path / "original", tmp_path / "relabelled"
    lerobot_demo.prepare(root, 1, original, FakeSelector(), confirmed_expert=True)
    selector = Selector()
    lerobot_demo.prepare(
        root, 1, relabelled, selector, confirmed_expert=True,
        instruction_override=INSTRUCTION, relabel_reason=REASON,
    )
    assert selector.instructions and set(selector.instructions) == {INSTRUCTION}
    source = json.loads((relabelled / "source.json").read_text())
    original_source = json.loads((original / "source.json").read_text())
    assert source["episode"] == original_source["episode"]
    assert source["episode"]["tasks"] == ["Sort objects."]
    assert source["source_hashes"] == original_source["source_hashes"]
    assert source["selection_cache_key"] != original_source["selection_cache_key"]
    assert (relabelled / "raw_endposes.jsonl").read_bytes() == (original / "raw_endposes.jsonl").read_bytes()
    compiled = json.loads((relabelled / "prepared/input.json").read_text())
    assert compiled["instruction"] == INSTRUCTION
    header = json.loads(compiled["content"][1])
    assert header["title"] == INSTRUCTION
    provenance = header["source"]["instruction_relabel"]
    assert provenance == source["instruction_relabel"]
    assert provenance["original_instruction"] == "Sort objects."
    assert provenance["replacement_instruction"] == INSTRUCTION
    assert provenance["reason"] == REASON
    assert provenance["native_target_success_verified"] is False
    assert header["source"]["episode_index"] == 1
    assert header["outcome"]["source"] == "dataset_owner_confirmation"
    assert header["outcome"]["success"] is True
    assert header["outcome"]["instruction_scope"] == "original_dataset_task"


@pytest.mark.parametrize(
    ("instruction", "reason"),
    [(INSTRUCTION, None), (None, REASON), (" ", REASON), (INSTRUCTION, "")],
)
def test_relabel_requires_instruction_and_provenance_before_writing(tmp_path, instruction, reason):
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="provenance"):
        lerobot_demo.prepare(
            tmp_path / "missing", 1, output, FakeSelector(),
            instruction_override=instruction, relabel_reason=reason,
        )
    assert not output.exists()


def test_completed_package_cannot_be_reused_for_another_relabel(tmp_path):
    import runpy

    module = runpy.run_path(
        str(Path(__file__).resolve().parents[1] / "scripts/prepare_robodojo_contexts.py")
    )
    source = {
        "dataset": str(tmp_path / "dataset"), "episode_index": 77,
        "instruction_relabel": {"replacement_instruction": INSTRUCTION, "reason": REASON},
    }
    (tmp_path / "reviewed.json").write_text(json.dumps({"source": source}))
    check = module["check_request_identity"]
    check(tmp_path, source["dataset"], 77, INSTRUCTION, REASON)
    for dataset, episode, instruction, reason in [
        (tmp_path / "other", 77, INSTRUCTION, REASON),
        (source["dataset"], 0, INSTRUCTION, REASON),
        (source["dataset"], 77, "A different goal", REASON),
        (source["dataset"], 77, INSTRUCTION, "different provenance"),
    ]:
        with pytest.raises(ValueError, match="identity"):
            check(tmp_path, dataset, episode, instruction, reason)
