"""Native-invalid substitution cannot discard a genuine failure or rewrite history."""
import copy
import importlib.util
from pathlib import Path

import pytest

from gpt_policy.simulation.robodojo.io import sha256, write_json

SPEC = importlib.util.spec_from_file_location(
    "layout_substitution", Path(__file__).parents[2] / "scripts/substitute_robodojo_layout.py"
)
sub = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(sub)


@pytest.fixture
def prior(tmp_path):
    cases = []
    for layout in range(5):
        invalid = layout == 4
        path = tmp_path / f"old_{layout}.json"
        write_json(path, {
            "task": "imitate_sorting_sequence", "seed": 0, "layout": layout,
            "status": "invalid_native_layout" if invalid else "native_completed",
            "complete": not invalid, "valid_for_success_rate": not invalid,
            "native_success": None if invalid else True,
        })
        cases.append({"layout": layout, "source": str(path),
                      "source_sha256": sha256(path),
                      "score_100": None if invalid else 100.0})
    return {"tasks": [{"task": "imitate_sorting_sequence", "name": "顺序模仿",
                       "cases": cases}], "reconstructed_cases": ["historical_marker"]}


def selection(prior):
    return sub.select(prior, "imitate_sorting_sequence", 4, 0)


@pytest.mark.parametrize("status", ["native_completed", "budget_censored", "runtime_error",
                                   "controller_interrupted", "model_give_up"])
def test_only_native_invalid_can_advance(prior, status):
    row = prior["tasks"][0]["cases"][4]
    result = sub.read(row["source"])
    result["status"] = status
    write_json(row["source"], result)
    row["source_sha256"] = sha256(row["source"])
    with pytest.raises(ValueError, match="native-invalid"):
        selection(prior)


def test_next_layout_and_source_identity(prior):
    old, spec = selection(prior)
    assert old["layout"] == 4 and spec["layout"] == 5
    with pytest.raises(ValueError, match="identity"):
        sub.select(prior, spec["task"], 4, 1)
    Path(old["source"]).write_text("{}")
    with pytest.raises(ValueError, match="changed"):
        selection(prior)


@pytest.mark.parametrize("status,success,score,failures,censored,invalid", [
    ("native_completed", True, 100.0, 0, 0, 0),
    ("native_completed", False, 40.0, 1, 0, 0),
    ("budget_censored", None, 60.0, 0, 1, 0),
    ("invalid_native_layout", None, None, 0, 0, 1),
])
def test_merge_keeps_every_outcome_and_all_history(
    tmp_path, prior, status, success, score, failures, censored, invalid
):
    original = copy.deepcopy(prior)
    hashes = sub.verify_sources(prior)
    old, spec = selection(prior)
    path = tmp_path / "new.json"
    write_json(path, {
        "case": {"task": spec["task"], "eval_seed": 0, "layout_id": 5},
        "status": status, "complete": status == "native_completed",
        "valid_for_success_rate": status == "native_completed",
        "native_success": success, "native_process_score_100": score,
    })
    manifest = {"replacement_spec": spec, "original_case": old,
                "prior_report": "old_report.json", "source_commit": "abc"}
    result = sub.merge(prior, manifest, path)
    task = result["tasks"][0]
    assert task["evaluated_layouts"] == [0, 1, 2, 3, 5]
    assert task["confirmed_successes"] == 4 + (success is True)
    assert task["native_failures"] == failures
    assert task["budget_censored"] == censored
    assert task["invalid_scenes"] == invalid
    assert task["mean_score_100"] == ((400 + score) / 5 if score is not None else 100)
    assert result["skipped_invalid_layouts"][0]["source"] == old["source"]
    assert result["reconstructed_cases"] == ["historical_marker"]
    assert prior == original and sub.verify_sources(prior) == hashes
    if status != "invalid_native_layout":
        with pytest.raises(ValueError, match="native-invalid"):
            sub.select(result, spec["task"], 5, 0)


def test_wrong_replacement_identity_is_rejected(tmp_path, prior):
    old, spec = selection(prior)
    p = tmp_path / "wrong.json"
    write_json(p, {"case": {"task": spec["task"], "eval_seed": 0, "layout_id": 6}})
    with pytest.raises(ValueError, match="identity"):
        sub.merge(prior, {"replacement_spec": spec, "original_case": old}, p)
