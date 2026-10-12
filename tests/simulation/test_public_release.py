"""Public export excludes private fields and preserves evaluation denominators."""
import importlib.util
import json
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).parents[2] / "scripts"
sys.path.insert(0, str(SCRIPTS))
try:
    spec = importlib.util.spec_from_file_location("release_export", SCRIPTS / "export_robodojo_release.py")
    export = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(export)
    from verify_robodojo_release import aggregate, local_path
finally:
    sys.path.remove(str(SCRIPTS))


def episode(success=True, **kwargs):
    e = {
        "task": "test", "task_name": "test", "layout": 0,
        "native_complete": True, "native_valid_for_success_rate": True,
        "native_success": success, "success": success,
        "outcome": "success" if success else "native_failure",
        "score_100": 100 if success else 0, "score_basis": "stored_native_process_score",
    }
    return dict(e, **kwargs)


def test_timeout_stays_in_denominator_and_reconstructed_score_is_explicit():
    cases = [episode(), episode(False),
             episode(False, native_complete=False, native_valid_for_success_rate=False,
                     native_success=None, outcome="timeout", score_100=50,
                     score_basis="offline_replay_reconstruction")]
    s = aggregate(cases)
    assert s["success_rate"] == pytest.approx(1 / 3)
    assert s["episode_mean_score_100"] == 50
    assert s["native_failures"] == s["timeouts"] == s["reconstructed_score_cases"] == 1


def test_cannot_count_unfinished_native_success():
    with pytest.raises(ValueError, match="eligibility"):
        aggregate([episode(native_complete=False)])
    with pytest.raises(ValueError, match="Outcome"):
        aggregate([episode(False, outcome="timeout")])


@pytest.mark.parametrize("score", [-1, 101, float("nan"), float("inf"), True, None])
def test_invalid_scores_cannot_enter_public_table(score):
    with pytest.raises(ValueError, match="score"):
        aggregate([episode(score_100=score)])


def test_action_projection_drops_unknown_nested_and_free_text():
    raw = {
        "step": 1, "command": [0.0] * 14,
        "requested": {
            "left": {"pose_xyzquat": [0, 0, 0, 0, 0, 0, 1], "gripper": 1,
                     "private_note": "must not ship"},
            "steps": 5, "note": "must not ship",
        },
        "observation_id": "private_session:0", "private_reasoning": "must not ship",
        "account_id": "must not ship",
    }
    public = export.public_action(raw)
    serialized = json.dumps(public)
    assert "private" not in serialized and "must not ship" not in serialized
    assert public["requested_targets"]["left"]["pose_xyzquat"][-1] == 1
    assert public["control_step"] == 1


def test_context_projection_keeps_episode_but_not_dataset_mount():
    source = {"source": {"dataset": "/private/machine/dataset/task", "episode_index": 77,
                         "private_auth": "must not ship"}, "title": "Sort objects"}
    context = export.public_context(
        {"context": {"sha256": "a" * 64, "image_count": 48}},
        {"content": [json.dumps(source)], "account_id": "must not ship"})
    assert context["expert_dataset_task"] == "task"
    assert context["expert_episode_index"] == 77
    assert context["timepoints"] == 16
    assert "/private" not in json.dumps(context)
    assert "must not ship" not in json.dumps(context)


@pytest.mark.parametrize("path", ["../outside", "/absolute/path", "a/../../outside"])
def test_release_links_cannot_escape(tmp_path, path):
    with pytest.raises(ValueError, match="Unsafe"):
        local_path(tmp_path, path)


def test_symlink_escape_is_rejected(tmp_path):
    (tmp_path / "link").symlink_to(tmp_path.parent)
    with pytest.raises(ValueError, match="escapes"):
        local_path(tmp_path, "link/outside")


def test_numerical_action_rejects_nonfinite_and_wrong_width():
    with pytest.raises(ValueError, match="width"):
        export.public_action({"step": 1, "command": [0] * 13})
    with pytest.raises(ValueError, match="Nonfinite"):
        export.public_action({"step": 1, "command": [float("nan")] + [0] * 13})



def test_standard_and_random_layout_ids_have_distinct_paths_and_identities():
    standard = export.public_identity("fold_clothes", {"layout": 0},
                                      {"task": "fold_clothes", "eval_seed": 0, "layout_id": 0})
    random = export.public_identity("fold_clothes", {"layout": 0, "variant": "random"},
                                    {"task": "fold_clothes_random", "variant": "random",
                                     "eval_seed": 0, "layout_id": 0})
    assert standard[2] != random[2]
    assert standard[3] != random[3]
    assert str(random[3]).endswith("/random_layout_0")
    with pytest.raises(ValueError, match="identity"):
        export.public_identity("fold_clothes", {"layout": 0, "variant": "random"},
                               {"task": "fold_clothes", "eval_seed": 0, "layout_id": 0})


def test_gallery_and_docs_use_current_metrics_and_scene_variants(tmp_path):
    cases = [episode(variant="standard"), episode(
        False, variant="random", native_complete=False,
        native_valid_for_success_rate=False, native_success=None,
        outcome="timeout", score_100=50, score_basis="offline_replay_reconstruction")]
    summary = aggregate(cases)
    html = export.gallery(cases, summary)
    assert "<strong>50%</strong>" in html and "<strong>75.00</strong>" in html
    assert "Success · 1 / 2" in html
    assert "1 historical stage scores" in html
    assert "e.variant" in html
    assert summary["tasks"][0]["scenes"] == [
        {"variant": "standard", "layout": 0}, {"variant": "random", "layout": 0}]
    (tmp_path / "protocol").mkdir()
    export.write_docs(tmp_path, summary, {}, {}, cases)
    readme = (tmp_path / "README.md").read_text()
    chinese = (tmp_path / "README.zh-CN.md").read_text()
    assert "Mean score 75.00" in readme and "50.00%" in chinese
    assert "\n\n" in chinese and r"\n" not in chinese
    assert "standard 0, random 0" in readme
    assert "1 historical stage scores" in readme
