"""Variant identity must reach native layouts without changing policy semantics."""

import hashlib

import pytest

from gpt_policy.simulation.robodojo.tasks import RANDOM_TASKS, case_identity
from gpt_policy.simulation.robodojo.tools import agent_context


@pytest.mark.parametrize("native,base", RANDOM_TASKS.items())
def test_random_uses_its_own_layout_and_case_id(tmp_path, native, base):
    folder = tmp_path / "Assets/Eval_Layout/RoboDojo/arx_x5/0"
    folder.mkdir(parents=True)
    standard = folder / f"{base}_0.json"
    standard.write_text('{"variant": "standard"}')
    # A standard layout must never silently stand in for a missing random layout.
    with pytest.raises(ValueError, match="Native layout missing"):
        case_identity(tmp_path, native, 0, 0)
    random = folder / f"{native}_0.json"
    random.write_text('{"variant": "random"}')
    actual = case_identity(tmp_path, native, 0, 0)
    ordinary = case_identity(tmp_path, base, 0, 0)
    assert actual["task"] == native and actual["base_task"] == base
    assert actual["variant"] == "random" and ordinary["variant"] == "standard"
    assert actual["case_id"] != ordinary["case_id"]
    assert actual["layout_path"] == str(random.resolve())
    assert actual["layout_sha256"] == hashlib.sha256(random.read_bytes()).hexdigest()
    assert actual["layout_sha256"] != ordinary["layout_sha256"]


@pytest.mark.parametrize("native,base", RANDOM_TASKS.items())
def test_random_preserves_standard_public_rules_and_action_contract(native, base):
    assert agent_context(native, extended_controls=True) == agent_context(
        base, extended_controls=True
    )


@pytest.mark.parametrize("task", ["classify_objects_random", "../fold_clothes", "fold_clothes_random_random"])
def test_unreviewed_variants_rejected(tmp_path, task):
    with pytest.raises(ValueError, match="Unknown task"):
        case_identity(tmp_path, task, 0, 0)


def test_asset_audit_includes_clutter_without_requiring_category_field(tmp_path):
    import importlib.util
    from pathlib import Path

    script = Path(__file__).resolve().parents[2] / "scripts/audit_robodojo_assets.py"
    spec = importlib.util.spec_from_file_location("variant_asset_audit", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    clutter = module.object_scope(
        tmp_path, "Rigid", "book", {"category_idx": 4, "type": "cluttered"}
    )
    assert clutter == tmp_path / "Object/RoboDojo/Clutter/book/00004"
    regular = module.object_scope(
        tmp_path, "Garment", "Top_Long", {"category_idx": 6}
    )
    assert regular == tmp_path / "Object/RoboDojo/Garment/Top_Long/00006"
