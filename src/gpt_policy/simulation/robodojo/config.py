"""Offline configuration and dependency preflight; no Isaac/model imports."""

import json
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path

from .arrival import (
    ARRIVAL_LIMITS,
    early_arrival_enabled,
    pass_through_enabled,
    stop_on_arrival_enabled,
)
from .budget import budget_config
from .execution import execution_config
from .io import sha256
from .tasks import SUPPORTED_TASKS, case_identity

PIN = "726e9aabfaa642203722eb126f5eaf0f37f3e1ad"
SUBMODULES = {
    "XPolicyLab": "bb9a0b5f5136a74503b679af830bfd0a3a837d5c",
    "third_party/IsaacLab": "afca7b09d60d8beb9c1cb28b43066499940b969b",
    "third_party/curobo": "d17b54ce32cba095c0b000c4c58777075d11de0e",
}


def git(root, *args):
    return subprocess.check_output(
        ["git", "-C", str(root), *args], text=True, timeout=30
    ).strip()


def decision_limit(config):
    """None (including an omitted key) means no model-decision count limit."""
    limit = config.get("max_decisions")
    if limit is not None and (type(limit) is not int or limit < 1):
        raise ValueError("max_decisions must be null or a positive integer")
    return limit


def simulation_agent_config(config):
    """Inherit the shared Codex profile unless the run explicitly overrides it."""
    from dataclasses import replace

    from ...harness.config import named_agent_config, with_live_image_window

    project = Path(__file__).resolve().parents[4]
    profile = named_agent_config("codex", project / "configs")
    profile = replace(
        profile, model=config["model"], executable=config["codex_bin"],
        effort=config["effort"], decision_only=True,
    )
    if "live_image_window" in config:
        profile = with_live_image_window(profile, config["live_image_window"])
    return profile


def load_config(path):
    config = json.loads(Path(path).read_text())
    for key in ("robodojo_root", "simulation_python", "output_root"):
        if not config.get(key):
            raise ValueError("Missing configuration: " + key)
    if config.get("robodojo_commit") != PIN:
        raise ValueError("Configuration must use the reviewed RoboDojo pin")
    decision_limit(config)
    execution_config(config)
    budget_config(config)
    early_arrival_enabled(config)
    stop_on_arrival_enabled(config)
    pass_through_enabled(config)
    if (config.get("distribute_closed_gripper_moves", False)
            and not early_arrival_enabled(config)):
        raise ValueError("distribute_closed_gripper_moves requires early_arrival")
    return config


def check(config, task="classify_objects", seed=0, layout=0):
    budgets = budget_config(config)
    arrival_stop = stop_on_arrival_enabled(config)
    profile = simulation_agent_config(config)
    root = Path(config["robodojo_root"]).resolve()
    revisions = {}
    for rel, sha in {".": PIN, **SUBMODULES}.items():
        actual = git(root / rel, "rev-parse", "HEAD")
        if actual != sha:
            raise ValueError(f"Dependency mismatch {rel}: {actual} != {sha}")
        dirty = git(root / rel, "diff", "HEAD", "--stat")
        if dirty:
            raise ValueError(f"Tracked dependency modifications in {rel}: {dirty}")
        revisions[rel] = actual
    if not Path(config["simulation_python"]).is_file():
        raise ValueError("Simulation interpreter missing")
    case = case_identity(root, task, seed, layout)
    project = Path(__file__).resolve().parents[4]
    source = {
        "commit": git(project, "rev-parse", "HEAD"),
        "status": git(project, "status", "--short"),
        "diff": git(project, "diff", "HEAD"),
        "python_sha256": {
            str(path.relative_to(project)): sha256(path)
            for path in sorted((project / "src/gpt_policy").rglob("*.py"))
        },
    }
    demos = config.get("prepared_inputs", {})
    prepared = demos.get(task)
    context = None
    if prepared:
        from ...input.manifest import ImagePart, load_manifest

        manifest = load_manifest(Path(prepared))
        images = [part for part in manifest.content if isinstance(part, ImagePart)]
        if len(images) != 48 and task == "classify_objects":
            raise ValueError("Fixed Classify context requires 48 images")
        context = {
            "path": prepared,
            "sha256": sha256(prepared),
            "image_count": len(images),
            "images": [
                {"path": str(part.path), "sha256": sha256(part.path)} for part in images
            ],
        }
    assets = {"root": str((root / "Assets").resolve()), "audit": None}
    audit_path = config.get("asset_audits", {}).get(task)
    if audit_path:
        audit = json.loads(Path(audit_path).read_text())
        if (
            audit["task"] != task
            or audit["seed"] != seed
            or layout not in [int(x) for x in audit["layouts"].split(",")]
        ):
            raise ValueError("Asset audit does not cover this case")
        if audit["missing"]:
            raise ValueError("Asset audit has missing local scopes")
        for entry in audit["files"]:
            if sha256(root / entry["path"]) != entry["sha256"]:
                raise ValueError("Asset changed after audit: " + entry["path"])
            if entry["official_digest"] is not None and not entry["matches"]:
                raise ValueError(
                    "Asset differs from official revision: " + entry["path"]
                )
        assets["audit"] = {
            "path": str(audit_path),
            "sha256": sha256(audit_path),
            "official_repository": audit["official_repository"],
            "revision": audit["revision"],
            "matched": audit["matched"],
            "total": audit["total"],
            "scope_note": audit["scope_note"],
            "local_extras": [e for e in audit["files"] if e["official_digest"] is None],
            "all_scoped_files_match": audit["all_match"],
        }
    executable = config.get("codex_bin")
    provider = {
        "model": config.get("model"),
        "effort": config.get("effort"),
        "executable": executable,
        "version": None,
        "live_image_window": profile.live_image_window,
    }
    if executable and Path(executable).is_file():
        provider["version"] = subprocess.check_output(
            [executable, "--version"],
            text=True,
            timeout=15,
        ).strip()
    host = {
        "python": sys.executable,
        "python_version": sys.version,
        "simulation_python_version": subprocess.check_output(
            [config["simulation_python"], "--version"],
            text=True,
            timeout=15,
        ).strip(),
        "packages": {
            name: version(name)
            for name in ("numpy", "scipy", "msgpack", "pillow", "jsonschema")
        },
    }
    return {
        "early_arrival": dict(ARRIVAL_LIMITS) if early_arrival_enabled(config) else None,
        "stop_on_arrival": arrival_stop,
        "pass_through": pass_through_enabled(config),
        "execution": execution_config(config),
        "decision_budget": decision_limit(config),
        "episode_budget": budgets,
        "provider": provider,
        "host": host,
        "hardware_opened": False,
        "simulation_started": False,
        "model_called": False,
        "source": source,
        "robodojo_root": str(root),
        "dependencies": revisions,
        "case": case,
        "context": context,
        "assets": assets,
        "tasks": list(SUPPORTED_TASKS),
    }
