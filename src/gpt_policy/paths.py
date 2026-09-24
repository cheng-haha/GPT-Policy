"""Shared filesystem locations for generated run artifacts."""

from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def generated_root() -> Path | None:
    """Return the optional root for generated artifacts."""
    value = os.environ.get("GPT_POLICY_GENERATED_ROOT", "").strip()
    return Path(value).expanduser() if value else None


def generated_var_path(*parts: str) -> Path:
    """Map the repository-local var tree into GPT_POLICY_GENERATED_ROOT."""
    root = generated_root()
    base = root / "var" if root is not None else PROJECT_ROOT / "var"
    return base.joinpath(*parts)
