"""Resolve command-line arguments and optional input manifests."""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import os
from pathlib import Path
import tempfile

from .manifest import ContentPart, InputManifest, TextPart, VideoPart, content_records, load_manifest
from .demo_mode import resolve_demo_mode


@dataclass(frozen=True)
class RunInput:
    instruction: str
    model: str
    content: tuple[ContentPart, ...] = ()
    manifest: InputManifest | None = None

    def record(self) -> dict[str, object]:
        payload = self.request()
        if self.manifest is not None:
            payload["manifest"] = self.manifest.record()
        return payload

    def request(self) -> dict[str, object]:
        return {"instruction": self.instruction, "model": self.model,
                "content": content_records(self.content or (TextPart(self.instruction),))}


def normalize_request(run_input: RunInput, demo: Path | None = None, mode: str | None = None) -> RunInput:
    """CLI and JSON become the same typed content before preprocessing."""
    content = run_input.content or (TextPart(run_input.instruction),)
    if demo is not None:
        content += (VideoPart(Path(demo).expanduser().resolve()),)
    normalized = []
    for part in content:
        if isinstance(part, VideoPart):
            source = part.path.expanduser().resolve()
            part = replace(part, path=source, mode=resolve_demo_mode(source, mode or part.mode))
            if part.mode == "video+action" and part.path.is_file() and part.path.suffix.lower() != ".json":
                raise ValueError("video+action requires a recorded run or demo.json; a video alone has no action data")
        normalized.append(part)
    return replace(run_input, content=tuple(normalized))


def save_request(run_input: RunInput, path: Path, *, overwrite: bool = True) -> Path:
    """Archive a replayable request with absolute source paths, atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            json.dump(run_input.request(), stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
            stream.close()
            if overwrite:
                temporary.replace(path)
            else:
                os.link(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
    return path


def save_task_request(run_input: RunInput, directory: Path, task_name: str) -> Path:
    """Keep reusable task requests directly in request_json, without run dates."""
    payload = run_input.request()
    if not task_name or Path(task_name).name != task_name or task_name in (".", ".."):
        raise ValueError("Task name must be a filename")
    target = directory / f"{task_name}.json"
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:8]
    variant = directory / f"{task_name}-{digest}.json"
    for existing in (target, variant):
        try:
            saved = normalize_request(resolve_run_input(None, existing, None, run_input.model))
        except (OSError, ValueError, TypeError):
            continue
        if saved.request() == payload:
            return existing
    if target.exists():
        target = variant
    try:
        return save_request(run_input, target, overwrite=False)
    except FileExistsError:
        # A concurrent writer may have published the same task after our lookup.
        saved = normalize_request(resolve_run_input(None, target, None, run_input.model))
        if saved.request() == payload:
            return target
        raise FileExistsError(f"A different request already exists: {target}") from None


def resolve_run_input(
    instruction: str | None,
    manifest_path: Path | None,
    model: str | None,
    default_model: str = "gpt-6-astra",
) -> RunInput:
    if manifest_path is not None and not isinstance(manifest_path, Path):
        manifest_path = Path(manifest_path)
    manifest = load_manifest(manifest_path) if manifest_path is not None else None
    resolved_instruction = instruction or (manifest.instruction if manifest else None)
    if not resolved_instruction or not resolved_instruction.strip():
        raise ValueError("请提供任务文字，或使用包含文字块的 --input-json 文件")
    resolved_model = model or (manifest.model if manifest else None) or default_model
    content = manifest.content if manifest is not None else ()
    return RunInput(resolved_instruction.strip(), resolved_model, content, manifest)
