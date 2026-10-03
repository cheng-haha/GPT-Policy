"""Parse the small, user-facing JSON format used for mixed model input."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeAlias


@dataclass(frozen=True)
class TextPart:
    text: str


@dataclass(frozen=True)
class ImagePart:
    path: Path
    label: str | None = None
    detail: str | None = None


@dataclass(frozen=True)
class VideoPart:
    path: Path
    label: str | None = None
    detail: str | None = None
    mode: str | None = None


ContentPart: TypeAlias = TextPart | ImagePart | VideoPart


def content_records(parts: tuple[ContentPart, ...]) -> list:
    """One public JSON representation for requests and recording metadata."""
    content = []
    for part in parts:
        if isinstance(part, TextPart):
            content.append(part.text)
            continue
        item = {"video" if isinstance(part, VideoPart) else "image": str(part.path)}
        if isinstance(part, VideoPart):
            item["mode"] = part.mode or "auto"
        if part.label is not None:
            item["label"] = part.label
        if part.detail is not None:
            item["detail"] = part.detail
        content.append(item)
    return content


@dataclass(frozen=True)
class InputManifest:
    """A validated request loaded from one JSON file."""

    source: Path
    content: tuple[ContentPart, ...]
    instruction: str
    model: str | None = None

    def record(self) -> dict[str, Any]:
        """Return a JSON-serializable representation for run metadata."""
        return {
            "source": str(self.source),
            "model": self.model,
            "instruction": self.instruction,
            "content": content_records(self.content),
        }


def load_manifest(path: Path) -> InputManifest:
    """Load and validate a user-facing mixed-content JSON manifest.

    Strings in ``content`` are text blocks. Objects with an ``image`` or
    ``video`` key are media blocks; paths are resolved relative to the
    manifest file.
    """
    source = path.expanduser().resolve()
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"输入 JSON 不存在: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"输入 JSON 格式错误: {path}: {exc.msg}") from exc

    if not isinstance(payload, dict):
        raise ValueError("输入 JSON 的根对象必须是 object")
    raw_content = payload.get("content")
    if raw_content is None and isinstance(payload.get("instruction"), str):
        raw_content = [payload["instruction"]]
    if not isinstance(raw_content, list) or not raw_content:
        raise ValueError("输入 JSON 必须包含非空 content 数组")

    content = tuple(
        _parse_part(item, source.parent, index)
        for index, item in enumerate(raw_content)
    )
    text = payload.get("instruction")
    if text is None:
        text = "\n".join(part.text for part in content if isinstance(part, TextPart)).strip()
    if not isinstance(text, str) or not text.strip():
        raise ValueError("输入 JSON 必须包含 instruction，或至少一个非空文字块")

    model = payload.get("model")
    if model is not None and (not isinstance(model, str) or not model.strip()):
        raise ValueError("输入 JSON 的 model 必须是非空字符串")
    return InputManifest(source, content, text.strip(), model.strip() if model else None)


def _parse_part(value: Any, base_dir: Path, index: int) -> ContentPart:
    if isinstance(value, str):
        if not value.strip():
            raise ValueError(f"content[{index}] 文字不能为空")
        return TextPart(value)
    if not isinstance(value, dict):
        raise ValueError(f"content[{index}] 必须是字符串或包含 image/video 的对象")

    media_keys = [key for key in ("image", "video") if key in value]
    if len(media_keys) != 1:
        raise ValueError(f"content[{index}] 必须且只能包含 image 或 video 之一")
    media_key = media_keys[0]
    raw_path = value.get(media_key)
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise ValueError(f"content[{index}] 的 {media_key} 必须是非空路径")
    media_path = (base_dir / Path(raw_path).expanduser()).resolve()
    if not media_path.is_file() and not (media_key == "video" and media_path.is_dir()):
        name = "视频" if media_key == "video" else "图片"
        raise ValueError(f"{name}文件不存在: {media_path}")

    label = value.get("label")
    if label is not None and (not isinstance(label, str) or not label.strip()):
        raise ValueError(f"content[{index}] 的 label 必须是非空字符串")
    detail = value.get("detail")
    if detail is not None and detail not in {"auto", "low", "high"}:
        raise ValueError(f"content[{index}] 的 detail 必须是 auto、low 或 high")
    if media_key == "video":
        mode = value.get("mode", "auto")
        if mode not in ("auto", "video", "video+action"):
            raise ValueError(f"content[{index}].mode must be auto, video or video+action")
        return VideoPart(media_path, label.strip() if label else None, detail, mode)
    if "mode" in value:
        raise ValueError(f"content[{index}].mode only applies to video inputs")
    return ImagePart(media_path, label.strip() if label else None, detail)
