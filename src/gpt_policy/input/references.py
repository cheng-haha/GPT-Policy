"""Resolve local media mentioned directly in the task text."""

import mimetypes
from pathlib import Path
import re

from .recorded_demo import read_json
from .mcap_demo import is_mcap_episode


_TOKENS = re.compile(
    r'''\w+://\S+|"([^"]+)"|'([^']+)'|`([^`]+)`|“([^”]+)”|‘([^’]+)’|'''
    r'''[^\s,，。!！?？;；:：、<>()（）\[\]{}"'`“”‘’]+'''
)
_VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}
_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tif", ".tiff"}
_MODE = re.compile(r"(?<![a-z0-9_./+-])video(?:\s*\+\s*action)?(?![a-z0-9_./+-])", re.I)


def instruction_text_without_media(instruction: str) -> str:
    """Remove media references without opening files or resolving their content."""
    def replace(match: re.Match[str]) -> str:
        raw = next((group for group in match.groups() if group is not None), match.group())
        if "/" in raw or Path(raw).suffix.lower() in _VIDEO_SUFFIXES | _IMAGE_SUFFIXES | {".json"}:
            return " "
        return match.group()

    return _TOKENS.sub(replace, instruction)


def instruction_mode(instruction: str | None) -> str | None:
    """Read explicit mode words without interpreting filenames or URLs."""
    modes = set()
    text = instruction_text_without_media(instruction or "")
    for match in _MODE.finditer(text):
        mode = "video+action" if "+" in match.group() else "video"
        # A generic phrase such as "video demonstration" is not a video-only
        # override: automatic selection must still retain its recorded actions.
        explicit = (mode == "video+action" or text.strip().lower() == "video"
                    or re.search(r"(?:--demo-mode|\bmode|模式)\s*[:=：]?\s*$", text[:match.start()], re.I)
                    or re.match(r"\s*(?:的?\s*模式|mode\b|only\b)", text[match.end():], re.I))
        if explicit:
            modes.add(mode)
    if len(modes) > 1:
        raise ValueError("Conflicting demonstration modes; specify only video or video+action")
    return next(iter(modes), None)


def instruction_source(instruction: str | None) -> tuple[Path | None, Path | None]:
    """Return (demo, input manifest), ignoring inline reference images."""
    demo, manifest, _ = instruction_media(instruction)
    return demo, manifest


def image_path(path: Path) -> Path:
    """Validate an explicit CLI image before starting a model or hardware."""
    path = path.expanduser().resolve()
    mime, _ = mimetypes.guess_type(path.name)
    if mime is None or not mime.startswith("image/"):
        raise ValueError(f"无法识别图片类型: {path}")
    if not path.is_file():
        raise ValueError(f"图片文件不存在: {path}")
    return path


def instruction_media(instruction: str | None) -> tuple[Path | None, Path | None, tuple[Path, ...]]:
    """Return (demo, input manifest, images) from paths in task text.

    Relative paths use the command's working directory. Space/punctuation in a
    filename requires quotes within the instruction. Explicit CLI input flags
    bypass this convenience resolver.
    """
    sources = set()
    images: dict[Path, None] = {}
    for match in _TOKENS.finditer(instruction or ""):
        raw = next((group for group in match.groups() if group is not None), match.group())
        if "://" in raw or raw.startswith("//"):
            continue
        quoted = any(group is not None for group in match.groups())
        raw = raw if quoted else raw.rstrip(".")
        path = Path(raw).expanduser()
        is_image = path.suffix.lower() in _IMAGE_SUFFIXES
        supported_file = path.suffix.lower() in _VIDEO_SUFFIXES | {".json"} or is_image
        explicit_path = raw.startswith(("/", "~/", "./", "../"))
        relative_dir = "/" in raw and path.is_dir()
        if not (supported_file or explicit_path or relative_dir):
            continue
        if not path.exists():
            kind = "Image" if is_image else "Demonstration"
            raise ValueError(f"{kind} path does not exist: {path}")
        if is_image:
            images[image_path(path)] = None
            continue
        if path.is_dir():
            # Both reviewed bundles and portable --prepare-only outputs work.
            if (path / "demo.json").is_file():
                path /= "demo.json"
            elif not (is_mcap_episode(path) or all((path / name).is_file() for name in
                         ("top.mp4", "config.json", "status.json", "events.jsonl", "video-frames.jsonl"))):
                if (path / "input.json").is_file():
                    path /= "input.json"
                else:
                    raise ValueError(f"Not a demonstration or recording directory: {path}")
        elif not supported_file or not path.is_file():
            raise ValueError(f"Unsupported demonstration file: {path}")
        sources.add(path.resolve())
    if len(sources) > 1:
        raise ValueError("Use one demonstration per task, or combine media with --input-json")
    if not sources:
        return None, None, tuple(images)
    source = sources.pop()
    if source.suffix.lower() == ".json":
        data = read_json(source)
        if isinstance(data, dict):
            if "keyframes" in data:
                return source, None, tuple(images)
            if "content" in data or "instruction" in data:
                return None, source, tuple(images)
        raise ValueError(f"Expected a demonstration or mixed input JSON: {source}")
    return source, None, tuple(images)
