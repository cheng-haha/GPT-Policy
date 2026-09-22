"""Expand raw video manifest parts into ordinary first-turn image parts."""

from __future__ import annotations

import json
import shutil
from copy import copy
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from .manifest import ImagePart, TextPart, VideoPart
from .demonstration import prepare_demonstration
from .request import RunInput
from .video import FfmpegVideoExtractor, FrameSelection, VideoExtraction
from .video_cache import CacheableFrameSelector, VideoProcessingCache


@dataclass(frozen=True)
class VideoPreparationResult:
    source: Path
    label: str
    output_dir: Path
    extraction: VideoExtraction
    selection: FrameSelection
    keyframes: tuple[ImagePart, ...]
    cache_key: str
    cache_hit: bool
    cache_dir: Path

    def record(self, relative_to: Path | None = None) -> dict[str, Any]:
        def local_path(path):
            return str(path.relative_to(relative_to)) if relative_to is not None else str(path)

        reasons = {item.index: item.reason for item in self.selection.selected}
        selected = []
        for image, choice in zip(self.keyframes, self.selection.selected):
            selected.append(
                {
                    "candidate_index": choice.index,
                    "timestamp_s": self.extraction.candidates[choice.index].timestamp_s,
                    "frame_index": self.extraction.candidates[choice.index].frame_index,
                    "reason": reasons[choice.index],
                    "path": local_path(image.path),
                    "subtask": choice.subtask or choice.stage,
                }
            )
        return {
            "source": str(self.source),
            "label": self.label,
            "mode": "video",
            "output_dir": local_path(self.output_dir),
            "cache_key": self.cache_key,
            "cache_hit": self.cache_hit,
            "cache_dir": str(self.cache_dir),
            "metadata": self.extraction.metadata.record(),
            "candidates": [item.record() for item in self.extraction.candidates],
            "selection": {
                "summary": self.selection.summary,
                "selected": selected,
                "wire": self.selection.wire,
            },
        }


@dataclass(frozen=True)
class DemonstrationPreparationResult:
    output_dir: Path
    metadata: dict

    def record(self, relative_to: Path | None = None) -> dict:
        directory = self.output_dir.relative_to(relative_to) if relative_to else self.output_dir
        return {**self.metadata, "output_dir": str(directory), "input": str(directory / "input.json")}


def has_video_input(run_input: RunInput) -> bool:
    return any(isinstance(part, VideoPart) for part in run_input.content)


def prepare_input_videos(
    run_input: RunInput,
    destination: Path,
    selector: CacheableFrameSelector,
    extractor: FfmpegVideoExtractor | None = None,
    cache: VideoProcessingCache | None = None,
) -> tuple[RunInput, tuple[VideoPreparationResult | DemonstrationPreparationResult, ...]]:
    """Replace each video in place, preserving surrounding content order."""

    if not has_video_input(run_input):
        return run_input, ()
    active_extractor = extractor or FfmpegVideoExtractor()
    active_cache = cache or VideoProcessingCache(
        Path(__file__).resolve().parents[3] / "var" / "cache" / "video-input"
    )
    destination.mkdir(parents=True, exist_ok=True)
    prepared = []
    results = []
    video_index = 0
    for part in run_input.content:
        if not isinstance(part, VideoPart):
            prepared.append(part)
            continue

        output_dir = destination / f"video-{video_index:03d}"
        video_index += 1
        # Normalized CLI/JSON requests explicitly select a demonstration mode.
        # Keep the low-level raw VideoPart helper available to existing callers.
        if part.mode is not None or part.path.is_dir() or part.path.suffix.lower() == ".json":
            demo, metadata = prepare_demonstration(
                replace(run_input, content=()), part.path, part.mode or "video", output_dir,
                selector, copy(active_extractor), active_cache, label=part.label,
            )
            if part.label:
                prepared.append(TextPart(part.label))
            prepared.extend(replace(p, detail=part.detail) if isinstance(p, ImagePart) and part.detail else p
                            for p in demo.content)
            results.append(DemonstrationPreparationResult(output_dir, metadata))
            continue
        cached = active_cache.resolve(
            run_input.instruction, part, active_extractor, selector
        )
        extraction, selection = cached.extraction, cached.selection
        keyframe_dir = output_dir / "keyframes"
        keyframe_dir.mkdir(parents=True, exist_ok=True)
        exported = []
        for order, frame in enumerate(cached.keyframes):
            target = keyframe_dir / (
                f"keyframe-{order:03d}-{frame.timestamp_s:010.3f}s.jpg"
            )
            shutil.copy2(frame.path, target)
            exported.append(replace(frame, path=target))
        label = part.label or part.path.name
        prepared.append(TextPart(f"Video '{label}' keyframe summary: {selection.summary}"))
        keyframes: list[ImagePart] = []
        for frame, choice in zip(exported, selection.selected):
            image = ImagePart(
                frame.path,
                (
                    f"Video '{label}' keyframe, timestamp_s={frame.timestamp_s:.3f}; "
                    f"selection reason: {choice.reason}; stage={choice.stage}; "
                    f"subtask={choice.subtask or choice.stage}; "
                    f"left={choice.left}; right={choice.right}; result={choice.result}"
                ),
                part.detail,
            )
            keyframes.append(image)
            prepared.append(image)
        result = VideoPreparationResult(
            part.path,
            label,
            output_dir,
            extraction,
            selection,
            tuple(keyframes),
            cached.cache_key,
            cached.cache_hit,
            cached.cache_dir,
        )
        _save_json(output_dir / "metadata.json", result.record(relative_to=output_dir))
        results.append(result)

    return replace(run_input, content=tuple(prepared)), tuple(results)


def _save_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
