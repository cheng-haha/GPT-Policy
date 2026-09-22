"""Extract timestamped video evidence with a bounded candidate density per window."""

from __future__ import annotations

import json
import math
from bisect import bisect_left
import shutil
import subprocess
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Sequence


@dataclass(frozen=True)
class VideoProcessingConfig:
    ffmpeg_bin: str = "ffmpeg"
    ffprobe_bin: str = "ffprobe"
    target_fps: float = 2.0
    max_candidates: int = 24
    max_keyframes: int = 8
    candidate_width: int = 768
    keyframe_width: int = 1280
    jpeg_quality: int = 3
    max_duration_s: float = 1800.0
    max_file_bytes: int = 2 * 1024 * 1024 * 1024
    command_timeout_s: float = 120.0
    window_s: float = 30.0

    def __post_init__(self) -> None:
        if not math.isfinite(self.target_fps) or self.target_fps <= 0:
            raise ValueError("video target_fps 必须是有限正数")
        for name in ("max_candidates", "max_keyframes", "candidate_width", "keyframe_width"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"video {name} 必须是正整数")
        if self.max_keyframes > self.max_candidates:
            raise ValueError("video max_keyframes 不能大于 max_candidates")
        if not 1 <= self.jpeg_quality <= 31:
            raise ValueError("video jpeg_quality 必须在 1 到 31 之间")
        if not math.isfinite(self.max_duration_s) or self.max_duration_s <= 0:
            raise ValueError("video max_duration_s 必须是有限正数")
        if self.max_file_bytes <= 0:
            raise ValueError("video max_file_bytes 必须是正整数")
        if not math.isfinite(self.command_timeout_s) or self.command_timeout_s <= 0:
            raise ValueError("video command_timeout_s 必须是有限正数")
        if not math.isfinite(self.window_s) or self.window_s <= 0:
            raise ValueError("video window_s must be finite and positive")


@dataclass(frozen=True)
class VideoMetadata:
    duration_s: float
    width: int
    height: int
    fps: float | None
    codec: str | None
    file_size: int

    def record(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CandidateFrame:
    index: int
    timestamp_s: float
    path: Path
    frame_index: int | None = None
    views: dict[str, CandidateFrame] = field(default_factory=dict)
    capture_delta_s: float | None = None

    def record(self, relative_to: Path | None = None) -> dict[str, Any]:
        return {
            "index": self.index,
            "timestamp_s": self.timestamp_s,
            "path": str(self.path.relative_to(relative_to) if relative_to else self.path),
            "frame_index": self.frame_index,
            "views": {name: frame.record(relative_to) for name, frame in self.views.items()},
            "capture_delta_s": self.capture_delta_s,
        }


@dataclass(frozen=True)
class SelectedFrame:
    index: int
    reason: str
    stage: str = ""
    left: str = ""
    right: str = ""
    result: str = ""
    # ``stage`` is the historical wire name.  Keep it for cache and bundle
    # compatibility while exposing the same annotation as an explicit
    # subtask to callers that build multi-view context.
    subtask: str = ""


@dataclass(frozen=True)
class FrameSelection:
    selected: tuple[SelectedFrame, ...]
    summary: str
    wire: dict[str, Any]


@dataclass(frozen=True)
class VideoExtraction:
    source: Path
    metadata: VideoMetadata
    candidates: tuple[CandidateFrame, ...]


Runner = Callable[..., subprocess.CompletedProcess[str]]


class FfmpegVideoExtractor:
    """Use FFprobe/FFmpeg through argument arrays, never through a shell."""

    def __init__(
        self,
        config: VideoProcessingConfig | None = None,
        runner: Runner = subprocess.run,
        event_times: Sequence[float] = (),
        end_time_s: float | None = None,
    ) -> None:
        self.config = config or VideoProcessingConfig()
        self._run_process = runner
        self.event_times = tuple(event_times)
        self.end_time_s = end_time_s
        self._timelines = {}
        # camera -> (video path, presentation times, shared capture-clock times).
        # Configured by recording importers; standalone videos keep one view.
        self.views = {}

    def view_identity(self):
        from .video_cache import _sha256_file
        return {name: {"sha256": _sha256_file(path), "pts": list(pts), "capture_times": list(clock)}
                for name, (path, pts, clock) in self.views.items()}

    def _with_views(self, frames, destination, width):
        """Pair nearest captures, never equal frame numbers across cameras."""
        if len(self.views) < 2:
            return frames
        from .recorded_demo import nearest_index
        reference = self.views["top"][2]
        grouped = [dict() for _ in frames]
        for name, (path, pts, clock) in self.views.items():
            if name == "top":
                continue
            indices = [nearest_index(clock, reference[f.frame_index]) for f in frames]
            if any(abs(clock[n] - reference[f.frame_index]) > .1 for n, f in zip(indices, frames)):
                raise ValueError(f"Demonstration camera {name} has no capture within 0.1 s of a selected top frame")
            decoded = self.extract_at(path, [pts[n] for n in indices], destination / name, width, pts)
            by_index = {f.frame_index: f for f in decoded}
            for group, n, reference_frame in zip(grouped, indices, frames):
                group[name] = replace(by_index[n], capture_delta_s=clock[n] - reference[reference_frame.frame_index])
        return tuple(replace(f, views=v) for f, v in zip(frames, grouped))

    def frame_times(self, source: Path) -> tuple[float, ...]:
        """Presentation timestamps in display order, never inferred from FPS."""
        stat = source.stat()
        key = (str(source.resolve()), stat.st_size, stat.st_mtime_ns)
        if key in self._timelines:
            return self._timelines[key]
        result = self._run([
            self._resolve(self.config.ffprobe_bin), "-v", "error", "-select_streams", "v:0",
            "-show_entries", "frame=best_effort_timestamp_time", "-of", "json", str(source),
        ])
        values = tuple(float(f["best_effort_timestamp_time"])
                       for f in json.loads(result.stdout).get("frames", []))
        if not values or any(not math.isfinite(t) for t in values):
            raise ValueError(f"Video has no valid frame timestamps: {source}")
        if any(b <= a for a, b in zip(values, values[1:])):
            raise ValueError(f"Video timestamps must increase: {source}")
        self._timelines[key] = values
        return values

    def extract_at(self, source: Path, times: Sequence[float], destination: Path,
                   width: int, timeline: Sequence[float] | None = None) -> tuple[CandidateFrame, ...]:
        """Decode requested presentation frames once; return their actual PTS."""
        pts = self.frame_times(source) if timeline is None else timeline
        indices = sorted({min(bisect_left(pts, t), len(pts) - 1) for t in times})
        destination.mkdir(parents=True, exist_ok=True)
        if not indices:
            return ()
        expression = "+".join(f"eq(n\\,{i})" for i in indices)
        self._run([
            self._resolve(self.config.ffmpeg_bin), "-nostdin", "-v", "error", "-y",
            "-i", str(source), "-map", "0:v:0", "-vf",
            f"select={expression},scale=min(iw\\,{width}):min(ih\\,{width}):force_original_aspect_ratio=decrease",
            "-vsync", "0", "-q:v", str(self.config.jpeg_quality),
            "-an", "-sn", "-dn", str(destination / "%06d.jpg"),
        ])
        frames = tuple(CandidateFrame(i, float(pts[n]), destination / f"{i+1:06d}.jpg", n)
                       for i, n in enumerate(indices))
        if any(not f.path.is_file() or not f.path.stat().st_size for f in frames):
            raise RuntimeError("FFmpeg did not export every selected presentation frame")
        return frames

    @staticmethod
    def _resolve(name: str) -> str:
        executable = shutil.which(name)
        if executable is None:
            raise RuntimeError(f"找不到视频处理程序: {name}")
        return executable

    def extract_candidates(self, source: Path, destination: Path) -> VideoExtraction:
        metadata = self.probe(source)
        pts = self.frame_times(source)
        end = min(pts[-1], self.end_time_s) if self.end_time_s is not None else pts[-1]
        times = [pts[0] + t for t in self._timestamps(metadata) if pts[0] + t <= end]
        times.append(end)
        for event in self.event_times:
            times.extend(max(pts[0], min(end, event + delta))
                         for delta in (-.3, 0, .3))
        candidates = self.extract_at(source, times, destination / "candidates", self.config.candidate_width, pts)
        return VideoExtraction(source, metadata, self._with_views(candidates, destination / "candidates", self.config.candidate_width))

    def probe(self, source: Path) -> VideoMetadata:
        size = source.stat().st_size
        if size > self.config.max_file_bytes:
            raise ValueError(
                f"视频文件过大: {size} bytes，最大允许 {self.config.max_file_bytes} bytes"
            )
        command = [
            self._resolve(self.config.ffprobe_bin),
            "-v", "error",
            "-select_streams", "v:0",
            "-show_entries",
            "stream=width,height,avg_frame_rate,r_frame_rate,codec_name,duration:format=duration",
            "-of", "json",
            str(source),
        ]
        completed = self._run(command)
        try:
            payload = json.loads(completed.stdout)
            stream = payload["streams"][0]
            duration = float(stream.get("duration") or payload["format"]["duration"])
            width, height = int(stream["width"]), int(stream["height"])
        except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError(f"无法读取视频流信息: {source}") from exc
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError(f"视频时长无效: {source}")
        if duration > self.config.max_duration_s:
            raise ValueError(
                f"视频时长 {duration:.3f}s 超过限制 {self.config.max_duration_s:.3f}s"
            )
        if width <= 0 or height <= 0:
            raise ValueError(f"视频分辨率无效: {source}")
        fps = _parse_rate(stream.get("avg_frame_rate")) or _parse_rate(
            stream.get("r_frame_rate")
        )
        codec = stream.get("codec_name")
        return VideoMetadata(
            duration, width, height, fps,
            str(codec) if codec is not None else None,
            size,
        )

    def export_keyframes(
        self,
        extraction: VideoExtraction,
        selected: Sequence[CandidateFrame],
        destination: Path,
    ) -> tuple[CandidateFrame, ...]:
        frames = self.extract_at(extraction.source, [f.timestamp_s for f in selected],
                                 destination / "keyframes", self.config.keyframe_width)
        frames = tuple(replace(frame, index=choice.index) for choice, frame in zip(selected, frames))
        return self._with_views(frames, destination / "keyframes", self.config.keyframe_width)

    def _timestamps(self, metadata: VideoMetadata) -> tuple[float, ...]:
        count = min(
            self.config.max_candidates * max(1, math.ceil(metadata.duration_s / self.config.window_s)),
            max(1, math.ceil(metadata.duration_s * self.config.target_fps) + 1),
        )
        if count == 1:
            return (0.0,)
        frame_duration = 1.0 / metadata.fps if metadata.fps else 0.04
        # Container duration usually points just past the last decodable frame.
        last = max(0.0, metadata.duration_s - frame_duration)
        return tuple(last * index / (count - 1) for index in range(count))

    def _run(self, command: list[str]) -> subprocess.CompletedProcess[str]:
        try:
            completed = self._run_process(
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=self.config.command_timeout_s,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(f"视频处理超时: {command[0]}") from exc
        if completed.returncode != 0:
            message = (completed.stderr or completed.stdout or "unknown error").strip()
            raise RuntimeError(f"视频处理失败: {message[:1000]}")
        return completed


def _parse_rate(value: Any) -> float | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        numerator, denominator = value.split("/", 1)
        result = float(numerator) / float(denominator)
    except (ValueError, ZeroDivisionError):
        return None
    return result if math.isfinite(result) and result > 0 else None
