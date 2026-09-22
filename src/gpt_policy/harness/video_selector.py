"""Use a short-lived Codex vision turn to select task-relevant video frames."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from typing import Any, Callable

from ..input.manifest import ImagePart, TextPart, VideoPart
from ..input.video import (
    FrameSelection,
    SelectedFrame,
    VideoExtraction,
    VideoProcessingConfig,
)
from .codex import CodexAppServer
from .config import AgentConfig


ClientFactory = Callable[..., CodexAppServer]


class CodexVideoSelector:
    """Select a small semantic subset without granting Codex any native tools."""

    def __init__(
        self,
        config: AgentConfig,
        processing: VideoProcessingConfig | None = None,
        client_factory: ClientFactory = CodexAppServer,
    ) -> None:
        if config.type != "codex":
            raise ValueError("视频关键帧选择必须使用 Codex 配置")
        self.config = config
        self.processing = processing or VideoProcessingConfig()
        self.client_factory = client_factory

    def cache_identity(self) -> dict[str, Any]:
        """Values which can change the semantic frame selection."""
        return {
            "selector": "codex-video-selector-v3-stages",
            "prompt_sha256": hashlib.sha256(
                _SELECTOR_INSTRUCTIONS.encode("utf-8")
            ).hexdigest(),
            "model": self.config.model or "gpt-6-astra",
            "effort": self.config.effort or "high",
            "max_keyframes": self.processing.max_keyframes,
            "candidate_detail": "auto",
        }

    def select(
        self,
        instruction: str,
        video: VideoPart,
        extraction: VideoExtraction,
    ) -> FrameSelection:
        # One bounded vision request per window. Global indices and chronological
        # order survive stitching; adjacent windows share their boundary frame.
        size = max(2, self.processing.max_candidates)
        if len(extraction.candidates) > size:
            selections, summaries, wires = {}, [], []
            for start in range(0, len(extraction.candidates) - 1, size - 1):
                window = extraction.candidates[start:start + size]
                local = replace(extraction, candidates=tuple(replace(f, index=i) for i, f in enumerate(window)))
                result = self.select(instruction, video, local)
                for choice in result.selected:
                    index = window[choice.index].index
                    selections[index] = replace(choice, index=index)
                summaries.append(f"{window[0].timestamp_s:.3f}–{window[-1].timestamp_s:.3f}s: {result.summary}")
                wires.append(result.wire)
            return FrameSelection(tuple(selections[i] for i in sorted(selections)), "\n".join(summaries), {"windows": wires})
        model = self.config.model or "gpt-6-astra"
        client = self.client_factory(
            model,
            self.config.executable,
            self.config.effort or "high",
            False,
            85,
        )
        try:
            client.start_thread(_SELECTOR_INSTRUCTIONS)
            decision = client.decide(
                _selection_request(instruction, video, extraction),
                _selection_schema(self.processing.max_keyframes),
                content=_candidate_content(extraction),
            )
        finally:
            client.close()
        return _validate_selection(
            decision, len(extraction.candidates), self.processing.max_keyframes
        )

    def review(self, instruction, video, frames, selection, limit=24):
        """Curate window selections into one short, chronological demonstration."""
        client = self.client_factory(self.config.model or "gpt-6-astra", self.config.executable,
                                     self.config.effort or "high", False, 85)
        parts = []
        for i, (frame, choice) in enumerate(zip(frames, selection.selected)):
            parts.append(TextPart(json.dumps({"index": i, "t_s": frame.timestamp_s,
                "stage": choice.stage, "subtask": choice.subtask or choice.stage,
                "left": choice.left, "right": choice.right,
                "reason": choice.reason, "result": choice.result}, ensure_ascii=False)))
            parts.extend(_frame_images(frame))
        try:
            client.start_thread(_REVIEW_INSTRUCTIONS)
            decision = client.decide(
                f"Task: {instruction}\nVideo: {video.label or video.path.name}\n"
                f"Select at most {limit} frames from 0..{len(frames)-1}. Include indices 0 and {len(frames)-1}.",
                _selection_schema(limit), content=tuple(parts))
        finally:
            client.close()
        reviewed = _validate_selection(decision, len(frames), limit)
        if not {0, len(frames) - 1} <= {c.index for c in reviewed.selected}:
            raise ValueError("Demonstration review must retain both timeline boundaries")
        return FrameSelection(tuple(replace(c, index=frames[c.index].index) for c in reviewed.selected),
                              reviewed.summary, {"windows": selection.wire, "review": reviewed.wire})

    def review_identity(self):
        return hashlib.sha256(_REVIEW_INSTRUCTIONS.encode()).hexdigest()


_REVIEW_INSTRUCTIONS = """Review the full historical demonstration before robot execution and return a concise, complete keyframe set.
Inputs are preliminary images and annotations selected per window, not yet globally deduplicated. Verify the images; do not blindly trust local annotations.
Each time may include top, left_wrist and right_wrist views. Use wrist close-ups to identify contact side, grasp direction, object-to-gripper orientation, gripper-to-table direction and recontact after release. Record these relationships in phase annotations; do not mislabel empty-gripper pressing as regrasping.
Usually retain 12-16 frames, never exceeding the supplied limit. Merge redundant holds and window boundaries while preserving the initial state, preparation, contact, grasp verification, arm-role changes, release and final outcome. Select important before/after changes separately; the host will not add neighboring images automatically. Preserve the order of repeated twists and regrasps rather than describing them as no motion.
summary briefly covers the full operation and uncertain phases; stage and subtask are the same short English subtask label; left/right describe roles; reason gives visual evidence; result states only visible outcomes. Explicitly report when success cannot be verified.
Retain the full sequence's first and last frames and return chronological order. Historical actions are references, not pending commands. You have no robot or file tools."""


_SELECTOR_INSTRUCTIONS = """Select demonstration keyframes from the supplied chronological video window and candidate images.
Candidates may contain multiple camera views. Combine wrist close-ups with the top overview; do not rely on an occluded top view alone. Record contact side, object-to-gripper orientation, gripper-to-table direction and distinctions such as empty-gripper recontact after release.
For the user's final task, select the smallest set that conveys the initial state, key actions, state changes and final outcome. Preserve before/after evidence for grasping, release and handoffs, and the order of repeated twists. Similar start/end poses do not imply no motion.
Use short English stage names and repeat the same value in subtask (for example approach, grasp, push, release, verify). left/right describe robot or human hand roles; stabilizing an object is an important action. result states only image-supported outcomes, including uncertainty or occlusion; gripper closure alone does not prove a grasp. reason explains the frame's evidence. Be concise and specific; do not invent unfamiliar objects or actions.
Cover the window's beginning and end with minimal redundant imagery. Select only supplied candidate indices and return chronological order.
You have no shell, file or robot-control tools. Return only JSON matching the output schema."""


def _candidate_content(extraction: VideoExtraction) -> tuple[TextPart | ImagePart, ...]:
    metadata = extraction.metadata
    parts: list[TextPart | ImagePart] = [
        TextPart(
            "Video metadata: "
            + json.dumps(metadata.record(), ensure_ascii=False, allow_nan=False)
        )
    ]
    for candidate in extraction.candidates:
        parts.append(
            TextPart(
                f"Candidate frame index={candidate.index}, "
                f"timestamp_s={candidate.timestamp_s:.3f}"
            )
        )
        parts.extend(_frame_images(candidate))
    return tuple(parts)


def _frame_images(frame):
    views = {"top" if frame.views else "video": frame, **frame.views}
    return tuple(ImagePart(f.path, f"{camera}, capture PTS={f.timestamp_s:.3f}s",
                           "high" if frame.views else "auto") for camera, f in views.items())


def _selection_request(
    instruction: str, video: VideoPart, extraction: VideoExtraction
) -> str:
    label = video.label or video.path.name
    return (
        f"User's final task: {instruction}\n"
        f"Current video: {label}\n"
        f"Select keyframes from indices 0..{len(extraction.candidates) - 1}."
    )


def _selection_schema(max_keyframes: int) -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["name", "arguments"],
        "properties": {
            "name": {"type": "string", "const": "select_video_frames"},
            "arguments": {
                "type": "object",
                "additionalProperties": False,
                "required": ["selected", "summary"],
                "properties": {
                    "selected": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": max_keyframes,
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": ["index", "reason", "stage", "left", "right", "result"],
                            "properties": {
                                "index": {"type": "integer", "minimum": 0},
                                "reason": {"type": "string", "minLength": 1},
                                **{key: {"type": "string"} for key in ("stage", "subtask", "left", "right", "result")},
                            },
                        },
                    },
                    "summary": {"type": "string", "minLength": 1},
                },
            },
        },
    }


def _validate_selection(
    decision: dict[str, Any], candidate_count: int, max_keyframes: int
) -> FrameSelection:
    if decision.get("name") != "select_video_frames":
        raise RuntimeError("Codex 返回了错误的视频处理动作")
    arguments = decision.get("arguments")
    if not isinstance(arguments, dict):
        raise RuntimeError("Codex 视频处理 arguments 必须是对象")
    raw_selected = arguments.get("selected")
    summary = arguments.get("summary")
    if not isinstance(raw_selected, list) or not 1 <= len(raw_selected) <= max_keyframes:
        raise RuntimeError(f"Codex 必须选择 1 到 {max_keyframes} 个关键帧")
    if not isinstance(summary, str) or not summary.strip():
        raise RuntimeError("Codex 视频摘要不能为空")

    selected: list[SelectedFrame] = []
    seen: set[int] = set()
    for item in raw_selected:
        if not isinstance(item, dict):
            raise RuntimeError("Codex 关键帧选择项必须是对象")
        index, reason = item.get("index"), item.get("reason")
        if isinstance(index, bool) or not isinstance(index, int):
            raise RuntimeError("Codex 关键帧 index 必须是整数")
        if not 0 <= index < candidate_count:
            raise RuntimeError(f"Codex 关键帧 index 越界: {index}")
        if index in seen:
            raise RuntimeError(f"Codex 重复选择关键帧: {index}")
        if not isinstance(reason, str) or not reason.strip():
            raise RuntimeError(f"Codex 关键帧 {index} 的 reason 不能为空")
        seen.add(index)
        details = {}
        for key in ("stage", "subtask", "left", "right", "result"):
            value = item.get(key, "")
            if not isinstance(value, str):
                raise RuntimeError(f"Codex keyframe {key} must be text")
            details[key] = value.strip()
        # ``stage`` was the original public field.  Keep old Codex responses
        # valid while exposing an explicit subtask label downstream.
        details["subtask"] = details["subtask"] or details["stage"]
        selected.append(SelectedFrame(index, reason.strip(), **details))

    selected.sort(key=lambda item: item.index)
    wire = decision.get("_wire", decision)
    return FrameSelection(
        tuple(selected),
        summary.strip(),
        wire if isinstance(wire, dict) else decision,
    )
