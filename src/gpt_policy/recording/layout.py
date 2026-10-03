"""Stable task/ICL directories for recordings, independent of outcome."""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

from ..input import ImagePart, RunInput, TextPart, VideoPart

ICL_DIRECTORIES = ("none_icl", "video_icl", "video_action_icl")


def _component(value: str) -> str:
    if not value or value in {".", ".."} or any(c in value for c in '/\\\0'):
        raise ValueError("Recording category/name must be a single directory name")
    return value


def task_category(run: RunInput, task_name: str, override: str | None = None) -> str:
    if override is not None:
        return _component(override.strip())
    # The current goal takes precedence over the referenced demonstration's title.
    goal = re.split(r"in the current scene[,.:]?", run.instruction, flags=re.I)[-1].lower()
    if (("plug" in goal and re.search(r"reinsert|re-insert|unplug|insert.{0,25}back", goal))
            or ("插头" in goal and re.search(r"重新插|插回|拔出|拔下", goal))):
        return "拔出并重新插回插头"
    if ("bottle" in goal and ("cap" in goal or "open_bottle" in goal)) or "瓶盖" in goal:
        return "拧瓶盖"
    if "plug" in goal or "插排" in goal or "插头插入" in goal:
        return "插入插排"
    rules = (
        (r"拿起胶棒|pick up (?:the |a )?glue stick", "拿起胶棒"),
        (r"胶棒.{0,5}盖|固体胶.{0,5}盖|(?:remove|pull).{0,30}glue.{0,20}cap", "拔下固体胶盖子"),
        (r"所有笔|all (?:the )?pens", "把所有笔放入笔筒"),
        (r"笔筒.{0,6}右边|(?:pencil|pen) holder.{0,20}right", "把笔筒移到右边"),
        (r"胶带.{0,6}桌子中央|tape.{0,20}(?:table center|center of the table)", "把胶带移到桌子中央"),
        (r"瓶子放到胶带里|bottle.{0,20}(?:into|inside).{0,10}tape", "把瓶子放到胶带里"),
        (r"双臂抬起|raise both arms", "双臂抬起"),
    )
    for pattern, category in rules:
        if re.search(pattern, goal):
            return category
    # Unknown tasks keep their semantic name; generic request filenames must not
    # merge unrelated goals into one task. Configuration can supply a stable label.
    name = re.sub(r"-[0-9a-f]{8}$", "", task_name)
    if name in {"task", "input", "request"}:
        digest = hashlib.sha256(run.instruction.encode()).hexdigest()[:8]
        name = f"未分类任务-{digest}"
    return _component(name)


def icl_directory(run: RunInput) -> str:
    videos = [part for part in run.content if isinstance(part, VideoPart)]
    if videos:
        return "video_action_icl" if any(p.mode == "video+action" for p in videos) else "video_icl"
    # Portable prepared inputs contain images and an explicit mode marker rather
    # than VideoPart. Ignore instruction wording and demo path names.
    if any(isinstance(part, ImagePart) for part in run.content):
        modes = {mode for part in run.content if isinstance(part, TextPart)
                 for mode in re.findall(r"Input mode: (video\+action|video)\.", part.text)}
        if modes:
            return "video_action_icl" if "video+action" in modes else "video_icl"
    return "none_icl"


def recording_directory(root: Path, category: str, icl: str, run_name: str) -> Path:
    """Create all three condition directories, including currently empty ones."""
    parent = root / _component(category)
    _component(run_name)
    if icl not in ICL_DIRECTORIES:
        raise ValueError(f"Unknown ICL directory: {icl}")
    for condition in ICL_DIRECTORIES:
        (parent / condition).mkdir(parents=True, exist_ok=True)
    return parent / icl / run_name
