"""Small, task-matched RoboDojo demonstrations for policy-server ICL.

The source dataset is a LeRobot-style collection containing long videos and
Parquet state/action tables.  This module reads only one or two episodes for
the active task and extracts a handful of synchronized camera frames, keeping
the live Codex context bounded.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
from typing import Any

from ..harness.video_selector import CodexVideoSelector
from ..input.manifest import VideoPart
from ..input.video import FfmpegVideoExtractor, VideoProcessingConfig


class RoboDojoICL:
    def __init__(self, root: str | os.PathLike[str], cache_dir: str | os.PathLike[str], *, keyframes: int = 6,
                 selector: CodexVideoSelector | None = None) -> None:
        self.root = Path(root).expanduser()
        self.cache_dir = Path(cache_dir).expanduser()
        self.keyframes = max(3, min(int(keyframes), 8))
        self._task_names: dict[str, str] | None = None
        self.selector = selector

    def available(self) -> bool:
        return self.root.is_dir()

    def task_for_instruction(self, instruction: str) -> str | None:
        if not self.available():
            return None
        if self._task_names is None:
            self._task_names = {}
            try:
                import pyarrow.parquet as parquet
                for task_dir in self.root.iterdir():
                    table_path = task_dir / "meta/tasks.parquet"
                    if not table_path.is_file():
                        continue
                    table = parquet.read_table(table_path, columns=["__index_level_0__"])
                    for value in table["__index_level_0__"].to_pylist():
                        self._task_names[str(value)] = task_dir.name
            except Exception:
                return None
        exact = self._task_names.get(instruction)
        if exact:
            return exact
        lowered = instruction.lower()
        return next((task for text, task in self._task_names.items() if text.lower() in lowered or lowered in text.lower()), None)

    def prepare(self, instruction: str, *, include_actions: bool = True) -> tuple[str, dict[str, Path]] | None:
        task = self.task_for_instruction(instruction)
        if task is None:
            return None
        task_root = self.root / task
        try:
            import pyarrow.parquet as parquet
            episodes = parquet.read_table(task_root / "meta/episodes/chunk-000/file-000.parquet").to_pylist()
            data = parquet.read_table(task_root / "data/chunk-000/file-000.parquet").to_pylist()
        except Exception:
            return None
        if not episodes or not data:
            return None
        episode = episodes[0]
        start, end = int(episode["dataset_from_index"]), int(episode["dataset_to_index"])
        rows = [row for row in data if start <= int(row["index"]) < end]
        if not rows:
            return None
        positions = [int(round(i * (len(rows) - 1) / (self.keyframes - 1))) for i in range(self.keyframes)]
        times = [float(rows[i]["timestamp"]) for i in positions]
        subtasks = [""] * len(times)
        selection_summary = ""
        if self.selector is not None:
            try:
                selected, _, selection_summary = self._codex_keyframes(task_root, task, rows, instruction)
                if selected:
                    positions, times, subtasks = zip(*selected)
                    positions, times, subtasks = list(positions), list(times), list(subtasks)
            except Exception:
                # A demonstration must never prevent the live policy from
                # starting. The deterministic fallback still supplies all
                # synchronized views and the existing action trajectory.
                pass
        cache = self.cache_dir / task / "episode-000"
        image_paths: dict[str, Path] = {}
        for camera, key in (("cam_high", "cam_high"), ("cam_left_wrist", "cam_left_wrist"), ("cam_right_wrist", "cam_right_wrist")):
            video = task_root / "videos" / f"observation.images.{camera}" / "chunk-000" / "file-000.mp4"
            if not video.is_file():
                continue
            for frame_index, timestamp in enumerate(times):
                output = cache / f"{key}-{frame_index:02d}.jpg"
                if not output.is_file():
                    output.parent.mkdir(parents=True, exist_ok=True)
                    try:
                        subprocess.run([
                            "ffmpeg", "-y", "-loglevel", "error", "-ss", f"{timestamp:.6f}",
                            "-i", str(video), "-frames:v", "1", "-vf", "scale=768:-1", str(output),
                        ], check=True, timeout=60)
                    except (OSError, subprocess.SubprocessError):
                        output.unlink(missing_ok=True)
                        return None
                image_paths[f"demo_{frame_index:02d}_{key}"] = output
        summary = self._summary(task, instruction, rows, positions, times, subtasks,
                                selection_summary, include_actions=include_actions)
        return summary, image_paths

    def _codex_keyframes(self, task_root: Path, task: str, rows: list[dict[str, Any]], instruction: str):
        """Select multi-view keyframes and subtask labels through Codex.

        The high camera is the timeline reference. Wrist videos are paired by
        the same dataset timestamps, so every selected frame remains a true
        multi-view observation while the action rows stay aligned to it.
        """
        videos = {
            name: task_root / "videos" / f"observation.images.{name}" / "chunk-000" / "file-000.mp4"
            for name in ("cam_high", "cam_left_wrist", "cam_right_wrist")
        }
        primary = videos["cam_high"]
        if not primary.is_file():
            return (), (), ""
        config = VideoProcessingConfig(max_candidates=24, max_keyframes=self.keyframes,
                                       candidate_width=768, keyframe_width=768)
        extractor = FfmpegVideoExtractor(config)
        timeline = extractor.frame_times(primary)
        # Dataset timestamps are the shared capture clock for all cameras.
        extractor.views = {"top": (primary, timeline, timeline)}
        for name, path in videos.items():
            if name == "cam_high" or not path.is_file():
                continue
            pts = extractor.frame_times(path)
            extractor.views[name] = (path, pts, pts)
        extraction = extractor.extract_candidates(primary, self.cache_dir / task / "codex-candidates")
        selection = self.selector.select(instruction or "Follow the demonstrated task.", VideoPart(primary, task), extraction)
        selected = []
        for choice in selection.selected:
            frame = extraction.candidates[choice.index]
            row_index = min(range(len(rows)), key=lambda i: abs(float(rows[i]["timestamp"]) - frame.timestamp_s))
            selected.append((row_index, float(rows[row_index]["timestamp"]), choice.subtask or choice.stage))
        return tuple(selected), tuple(t for _, t, _ in selected), selection.summary

    @staticmethod
    def _summary(task: str, instruction: str, rows: list[dict[str, Any]], positions: list[int], times: list[float],
                 subtasks: list[str] | tuple[str, ...] = (), selection_summary: str = "",
                 *, include_actions: bool = True) -> str:
        samples = []
        for n, (index, timestamp) in enumerate(zip(positions, times)):
            row = rows[index]
            action = row.get("actions.eef") or row.get("action")
            state = row.get("observation.states.eef") or row.get("observation.state")
            sample = {"t_s": round(timestamp, 3), "subtask": subtasks[n] if n < len(subtasks) else ""}
            if include_actions:
                sample.update({"state_eef": _round_list(state), "action_eef": _round_list(action)})
            samples.append(sample)
        return json.dumps({
            "type": "historical_robodojo_demonstration",
            "task": task,
            "instruction": instruction,
            "source_robot": "arx_x5",
            "coordinate_frame": "Historical RoboDojo source EE-link poses, not live grasp-center TCP poses. Adapt using current world/base calibration and tcp_from_source_link before comparing positions.",
            "eef_sample_layout": "left_xyz + left_rot6d[6] + left_gripper + right_xyz + right_rot6d[6] + right_gripper",
            "joint_action_layout": "left six joints + left_gripper + right six joints + right_gripper; radians and normalized openness",
            "episode": 0,
            "samples": samples,
            "subtask_annotations": [s for s in subtasks if s],
            "visual_summary": selection_summary,
            "warning": "Historical actions are references for stage order, arm role, contact direction and scale. Do not replay old absolute joint commands; verify every current target from live state and images.",
        }, ensure_ascii=False, separators=(",", ":"))


def _round_list(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, list):
        return [round(float(item), 4) for item in value]
    return value
