from __future__ import annotations

import json
import subprocess
import re
from pathlib import Path
from unittest.mock import patch

import pytest

from gpt_policy.harness.config import AgentConfig
from gpt_policy.harness.video_selector import (
    CodexVideoSelector,
    FrameSelection,
    SelectedFrame,
)
from gpt_policy.input import ImagePart, TextPart, VideoPart
from gpt_policy.input.preparation import prepare_input_videos
from gpt_policy.input.request import RunInput
from gpt_policy.input.video import (
    CandidateFrame,
    FfmpegVideoExtractor,
    VideoExtraction,
    VideoMetadata,
    VideoProcessingConfig,
)
from gpt_policy.input.video_cache import VideoProcessingCache
from gpt_policy.recording.trace import RunRecorder


class FakeMediaRunner:
    def __init__(self, duration: float = 2.0):
        self.duration = duration
        self.commands: list[list[str]] = []

    def __call__(self, command, **_kwargs):
        self.commands.append(command)
        if "ffprobe" in Path(command[0]).name:
            if "frame=best_effort_timestamp_time" in command:
                return subprocess.CompletedProcess(command, 0, json.dumps({"frames": [
                    {"best_effort_timestamp_time": str(i / 30)} for i in range(round(self.duration * 30))]}), "")
            stdout = json.dumps({
                "streams": [{
                    "width": 1920,
                    "height": 1080,
                    "avg_frame_rate": "30/1",
                    "r_frame_rate": "30/1",
                    "codec_name": "h264",
                }],
                "format": {"duration": str(self.duration)},
            })
            return subprocess.CompletedProcess(command, 0, stdout, "")
        for i, _ in enumerate(re.findall(r"eq\(n\\,(\d+)\)", command[command.index("-vf") + 1])):
            Path(command[-1].replace("%06d", f"{i+1:06d}")).write_bytes(b"jpeg-frame")
        return subprocess.CompletedProcess(command, 0, "", "")


class FakeSelector:
    def __init__(self):
        self.calls = 0

    @staticmethod
    def cache_identity():
        return {"selector": "fake-v1"}

    def select(self, _instruction, _video, extraction):
        self.calls += 1
        last = len(extraction.candidates) - 1
        return FrameSelection(
            (SelectedFrame(0, "初始状态"), SelectedFrame(last, "最终状态")),
            "覆盖任务前后的状态",
            {"name": "select_video_frames"},
        )

    def review_identity(self):
        return "fake-review-v1"

    def review(self, instruction, video, frames, selection, limit):
        return selection


def test_action_events_are_candidates_not_forced_semantic_keyframes(tmp_path, monkeypatch):
    monkeypatch.setattr("gpt_policy.input.video.shutil.which", lambda name: name)
    video = tmp_path / "video.mp4"
    video.write_bytes(b"video")
    extractor = FfmpegVideoExtractor(runner=FakeMediaRunner(), event_times=(.4, .8, 1.2, 1.6))
    selector = FakeSelector()
    cache = VideoProcessingCache(tmp_path / "cache")
    result = cache.resolve("task", VideoPart(video), extractor, selector)
    assert len(result.extraction.candidates) > 4
    assert len(result.keyframes) == 2
    # Migrate old cached host-forced event anchors without another model call.
    path = result.cache_dir / "cache.json"
    data = json.loads(path.read_text())
    data["selection"]["selected"].insert(1, {"index": 1,
        "reason": "Timeline boundary or recorded event; verify the visible outcome.",
        "stage": "", "left": "", "right": "", "result": ""})
    data["keyframes"].insert(1, data["candidates"][1])
    path.write_text(json.dumps(data))
    reused = cache.resolve("task", VideoPart(video), extractor, selector)
    assert reused.cache_hit and selector.calls == 1
    assert len(reused.keyframes) == 2


def test_ffmpeg_extractor_probes_and_uniformly_bounds_candidates(tmp_path):
    source = tmp_path / "demo.mp4"
    source.write_bytes(b"video")
    runner = FakeMediaRunner(duration=10.0)
    config = VideoProcessingConfig(target_fps=2, max_candidates=4, max_keyframes=4)
    with patch("gpt_policy.input.video.shutil.which", side_effect=lambda name: f"/tools/{name}"):
        extraction = FfmpegVideoExtractor(config, runner).extract_candidates(
            source, tmp_path / "output"
        )

    assert extraction.metadata == VideoMetadata(10.0, 1920, 1080, 30.0, "h264", 5)
    assert len(extraction.candidates) == 4
    assert extraction.candidates[0].timestamp_s == 0
    assert extraction.candidates[-1].timestamp_s < 10
    ffmpeg_commands = [item for item in runner.commands if "ffmpeg" in Path(item[0]).name]
    assert len(ffmpeg_commands) == 1
    assert extraction.candidates[-1].frame_index == 299
    assert "select=" in ffmpeg_commands[0][ffmpeg_commands[0].index("-vf") + 1]


def test_preparation_reuses_content_addressed_codex_result(tmp_path):
    source = tmp_path / "demo.mp4"
    source.write_bytes(b"same-video-content")
    raw = RunInput(
        "模仿视频完成放置",
        "gpt-test",
        (TextPart("before"), VideoPart(source, "演示", "high"), TextPart("after")),
    )
    runner = FakeMediaRunner()
    config = VideoProcessingConfig(target_fps=1, max_candidates=4, max_keyframes=2)
    extractor = FfmpegVideoExtractor(config, runner)
    selector = FakeSelector()
    cache = VideoProcessingCache(tmp_path / "cache")

    with patch("gpt_policy.input.video.shutil.which", side_effect=lambda name: f"/tools/{name}"):
        first, first_results = prepare_input_videos(
            raw, tmp_path / "run-1", selector, extractor, cache
        )
        command_count = len(runner.commands)
    with patch(
        "gpt_policy.input.video.shutil.which",
        side_effect=AssertionError("cache hit must not resolve FFmpeg"),
    ):
        second, second_results = prepare_input_videos(
            raw, tmp_path / "run-2", selector, extractor, cache
        )

    assert selector.calls == 1
    assert len(runner.commands) == command_count
    assert first_results[0].cache_hit is False
    assert second_results[0].cache_hit is True
    assert first_results[0].cache_key == second_results[0].cache_key
    assert [type(item) for item in first.content] == [
        TextPart, TextPart, ImagePart, ImagePart, TextPart,
    ]
    assert all(item.path.is_file() for item in second.content if isinstance(item, ImagePart))
    metadata = json.loads((tmp_path / "run-2" / "video-000" / "metadata.json").read_text())
    assert metadata["cache_hit"] is True
    assert metadata["selection"]["selected"][0]["reason"] == "初始状态"


@pytest.mark.parametrize("relative", [False, True])
def test_saved_video_keyframes_remain_accessible_after_outcome_rename(tmp_path, monkeypatch, relative):
    monkeypatch.chdir(tmp_path)
    source = tmp_path / "demo.mp4"
    source.write_bytes(b"video")
    root = Path("run") if relative else tmp_path / "run"
    recorder = RunRecorder(root, {"task_name": "place-block"})
    config = VideoProcessingConfig(target_fps=1, max_candidates=4, max_keyframes=2)
    with patch("gpt_policy.input.video.shutil.which", side_effect=lambda name: f"/tools/{name}"):
        _, results = prepare_input_videos(
            RunInput("放置积木", "test", (VideoPart(source),)), root / "input-videos",
            FakeSelector(), FfmpegVideoExtractor(config, FakeMediaRunner()),
            VideoProcessingCache(tmp_path / "cache"),
        )
    recorder.write("video_preprocessed", results[0].record(relative_to=root))
    recorder.write("human_evaluation", {"outcome": "success"})
    final_root = recorder.close("completed")
    assert final_root.name == "run_success"
    assert not root.exists()
    events = [json.loads(line) for line in (final_root / "events.jsonl").read_text().splitlines()]
    event = next(row for row in events if row["event"] == "video_preprocessed")
    exported = final_root / event["output_dir"]
    metadata = json.loads((exported / "metadata.json").read_text())
    for base, record in ((final_root, event), (exported, metadata)):
        assert not Path(record["output_dir"]).is_absolute()
        for selected in record["selection"]["selected"]:
            assert not Path(selected["path"]).is_absolute()
            assert (base / selected["path"]).read_bytes() == b"jpeg-frame"
    assert source.read_bytes() == b"video"


def test_instruction_change_invalidates_semantic_cache(tmp_path):
    source = tmp_path / "demo.mp4"
    source.write_bytes(b"video")
    runner = FakeMediaRunner(duration=1.0)
    config = VideoProcessingConfig(target_fps=1, max_candidates=2, max_keyframes=2)
    extractor, selector = FfmpegVideoExtractor(config, runner), FakeSelector()
    cache = VideoProcessingCache(tmp_path / "cache")
    with patch("gpt_policy.input.video.shutil.which", side_effect=lambda name: f"/tools/{name}"):
        prepare_input_videos(
            RunInput("任务 A", "model", (VideoPart(source),)),
            tmp_path / "run-a", selector, extractor, cache,
        )
        prepare_input_videos(
            RunInput("任务 B", "model", (VideoPart(source),)),
            tmp_path / "run-b", selector, extractor, cache,
        )
    assert selector.calls == 2
    # Candidate decode is shared; only selected full-size exports run again.
    candidate_decodes = [c for c in runner.commands if "ffmpeg" in Path(c[0]).name and "candidates" in c[-1]]
    assert len(candidate_decodes) == 1


class FakeCodexClient:
    def __init__(self, decision):
        self.decision = decision
        self.started = None
        self.call = None
        self.closed = False

    def start_thread(self, instructions):
        self.started = instructions

    def decide(self, observation, schema, **kwargs):
        self.call = (observation, schema, kwargs)
        return self.decision

    def close(self):
        self.closed = True


def test_global_review_reuses_window_cache_and_keeps_real_indices(tmp_path, monkeypatch):
    from dataclasses import replace
    monkeypatch.setattr("gpt_policy.input.video.shutil.which", lambda name: name)
    video = tmp_path / "long.mp4"
    video.write_bytes(b"video")
    class Selector(FakeSelector):
        reviews = 0
        def select(self, instruction, video, extraction):
            self.calls += 1
            return FrameSelection(tuple(SelectedFrame(f.index, "stage") for f in extraction.candidates),
                                  "window notes", {"windows": ["original selection"]})
        def review_identity(self): return "review-test-v1"
        def review(self, instruction, video, frames, selection, limit):
            self.reviews += 1
            assert len(frames) > limit
            return replace(selection, selected=(selection.selected[0], selection.selected[15], selection.selected[-1]),
                           summary="whole task", wire={"review": "curated", "windows": selection.wire})
    selector = Selector()
    cache = VideoProcessingCache(tmp_path / "cache")
    extractor = FfmpegVideoExtractor(runner=FakeMediaRunner(40))
    first = cache.resolve("task", VideoPart(video), extractor, selector)
    second = cache.resolve("task", VideoPart(video), extractor, selector)
    assert selector.calls == selector.reviews == 1
    assert not first.cache_hit and second.cache_hit
    assert first.selection.summary == "whole task"
    assert [f.index for f in second.keyframes] == [0, 15, len(second.extraction.candidates)-1]
    # Corrupt only final review: reuse initial selection and repeat only review.
    (second.cache_dir / "review-v1.json").write_text("invalid")
    cache.resolve("task", VideoPart(video), extractor, selector)
    assert selector.calls == 1 and selector.reviews == 2


def test_global_review_receives_annotations_images_and_keeps_endpoints(tmp_path):
    frames = tuple(CandidateFrame(10+i, float(i), tmp_path / f"{i}.jpg", i) for i in range(30))
    original = FrameSelection(tuple(SelectedFrame(f.index, "evidence", "hold", "hold bottle", "turn cap")
                                    for f in frames), "many window summaries", {"windows": []})
    client = FakeCodexClient({"name": "select_video_frames", "arguments": {
        "selected": [{"index": i, "reason": "observed"} for i in (0, 15, 29)], "summary": "Hold, turn, separate."}})
    selector = CodexVideoSelector(AgentConfig(), client_factory=lambda *_: client)
    result = selector.review("Open bottle", VideoPart(tmp_path / "video.mp4"), frames, original)
    assert client.started.isascii()
    assert "gripper-to-table direction" in client.started
    assert "do not mislabel empty-gripper pressing as regrasping" in client.started
    assert "Preserve the order of repeated twists and regrasps" in client.started
    assert "Retain the full sequence's first and last frames" in client.started
    assert [c.index for c in result.selected] == [10, 25, 39]
    assert sum(isinstance(p, ImagePart) for p in client.call[2]["content"]) == 30
    assert "hold bottle" in client.call[2]["content"][0].text
    assert client.closed
    client.decision["arguments"]["selected"].pop()
    with pytest.raises(ValueError, match="boundaries"):
        selector.review("Open bottle", VideoPart(tmp_path / "video.mp4"), frames, original)


def test_codex_selector_receives_timestamped_images_and_validates_output(tmp_path):
    images = []
    for index in range(3):
        path = tmp_path / f"{index}.jpg"
        path.write_bytes(b"jpeg")
        images.append(CandidateFrame(index, float(index), path))
    extraction = VideoExtraction(
        tmp_path / "demo.mp4",
        VideoMetadata(3, 640, 480, 30, "h264", 100),
        tuple(images),
    )
    client = FakeCodexClient({
        "name": "select_video_frames",
        "arguments": {
            "selected": [
                {"index": 2, "reason": "结果"},
                {"index": 0, "reason": "初始"},
            ],
            "summary": "完整过程",
        },
        "_wire": {"name": "select_video_frames"},
    })
    selector = CodexVideoSelector(
        AgentConfig(model="gpt-test"),
        VideoProcessingConfig(max_keyframes=2),
        lambda *_args: client,
    )
    result = selector.select("完成任务", VideoPart(extraction.source), extraction)

    assert client.started.isascii()
    assert "gripper-to-table direction" in client.started
    assert "gripper closure alone does not prove a grasp" in client.started
    assert "You have no shell, file or robot-control tools" in client.started
    assert [item.index for item in result.selected] == [0, 2]
    assert client.closed
    observation, schema, kwargs = client.call
    assert "完成任务" in observation
    assert "User's final task:" in observation
    assert schema["properties"]["arguments"]["properties"]["selected"]["maxItems"] == 2
    content = kwargs["content"]
    assert content[0].text.startswith("Video metadata: ")
    assert content[1].text.startswith("Candidate frame index=0, ")
    assert len([item for item in content if isinstance(item, ImagePart)]) == 3
    assert all(item.detail == "auto" for item in content if isinstance(item, ImagePart))


def test_codex_selector_rejects_duplicate_indices(tmp_path):
    image = tmp_path / "0.jpg"
    image.write_bytes(b"jpeg")
    extraction = VideoExtraction(
        tmp_path / "demo.mp4",
        VideoMetadata(1, 1, 1, 1, "mjpeg", 1),
        (CandidateFrame(0, 0, image),),
    )
    client = FakeCodexClient({
        "name": "select_video_frames",
        "arguments": {
            "selected": [
                {"index": 0, "reason": "a"},
                {"index": 0, "reason": "b"},
            ],
            "summary": "duplicate",
        },
    })
    selector = CodexVideoSelector(
        AgentConfig(), VideoProcessingConfig(max_keyframes=2), lambda *_args: client
    )
    with pytest.raises(RuntimeError, match="重复"):
        selector.select("task", VideoPart(extraction.source), extraction)
    assert client.closed
