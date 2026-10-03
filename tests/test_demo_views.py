"""Regressions for the top-only, unreviewed insert-chapai demonstration."""
import json

from gpt_policy.harness.video_selector import _candidate_content
from gpt_policy.input.demonstration import prepare_demonstration
from gpt_policy.input.manifest import ImagePart, VideoPart
from gpt_policy.input.request import RunInput
from gpt_policy.input.video import FfmpegVideoExtractor, FrameSelection, SelectedFrame
from gpt_policy.input.video_cache import VideoProcessingCache
from test_video_input import FakeMediaRunner, FakeSelector


class ReviewedSelector(FakeSelector):
    reviews = 0

    def review_identity(self):
        return "test-whole-demo-v1"

    def review(self, instruction, video, frames, selection, limit):
        self.reviews += 1
        assert limit >= 2
        return FrameSelection(selection.selected, "Grasp, insert, release, press, withdraw.",
                              {"windows": selection.wire, "review": True})


def test_short_demonstration_is_reviewed_once_and_cached(tmp_path, monkeypatch):
    monkeypatch.setattr("gpt_policy.input.video.shutil.which", lambda name: name)
    video = tmp_path / "demo.mp4"
    video.write_bytes(b"video")
    selector = ReviewedSelector()
    cache = VideoProcessingCache(tmp_path / "cache")
    extractor = FfmpegVideoExtractor(runner=FakeMediaRunner())
    first = cache.resolve("insert plug", VideoPart(video), extractor, selector)
    second = cache.resolve("insert plug", VideoPart(video), extractor, selector)
    assert selector.reviews == 1
    assert selector.calls == 1 and second.cache_hit
    assert first.selection.summary == "Grasp, insert, release, press, withdraw."


def test_episode_views_reach_selection_review_and_saved_context(tmp_path, monkeypatch):
    monkeypatch.setattr("gpt_policy.input.video.shutil.which", lambda name: name)
    root = tmp_path / "episode"
    root.mkdir()
    cameras = ("top", "left_wrist", "right_wrist")
    for camera in cameras:
        (root / f"camera_{camera}.mp4").write_bytes(camera.encode())
    # Video-only preparation must not open action records.
    (root / "episode.mcap").write_bytes(b"not readable as MCAP")
    (root / "session_summary.json").write_text(json.dumps({
        "schema": "any_robo_episode_media_v1", "ok": True,
        "timing_mode": "recorded_timestamps", "timeline_origin_ns": 0,
        "cameras": {c: {"frames": 60, "timing_mode": "recorded_timestamps",
                         "timeline_origin_ns": 0} for c in cameras}}))

    class Selector(ReviewedSelector):
        def select(self, instruction, video, extraction):
            content = _candidate_content(extraction)
            assert sum(isinstance(p, ImagePart) for p in content) == 3 * len(extraction.candidates)
            return super().select(instruction, video, extraction)

        def review(self, instruction, video, frames, selection, limit):
            assert all(set(f.views) == {"left_wrist", "right_wrist"} for f in frames)
            assert limit == 16  # Three views share the existing 48-image budget.
            return super().review(instruction, video, frames, selection, limit)

    selector = Selector()
    extractor = FfmpegVideoExtractor(runner=FakeMediaRunner())
    cache = VideoProcessingCache(tmp_path / "cache")
    for i in range(2):
        out = tmp_path / f"prepared-{i}"
        run, meta = prepare_demonstration(RunInput("insert plug", "test"), root, "video", out,
                                          selector, extractor, cache)
        assert meta["unique_images"] == 6
        assert sum(isinstance(p, ImagePart) for p in run.content) == 6
        frames = json.loads((out / "demo.json").read_text())["keyframes"]
        assert all(set(f["images"]) == set(cameras) for f in frames)
        assert all("state" not in f and "action" not in f for f in frames)
        assert all((out / p).is_file() for f in frames for p in f["images"].values())
    assert selector.calls == selector.reviews == 1
    (root / "camera_right_wrist.mp4").write_bytes(b"changed wrist recording")
    prepare_demonstration(RunInput("insert plug", "test"), root, "video", tmp_path / "changed",
                          selector, extractor, cache)
    assert selector.calls == selector.reviews == 2


def test_views_align_by_capture_time_not_equal_pts_or_frame_index(tmp_path):
    from gpt_policy.input.video import CandidateFrame
    extractor = FfmpegVideoExtractor()
    extractor.views = {
        "top": (tmp_path / "top.mp4", [0., 1.], [100., 102.]),
        "right_wrist": (tmp_path / "right.mp4", [0., .5, 1.], [99., 100.03, 102.02]),
    }
    def decode(path, times, destination, width, timeline):
        return tuple(CandidateFrame(i, t, destination / f"{i}.jpg", timeline.index(t))
                     for i, t in enumerate(sorted(set(times))))
    extractor.extract_at = decode
    top = (CandidateFrame(0, 0., tmp_path / "top0.jpg", 0), CandidateFrame(1, 1., tmp_path / "top1.jpg", 1))
    paired = extractor._with_views(top, tmp_path / "images", 640)
    assert [f.views["right_wrist"].frame_index for f in paired] == [1, 2]
    assert [f.views["right_wrist"].timestamp_s for f in paired] == [.5, 1.]
    extractor.views["right_wrist"] = (tmp_path / "right.mp4", [0.], [101.])
    import pytest
    with pytest.raises(ValueError, match="no capture"):
        extractor._with_views(top, tmp_path / "images", 640)
