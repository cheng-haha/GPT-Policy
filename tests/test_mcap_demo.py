"""Asynchronous YAM episodes use capture clocks, not equal row numbers."""

import json
from types import SimpleNamespace

import pytest
from google.protobuf.descriptor_pb2 import FileDescriptorProto, FileDescriptorSet
from google.protobuf.descriptor_pool import DescriptorPool
from google.protobuf.message_factory import GetMessageClass
from mcap.writer import Writer

from gpt_policy.input.demonstration import demonstration_instruction
from gpt_policy.input.references import instruction_source


def episode(root, *, missing=None, gap=False, pose_format="ee_pos_quat=[x,y,z,qw,qx,qy,qz]"):
    root.mkdir()
    origin = 1789002530000000000
    pts = [.03, .08, .19, .33, .41]
    (root / "camera_top.mp4").write_bytes(b"video")
    (root / "session_summary.json").write_text(json.dumps({
        "schema": "any_robo_episode_media_v1", "ok": True,
        "timing_mode": "recorded_timestamps", "timeline_origin_ns": origin,
        "cameras": {"top": {"frames": len(pts), "processing": "direct_remux"}},
    }))
    (root / "finalization.json").write_text('{"state": "completed"}')
    fd = FileDescriptorProto(name="sample.proto", syntax="proto3")
    msg = fd.message_type.add(name="Sample")
    for n, name in enumerate(("position", "pose"), 1):
        msg.field.add(name=name, number=n, type=1, label=3)
    pool = DescriptorPool()
    pool.Add(fd)
    cls = GetMessageClass(pool.FindMessageTypeByName("Sample"))
    with (root / "episode.mcap").open("wb") as stream:
        writer = Writer(stream)
        writer.start()
        writer.add_metadata("session-metadata", {"recording_format": "episode_mcap_v1",
                            "robot_state_pose_format": pose_format,
                            "instruction": "Stale unrelated task label"})
        schema = writer.register_schema("Sample", "protobuf", FileDescriptorSet(file=[fd]).SerializeToString())
        camera = writer.register_channel("/top-camera", "video", 0)
        sync = writer.register_channel("/sync", "json", 0)
        for t in pts:
            writer.add_message(camera, origin + round(t * 1e9), b"packet", origin + round(t * 1e9))
        for i in range(10):
            t = origin + round((.04 + i * .04) * 1e9)
            writer.add_message(sync, t, b"{}", t)
        for side in ("left", "right"):
            for part in ("arm", "ee"):
                for kind in ("state", "action"):
                    topic = f"/{side}-{part}-{kind}"
                    if topic == missing:
                        continue
                    channel = writer.register_channel(topic, "protobuf", schema)
                    for i in range(10):
                        if gap and topic.endswith("state") and i > 1:
                            continue
                        # Command and measurement occur at distinct times.
                        t = origin + round((.04 + i * .04 + (.006 if kind == "state" else -.005)) * 1e9)
                        q = i / 10 + (1 if kind == "action" else 0)
                        grip = .2 if kind == "action" else .6
                        value = cls(position=[q] * 6 + ([grip] if kind == "state" else [])
                                    if part == "arm" else [grip],
                                    pose=[.2, .1, .3, 1, 0, 0, 0] if part == "arm" else [])
                        writer.add_message(channel, t, value.SerializeToString(), t)
        writer.finish()
    return SimpleNamespace(frame_times=lambda _: tuple(pts))


def test_inline_raw_episode_is_recognized_without_config_json(tmp_path):
    root = tmp_path / "episode"
    episode(root)
    assert instruction_source(f"{root} 以video+action的模式模仿这个行为打开瓶盖") == (root, None)
    assert "Stale" not in demonstration_instruction(root)


def test_async_mcap_import_separates_targets_feedback_and_retains_alignment(tmp_path):
    from gpt_policy.input.mcap_demo import McapDemo
    root = tmp_path / "episode"
    extractor = episode(root)
    demo = McapDemo(root, extractor, True)
    assert len(demo.samples) == 10  # Camera has only five frames.
    assert demo.metadata["outcome"] == "unverified"
    assert demo.metadata["finalization_state"] == "completed"
    frames = [{"video_frame_index": n} for n in (0, 2, 4)]
    demo.attach(frames)
    assert frames[0]["state"]["left"]["gripper_measured"] == .6
    assert "joint_target_rad" not in frames[0]["state"]["left"]
    sample = frames[0]["action"]["samples"][0]
    assert sample["left"]["joint_target_rad"][0] == 1
    assert sample["left"]["joint_measured_rad"][0] == 0
    assert sample["left"]["eef_target_xyz_xyzw"] == [.2, .1, .3, 0, 0, 0, 1]
    assert sample["left"]["gripper_command"] == .2
    assert sample["source_times_s"]["/left-arm-action"] == pytest.approx(.035)
    assert sample["source_times_s"]["/left-arm-state"] == pytest.approx(.046)
    assert frames[0]["alignment"]["exposure_synchronized"] is False
    demo.verify_sources()
    (root / "episode.mcap").write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="changed"):
        demo.verify_sources()


def test_video_does_not_read_mcap_or_require_action_metadata(tmp_path):
    from gpt_policy.input.mcap_demo import McapDemo
    root = tmp_path / "episode"
    extractor = episode(root)
    (root / "episode.mcap").write_bytes(b"not read in video mode")
    demo = McapDemo(root, extractor, False)
    assert not demo.samples
    assert "episode.mcap" not in demo.source_hashes


@pytest.mark.parametrize("case, message", [("missing", "Missing"), ("pose", "pose convention"), ("frames", "frame"), ("clock", "timestamps")])
def test_incomplete_or_ambiguous_actions_fail_explicitly(tmp_path, case, message):
    from gpt_policy.input.mcap_demo import McapDemo
    root = tmp_path / "episode"
    extractor = episode(root, missing="/right-arm-action" if case == "missing" else None,
                        pose_format="unknown" if case == "pose" else "ee_pos_quat=[x,y,z,qw,qx,qy,qz]")
    if case == "frames":
        extractor.frame_times = lambda _: (0., 1.)
    if case == "clock":
        extractor.frame_times = lambda _: (.03, .08, .15, .33, .41)
    with pytest.raises(ValueError, match=message):
        McapDemo(root, extractor, True)


def test_stale_state_is_not_attached_to_later_images(tmp_path):
    from gpt_policy.input.mcap_demo import McapDemo
    root = tmp_path / "episode"
    demo = McapDemo(root, episode(root, gap=True), True)
    frames = [{"video_frame_index": 4}]
    demo.attach(frames)
    assert frames[0]["state"] is None
    assert "/left-arm-state" in frames[0]["alignment"]["missing_streams"]


@pytest.mark.parametrize("mode", ["video", "video+action"])
def test_cli_prepares_raw_episode_as_portable_context_without_hardware(tmp_path, monkeypatch, mode):
    from gpt_policy import main
    from gpt_policy.input.video import FfmpegVideoExtractor
    from gpt_policy.input.video_cache import VideoProcessingCache
    from test_video_input import FakeMediaRunner, FakeSelector

    root = tmp_path / "episode"
    timeline = episode(root)
    output = tmp_path / "prepared"
    instruction = f"{root} 以{mode}的模式模仿这个行为打开瓶盖"
    monkeypatch.setattr("sys.argv", ["gpt-policy", instruction, "--prepare-only", str(output)])
    def forbidden(*_, **__):
        raise AssertionError("Input preparation must not open hardware or make task decisions")
    for name in ("preflight_agent", "select_task_request", "CameraSet", "ArxRobot", "_open_yam"):
        monkeypatch.setattr(main, name, forbidden)
    def extractor(config):
        result = FfmpegVideoExtractor(config, FakeMediaRunner(.45))
        result.frame_times = timeline.frame_times
        return result
    monkeypatch.setattr(main, "FfmpegVideoExtractor", extractor)
    monkeypatch.setattr(main, "CodexVideoSelector", lambda *_: FakeSelector())
    monkeypatch.setattr("gpt_policy.input.video.shutil.which", lambda name: name)
    monkeypatch.setattr("gpt_policy.input.preparation.VideoProcessingCache", lambda *_: VideoProcessingCache(tmp_path / "cache"))
    main.main()
    data = json.loads((output / "input.json").read_text())
    assert data["instruction"] == instruction
    assert "Stale unrelated" not in json.dumps(data)
    demo = json.loads((output / "input-videos/video-000/demo.json").read_text())
    assert len(demo["keyframes"]) == 2
    assert (output / "input-videos/video-000/actions.jsonl").exists() == (mode == "video+action")
    if mode == "video+action":
        assert demo["keyframes"][0]["action"]["samples"][0]["right"]["joint_target_rad"][0] == 1
    else:
        assert all("state" not in frame and "action" not in frame for frame in demo["keyframes"])
