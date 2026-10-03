from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from gpt_policy.input import ImagePart, TextPart, VideoPart, load_manifest, resolve_run_input


class InputManifestTest(unittest.TestCase):
    def test_strings_and_images_keep_their_order(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "frame.png"
            image.write_bytes(b"png")
            manifest_path = root / "request.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "model": "gpt-6-test",
                        "content": [
                            "先看这张图",
                            {"image": "frame.png", "label": "测试图", "detail": "high"},
                            "然后给出结论",
                        ],
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            manifest = load_manifest(manifest_path)

        self.assertEqual(manifest.model, "gpt-6-test")
        self.assertIsInstance(manifest.content[0], TextPart)
        self.assertIsInstance(manifest.content[1], ImagePart)
        self.assertEqual(manifest.content[1].path, image.resolve())
        self.assertEqual(manifest.instruction, "先看这张图\n然后给出结论")

    def test_instruction_can_be_used_for_image_only_content(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "frame.jpg"
            image.write_bytes(b"jpg")
            path = root / "request.json"
            path.write_text(
                json.dumps({"instruction": "描述图片", "content": [{"image": "frame.jpg"}]}),
                encoding="utf-8",
            )
            run_input = resolve_run_input(None, path, None)

        self.assertEqual(run_input.instruction, "描述图片")
        self.assertEqual(run_input.model, "gpt-6-astra")

    def test_local_video_path_is_resolved_relative_to_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "assets" / "demo.mp4"
            video.parent.mkdir()
            video.write_bytes(b"video")
            path = root / "request.json"
            path.write_text(
                json.dumps({
                    "instruction": "参考演示完成任务",
                    "content": [{
                        "video": "assets/demo.mp4",
                        "label": "演示",
                        "detail": "high",
                    }],
                }, ensure_ascii=False),
                encoding="utf-8",
            )
            manifest = load_manifest(path)

        self.assertIsInstance(manifest.content[0], VideoPart)
        self.assertEqual(manifest.content[0].path, video.resolve())
        self.assertEqual(manifest.record()["content"][0]["video"], str(video.resolve()))

    def test_media_object_must_choose_exactly_one_type(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            media = root / "media.bin"
            media.write_bytes(b"media")
            for item in ({}, {"image": "media.bin", "video": "media.bin"}):
                path = root / "request.json"
                path.write_text(json.dumps({
                    "instruction": "task", "content": [item],
                }), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "只能包含 image 或 video"):
                    load_manifest(path)

    def test_missing_task_is_rejected_without_manifest(self) -> None:
        with self.assertRaisesRegex(ValueError, "请提供任务文字"):
            resolve_run_input(None, None, None)


if __name__ == "__main__":
    unittest.main()
