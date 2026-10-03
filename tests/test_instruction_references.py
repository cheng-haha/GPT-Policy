import json
import shutil

import pytest
from PIL import Image

from gpt_policy import main
from gpt_policy.input.manifest import ImagePart, load_manifest
from gpt_policy.input.references import instruction_media, instruction_source
from test_demonstration import reviewed


@pytest.mark.parametrize("text", [
    "参考 ./clip.mp4，拧开瓶盖", "参考：~/clip.mp4。拧开瓶盖", "Use clip.mp4. Open bottle.",
    "参考（./clip.mp4），拧开瓶盖", "[示范](./clip.mp4)",
])
def test_video_paths_use_cwd_home_and_prose_delimiters(tmp_path, monkeypatch, text):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))
    video = tmp_path / "clip.mp4"
    video.touch()
    assert instruction_source(text) == (video, None)


@pytest.mark.parametrize("quotes", ["''", '""', "``", "“”", "‘’"])
def test_quoted_paths_allow_spaces_and_punctuation(tmp_path, quotes):
    video = tmp_path / "瓶子 (demo)，01.mp4"
    video.touch()
    assert instruction_source(f"参考 {quotes[0]}{video}{quotes[1]}，拧开瓶盖") == (video, None)


def test_recordings_take_precedence_over_their_original_input_snapshot(tmp_path):
    for name in ("top.mp4", "config.json", "status.json", "events.jsonl", "video-frames.jsonl"):
        (tmp_path / name).touch()
    (tmp_path / "input.json").write_text('{"content": ["original input"]}')
    assert instruction_source(f"参考 {tmp_path}，重复任务") == (tmp_path, None)


def test_bundle_aliases_are_deduplicated_and_prepared_inputs_are_recognized(tmp_path, monkeypatch):
    source = reviewed(tmp_path, 2)
    monkeypatch.chdir(tmp_path)
    assert instruction_source(f"参考 ./source/，示范文件是 {source}") == (source, None)
    package = tmp_path / "prepared"
    package.mkdir()
    manifest = package / "input.json"
    manifest.write_text('{"content": ["task"]}')
    assert instruction_source("参考 prepared/，拧开瓶盖") == (None, manifest)
    assert instruction_source(f"参考 {manifest}，拧开瓶盖") == (None, manifest)


@pytest.mark.parametrize("text", [None, "拿起水果", "左/右臂配合，转动 1/2 圈", "参考 https://example.com/demo.mp4"])
def test_ordinary_text_and_web_urls_are_not_local_inputs(text):
    assert instruction_source(text) == (None, None)


def test_inline_images_preserve_order_and_deduplicate_aliases(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    first = tmp_path / "first image.png"
    second = tmp_path / "second.jpg"
    first.touch()
    second.touch()
    instruction = f'参考 "{first.name}"，再参考 {second.name}，以及 ./second.jpg'
    assert instruction_media(instruction) == (None, None, (first, second))
    assert instruction_source(instruction) == (None, None)


def test_image_filename_does_not_set_demonstration_mode():
    from gpt_policy.input.references import instruction_mode
    assert instruction_mode('参考 "video+action.png"，抓取水果') is None


def test_invalid_and_multiple_sources_fail_clearly(tmp_path):
    with pytest.raises(ValueError, match="does not exist"):
        instruction_source(f"参考 {tmp_path}/missing.mp4，拿起瓶子")
    with pytest.raises(ValueError, match="Not a demonstration"):
        instruction_source(f"参考 {tmp_path}，拿起瓶子")
    config = tmp_path / "config.json"
    config.write_text('{"runtime": {}}')
    with pytest.raises(ValueError, match="Expected a demonstration"):
        instruction_source(f"参考 {config}，拿起瓶子")
    first, second = tmp_path / "a.mp4", tmp_path / "b.mp4"
    first.touch()
    second.touch()
    with pytest.raises(ValueError, match="one demonstration"):
        instruction_source(f"参考 {first} 和 {second}，拿起瓶子")


@pytest.fixture
def no_hardware_or_models(monkeypatch):
    def forbidden(*_, **__):
        raise AssertionError("Reviewed input must not initialize hardware/model")
    for name in ("preflight_agent", "select_task_request", "CameraSet", "ArxRobot", "_open_yam"):
        monkeypatch.setattr(main, name, forbidden)


@pytest.mark.parametrize("mode", ["video", "video+action"])
def test_cli_inline_demo_preserves_instruction_and_portable_reuse(tmp_path, monkeypatch, no_hardware_or_models, mode):
    source = reviewed(tmp_path, 2)
    output = tmp_path / "output"
    instruction = f"参考 {source}，拧开瓶盖"
    monkeypatch.setattr("sys.argv", ["gpt-policy", instruction, "--demo-mode", mode, "--prepare-only", str(output)])
    main.main()
    manifest = load_manifest(output / "input.json")
    assert manifest.instruction == instruction
    assert len([p for p in manifest.content if isinstance(p, ImagePart)]) == 2
    assert (output / "input-videos/video-000/actions.jsonl").exists() == (mode == "video+action")
    shutil.rmtree(source.parent)
    replay = tmp_path / "replay"
    monkeypatch.setattr("sys.argv", ["gpt-policy", "--input-json", str(output / "input.json"),
                                    "--prepare-only", str(replay)])
    main.main()
    assert load_manifest(replay / "input.json").instruction == instruction
    # Referencing the prepared package in a new task also uses its existing images.
    reuse = tmp_path / "reuse"
    instruction = f"参考 {output}，拧开另一个瓶盖"
    monkeypatch.setattr("sys.argv", ["gpt-policy", instruction, "--prepare-only", str(reuse)])
    main.main()
    reused = load_manifest(reuse / "input.json")
    assert reused.instruction == instruction
    assert len([p for p in reused.content if isinstance(p, ImagePart)]) == 2


@pytest.mark.parametrize("flag", ["--demo", "--input-json"])
def test_explicit_input_bypasses_paths_mentioned_in_prose(tmp_path, monkeypatch, no_hardware_or_models, flag):
    source = reviewed(tmp_path, 2)
    if flag == "--input-json":
        source = tmp_path / "input.json"
        source.write_text(json.dumps({"content": ["prepared context"]}))
    instruction = f"不用 {tmp_path}/missing.mp4，拧开瓶盖"
    output = tmp_path / "output"
    monkeypatch.setattr("sys.argv", ["gpt-policy", instruction, flag, str(source), "--prepare-only", str(output)])
    main.main()
    assert load_manifest(output / "input.json").instruction == instruction


def test_missing_inline_source_fails_before_model_or_hardware(tmp_path, monkeypatch, no_hardware_or_models):
    monkeypatch.setattr("sys.argv", ["gpt-policy", f"参考 {tmp_path}/missing.mp4，拿起瓶子"])
    with pytest.raises(ValueError, match="does not exist"):
        main.main()


def test_cli_inline_images_are_archived_and_portable(tmp_path, monkeypatch, no_hardware_or_models):
    monkeypatch.setattr(main, "REQUEST_DIRECTORY", tmp_path / "requests")
    first = tmp_path / "target image.png"
    second = tmp_path / "grasp.jpg"
    Image.new("RGB", (2, 2), "red").save(first)
    Image.new("RGB", (2, 2), "blue").save(second)
    instruction = f'参考 "{first}" 和 {second}，抓取相同物体'
    output = tmp_path / "prepared"
    monkeypatch.setattr("sys.argv", ["gpt-policy", instruction, "--prepare-only", str(output)])
    main.main()
    request = json.loads((output / "request.json").read_text())
    assert request["content"] == [instruction,
                                  {"image": str(first), "label": first.name},
                                  {"image": str(second), "label": second.name}]
    manifest = load_manifest(output / "input.json")
    images = [part for part in manifest.content if isinstance(part, ImagePart)]
    assert [part.path.read_bytes() for part in images] == [first.read_bytes(), second.read_bytes()]
    first.unlink()
    second.unlink()
    assert all(part.path.is_file() for part in load_manifest(output / "input.json").content
               if isinstance(part, ImagePart))


def test_cli_image_flag_combines_with_json_and_validates_before_hardware(tmp_path, monkeypatch, no_hardware_or_models):
    image = tmp_path / "reference.png"
    Image.new("RGB", (2, 2), "green").save(image)
    source = tmp_path / "request.json"
    source.write_text('{"instruction": "拿起水果", "content": ["看目标形状"]}')
    output = tmp_path / "prepared"
    monkeypatch.setattr("sys.argv", ["gpt-policy", "--input-json", str(source),
                                    "--image", str(image), "--prepare-only", str(output)])
    main.main()
    content = load_manifest(output / "input.json").content
    assert content[0].text == "看目标形状"
    assert isinstance(content[1], ImagePart)
    assert content[1].path.read_bytes() == image.read_bytes()
    monkeypatch.setattr("sys.argv", ["gpt-policy", "拿起水果", "--image", str(tmp_path / "missing.png")])
    with pytest.raises(ValueError, match="图片文件不存在"):
        main.main()
