import base64
from io import BytesIO

from PIL import Image

from gpt_policy.hardware.camera import CapturedImage
from gpt_policy.harness.media import content_blocks
from gpt_policy.harness.models import AgentTurn
from gpt_policy.input import ImagePart, TextPart


def test_jpeg_option_only_converts_live_frames_and_preserves_input_order(tmp_path):
    reference = tmp_path / "reference.png"
    reference.write_bytes(b"unchanged-reference")
    frame = CapturedImage("left", b"source-png", "image/png", 2, 1, 0,
                          bytes((255, 0, 0, 0, 255, 0)))
    blocks = content_blocks(AgentTurn("observation", {"left": frame}, (
        TextPart("before"), ImagePart(reference, "reference", "high"), TextPart("after"),
    )), True, 85)
    assert blocks[0].text == "before" and blocks[1].text == "Image: reference"
    assert blocks[2].mime_type == "image/png"
    assert base64.b64decode(blocks[2].data) == b"unchanged-reference"
    assert [blocks[i].text for i in (3, 4, 5)] == ["after", "observation", "Camera image: left"]
    assert blocks[6].mime_type == "image/jpeg"
    with Image.open(BytesIO(base64.b64decode(blocks[6].data))) as image:
        assert image.size == (2, 1) and image.format == "JPEG"
    assert frame.data == b"source-png"


def test_existing_jpeg_and_disabled_conversion_preserve_bytes():
    jpeg = CapturedImage("left", b"jpeg", "image/jpeg", 1, 1, 0)
    png = CapturedImage("left", b"png", "image/png", 1, 1, 0)
    for image, convert in ((jpeg, True), (png, False)):
        blocks = content_blocks(AgentTurn("observation", {"left": image}), convert, 85)
        assert base64.b64decode(blocks[-1].data) == image.data
        assert blocks[-1].mime_type == image.mime_type
