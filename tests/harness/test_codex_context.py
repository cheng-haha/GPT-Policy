import json
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from gpt_policy.hardware.camera import CapturedImage
from gpt_policy.harness.codex import CodexAppServer
from gpt_policy.harness.config import AgentConfig, named_agent_config
from gpt_policy.harness.errors import AgentOverloadedError
from gpt_policy.harness.models import AgentContext, AgentTurn
from gpt_policy.harness.providers.codex import CodexSession
from gpt_policy.input.manifest import ImagePart, TextPart


class Native:
    _input = staticmethod(CodexAppServer._input)
    convert_camera_images_to_jpeg = False
    camera_jpeg_quality = 85
    thread_id = "thread-0"

    def __init__(self, *_):
        self.calls = []
        self.starts = []

    def start_thread(self, *args): self.starts.append(args)

    def decide(self, observation, schema, images, content, **kwargs):
        self.calls.append({"observation": observation, "content": content, **kwargs})
        wire = {"name": "move_to", "arguments": {"note": f"decision-{len(self.calls)}"}}
        return {**wire, "_wire": wire}

    def close(self): pass


def test_long_native_conversation_never_refreshes_or_replays_by_image_count(tmp_path):
    demo_image = tmp_path / "demo.jpg"
    demo_image.write_bytes(b"historical")
    context = AgentContext("original instructions", [], {})
    with patch("gpt_policy.harness.providers.codex.CodexAppServer", Native):
        session = CodexSession(AgentConfig(), "test", False, 85)
    session.start(context)
    static = (TextPart("historical demonstration"), ImagePart(demo_image, "demo"))
    for step in range(400):
        images = {"top": CapturedImage("top", b"frame", "image/jpeg", 1, 1, step)}
        session.decide(AgentTurn("observation" * 800, images, static if step == 0 else None))
        assert "replay" not in session.client.calls[-1]
        assert session.last_context_refresh is None
    assert len(session.client.starts) == 1
    assert session.client.calls[0]["content"] == static
    assert all(c["content"] is None for c in session.client.calls[1:])
    assert not hasattr(session, "_history")


@pytest.mark.parametrize("window", [0, 2, 3, 8, 16, True, 3.5, "8"])
def test_invalid_live_window_rejected_by_configuration(tmp_path, window):
    (tmp_path / "agents").mkdir()
    (tmp_path / "agents/codex.json").write_text(json.dumps({"live_image_window": window}))
    with pytest.raises(ValueError, match="disabled"):
        named_agent_config("codex", tmp_path)


def test_current_input_is_size_checked_before_turn_start():
    native = object.__new__(CodexAppServer)
    native.model = "gpt-test"
    native.effort = "high"
    native.project_root = Path("/robot")
    native.thread_id = "thread-1"
    native.convert_camera_images_to_jpeg = False
    native.camera_jpeg_quality = 85
    native._request = Mock(return_value={"turn": {"id": "turn-1"}})
    with pytest.raises(ValueError, match="too large"):
        native.decide("current", {}, content=(TextPart("x" * 1048576),))
    native._request.assert_not_called()


@pytest.mark.parametrize("previous_steps", [0, 1, 3])
def test_overload_keeps_the_original_thread(previous_steps):
    with patch("gpt_policy.harness.providers.codex.CodexAppServer", Native):
        session = CodexSession(AgentConfig(), "test", False, 85)
    session.start(AgentContext("same instructions", [], {}))
    for step in range(previous_steps):
        session.decide(AgentTurn(f"success-{step}"))
    decide = session.client.decide

    def overloaded(*args, **kwargs):
        raise AgentOverloadedError("at capacity", provider="codex", code="serverOverloaded")

    session.client.decide = overloaded
    for _ in range(2):
        with pytest.raises(AgentOverloadedError):
            session.decide(AgentTurn("pending observation"))
    session.client.decide = decide
    session.decide(AgentTurn("pending observation"))
    assert len(session.client.starts) == 1
    assert session.client.thread_id == "thread-0"
    assert session.client.calls[-1]["observation"] == "pending observation"
    assert "replay" not in session.client.calls[-1]


@pytest.mark.parametrize("window", [1, 3, 8, 100, True, "8"])
def test_programmatic_configuration_cannot_enable_window(window):
    with pytest.raises(ValueError, match="disabled"):
        AgentConfig(live_image_window=window)
