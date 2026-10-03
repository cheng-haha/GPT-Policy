import base64
import json
import os
from unittest.mock import patch

import pytest

from gpt_policy.hardware.camera import CapturedImage
from gpt_policy.harness.config import AgentConfig
from gpt_policy.harness.errors import AgentProtocolError
from gpt_policy.harness.models import AgentTurn
from gpt_policy.harness.providers.gemini_cli.session import GeminiCliSession
from gpt_policy.input import TextPart


def session():
    return GeminiCliSession(AgentConfig("gemini_cli", "gemini-test", "gemini", None, 30),
                            "gemini-test", False, 85)


def reply(request_id, result):
    return {"jsonrpc": "2.0", "id": request_id, "result": result}


def startup(image=True):
    return [reply(1, {"protocolVersion": 1, "agentCapabilities": {"promptCapabilities": {"image": image}}}),
            reply(2, {"sessionId": "session-1"})]


def update(kind, text="", session_id="session-1"):
    return {"jsonrpc": "2.0", "method": "session/update", "params": {
        "sessionId": session_id, "update": {"sessionUpdate": kind, "content": {"type": "text", "text": text}},
    }}


def test_persistent_session_native_images_and_fragmented_output(context, selection):
    serialized = json.dumps(selection)
    captured = CapturedImage("left", b"live", "image/png", 1, 1, 0)
    with patch("gpt_policy.harness.providers.gemini_cli.session.JsonProcess") as process:
        process.return_value.receive.side_effect = startup() + [
            update("agent_message_chunk", "ignore", "other-session"),
            update("agent_thought_chunk", "reasoning"),
            update("agent_message_chunk", serialized[:25]),
            update("agent_message_chunk", serialized[25:]),
            reply(3, {"stopReason": "end_turn"}),
            update("agent_message_chunk", serialized), reply(4, {"stopReason": "end_turn"}),
        ]
        original_environment = dict(os.environ)
        agent = session()
        agent.start(context)
        workspace = agent.workspace.path
        try:
            assert agent.decide(AgentTurn("first", {"left": captured}, (TextPart("/task"),)))["name"] == "done"
            agent.decide(AgentTurn("second", {"left": captured}))
            process.assert_called_once()
            sent = [call.args[0] for call in process.return_value.send.call_args_list]
            assert [x["method"] for x in sent] == ["initialize", "session/new", "session/prompt", "session/prompt"]
            assert all(x["params"]["sessionId"] == "session-1" for x in sent[2:])
            prompt = sent[2]["params"]["prompt"]
            assert not prompt[0]["text"].startswith("/")
            assert prompt[1]["text"] == "/task"
            assert prompt[-1]["type"] == "image"
            assert base64.b64decode(prompt[-1]["data"]) == b"live"
            assert sent[3]["params"]["prompt"][1]["text"] == "second"
            command, cwd, environment = process.call_args.args
            assert "--acp" in command and "--admin-policy" in command
            assert cwd == workspace
            settings = json.loads(agent.workspace.settings.read_text())
            assert settings["tools"]["core"] == []
            assert settings["hooksConfig"]["enabled"] is False
            assert settings["admin"]["mcp"]["enabled"] is False
            assert environment["GEMINI_SYSTEM_MD"] == str(agent.workspace.system_prompt)
            system_prompt = agent.workspace.system_prompt.read_text()
            assert "native Codex harness" not in system_prompt
            assert system_prompt.count(context.output_schema["description"]) == 1
            schema_text = system_prompt.split(
                "Return exactly one JSON object matching this schema:\n", 1,
            )[1]
            assert json.loads(schema_text) == context.output_schema
            assert dict(os.environ) == original_environment
        finally:
            agent.close()
        assert not workspace.exists()
        process.return_value.close.assert_called_once()


def test_missing_image_capability_fails_start_and_cleans_up(context):
    with patch("gpt_policy.harness.providers.gemini_cli.session.JsonProcess") as process:
        process.return_value.receive.side_effect = startup(False)
        agent = session()
        with pytest.raises(AgentProtocolError, match="图片"):
            agent.start(context)
        assert agent.workspace is None and agent.process is None
        process.return_value.close.assert_called_once()


@pytest.mark.parametrize("event", [
    update("tool_call"), reply(3, {"stopReason": "cancelled"}),
    {"jsonrpc": "2.0", "id": 3, "error": {"code": -32000, "message": "failed"}},
    reply(99, {}),
])
def test_failed_turn_never_produces_a_robot_decision(context, event):
    with patch("gpt_policy.harness.providers.gemini_cli.session.JsonProcess") as process:
        process.return_value.receive.side_effect = startup() + [event]
        agent = session()
        agent.start(context)
        with pytest.raises(AgentProtocolError):
            agent.decide(AgentTurn("observation"))
        assert agent.process is None
        process.return_value.close.assert_called_once()


def test_permission_request_is_explicitly_cancelled(context):
    with patch("gpt_policy.harness.providers.gemini_cli.session.JsonProcess") as process:
        process.return_value.receive.side_effect = startup() + [{
            "jsonrpc": "2.0", "id": "permission-1", "method": "session/request_permission",
            "params": {"sessionId": "session-1", "options": []},
        }]
        agent = session()
        agent.start(context)
        with pytest.raises(AgentProtocolError):
            agent.decide(AgentTurn("observation"))
        assert process.return_value.send.call_args.args[0] == reply(
            "permission-1", {"outcome": {"outcome": "cancelled"}},
        )
