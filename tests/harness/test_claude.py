import base64
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from gpt_policy.hardware.camera import CapturedImage
from gpt_policy.harness.config import AgentConfig
from gpt_policy.harness.errors import AgentProtocolError, AgentTimeoutError
from gpt_policy.harness.models import AgentTurn
from gpt_policy.harness.providers.claude_code.session import ClaudeCodeSession
from gpt_policy.input import ImagePart, TextPart


def session():
    config = AgentConfig(
        "claude_code", "sonnet", "claude", None, 30,
        {"ANTHROPIC_BASE_URL": "https://gateway.example.com", "ANTHROPIC_AUTH_TOKEN": "secret"},
    )
    return ClaudeCodeSession(config, "sonnet", False, 85)


@patch.dict("os.environ", {
    "CLAUDE_CODE_USE_BEDROCK": "1",
    "ANTHROPIC_API_KEY": "stale-key",
    "ANTHROPIC_MODEL": "stale-model",
})
def test_two_turns_keep_one_process_and_forward_ordered_images(context, selection, tmp_path):
    image = tmp_path / "reference.png"
    image.write_bytes(b"reference")
    captured = CapturedImage("left", b"live", "image/png", 1, 1, 0)
    with patch("gpt_policy.harness.providers.claude_code.session.JsonProcess") as process:
        process.return_value.receive.side_effect = [
            {"type": "system", "subtype": "init"},
            {"type": "result", "subtype": "success", "structured_output": selection},
            {"type": "result", "subtype": "success", "structured_output": selection},
        ]
        agent = session()
        agent.start(context)
        try:
            command = process.call_args.args[0]
            assert command[command.index("--tools") + 1] == ""
            assert "--safe-mode" in command and "--strict-mcp-config" in command
            assert "--no-session-persistence" in command
            assert json.loads(command[command.index("--json-schema") + 1]) == context.output_schema
            prompt = command[command.index("--system-prompt") + 1]
            assert "工具选择输出 schema" not in prompt
            assert context.output_schema["description"] not in prompt
            environment = process.call_args.kwargs["env"]
            assert environment["ANTHROPIC_BASE_URL"] == "https://gateway.example.com"
            assert environment["ANTHROPIC_AUTH_TOKEN"] == "secret"
            assert "CLAUDE_CODE_USE_BEDROCK" not in environment
            assert "ANTHROPIC_API_KEY" not in environment
            assert "ANTHROPIC_MODEL" not in environment
            agent.decide(AgentTurn("first", {"left": captured}, (TextPart("task"), ImagePart(image))))
            agent.decide(AgentTurn("second", {"left": captured}))
            process.assert_called_once()
            messages = [call.args[0]["message"]["content"] for call in process.return_value.send.call_args_list]
            assert [x["type"] for x in messages[0]] == ["text", "image", "text", "text", "image"]
            assert base64.b64decode(messages[0][1]["source"]["data"]) == b"reference"
            assert base64.b64decode(messages[0][-1]["source"]["data"]) == b"live"
            assert messages[1][0] == {"type": "text", "text": "second"}
            assert len(messages[1]) == 3
        finally:
            agent.close()
        process.return_value.close.assert_called_once()


@pytest.mark.parametrize("event", [
    {"type": "result", "subtype": "error_max_turns", "is_error": True},
    {"type": "result", "subtype": "success", "structured_output": []},
    {"type": "assistant", "message": {"content": [{"type": "tool_use", "name": "Bash"}]}},
    {"type": "control_request", "request_id": "request-1"},
])
def test_invalid_or_native_tool_output_closes_session(context, event):
    with patch("gpt_policy.harness.providers.claude_code.session.JsonProcess") as process:
        process.return_value.receive.return_value = event
        agent = session()
        agent.start(context)
        with pytest.raises(AgentProtocolError):
            agent.decide(AgentTurn("observation"))
        assert agent.process is None and agent.workspace is None
        process.return_value.close.assert_called_once()


def test_timeout_closes_process(context):
    with patch("gpt_policy.harness.providers.claude_code.session.JsonProcess") as process:
        process.return_value.receive.side_effect = AgentTimeoutError("timeout")
        agent = session()
        agent.start(context)
        with pytest.raises(AgentTimeoutError):
            agent.decide(AgentTurn("observation"))
        process.return_value.close.assert_called_once()


def test_profiles_isolate_routing_and_config_directories(context, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://wrong.example")
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "wrong-token")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "wrong-key")
    monkeypatch.setenv("ANTHROPIC_DEFAULT_SONNET_MODEL", "wrong-model")
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "wrong-oauth")
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", "/shared/config")
    agents = [session(), session()]
    with patch("gpt_policy.harness.providers.claude_code.session.JsonProcess") as process:
        try:
            for agent in agents:
                agent.start(context)
            environments = [call.kwargs["env"] for call in process.call_args_list]
            directories = [Path(env["CLAUDE_CONFIG_DIR"]) for env in environments]
            assert directories[0] != directories[1]
            for env, directory, agent in zip(environments, directories, agents):
                assert directory.is_dir() and directory.is_relative_to(agent.workspace.name)
                assert directory.stat().st_mode & 0o777 == 0o700
                assert env["ANTHROPIC_AUTH_TOKEN"] == "secret"
                assert env["ANTHROPIC_BASE_URL"] == "https://gateway.example.com"
                assert "ANTHROPIC_API_KEY" not in env
                assert "ANTHROPIC_DEFAULT_SONNET_MODEL" not in env
                assert "CLAUDE_CODE_OAUTH_TOKEN" not in env
        finally:
            for agent in agents:
                agent.close()
        assert all(not directory.exists() for directory in directories)


def test_failed_start_cleans_private_configuration(context):
    with patch("gpt_policy.harness.providers.claude_code.session.JsonProcess", side_effect=OSError("cannot start")) as process:
        agent = session()
        with pytest.raises(OSError, match="cannot start"):
            agent.start(context)
        assert not Path(process.call_args.kwargs["env"]["CLAUDE_CONFIG_DIR"]).exists()
        assert agent.workspace is None and agent.process is None
