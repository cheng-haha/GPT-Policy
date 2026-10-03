import json
from pathlib import Path
from unittest.mock import patch

import pytest

from gpt_policy.harness.config import (
    AGENT_NAMES,
    AgentConfig,
    agent_config,
    load_claude_key,
    named_agent_config,
    save_claude_key,
)
from gpt_policy.harness.factory import create_agent, preflight_agent


def write_agent(directory, name, value):
    agents = directory / "agents"
    agents.mkdir()
    (agents / f"{name}.json").write_text(json.dumps(value), encoding="utf-8")


def test_codex_selector_loads_separate_config(tmp_path):
    write_agent(tmp_path, "codex", {
        "model": "gpt-custom",
        "executable": "/opt/codex",
        "effort": "xhigh",
    })

    assert agent_config({"agent": "codex"}, tmp_path) == AgentConfig(
        "codex", "gpt-custom", "/opt/codex", "xhigh", None,
    )


def test_video_preprocessor_can_load_codex_independently_of_selected_agent(tmp_path):
    write_agent(tmp_path, "codex", {"model": "gpt-video", "effort": "high"})

    assert named_agent_config("codex", tmp_path) == AgentConfig(
        "codex", "gpt-video", "codex", "high", None,
    )


def test_naming_override_loads_without_changing_robot_model(tmp_path):
    write_agent(tmp_path, "codex", {
        "model": "gpt-6-astra", "effort": "medium",
        "task_name_model": "gpt-5.3-codex-spark", "task_name_effort": "low",
    })
    config = agent_config({"agent": "codex"}, tmp_path)
    assert (config.model, config.effort) == ("gpt-6-astra", "medium")
    assert (config.task_name_model, config.task_name_effort) == ("gpt-5.3-codex-spark", "low")


def test_claude_selector_maps_to_claude_code_provider(tmp_path):
    write_agent(tmp_path, "claude", {
        "model": "claude-test",
        "executable": "/opt/claude",
        "effort": "high",
        "timeout_s": 30,
        "base_url": "https://gateway.example.com/",
    })
    credentials = tmp_path / "credentials.json"
    save_claude_key("secret-key", credentials)

    assert agent_config({"agent": "claude"}, tmp_path, credentials) == AgentConfig(
        "claude_code", "claude-test", "/opt/claude", "high", 30.0,
        {
            "ANTHROPIC_BASE_URL": "https://gateway.example.com",
            "ANTHROPIC_AUTH_TOKEN": "secret-key",
        },
    )


def test_repository_main_and_agent_configs_resolve_together(tmp_path):
    configs = Path(__file__).resolve().parents[1] / "configs"
    from gpt_policy.settings import load_settings
    settings = load_settings(configs / "default.json")
    credentials = tmp_path / "credentials.json"
    save_claude_key("secret-key", credentials)

    selected = settings["agent"]
    selected_file = json.loads(
        (configs / "agents" / f"{selected}.json").read_text(encoding="utf-8")
    )
    assert selected in {"codex", "claude"}
    assert agent_config(settings, configs, credentials).model == selected_file["model"]
    assert agent_config({"agent": "codex"}, configs).type == "codex"


def test_claude_key_is_stored_privately_and_not_shown_in_config_repr(tmp_path):
    credentials = save_claude_key("secret-key", tmp_path / "claude.json")

    assert credentials.stat().st_mode & 0o777 == 0o600
    assert load_claude_key(credentials) == "secret-key"
    assert "secret-key" not in repr(AgentConfig(environment={"TOKEN": "secret-key"}))


def test_default_credentials_read_legacy_then_prefer_new_location(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    legacy = tmp_path / "arx-gpt" / "claude.json"
    save_claude_key("old-key", legacy)
    assert load_claude_key() == "old-key"

    current = save_claude_key("new-key")
    assert current == tmp_path / "gpt-policy" / "claude.json"
    assert load_claude_key() == "new-key"
    assert load_claude_key(legacy) == "old-key"


def test_explicit_credentials_path_does_not_fall_back(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    save_claude_key("old-key", tmp_path / "arx-gpt" / "claude.json")
    with pytest.raises(ValueError, match="--set-claude-key"):
        load_claude_key(tmp_path / "missing.json")


def test_claude_key_rejects_permissions_visible_to_other_users(tmp_path):
    credentials = save_claude_key("secret-key", tmp_path / "claude.json")
    credentials.chmod(0o644)

    with pytest.raises(ValueError, match="chmod 600"):
        load_claude_key(credentials)


def test_claude_requires_one_time_key_setup(tmp_path):
    write_agent(tmp_path, "claude", {
        "model": "claude-test",
        "base_url": "https://gateway.example.com",
    })

    with pytest.raises(ValueError, match="--set-claude-key"):
        agent_config({"agent": "claude"}, tmp_path, tmp_path / "missing.json")


@pytest.mark.parametrize("agent", [None, "", "claude_code", "gemini", {"type": "codex"}])
def test_selector_must_be_a_known_profile(tmp_path, agent):
    with pytest.raises(ValueError, match="agent"):
        agent_config({"agent": agent}, tmp_path)


def test_missing_selected_agent_file_is_reported(tmp_path):
    with pytest.raises(ValueError, match="找不到 agent 配置文件"):
        agent_config({"agent": "codex"}, tmp_path)


@pytest.mark.parametrize("value", [
    [],
    {"type": "codex", "model": "gpt"},
    {"model": ""},
    {"model": "gpt", "timeout_s": 180},
    {"task_name_model": ""},
    {"task_name_effort": 2},
])
def test_bad_codex_file_fails_before_starting_hardware(tmp_path, value):
    write_agent(tmp_path, "codex", value)
    with pytest.raises(ValueError):
        agent_config({"agent": "codex"}, tmp_path)


@pytest.mark.parametrize("value", [None, 0, -1, True, float("inf"), float("nan"), "180"])
def test_claude_timeout_must_be_finite_positive_number(tmp_path, value):
    write_agent(tmp_path, "claude", {
        "model": "claude", "timeout_s": value, "base_url": "https://gateway.example.com",
    })
    with pytest.raises(ValueError, match="timeout_s"):
        agent_config({"agent": "claude"}, tmp_path)


def test_claude_model_is_required(tmp_path):
    write_agent(tmp_path, "claude", {"base_url": "https://gateway.example.com"})
    with pytest.raises(ValueError, match="model"):
        agent_config({"agent": "claude"}, tmp_path)


def test_factory_creates_only_the_selected_provider():
    with patch("gpt_policy.harness.providers.codex.CodexAppServer") as native:
        session = create_agent(AgentConfig(), "gpt-test", True, 90)
        native.assert_called_once_with("gpt-test", "codex", "medium", True, 90)
        session.close()


def test_missing_executable_fails_preflight():
    with patch("gpt_policy.harness.factory.shutil.which", return_value=None):
        with pytest.raises(ValueError, match="可执行文件"):
            preflight_agent(AgentConfig(type="claude_code", executable="missing"))


@pytest.mark.parametrize("name", ["claude", "kimi"])
@pytest.mark.parametrize("auth_key", ["ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_API_KEY"])
def test_cli_settings_import_only_connection_fields(tmp_path, monkeypatch, name, auth_key):
    monkeypatch.setenv("HOME", str(tmp_path))
    source = tmp_path / "native.json"
    source.write_text(json.dumps({
        "env": {
            "ANTHROPIC_BASE_URL": "https://selected.example/",
            auth_key: "selected-secret",
            "NO_PROXY": "selected.example",
            "ANTHROPIC_MODEL": "unselected-model",
            "ANTHROPIC_DEFAULT_SONNET_MODEL": "unselected-model",
            "NODE_OPTIONS": "--require untrusted.js",
            "CLAUDE_CONFIG_DIR": "/shared/config",
        },
        "model": "unselected-model",
        "hooks": {"SessionStart": [{"command": "untrusted"}]},
    }))
    write_agent(tmp_path, name, {"model": "selected-model", "settings_file": "~/native.json"})
    before = source.read_bytes()
    config = named_agent_config(name, tmp_path, tmp_path / "missing-legacy-key.json")
    assert config.type == "claude_code" and config.executable == "claude"
    assert config.model == "selected-model"
    assert config.environment == {
        "ANTHROPIC_BASE_URL": "https://selected.example",
        auth_key: "selected-secret", "NO_PROXY": "selected.example",
    }
    assert "selected-secret" not in repr(config)
    assert source.read_bytes() == before
    from gpt_policy.harness.providers.claude_code.session import ClaudeCodeSession
    assert isinstance(create_agent(config, config.model), ClaudeCodeSession)


def test_settings_file_resolves_relative_to_agent_json(tmp_path):
    write_agent(tmp_path, "kimi", {"model": "kimi", "settings_file": "connection.json"})
    (tmp_path / "agents/connection.json").write_text(json.dumps({"env": {
        "ANTHROPIC_BASE_URL": "https://kimi.example", "ANTHROPIC_AUTH_TOKEN": "kimi-secret",
    }}))
    assert named_agent_config("kimi", tmp_path).environment["ANTHROPIC_AUTH_TOKEN"] == "kimi-secret"


@pytest.mark.parametrize("contents", [
    "not JSON", "[]", "{}", '{"env":[]}',
    '{"env":{"ANTHROPIC_BASE_URL":"https://example"}}',
    '{"env":{"ANTHROPIC_BASE_URL":"https://example","ANTHROPIC_AUTH_TOKEN":123}}',
    '{"env":{"ANTHROPIC_BASE_URL":"https://example","ANTHROPIC_AUTH_TOKEN":""}}',
    '{"env":{"ANTHROPIC_BASE_URL":"https://example","ANTHROPIC_AUTH_TOKEN":"token","ANTHROPIC_API_KEY":"key"}}',
])
def test_invalid_settings_file_cannot_fall_back_to_another_credential(tmp_path, contents):
    source = tmp_path / "native.json"
    source.write_text(contents)
    write_agent(tmp_path, "kimi", {"model": "kimi", "settings_file": str(source)})
    with pytest.raises(ValueError, match="settings_file"):
        named_agent_config("kimi", tmp_path)


@pytest.mark.parametrize("name,value", [
    ("codex", {"settings_file": "native.json"}),
    ("kimi", {"model": "kimi", "base_url": "https://example"}),
    ("claude", {"model": "claude", "base_url": "https://example", "settings_file": "native.json"}),
    ("kimi", {"model": "kimi", "settings_file": "missing.json"}),
])
def test_settings_source_must_be_explicit_and_unambiguous(tmp_path, name, value):
    write_agent(tmp_path, name, value)
    with pytest.raises(ValueError, match="settings_file"):
        named_agent_config(name, tmp_path)


def test_repository_native_profiles_resolve_without_legacy_credentials(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    for directory, token in [(".claude", "claude-key"), (".claude-volcengine", "kimi-key")]:
        source = tmp_path / directory / "settings.json"
        source.parent.mkdir()
        source.write_text(json.dumps({"env": {
            "ANTHROPIC_BASE_URL": "https://example", "ANTHROPIC_AUTH_TOKEN": token,
        }}))
    configs = Path(__file__).resolve().parents[1] / "configs"
    assert AGENT_NAMES == ("codex", "claude", "kimi")
    for name in ["claude", "kimi"]:
        config = named_agent_config(name, configs, tmp_path / "missing.json")
        assert config.environment["ANTHROPIC_AUTH_TOKEN"] == f"{name}-key"
        assert config.type == "claude_code"
