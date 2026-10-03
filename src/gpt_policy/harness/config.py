"""Load the selected agent from its provider-specific JSON file."""

from __future__ import annotations

import json
import math
import os
import tempfile
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping

AGENT_NAMES = ("codex", "claude", "kimi")
_PROVIDER_TYPES = {"codex": "codex", "claude": "claude_code", "kimi": "claude_code"}

# Import connection settings only; CLI tools, hooks, model aliases and arbitrary
# environment entries must not become part of the host's decision session.
_CONNECTION_ENVIRONMENT = frozenset({
    "ANTHROPIC_BASE_URL", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_API_KEY",
    "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY",
    "http_proxy", "https_proxy", "all_proxy", "no_proxy",
})


@dataclass(frozen=True)
class AgentConfig:
    type: str = "codex"
    model: str | None = None
    executable: str = "codex"
    effort: str | None = "medium"
    timeout_s: float | None = None
    environment: Mapping[str, str] = field(default_factory=dict, repr=False)
    task_name_model: str | None = None
    task_name_effort: str | None = None
    live_image_window: int | None = 8


def _validate_live_image_window(window: int | None) -> None:
    if window is not None and (isinstance(window, bool) or not isinstance(window, int) or window < 3):
        raise ValueError("live_image_window must be an integer >= 3 or null")


def with_live_image_window(config: AgentConfig, window: int | None) -> AgentConfig:
    """Apply a per-run override without rewriting the shared agent profile."""
    if config.type != "codex":
        raise ValueError("live_image_window is only supported by Codex")
    _validate_live_image_window(window)
    return replace(config, live_image_window=window)


def agent_config(
    settings: dict[str, Any],
    base_dir: Path,
    credentials_path: Path | None = None,
) -> AgentConfig:
    """Resolve ``agent`` through ``<base_dir>/agents/<agent>.json``."""

    selected = _text(settings.get("agent"), "agent")
    return named_agent_config(selected, Path(settings.get("agent_config_dir", base_dir)), credentials_path)


def named_agent_config(
    selected: str,
    base_dir: Path,
    credentials_path: Path | None = None,
) -> AgentConfig:
    """Load one named agent independently of the main selector."""

    if selected not in AGENT_NAMES:
        raise ValueError(f"不支持的 agent: {selected}; 可选: {', '.join(AGENT_NAMES)}")

    filename = base_dir / "agents" / f"{selected}.json"
    try:
        with filename.open(encoding="utf-8") as stream:
            value = json.load(stream)
    except FileNotFoundError as exc:
        raise ValueError(f"找不到 agent 配置文件: {filename}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"agent 配置文件不是有效 JSON: {filename}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"agent 配置文件根节点必须是对象: {filename}")

    unknown = set(value) - {"model", "executable", "effort", "timeout_s", "base_url", "settings_file", "task_name_model", "task_name_effort", "live_image_window"}
    if unknown:
        raise ValueError(f"{filename} 中存在未知字段: {sorted(unknown)}")

    kind = _PROVIDER_TYPES[selected]
    codex = selected == "codex"
    window = value.get("live_image_window", 8)
    if "live_image_window" in value and not codex:
        raise ValueError("live_image_window is only supported by Codex")
    _validate_live_image_window(window)
    model = value.get("model")
    if model is not None:
        model = _text(model, f"{selected}.model")
    elif not codex:
        raise ValueError(f"{selected}.model 必须显式设置")
    executable = _text(
        value.get("executable", "codex" if codex else "claude"),
        f"{selected}.executable",
    )
    effort = value.get("effort", "medium")
    if effort is not None:
        effort = _text(effort, f"{selected}.effort")
    naming = {
        key: _text(value[key], f"{selected}.{key}") if value.get(key) is not None else None
        for key in ("task_name_model", "task_name_effort")
    }
    timeout = value.get("timeout_s", None if codex else 180.0)
    if timeout is not None:
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
            raise ValueError(f"{selected}.timeout_s 必须是正数")
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError(f"{selected}.timeout_s 必须是有限正数")
        timeout = float(timeout)
    if codex and timeout is not None:
        raise ValueError("Codex 沿用原生 app-server 等待行为，请省略 codex.timeout_s")
    if not codex and timeout is None:
        raise ValueError(f"{selected}.timeout_s 必须是有限正数")

    environment: Mapping[str, str] = {}
    if codex:
        if set(value) & {"base_url", "settings_file"}:
            raise ValueError("codex.json 不支持 base_url 或 settings_file")
    elif "settings_file" in value:
        if "base_url" in value:
            raise ValueError(f"{selected}.settings_file 与 base_url 不能同时设置")
        source = Path(_text(value["settings_file"], f"{selected}.settings_file")).expanduser()
        if not source.is_absolute():
            source = filename.parent / source
        environment = _claude_environment(source)
    elif selected == "kimi":
        raise ValueError("kimi.settings_file 必须指向 Kimi 的 Claude Code 配置")
    else:
        base_url = _text(value.get("base_url"), "claude.base_url").rstrip("/")
        api_key = load_claude_key(credentials_path)
        environment = {
            "ANTHROPIC_BASE_URL": base_url,
            "ANTHROPIC_AUTH_TOKEN": api_key,
        }
    return AgentConfig(kind, model, executable, effort, timeout, environment, **naming, live_image_window=window)


def _claude_environment(filename: Path) -> dict[str, str]:
    """Read a selected CLI's connection settings without loading customizations."""
    try:
        value = json.loads(filename.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"无法读取 Claude Code settings_file: {filename}") from exc
    except (json.JSONDecodeError, UnicodeError) as exc:
        raise ValueError(f"Claude Code settings_file 不是有效 JSON: {filename}") from exc
    if not isinstance(value, dict) or not isinstance(value.get("env"), dict):
        raise ValueError(f"Claude Code settings_file 必须包含 env 对象: {filename}")
    environment = {
        key: _text(item, f"settings_file.env.{key}")
        for key, item in value["env"].items() if key in _CONNECTION_ENVIRONMENT
    }
    environment["ANTHROPIC_BASE_URL"] = _text(
        environment.get("ANTHROPIC_BASE_URL"), "settings_file.env.ANTHROPIC_BASE_URL",
    ).rstrip("/")
    auth = set(environment) & {"ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_API_KEY"}
    if len(auth) != 1:
        raise ValueError("settings_file.env 必须设置且只设置一个 ANTHROPIC_AUTH_TOKEN 或 ANTHROPIC_API_KEY")
    return environment


def claude_credentials_path() -> Path:
    """Return the user-private credential file outside the repository."""

    configured = os.environ.get("XDG_CONFIG_HOME")
    root = Path(configured).expanduser() if configured else Path.home() / ".config"
    return root / "gpt-policy" / "claude.json"


def load_claude_key(path: Path | None = None) -> str:
    filename = path or claude_credentials_path()
    # Existing ARX installations can keep their saved key until it is replaced.
    if path is None and not filename.exists():
        filename = filename.parent.parent / "arx-gpt" / "claude.json"
    try:
        mode = filename.stat().st_mode
        with filename.open(encoding="utf-8") as stream:
            value = json.load(stream)
    except FileNotFoundError as exc:
        raise ValueError(
            "尚未配置 Claude API Key；请运行 gpt-policy --set-claude-key"
        ) from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Claude 凭证文件不是有效 JSON: {filename}: {exc}") from exc
    if mode & 0o077:
        raise ValueError(f"Claude 凭证文件权限过宽，请执行: chmod 600 {filename}")
    if not isinstance(value, dict) or set(value) != {"api_key"}:
        raise ValueError(f"Claude 凭证文件只能包含 api_key: {filename}")
    return _text(value["api_key"], "Claude API Key")


def save_claude_key(api_key: str, path: Path | None = None) -> Path:
    """Atomically save a Claude key in a mode-0600 user file."""

    key = _text(api_key, "Claude API Key")
    filename = path or claude_credentials_path()
    filename.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(filename.parent, 0o700)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=filename.parent, prefix=".claude-", suffix=".tmp", text=True,
    )
    temporary = Path(temporary_name)
    try:
        os.chmod(temporary, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump({"api_key": key}, stream)
            stream.write("\n")
        os.replace(temporary, filename)
        os.chmod(filename, 0o600)
    except BaseException:
        try:
            os.close(descriptor)
        except OSError:
            pass
        temporary.unlink(missing_ok=True)
        raise
    return filename


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} 必须是非空字符串")
    return value.strip()
