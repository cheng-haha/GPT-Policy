"""Read-only context requests may reconnect on timeout without changing input."""

import runpy
from pathlib import Path
from unittest.mock import Mock

import pytest

from gpt_policy.harness.errors import AgentTimeoutError, AgentUsageLimitError

MODULE = runpy.run_path(
    str(Path(__file__).resolve().parents[1] / "scripts/prepare_robodojo_contexts.py")
)


def client(monkeypatch, failures):
    waits = []
    monkeypatch.setattr(MODULE["time"], "sleep", waits.append)
    sessions = [Mock() for _ in failures]
    for session, result in zip(sessions, failures):
        if isinstance(result, Exception):
            session.decide.side_effect = result
        else:
            session.decide.return_value = result
    value = object.__new__(MODULE["ContextClient"])
    value.instructions = "read-only historical context"
    value.session = sessions[0]
    value.factory = Mock(side_effect=sessions[1:])
    return value, sessions, waits


def test_context_timeout_retries_identical_request_in_new_session(monkeypatch):
    answer = {"name": "inspect_context", "arguments": {}}
    value, sessions, waits = client(monkeypatch, [AgentTimeoutError("timeout"), answer])
    schema, images, content = {"type": "object"}, {}, ()
    assert value.decide("same observation", schema, images, content) == answer
    assert waits == [15] and value.factory.call_count == 1
    sessions[0].close.assert_called_once()
    assert sessions[0].decide.call_args == sessions[1].decide.call_args
    assert sessions[0].start.call_args == sessions[1].start.call_args
    assert sessions[1].decide.call_args.args[0].observation == "same observation"


def test_context_timeout_stops_after_three_attempts(monkeypatch):
    value, sessions, waits = client(
        monkeypatch, [AgentTimeoutError("timeout") for _ in range(3)]
    )
    with pytest.raises(AgentTimeoutError):
        value.decide("same observation", {})
    assert waits == [15, 15] and value.factory.call_count == 2
    assert all(s.decide.call_count == 1 and s.close.call_count == 1 for s in sessions)


def test_context_quota_never_reconnects_or_resets(monkeypatch):
    quota = AgentUsageLimitError("quota", provider="codex", code="usageLimitExceeded")
    value, _sessions, waits = client(monkeypatch, [quota])
    with pytest.raises(AgentUsageLimitError):
        value.decide("same observation", {})
    value.factory.assert_not_called()
    assert waits == []
