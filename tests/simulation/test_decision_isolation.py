"""Simulation decision isolation preserves hardware defaults and rejects extra tools."""

import queue

import pytest

from gpt_policy.harness.codex import CodexAppServer


def client(decision_only):
    value = CodexAppServer.__new__(CodexAppServer)
    value.decision_only = decision_only
    value.decision_only_overrides = {
        "features.shell_tool": False,
        "web_search": "disabled",
    }
    value.model = "test"
    value.project_root = "/example"
    value._events = queue.Queue()
    return value


def test_only_simulation_start_disables_environments_and_inherited_mcp():
    strict = client(True)
    calls = []

    def request(method, params):
        calls.append((method, params))
        if method == "config/read":
            return {
                "config": {"mcp_servers": {"example": {"token": "NOT_TO_BE_COPIED"}}}
            }
        return {"thread": {"id": "test"}}

    strict._request = request
    strict.start_thread("policy")
    params = calls[-1][1]
    assert (
        params["environments"]
        == params["dynamicTools"]
        == params["selectedCapabilityRoots"]
        == []
    )
    assert params["config"]["mcp_servers.example.enabled"] is False
    assert "NOT_TO_BE_COPIED" not in str(params)
    normal = client(False)
    calls.clear()
    normal._request = request
    normal.start_thread("hardware")
    assert len(calls) == 1
    assert "environments" not in calls[0][1] and "config" not in calls[0][1]


@pytest.mark.parametrize(
    "kind",
    [
        "commandExecution",
        "mcpToolCall",
        "webSearch",
        "imageView",
        "dynamicToolCall",
        "collabAgentToolCall",
    ],
)
def test_unexpected_native_tool_event_aborts_before_robot_decision(kind):
    strict = client(True)
    strict._events.put({"method": "item/started", "params": {"item": {"type": kind}}})
    with pytest.raises(RuntimeError, match="Decision-only"):
        strict._receive()


@pytest.mark.parametrize(
    "kind", ["agentMessage", "reasoning", "contextCompaction", "userMessage"]
)
def test_expected_items_are_accepted_without_recording_private_content(kind):
    strict = client(True)
    event = {"method": "item/completed", "params": {"item": {"type": kind}}}
    strict._events.put(event)
    assert strict._receive() == event


def test_native_resume_preserves_identity_and_decision_isolation():
    strict = client(True)
    calls = []

    def request(method, params):
        calls.append((method, params))
        if method == "config/read":
            return {"config": {"mcp_servers": {"example": {"secret": "PRIVATE"}}}}
        return {"thread": {"id": "original-thread"}}

    strict._request = request
    strict.resume_thread("original-thread")
    method, params = calls[-1]
    assert method == "thread/resume" and params["threadId"] == "original-thread"
    assert params["excludeTurns"] is True
    assert params["runtimeWorkspaceRoots"] == []
    assert params["config"]["mcp_servers.example.enabled"] is False
    assert params["approvalPolicy"] == "never" and params["sandbox"] == "read-only"
    assert "history" not in params and "PRIVATE" not in str(params)
    assert strict.thread_id == "original-thread"


def test_resume_rejects_changed_identity():
    native = client(False)
    native._request = lambda *args: {"thread": {"id": "different-thread"}}
    with pytest.raises(RuntimeError, match="different thread"):
        native.resume_thread("original-thread")
