"""Freeze the established app-server wire protocol before adding adapters."""

import json
import signal
import subprocess
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from gpt_policy.harness.codex import CodexAppServer
from gpt_policy.harness.config import AgentConfig
from gpt_policy.harness.errors import AgentOverloadedError, AgentUsageLimitError
from gpt_policy.harness.models import AgentContext, AgentTurn
from gpt_policy.harness.providers.codex import CodexSession
from gpt_policy.harness.prompts import robot_instructions
from gpt_policy.input import TextPart


def client():
    result = object.__new__(CodexAppServer)
    result.model = "gpt-test"
    result.effort = "high"
    result.project_root = Path("/robot")
    result.thread_id = "thread-1"
    result.convert_camera_images_to_jpeg = False
    result.camera_jpeg_quality = 85
    result._request = Mock(return_value={"turn": {"id": "turn-1"}})
    return result


def final_events(answer):
    return [
        {"method": "item/completed", "params": {
            "threadId": "unrelated", "item": {"type": "agentMessage", "text": "ignore"},
        }},
        {"method": "item/completed", "params": {
            "threadId": "thread-1", "item": {
                "type": "agentMessage", "phase": "commentary", "text": "ignore",
            },
        }},
        {"method": "item/completed", "params": {
            "threadId": "thread-1", "item": {
                "type": "agentMessage", "phase": "final_answer", "text": answer,
            },
        }},
        {"method": "turn/completed", "params": {
            "threadId": "thread-1", "turn": {"id": "turn-1", "status": "completed"},
        }},
    ]


def test_native_thread_settings_and_prompt_are_unchanged():
    native = client()
    native._request.return_value = {"thread": {"id": "new-thread"}}
    tools = [{"type": "function", "function": {"name": "done"}}]
    native.start_thread("机器人提示", tools)
    native._request.assert_called_once_with("thread/start", {
        "model": "gpt-test", "cwd": "/robot", "approvalPolicy": "never",
        "sandbox": "read-only", "ephemeral": True, "serviceName": "gpt-policy",
        "baseInstructions": "机器人提示\n\nRobot tool catalog:\n" + json.dumps(tools, ensure_ascii=False),
    })
    assert native.thread_id == "new-thread"


def test_output_schema_is_sent_with_turn_instead_of_base_instructions(context, selection):
    native = client()
    native._request.side_effect = [
        {"thread": {"id": "thread-1"}}, {"turn": {"id": "turn-1"}},
    ]
    native._receive = Mock(side_effect=final_events(json.dumps(selection)))

    native.start_thread(context.instructions, context.tools)
    native.decide("observation", context.output_schema)

    start, turn = native._request.call_args_list
    assert start.args[1]["baseInstructions"] == (
        context.instructions + "\n\nRobot tool catalog:\n"
        + json.dumps(context.tools, ensure_ascii=False)
    )
    assert "Codex 外层输出 schema" not in start.args[1]["baseInstructions"]
    assert turn.args[1]["outputSchema"] == context.output_schema


@pytest.mark.parametrize("legacy", [False, True])
def test_native_decision_preserves_wire_and_legacy_arguments(legacy):
    native = client()
    arguments = {"summary": "完成"}
    selection = {"name": "done", "arguments": json.dumps(arguments) if legacy else arguments}
    native._receive = Mock(side_effect=final_events(json.dumps(selection)))
    schema = {"type": "object"}
    decision = native.decide("实时观察", schema, content=(TextPart("静态输入"),))
    assert decision == {"name": "done", "arguments": arguments, "_wire": selection}
    native._request.assert_called_once_with("turn/start", {
        "threadId": "thread-1", "input": [
            {"type": "text", "text": "静态输入"}, {"type": "text", "text": "实时观察"},
        ], "outputSchema": schema, "effort": "high",
    })


@pytest.mark.parametrize("failed", [False, True])
def test_client_timing_separates_input_preparation_from_request_wait(monkeypatch, failed):
    clock = [0.0]
    monkeypatch.setattr("gpt_policy.harness.codex.time.perf_counter", lambda: clock[0])
    native = client()
    original_input = native._input

    def inputs(*args):
        clock[0] += 2
        return original_input(*args)

    events = iter(final_events('{"name":"done","arguments":{"summary":"ok"}}'))

    def receive():
        clock[0] += 1
        if failed:
            raise RuntimeError("provider disconnected")
        return next(events)

    native._input = inputs
    native._receive = receive
    if failed:
        with pytest.raises(RuntimeError, match="disconnected"):
            native.decide("observation", {})
    else:
        native.decide("observation", {})
    assert native.last_request_timing == {
        "input_prepare_s": 2.0, "request_elapsed_s": 1.0 if failed else 4.0,
    }


@pytest.mark.parametrize("selection", [[], {"name": "done", "arguments": []}])
def test_native_invalid_output_still_fails(selection):
    native = client()
    native._receive = Mock(side_effect=final_events(json.dumps(selection)))
    with pytest.raises(RuntimeError):
        native.decide("{}", {})


def test_adapter_forwards_original_objects_without_extra_validation():
    context = AgentContext("same prompt", [], {"type": "object"})
    turn = AgentTurn("same observation", {}, (TextPart("task"),))
    with patch("gpt_policy.harness.providers.codex.CodexAppServer") as constructor:
        session = CodexSession(AgentConfig(executable="custom-codex", effort="medium"),
                               "gpt-test", True, 82)
        constructor.assert_called_once_with("gpt-test", "custom-codex", "medium", True, 82)
        session.start(context)
        constructor.return_value.start_thread.assert_called_once_with(context.instructions, context.tools)
        decision = session.decide(turn)
        assert decision is constructor.return_value.decide.return_value
        constructor.return_value.decide.assert_called_once_with(
            turn.observation, context.output_schema, turn.images, turn.content,
        )
        session.close()
        constructor.return_value.close.assert_called_once_with()


def test_codex_prompt_is_not_rewritten(context):
    assert robot_instructions(context.instructions, "codex") is context.instructions
    assert "native Codex harness" in context.instructions
    for provider, name in (("claude_code", "Claude Code"), ("gemini_cli", "Gemini CLI")):
        adapted = robot_instructions(context.instructions, provider)
        assert "native Codex harness" not in adapted
        assert f"{name} harness" in adapted


def test_oversize_input_fails_before_turn_start():
    native = client()
    with pytest.raises(ValueError, match="input.*too large"):
        native.decide("live", {}, content=(TextPart("x" * 1048576),))
    native._request.assert_not_called()


def test_close_bounds_wait_for_a_provider_ignoring_termination():
    native = client()
    native.process = process = Mock(pid=12345)
    process.poll.return_value = None
    process.wait.side_effect = [subprocess.TimeoutExpired("codex", 2), 0]
    with patch("os.killpg") as kill:
        native.close()
        native.close()
    assert [call.args for call in kill.call_args_list] == [(12345, signal.SIGTERM), (12345, signal.SIGKILL)]
    assert [call.kwargs for call in process.wait.call_args_list] == [{"timeout": 2}, {"timeout": 2}]


@pytest.mark.parametrize("code,status,retryable", [
    ("serverOverloaded", "failed", True),
    ("serverOverloaded", "interrupted", False),
    ("usageLimitExceeded", "failed", False),
    ("usageLimitExceeded", "interrupted", False),
    ("contextWindowExceeded", "failed", False),
    ("unauthorized", "failed", False),
    (None, "failed", False),
])
def test_only_explicitly_failed_overloaded_turn_is_retryable(code, status, retryable):
    native = client()
    events = final_events('{"name":"move_to","arguments":{"note":"partial output"}}')
    events[-1]['params']['turn'].update(status=status, error={
        'message': 'Selected model is at capacity. Please try a different model.',
        'codexErrorInfo': code,
    })
    native._receive = Mock(side_effect=events)
    with pytest.raises(RuntimeError) as caught:
        native.decide('observation', {})
    assert isinstance(caught.value, AgentOverloadedError) == retryable
    assert isinstance(caught.value, AgentUsageLimitError) == (code == 'usageLimitExceeded' and status == 'failed')
    if isinstance(caught.value, AgentUsageLimitError):
        assert caught.value.provider == 'codex'
        assert caught.value.code == 'usageLimitExceeded'
    if retryable:
        assert caught.value.provider == 'codex'
        assert caught.value.code == 'serverOverloaded'
    native._request.assert_called_once()  # Recovery belongs to the observing host, not transport.


def test_stale_turn_answer_cannot_supply_a_new_decision():
    native = client()
    events = final_events('{"name":"move_to","arguments":{}}')
    for event in events[:-1]:
        event['params']['turnId'] = 'previous-turn'
    native._receive = Mock(side_effect=events)
    with pytest.raises(RuntimeError, match='没有返回工具选择'):
        native.decide('observation', {})
