import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from gpt_policy.harness.codex import CodexAppServer
from gpt_policy.harness.config import AgentConfig
from gpt_policy.harness.errors import AgentProtocolError
from gpt_policy.harness.models import AgentTurn
from gpt_policy.harness.providers.claude_code.session import ClaudeCodeSession
from gpt_policy.harness.usage import (
    collect_usage, estimate_cost, model_call, normalize_usage, usage_phase,
)
from gpt_policy.recording.trace import RunRecorder


RAW = {"inputTokens": 10000, "cachedInputTokens": 8000, "outputTokens": 1000,
       "reasoningOutputTokens": 400, "totalTokens": 11000}


def native():
    client = object.__new__(CodexAppServer)
    client.model, client.effort, client.thread_id = "gpt-6-astra", "medium", "thread-1"
    client._next_id = 0
    client._send = Mock()
    client.convert_camera_images_to_jpeg, client.camera_jpeg_quality = False, 85
    return client


def usage_event(turn="turn-1", thread="thread-1", raw=None):
    return {"method": "thread/tokenUsage/updated", "params": {
        "threadId": thread, "turnId": turn,
        "tokenUsage": {"last": raw or RAW, "total": {"inputTokens": 999999}},
    }}


def finish(turn="turn-1", status="completed"):
    return {"method": "turn/completed", "params": {
        "threadId": "thread-1", "turn": {"id": turn, "status": status},
    }}


def answer():
    return {"method": "item/completed", "params": {"threadId": "thread-1", "item": {
        "type": "agentMessage", "text": '{"name":"done","arguments":{"summary":"ok"}}',
    }}}


def test_cached_and_reasoning_tokens_are_not_charged_twice():
    usage = normalize_usage(RAW, "codex")
    assert usage["total_tokens"] == 11000
    assert estimate_cost("gpt-6-astra", usage)["estimated_cost_usd"] == pytest.approx(.078)
    usage = normalize_usage({**RAW, "cacheWriteInputTokens": 1000}, "codex")
    assert estimate_cost("gpt-6-astra", usage)["estimated_cost_usd"] == pytest.approx(.0805)


def test_long_context_uses_whole_request_and_strict_threshold():
    usage = normalize_usage({"inputTokens": 300000, "cachedInputTokens": 200000, "outputTokens": 2000}, "codex")
    price = estimate_cost("gpt-6-astra", usage)
    assert price["long_context_pricing"] and price["estimated_cost_usd"] == pytest.approx(2.55)
    usage["input_tokens"] = 272000
    assert not estimate_cost("gpt-6-astra", usage)["long_context_pricing"]


def test_claude_cache_writes_are_added_to_input_and_use_their_own_rates():
    usage = normalize_usage({"input_tokens": 10000, "cache_read_input_tokens": 20000,
        "cache_creation_input_tokens": 5000, "cache_creation": {"ephemeral_1h_input_tokens": 1000},
        "output_tokens": 1000}, "claude_code")
    assert usage["input_tokens"] == 35000 and usage["total_tokens"] == 36000
    assert estimate_cost("claude-fable-5-1[1m]", usage)["estimated_cost_usd"] == pytest.approx(.225)
    assert estimate_cost("kimi-k3[1m]", usage)["estimated_cost_usd"] is None


@pytest.mark.parametrize("model", ["gpt-5.3-codex-spark", "unknown-model"])
def test_unpriced_models_are_unknown_not_free(model):
    assert estimate_cost(model, normalize_usage(RAW, "codex"))["estimated_cost_usd"] is None


def test_codex_usage_before_rpc_reply_duplicates_stale_turns_and_refresh(tmp_path):
    client = native()
    client._receive = Mock(side_effect=[
        usage_event(), {"id": 1, "result": {"turn": {"id": "turn-1"}}},
        usage_event(), usage_event(thread="other"), usage_event(turn="stale"), answer(), finish(),
        # A late previous-turn update arrives while the next RPC is pending.
        usage_event(), {"id": 2, "result": {"turn": {"id": "turn-2"}}},
        usage_event(turn="turn-2"), answer(), finish("turn-2"),
    ])
    with collect_usage() as ledger:
        ledger.attach(tmp_path)
        first = client.decide("one", {})
        client.decide("two", {})
        assert first == {"name": "done", "arguments": {"summary": "ok"},
                         "_wire": {"name": "done", "arguments": {"summary": "ok"}}}
        assert ledger.summary()["calls"] == 2
        assert ledger.summary()["tokens"]["input_tokens"] == 20000
        assert ledger.summary()["estimated_cost_usd"] == pytest.approx(.156)
        assert ledger.calls[1]["turn_id"] == "turn-2"


@pytest.mark.parametrize("failure", ["failed", "interrupt", "no_usage"])
def test_failed_and_interrupted_codex_turns_save_available_usage(tmp_path, failure):
    client = native()
    events = [{"id": 1, "result": {"turn": {"id": "turn-1"}}}]
    if failure != "no_usage":
        events.append(usage_event())
    events.append(KeyboardInterrupt() if failure == "interrupt" else finish(status="failed"))
    client._receive = Mock(side_effect=events)
    with collect_usage() as ledger:
        ledger.attach(tmp_path)
        with pytest.raises(KeyboardInterrupt if failure == "interrupt" else RuntimeError):
            client.decide("observation", {})
        summary = ledger.summary()
        assert summary["calls"] == summary["failed_calls"] == 1
        assert summary["token_totals_complete"] == (failure == "failed")
        assert (summary["estimated_cost_usd"] is None) == (failure != "failed")
    assert json.loads((tmp_path / "usage.json").read_text())["calls"] == 1


def test_validation_rejection_is_not_a_model_call(tmp_path):
    with collect_usage() as ledger:
        ledger.attach(tmp_path)
        with pytest.raises(ValueError, match="too large"):
            native().decide("x" * 1048577, {})
        assert ledger.summary()["calls"] == 0


def test_naming_before_recorder_and_all_phases_survive_final_directory_rename(tmp_path):
    with collect_usage() as ledger:
        with usage_phase("task_name"), model_call("codex", "gpt-5.3-codex-spark") as call:
            call.update(raw_usage=RAW, request_started=True, usage_final=True)
        recorder = RunRecorder(tmp_path / "task", {})
        for phase in ("demonstration", "decision"):
            with usage_phase(phase), model_call("codex", "gpt-6-astra") as call:
                call.update(raw_usage=RAW, request_started=True, usage_final=True)
        destination = recorder.close("interrupted")
        summary = ledger.summary()
        assert summary["calls"] == 3 and summary["unpriced_calls"] == 1
        assert summary["cost_status"] == "partial" and summary["estimated_cost_usd"] is None
        assert summary["priced_cost_usd"] == pytest.approx(.156)
        assert set(summary["by_phase"]) == {"task_name", "demonstration", "decision"}
    rows = (destination / "usage.jsonl").read_text().splitlines()
    assert len(rows) == 3
    assert json.loads((destination / "status.json").read_text())["usage"] == summary
    assert not (tmp_path / "task").exists()


def test_naming_failure_before_directory_creation_is_preserved(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(RuntimeError), collect_usage():
        with usage_phase("task_name"), model_call("codex", "gpt-5.3-codex-spark") as call:
            call.update(request_started=True)
            raise RuntimeError("naming failed")
    summaries = list(tmp_path.glob("var/runs/gpt/*-initialization_failed/usage.json"))
    assert len(summaries) == 1
    assert json.loads(summaries[0].read_text())["usage_missing_calls"] == 1


def claude(events, context):
    agent = ClaudeCodeSession(AgentConfig(timeout_s=5), "claude-fable-5-1[1m]", False, 85)
    agent.context = context
    agent.process = SimpleNamespace(send=Mock(), receive=Mock(side_effect=events), close=Mock())
    return agent


def test_claude_result_is_authoritative_per_turn_and_error_usage_is_preserved(tmp_path, context, selection):
    raw = {"input_tokens": 100, "cache_read_input_tokens": 1000, "output_tokens": 10}
    agent = claude([
        {"type": "assistant", "message": {"id": "one", "usage": raw}},
        {"type": "assistant", "message": {"id": "one", "usage": raw}},
        {"type": "result", "subtype": "success", "usage": raw, "total_cost_usd": .5, "structured_output": selection},
        {"type": "result", "subtype": "error_during_execution", "is_error": True, "usage": raw, "total_cost_usd": .7},
    ], context)
    with collect_usage() as ledger:
        ledger.attach(tmp_path)
        agent.decide(AgentTurn("one"))
        with pytest.raises(AgentProtocolError):
            agent.decide(AgentTurn("two"))
        assert ledger.summary()["tokens"]["input_tokens"] == 2200
        assert ledger.summary()["calls"] == 2
        assert [c["provider_reported_cost_usd"] for c in ledger.calls] == [.5, .7]
        assert ledger.summary()["estimated_cost_usd"] == pytest.approx(.0035)


def test_claude_interruption_deduplicates_message_ids_and_keeps_partial_counts(tmp_path, context):
    messages = [{"type": "assistant", "message": {"id": "same", "usage": {
        "input_tokens": 100, "output_tokens": out}}} for out in (10, 15, 12)]
    agent = claude([*messages, KeyboardInterrupt()], context)
    with collect_usage() as ledger:
        ledger.attach(tmp_path)
        with pytest.raises(KeyboardInterrupt):
            agent.decide(AgentTurn("one"))
        assert ledger.calls[0]["usage"]["output_tokens"] == 15
        assert ledger.summary()["token_totals_complete"] is False
        assert ledger.summary()["estimated_cost_usd"] is None
