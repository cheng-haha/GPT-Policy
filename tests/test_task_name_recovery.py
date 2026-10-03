import json
import queue
from unittest.mock import Mock

import pytest

from gpt_policy.harness import task_name as naming
from gpt_policy.harness.config import AgentConfig
from gpt_policy.harness.errors import AgentOverloadedError, AgentTimeoutError, AgentUsageLimitError
from gpt_policy.harness.waiting import receive_event
from gpt_policy.input.manifest import VideoPart
from gpt_policy.input.request import RunInput, normalize_request


def overloaded():
    return AgentOverloadedError("Selected model is at capacity.",
                                provider="codex", code="serverOverloaded")


def usage_limited():
    return AgentUsageLimitError("You've hit your usage limit for GPT-5.3-Codex-Spark.",
                               provider="codex", code="usageLimitExceeded")


def decision(**arguments):
    return {"name": "name_recording", "arguments": {"task_name": "insert-plug", **arguments}}


@pytest.fixture
def recovery(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(naming.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(naming.random, "uniform", lambda *_: 0)

    def advance(seconds):
        clock[0] += seconds

    wait = Mock(side_effect=advance)
    monkeypatch.setattr(naming, "wait_with_health", wait)
    return clock, wait


@pytest.mark.parametrize("failure", [overloaded(), usage_limited()])
def test_unavailable_model_fallback_matches_saved_demo_and_preserves_goal(tmp_path, monkeypatch, recovery, caplog, failure):
    demo = tmp_path / "demo.json"
    demo.write_text('{"keyframes": []}')
    saved = tmp_path / "insert-plug.json"
    saved.write_text(json.dumps({"instruction": "Insert the plug and release it fully seated.",
                                "content": [{"video": str(demo), "mode": "video"}]}))
    run = normalize_request(RunInput("Use the demo to insert the plug and leave it seated after release.",
                                     "robot-model", (VideoPart(demo),)))
    first, fallback = Mock(), Mock()
    first.decide.side_effect = failure
    fallback.decide.return_value = decision(request_json=saved.name)
    factory = Mock(side_effect=[first, fallback])
    monkeypatch.setattr(naming, "create_agent", factory)
    config = AgentConfig(model="robot-model", task_name_model="naming-model", effort="medium")

    name, path, selected = naming.select_task_request(run, tmp_path, config, "video")

    assert (name, path) == ("insert-plug", saved)
    assert selected.instruction == run.instruction and selected.model == run.model
    assert selected.content == run.content
    assert [c.args[1] for c in factory.call_args_list] == ["naming-model", "robot-model"]
    assert all(c.args[0].effort == "low" for c in factory.call_args_list)
    assert config.model == "robot-model" and config.effort == "medium"
    assert first.decide.call_args == fallback.decide.call_args
    assert first.start.call_args == fallback.start.call_args
    first.close.assert_called_once()
    fallback.close.assert_called_once()
    recovery[1].assert_called_once_with(2.0)
    assert f"naming-model unavailable ({failure.code}); retrying with robot-model" in caplog.text


@pytest.mark.parametrize("separate_model", [True, False])
def test_exhausted_quota_without_available_fallback_is_not_retried(monkeypatch, recovery, separate_model):
    agents = [Mock() for _ in range(2 if separate_model else 1)]
    failure = usage_limited()
    for agent in agents:
        agent.decide.side_effect = failure
    factory = Mock(side_effect=agents)
    monkeypatch.setattr(naming, "create_agent", factory)
    config = AgentConfig(model="robot-model", task_name_model="spark" if separate_model else "robot-model")
    with pytest.raises(AgentUsageLimitError) as caught:
        naming.generate_task_name("Insert the plug", config, "robot-model")
    assert caught.value is failure
    assert factory.call_count == len(agents)
    assert recovery[1].call_count == int(separate_model)
    for agent in agents:
        agent.close.assert_called_once()


@pytest.mark.parametrize("configured_model", [None, "naming-model", "robot-model"])
def test_persistent_overload_has_bounded_retries_and_closes_every_session(monkeypatch, recovery, configured_model):
    failure = overloaded()
    agents = [Mock() for _ in range(4)]
    for agent in agents:
        agent.decide.side_effect = failure
    factory = Mock(side_effect=agents)
    monkeypatch.setattr(naming, "create_agent", factory)
    config = AgentConfig(model=configured_model, task_name_model="naming-model")

    with pytest.raises(AgentOverloadedError) as caught:
        naming.generate_task_name("Insert the plug", config, "naming-model")

    assert caught.value is failure
    assert [c.args[1] for c in factory.call_args_list] == ["naming-model"] + [configured_model or "naming-model"] * 3
    assert [c.args[0] for c in recovery[1].call_args_list] == [2, 4, 8]
    for agent in agents:
        agent.close.assert_called_once()


@pytest.mark.parametrize("late_answer", [True, False])
def test_recovery_deadline_bounds_late_or_silent_fallback(monkeypatch, recovery, late_answer):
    clock, _ = recovery
    first, fallback = Mock(), Mock()
    first.decide.side_effect = overloaded()

    class SilentQueue:
        def get(self, timeout=None):
            clock[0] += timeout
            raise queue.Empty

    def decide(_):
        if late_answer:
            clock[0] += 61
            return decision()
        return receive_event(SilentQueue())

    fallback.decide.side_effect = decide
    factory = Mock(side_effect=[first, fallback])
    monkeypatch.setattr(naming, "create_agent", factory)

    with pytest.raises(AgentTimeoutError):
        naming.generate_task_name("Insert the plug", AgentConfig(model="robot-model"), "naming-model")

    assert factory.call_count == 2
    assert clock[0] <= (63 if late_answer else 60.11)
    first.close.assert_called_once()
    fallback.close.assert_called_once()


@pytest.mark.parametrize("failure", [KeyboardInterrupt(), RuntimeError("connection lost")])
def test_non_overload_failure_during_recovery_is_not_retried(monkeypatch, recovery, failure):
    first, fallback = Mock(), Mock()
    first.decide.side_effect = overloaded()
    fallback.decide.side_effect = failure
    factory = Mock(side_effect=[first, fallback])
    monkeypatch.setattr(naming, "create_agent", factory)
    with pytest.raises(type(failure)) as caught:
        naming.generate_task_name("Insert the plug", AgentConfig(model="robot-model"), "naming-model")
    assert caught.value is failure
    assert factory.call_count == 2
    first.close.assert_called_once()
    fallback.close.assert_called_once()


def test_interrupt_during_backoff_prevents_new_session(monkeypatch, recovery):
    agent = Mock()
    agent.decide.side_effect = overloaded()
    factory = Mock(return_value=agent)
    monkeypatch.setattr(naming, "create_agent", factory)
    recovery[1].side_effect = KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt):
        naming.generate_task_name("Insert the plug", AgentConfig(model="robot-model"), "naming-model")
    factory.assert_called_once()
    agent.close.assert_called_once()
