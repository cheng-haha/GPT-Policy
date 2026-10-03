import queue
import time
from unittest.mock import Mock

import pytest

from gpt_policy.hardware.motion_control import MotionFault
from gpt_policy.harness.codex import CodexAppServer
from gpt_policy.harness.errors import AgentTimeoutError
from gpt_policy.harness.process import JsonProcess
from gpt_policy.harness.waiting import monitor_health, receive_event, wait_with_health


@pytest.mark.parametrize("transport", [CodexAppServer, JsonProcess])
def test_provider_wait_checks_health_after_empty_poll(transport):
    process = object.__new__(transport)
    process._events = queue.Queue()
    process._closed = False
    fault = MotionFault({"reason": "motor_overtemperature"})
    check = Mock(side_effect=[None, None, fault])
    with pytest.raises(MotionFault) as caught, monitor_health(check):
        if transport is CodexAppServer:
            process._receive()
        else:
            process.receive(time.monotonic() + 5)
    assert caught.value is fault
    assert check.call_count == 3


def test_fault_wins_over_a_queued_response_and_monitor_is_cleared():
    events = queue.Queue()
    events.put({"answer": "must not execute"})
    fault = MotionFault({"reason": "motor_overtemperature"})
    check = Mock(side_effect=[None, None, fault])
    with pytest.raises(MotionFault), monitor_health(check):
        receive_event(events)
    events.put({"next_run": True})
    assert receive_event(events) == {"next_run": True}
    assert check.call_count == 3


def test_monitored_provider_preserves_total_deadline():
    process = object.__new__(JsonProcess)
    process._events, process._closed = queue.Queue(), False
    with monitor_health(lambda: None), pytest.raises(AgentTimeoutError):
        process.receive(time.monotonic() + .01)


def test_codex_recovery_deadline_interrupts_silent_provider_and_is_cleared():
    process = object.__new__(CodexAppServer)
    process._events = queue.Queue()
    with monitor_health(None, deadline=time.monotonic() + .01), pytest.raises(AgentTimeoutError):
        process._receive()
    process._events.put({'next_run': True})
    assert process._receive() == {'next_run': True}


@pytest.mark.parametrize('error', [MotionFault({'reason':'motor_overtemperature'}), KeyboardInterrupt()])
def test_backoff_remains_interruptible_by_health_fault_or_ctrl_c(error):
    check = Mock(side_effect=[None, None, error])
    with monitor_health(check), pytest.raises(type(error)) as caught:
        wait_with_health(5)
    assert caught.value is error
    assert check.call_count == 3


def test_backoff_obeys_total_recovery_deadline():
    with monitor_health(None, deadline=time.monotonic() + .01), pytest.raises(AgentTimeoutError):
        wait_with_health(5)


def test_healthy_backoff_finishes_without_a_timeout_error():
    check = Mock()
    with monitor_health(check):
        wait_with_health(.001)
    assert check.call_count >= 2
