import json
import queue
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from gpt_policy.hardware.motion_control import MotionFault
from gpt_policy.harness.codex import CodexAppServer
from gpt_policy.input.request import RunInput
from gpt_policy.runtime.runner import run_loop
from gpt_policy.settings import runtime_config


def test_background_fault_interrupts_silent_model_before_it_returns():
    started, tripped, finished, fallback = [threading.Event() for _ in range(4)]
    fault = MotionFault({"reason": "motor_overtemperature", "motor": "J2", "value": 80, "limit": 80})
    native = object.__new__(CodexAppServer)
    native.model, native.effort, native.thread_id = "test", "high", "thread-1"
    native.convert_camera_images_to_jpeg, native.camera_jpeg_quality = False, 85
    native._events = queue.Queue()

    def request(*_):
        started.set()
        assert tripped.wait(1)
        return {"turn": {"id": "turn-1"}}

    native._request = request

    def health_check():
        if tripped.is_set():
            raise fault

    def model():
        assert started.wait(1)
        tripped.set()
        if not finished.wait(.6):
            # Bound the regression test even with the old blocking receive().
            fallback.set()
            native._events.put({"method": "item/completed", "params": {"threadId": "thread-1",
                "item": {"type": "agentMessage", "text": json.dumps({"name": "done", "arguments": {}})}}})
            native._events.put({"method": "turn/completed", "params": {"threadId": "thread-1",
                "turn": {"id": "turn-1", "status": "completed"}}})

    model_thread = threading.Thread(target=model, daemon=True)
    model_thread.start()
    robot, cameras, video, executor, recorder = [Mock() for _ in range(5)]
    video.snapshot.return_value = {}
    agent = SimpleNamespace(decide=lambda turn: native.decide(turn.observation, {}, turn.images))
    try:
        with pytest.raises(MotionFault) as caught:
            run_loop(runtime_config({}), RunInput("task", "test"), robot, cameras, video,
                     agent, executor, recorder, lambda *_: "observation", health_check=health_check)
        assert caught.value is fault
        assert not fallback.is_set(), "host waited for a model decision after the robot had faulted"
        executor.execute.assert_not_called()
        robot.return_home.assert_not_called()
    finally:
        finished.set()
        model_thread.join(timeout=1)
    assert not model_thread.is_alive()
