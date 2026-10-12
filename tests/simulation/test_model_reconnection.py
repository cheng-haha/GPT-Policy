"""Real Codex JSON-RPC child restart against a deterministic robot stand-in."""

import json
import os
import sys
import time

import numpy as np
import pytest
from PIL import Image

from gpt_policy.harness.providers import codex as provider
from gpt_policy.simulation.robodojo import runner

FAKE_SERVER = r"""
import json, os, sys
from pathlib import Path
if "--version" in sys.argv:
    print("engineering-fault-injector 1 (no remote model calls)")
    raise SystemExit(0)
root = Path(__file__).parent
count_file = root / "model_calls"
failure = (root / "failure").read_text()
failure_turn = 4 if failure == "long_history" else 2
def send(value):
    print(json.dumps(value), flush=True)
for line in sys.stdin:
    request = json.loads(line)
    method, identity = request["method"], request.get("id")
    if identity is None:
        continue
    params = request.get("params", {})
    if method == "initialize":
        result = {}
    elif method == "config/read":
        result = {"config": {}}
    elif method == "thread/start":
        assert params["environments"] == [] and params["dynamicTools"] == []
        assert params["ephemeral"] is False
        assert not (root / "thread_id").exists(), "must not replace original thread"
        thread_id = str(os.getpid())
        (root / "thread_id").write_text(thread_id)
        result = {"thread": {"id": thread_id}}
    elif method == "thread/resume":
        thread_id = (root / "thread_id").read_text()
        assert params["threadId"] == thread_id
        assert params["excludeTurns"] is True and "history" not in params
        assert params["config"]["features.shell_tool"] is False
        result = {"thread": {"id": thread_id}}
    elif method == "turn/start":
        count = int(count_file.read_text()) + 1 if count_file.exists() else 1
        count_file.write_text(str(count))
        with (root / "requests.jsonl").open("a") as stream:
            stream.write(json.dumps({"pid": os.getpid(), "params": params}) + "\n")
        turn = str(count)
        if count == 2 and failure == "turn_start_error":
            send({"id": identity, "error": {"code": -32000, "message": '{"detail":"Bad Request"}'}})
            continue
        send({"id": identity, "result": {"turn": {"id": turn}}})
        if (count == failure_turn and failure in ("bad_request", "quota", "long_history")) or (failure == "repeated_bad_request" and 2 <= count <= 5):
            send({"method": "turn/completed", "params": {"threadId": thread_id,
                  "turn": {"id": turn, "status": "failed", "error": {
                      "message": "{\"detail\":\"Bad Request\"}", "codexErrorInfo": "usageLimitExceeded" if failure == "quota" else "other"}}}})
            continue
        # The last text before Camera image labels is the current host observation.
        observations = []
        for item in params["input"]:
            if item.get("type") != "text":
                continue
            try:
                value = json.loads(item["text"])
                if isinstance(value, dict) and "observation_id" in value:
                    observations.append(value)
            except ValueError:
                pass
        obs = observations[-1]
        args = {"observation_id": obs["observation_id"], "note": "hold current pose",
                "waypoints": [{"steps": 1, **{a: {
                    "pose_xyzquat": obs["arms"][a]["link6_pose_xyzquat"], "gripper": 1.
                } for a in ("left", "right")}}]}
        answer = {"name": "move_eef_chunk", "arguments": args}
        if failure == "give_up":
            if count == 2:
                answer = {"name": "give_up", "arguments": {
                    "observation_id": obs["observation_id"], "note": "visible difficulty"}}
            if count == 3:
                assert obs["previous_result"]["termination_rejected"]
                assert obs["step_id"] == 1
        send({"method": "item/completed", "params": {"threadId": thread_id, "turnId": turn,
              "item": {"type": "agentMessage", "phase": "final_answer", "text": json.dumps(answer)}}})
        if count == 2 and failure == "disconnect":
            # Lose the stream AFTER a plausible action, BEFORE turn completion.
            os._exit(9)
        send({"method": "turn/completed", "params": {"threadId": thread_id,
              "turn": {"id": turn, "status": "completed"}}})
        continue
    else:
        send({"id": identity, "error": {"code": -32601, "message": "unsupported"}})
        continue
    send({"id": identity, "result": result})
"""


@pytest.mark.parametrize("failure", ["bad_request", "disconnect", "turn_start_error", "quota", "give_up", "long_history", "repeated_bad_request"])
def test_reconnect_child_keeps_episode_and_ack_history(tmp_path, monkeypatch, failure):
    executable = tmp_path / "fake-codex"
    executable.write_text("#!" + sys.executable + "\n" + FAKE_SERVER)
    executable.chmod(0o700)
    (tmp_path / "failure").write_text(failure)
    image = tmp_path / "demo.png"
    Image.new("RGB", (2, 2), (0, 0, 255)).save(image)
    prepared = tmp_path / "input.json"
    prepared.write_text(
        json.dumps({"content": ["fixed successful ICL", {"image": str(image)}]})
    )
    instances, acks, waits, agents = [], [], [], []
    terminal_step = 4 if failure == "long_history" else 2
    deadline = time.monotonic() + 9000

    def observation(step):
        return {
            "episode_id": "same-episode",
            "observation_id": f"same-episode:{step}",
            "step_id": step,
            "native_terminal": step == terminal_step,
            "episode_deadline_monotonic": deadline,
            "tool_geometry": {"fixed_calibration": "x" * 400000} if failure == "long_history" else {},
            "remaining_controls": None,
            "instruction": "hold",
            "arms": {
                a: {"link6_pose_xyzquat": [0, 0, 1, 0, 0, 0, 1], "gripper": 1.0}
                for a in ("left", "right")
            },
            "images": {
                c: np.full((2, 2, 3), step, np.uint8)
                for c in ("cam_high", "cam_left_wrist", "cam_right_wrist")
            },
        }

    class Sim:
        client = True

        def __init__(self, *args):
            instances.append(self)
            self.observation = observation(0)
            self.closed = False

        def check_alive(self):
            assert not self.closed

        def move(self, arguments):
            assert arguments["observation_id"] == self.observation["observation_id"]
            acks.append(arguments["observation_id"])
            self.observation = observation(self.observation["step_id"] + 1)
            return {"observation": self.observation, "acknowledged_controls": 1}

        def finish(self, reason):
            return {
                "complete": self.observation["native_terminal"],
                "valid_for_success_rate": self.observation["native_terminal"],
                "native_success": self.observation["native_terminal"],
                "native_score": float(self.observation["native_terminal"]),
                "reason": reason,
            }

        def close(self):
            self.closed = True

    create = runner.create_agent

    def create_agent(*args):
        session = create(*args)
        agents.append(session)
        return session

    def backoff(delay):
        waits.append((delay, instances[0].observation["step_id"], len(acks)))
        instances[0].check_alive()

    monkeypatch.setattr(runner, "Simulator", Sim)
    monkeypatch.setattr(runner, "create_agent", create_agent)
    monkeypatch.setattr(provider, "wait_with_health", backoff)
    monkeypatch.setattr(runner, "wait_with_health", backoff)
    def recover(config, *, health_check):
        assert failure == "quota"
        assert len(instances) == 1 and instances[0].observation["step_id"] == 1
        assert acks == ["same-episode:0"]
        health_check()
        return True
    monkeypatch.setattr(runner, "recover_quota", recover)
    config = {
        "model": "test",
        "effort": "medium",
        "codex_bin": str(executable),
        "control_step_limit": None,
        "episode_wall_time_s": 9000,
        "prepared_inputs": {"classify_objects": str(prepared)},
    }
    result = runner.run_episode(
        config, {"case": {"task": "classify_objects"}}, tmp_path / "run", 0
    )
    assert result["native_success"] is True
    assert result["decisions"] == (3 if failure == "give_up" else terminal_step)
    assert result["provider_recoveries"] == (4 if failure == "repeated_bad_request" else 0 if failure in ("quota", "give_up") else 1)
    assert result["quota_recoveries"] == (1 if failure == "quota" else 0)
    assert result["usage"]["calls"] == terminal_step + (4 if failure == "repeated_bad_request" else 1)
    assert result["usage"]["failed_calls"] == (4 if failure == "repeated_bad_request" else 0 if failure == "give_up" else 1)
    assert len(instances) == 1 and instances[0].closed
    assert acks == [f"same-episode:{i}" for i in range(terminal_step)]
    assert waits == ([(d, 1, 1) for d in (5, 15, 30, 60)] if failure == "repeated_bad_request" else [] if failure == "give_up" else [(5, terminal_step - 1, terminal_step - 1)])
    requests = [
        json.loads(s) for s in (tmp_path / "requests.jsonl").read_text().splitlines()
    ]
    assert len(requests) == terminal_step + (4 if failure == "repeated_bad_request" else 1)
    assert requests[0]["pid"] == requests[1]["pid"]
    assert (requests[-2]["pid"] == requests[-1]["pid"]) is (failure != "disconnect")
    assert len({x["params"]["threadId"] for x in requests}) == 1
    inputs = requests[-1]["params"]["input"]
    texts = "\n".join(x["text"] for x in inputs if x["type"] == "text")
    assert "fixed successful ICL" not in texts
    assert "Historical model decision" not in texts
    assert sum(x["type"] == "image" for x in inputs) == 3
    if failure == "give_up":
        assert "termination_rejected" in texts
        assert "bounded corrective action" in texts
    else:
        assert requests[-2]["params"] == requests[-1]["params"]
        if failure == "repeated_bad_request":
            assert all(r["params"] == requests[1]["params"] for r in requests[1:])
        events = [
            json.loads(s)
            for s in (tmp_path / "run/policy/events.jsonl").read_text().splitlines()
        ]
        assert any(
            ("quota_recovery" if failure == "quota" else "provider_recovery") in json.dumps(e)
            and f"same-episode:{terminal_step - 1}" in json.dumps(e)
            for e in events
        )
    if failure == "long_history":
        # Aggregate history exceeds the per-request guard; native thread owns it.
        assert sum(len(json.dumps(x["params"]["input"])) for x in requests[:-1]) > 1048576
        assert len(texts) < 1048576
        assert not hasattr(agents[0], "_history")
    for pid in {x["pid"] for x in requests}:
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)
