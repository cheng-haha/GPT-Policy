"""Real image transport, context lifetime, hidden-state isolation and cleanup."""

import json
from io import BytesIO
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from gpt_policy.harness.input_content import to_app_server_items
from gpt_policy.simulation.robodojo import runner


def observation(step=0):
    pose = [0, 0, 1, 0, 0, 0, 1]
    return {
        "observation_id": f"e:{step}",
        "step_id": step,
        "native_terminal": step == 2,
        "instruction": "Sort visible objects",
        "remaining_controls": 1100 - step,
        "arms": {
            a: {"link6_pose_xyzquat": pose, "gripper": 1} for a in ("left", "right")
        },
        "initial_robot_state": {},
        "camera_calibration": {},
        "frame": "environment_origin",
        "site": "link6",
        "quaternion": "xyzw",
        "images": {
            k: np.full((8, 8, 3), [255, 0, 0], np.uint8)
            for k in ("cam_high", "cam_left_wrist", "cam_right_wrist")
        },
        "hidden_object_pose": "SECRET",
        "terminal_conditions": "SECRET",
        "case": "SECRET",
    }


def test_model_observation_allowlist_and_rgb_payload():
    obs = observation()
    obs["native_stop_reason"] = "SECRET"
    assert "SECRET" not in runner.model_observation(obs)
    images = runner.image_inputs(obs)
    assert len(images) == 3
    for image in images.values():
        assert tuple(np.asarray(Image.open(BytesIO(image.data)))[0, 0]) == (255, 0, 0)
        assert image.data_url().startswith("data:image/png;base64,")


@pytest.mark.parametrize(
    "budget,terminal_after,expected_decisions,expected_reason,complete",
    [
        (3, 2, 2, "native_terminal", True),
        (None, 105, 105, "native_terminal", True),
        ("omitted", 105, 105, "native_terminal", True),
        (1, 2, 1, "decision_budget", False),
        (None, 1, 1, "invalid_native_layout", False),
    ],
)
def test_first_turn_only_reference_and_cleanup(
    tmp_path,
    monkeypatch,
    budget,
    terminal_after,
    expected_decisions,
    expected_reason,
    complete,
):
    reference = tmp_path / "demo.png"
    Image.new("RGB", (8, 8), (0, 0, 255)).save(reference)
    prepared = tmp_path / "input.json"
    prepared.write_text(
        json.dumps(
            {
                "instruction": "historical",
                "content": ["historical reference", {"image": str(reference)}],
            }
        )
    )
    turns = []
    closed = []

    class Sim:
        def __init__(self, config, case, output, gpu):
            self.observation = observation()
            self.client = True

        def check_alive(self):
            pass

        def move(self, args):
            self.observation = observation(self.observation["step_id"] + 1)
            self.observation["native_terminal"] = (
                self.observation["step_id"] >= terminal_after
            )
            if expected_reason == "invalid_native_layout":
                self.observation["native_terminal"] = False
                self.observation["native_stop_reason"] = "invalid_native_layout"
            return {"observation": self.observation, "acknowledged_controls": 1}

        def finish(self, reason):
            return {
                "complete": self.observation["native_terminal"],
                "valid_for_success_rate": self.observation["native_terminal"],
                "native_success": False
                if self.observation["native_terminal"]
                else None,
                "native_score": 0 if self.observation["native_terminal"] else None,
                "reason": reason,
            }

        def close(self):
            closed.append("sim")

    class Agent:
        def __init__(self):
            self.client = SimpleNamespace(thread_id="t")
            self.last_context_refresh = self.last_decision_timing = None

        def start(self, context):
            if budget in (None, "omitted"):
                assert (
                    "model decisions have no fixed count limit" in context.instructions
                )
                assert "capped at" not in context.instructions
            else:
                assert f"capped at {budget}" in context.instructions

        def decide(self, turn):
            turns.append(turn)
            obs = json.loads(turn.observation)
            args = {
                "observation_id": obs["observation_id"],
                "note": "hold",
                "waypoints": [
                    dict(
                        steps=1,
                        **{
                            a: {
                                "pose_xyzquat": obs["arms"][a]["link6_pose_xyzquat"],
                                "gripper": 1.0,
                            }
                            for a in ("left", "right")
                        },
                    )
                ],
            }
            value = {"name": "move_eef_chunk", "arguments": args}
            return dict(value, _wire=value)

        def close(self):
            closed.append("agent")

    monkeypatch.setattr(runner, "Simulator", Sim)
    monkeypatch.setattr(runner, "create_agent", lambda *a: Agent())
    config = {
        "model": "test",
        "effort": "medium",
        "codex_bin": "test",
        "max_decisions": budget,
        "prepared_inputs": {"classify_objects": str(prepared)},
    }
    if budget == "omitted":
        del config["max_decisions"]
    result = runner.run_episode(
        config, {"case": {"task": "classify_objects"}}, tmp_path / "run", 0
    )
    assert len(turns) == expected_decisions
    assert result["decisions"] == expected_decisions
    assert result["reason"] == expected_reason
    assert turns[0].content and all(t.content is None for t in turns[1:])
    assert (
        len([i for i in to_app_server_items(turns[0].content) if i["type"] == "image"])
        == 1
    )
    assert result["native_success"] is (False if complete else None)
    assert result["valid_for_success_rate"] is complete
    assert closed == ["agent", "sim"]
    assert (tmp_path / "run/result.json").is_file()


@pytest.mark.parametrize("limit", [0, -1, True, 1.5, "unlimited"])
def test_invalid_decision_limits_fail_before_starting_a_run(tmp_path, limit):
    with pytest.raises(ValueError, match="null or a positive integer"):
        runner.run_episode({"max_decisions": limit}, {}, tmp_path / "invalid", 0)
    assert not (tmp_path / "invalid").exists()


def test_window_override_rejected_before_simulator_or_output(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "Simulator", lambda *a: pytest.fail("opened simulator"))
    config = {"model": "test", "effort": "medium", "codex_bin": "test", "live_image_window": 8}
    with pytest.raises(ValueError, match="disabled"):
        runner.run_episode(config, {}, tmp_path / "invalid", 0)
    assert not (tmp_path / "invalid").exists()


def test_wall_deadline_interrupts_model_wait_and_cleans_up(tmp_path, monkeypatch):
    import queue

    from gpt_policy.harness.waiting import receive_event

    clock = [0.0]
    closed = []
    monkeypatch.setattr(runner.time, "monotonic", lambda: clock[0])
    prepared = tmp_path / "input.json"
    prepared.write_text('{"content":["historical reference"]}')
    class Sim:
        client = True
        def __init__(self, *args):
            self.observation = {**observation(), "episode_deadline_monotonic": 10}
        def check_alive(self): pass
        def move(self, args): pytest.fail("No action after expired model wait")
        def finish(self, reason):
            assert reason == "wall_time_budget"
            return {"complete": False, "valid_for_success_rate": False,
                    "native_success": None, "native_score": None, "reason": reason}
        def close(self): closed.append("sim")
    class Agent:
        client = SimpleNamespace(thread_id="test")
        def start(self, context): pass
        def decide(self, turn):
            clock[0] = 10
            receive_event(queue.Queue())
            pytest.fail("Expired provider wait returned")
        def close(self): closed.append("agent")
    monkeypatch.setattr(runner, "Simulator", Sim)
    monkeypatch.setattr(runner, "create_agent", lambda *args: Agent())
    config = {"model": "test", "effort": "medium", "codex_bin": "test",
              "control_step_limit": None, "episode_wall_time_s": 10,
              "prepared_inputs": {"classify_objects": str(prepared)}}
    result = runner.run_episode(config, {"case": {"task": "classify_objects"}},
                                tmp_path / "run", 0)
    assert result["reason"] == "wall_time_budget"
    assert result["decisions"] == 0 and result["native_success"] is None
    assert closed == ["agent", "sim"]


@pytest.mark.parametrize("native_success", [True, False])
def test_difficulty_reports_preserve_episode_and_require_valid_correction(
    tmp_path, monkeypatch, native_success
):
    from gpt_policy.simulation.robodojo.control import validate_path

    prepared = tmp_path / "input.json"
    prepared.write_text('{"content":["fixed historical context"]}')
    turns, instances, acks, finishes, rejected = [], [], [], [], []

    class Sim:
        client = True
        def __init__(self, *args):
            instances.append(self)
            self.observation = observation()
        def check_alive(self):
            pass
        def move(self, args):
            try:
                validate_path(args, self.observation)
            except ValueError:
                rejected.append(args)
                raise
            acks.append(args["observation_id"])
            self.observation = observation(self.observation["step_id"] + 1)
            return {"acknowledged_controls": 1, "observation": self.observation}
        def finish(self, reason):
            finishes.append(reason)
            assert self.observation["native_terminal"]
            return {"complete": True, "valid_for_success_rate": True,
                    "native_success": native_success,
                    "native_score": float(native_success), "reason": reason}
        def close(self):
            pass

    class Agent:
        client = SimpleNamespace(thread_id="unchanged-model-thread")
        last_context_refresh = last_decision_timing = None
        def start(self, context):
            assert "Do not abandon" in context.instructions
            description = next(t["function"]["description"] for t in context.tools
                               if t["function"]["name"] == "give_up")
            assert "does not terminate" in description
        def decide(self, turn):
            turns.append(turn)
            obs = json.loads(turn.observation)
            assert "SECRET" not in turn.observation
            n = len(turns)
            args = {"observation_id": obs["observation_id"], "note": "visible difficulty"}
            if n <= 2:
                value = {"name": "give_up", "arguments": args}
            else:
                args["waypoints"] = [dict(steps=1, **{
                    arm: {"pose_xyzquat": list(obs["arms"][arm]["link6_pose_xyzquat"]),
                          "gripper": 1.0} for arm in ("left", "right")
                })]
                if n == 3:
                    # Continuing does not bypass the existing 5 cm path limit.
                    args["waypoints"][0]["left"]["pose_xyzquat"][0] += 0.06
                value = {"name": "move_eef_chunk", "arguments": args}
            return dict(value, _wire=value)
        def close(self):
            pass

    monkeypatch.setattr(runner, "Simulator", Sim)
    monkeypatch.setattr(runner, "create_agent", lambda *args: Agent())
    result = runner.run_episode(
        {"model": "test", "effort": "medium", "codex_bin": "test",
         "control_step_limit": None, "episode_wall_time_s": 9000,
         "prepared_inputs": {"classify_objects": str(prepared)}},
        {"case": {"task": "classify_objects"}}, tmp_path / "run", 0,
    )
    assert len(instances) == 1 and finishes == ["native_terminal"]
    assert result["decisions"] == 5 and result["native_success"] is native_success
    assert result["native_score"] == float(native_success)
    assert acks == ["e:0", "e:1"] and len(rejected) == 1
    assert [json.loads(t.observation)["step_id"] for t in turns] == [0, 0, 0, 0, 1]
    assert turns[0].content and all(t.content is None for t in turns[1:])
    for turn in turns[1:3]:
        feedback = json.loads(turn.observation)["previous_result"]
        assert feedback["termination_rejected"]
        assert feedback["native_episode_still_running"]
        assert "revise" in feedback["instruction"] or "change" in feedback["instruction"]
        assert all(turn.images[k].data == turns[0].images[k].data for k in turn.images)
    assert "tool_rejected" in json.loads(turns[3].observation)["previous_result"]


def test_repeated_difficulty_reports_still_obey_wall_deadline(tmp_path, monkeypatch):
    clock, turns, finishes = [0.0], [], []
    monkeypatch.setattr(runner.time, "monotonic", lambda: clock[0])
    prepared = tmp_path / "input.json"
    prepared.write_text('{"content":["fixed historical context"]}')

    class Sim:
        client = True
        def __init__(self, *args):
            self.observation = {**observation(), "episode_deadline_monotonic": 3}
        def check_alive(self):
            pass
        def move(self, args):
            pytest.fail("A declaration must never advance physics")
        def finish(self, reason):
            finishes.append(reason)
            return runner.empty_result(reason)
        def close(self):
            pass

    class Agent:
        client = SimpleNamespace(thread_id="same")
        last_context_refresh = last_decision_timing = None
        def start(self, context):
            pass
        def decide(self, turn):
            turns.append(turn)
            clock[0] += 1
            args = {"observation_id": "e:0", "note": "cannot proceed"}
            value = {"name": "give_up", "arguments": args}
            return dict(value, _wire=value)
        def close(self):
            pass

    monkeypatch.setattr(runner, "Simulator", Sim)
    monkeypatch.setattr(runner, "create_agent", lambda *args: Agent())
    result = runner.run_episode(
        {"model": "test", "effort": "medium", "codex_bin": "test",
         "control_step_limit": None, "episode_wall_time_s": 3,
         "prepared_inputs": {"classify_objects": str(prepared)}},
        {"case": {"task": "classify_objects"}}, tmp_path / "run", 0,
    )
    assert len(turns) == 3 and finishes == ["wall_time_budget"]
    assert result["reason"] == "wall_time_budget"
    assert result["native_success"] is None and not result["valid_for_success_rate"]



@pytest.mark.parametrize("failure", ["runtime_error", "quota_error", "interrupted"])
@pytest.mark.parametrize("native_complete", [False, True])
@pytest.mark.parametrize("finish_fails", [False, True])
def test_infrastructure_stop_preserves_confirmed_native_accounting_only(
    tmp_path, monkeypatch, failure, native_complete, finish_fails
):
    from gpt_policy.harness.errors import AgentUsageLimitError

    prepared = tmp_path / "input.json"
    prepared.write_text('{"content":["fixed ICL"]}')
    closed, finishes = [], []

    class Sim:
        client = True
        def __init__(self, *args):
            # The final receipt can confirm more controls than the last observation.
            self.observation = observation(5)
        def check_alive(self):
            pass
        def move(self, args):
            pytest.fail("No action may follow a failed model call")
        def finish(self, reason):
            finishes.append(reason)
            if finish_fails:
                raise TimeoutError("final receipt unavailable")
            return {
                "complete": native_complete,
                "valid_for_success_rate": native_complete,
                "native_success": True if native_complete else None,
                "native_score": 1.0 if native_complete else None,
                "native_control_steps": 7,
                "native_step_limit": 10**9,
                "evaluation_budget": {"control_step_limit": None, "episode_wall_time_s": 9000},
                "task": "classify_objects",
                "episode_id": "original-episode",
            }
        def close(self):
            closed.append("sim")

    class Agent:
        client = SimpleNamespace(thread_id="original-thread")
        def start(self, context):
            pass
        def decide(self, turn):
            if failure == "quota_error":
                raise AgentUsageLimitError("unavailable", provider="codex",
                                           code="sessionBudgetExceeded")
            if failure == "interrupted":
                raise KeyboardInterrupt("explicit interrupt")
            raise RuntimeError("provider failure after previous progress")
        def close(self):
            closed.append("agent")

    monkeypatch.setattr(runner, "Simulator", Sim)
    monkeypatch.setattr(runner, "create_agent", lambda *args: Agent())
    result = runner.run_episode(
        {"model": "test", "effort": "medium", "codex_bin": "test",
         "control_step_limit": None, "episode_wall_time_s": 9000,
         "prepared_inputs": {"classify_objects": str(prepared)}},
        {"case": {"task": "classify_objects"}}, tmp_path / "run", 0,
    )
    assert finishes == [failure] and closed == ["agent", "sim"]
    assert result["reason"] == result["status"] == failure
    assert result["complete"] is False and result["valid_for_success_rate"] is False
    assert result["native_success"] is None and result["native_score"] is None
    if finish_fails:
        # The stale observation's step 5 must not be invented as the final count.
        assert "native_control_steps" not in result
        events = (tmp_path / "run/policy/events.jsonl").read_text()
        assert "final receipt unavailable" in events
    else:
        assert result["native_control_steps"] == 7
        assert result["native_step_limit"] == 10**9
        assert result["evaluation_budget"]["episode_wall_time_s"] == 9000
        assert result["episode_id"] == "original-episode"
    assert json.loads((tmp_path / "run/result.json").read_text()) == result


@pytest.mark.parametrize("stop", ["deadline", "simulator_failure", "no_deadline"])
def test_exhausted_provider_retries_preserve_scene_until_health_boundary(
    tmp_path, monkeypatch, stop
):
    import queue

    from gpt_policy.harness.errors import AgentRecoverableError
    from gpt_policy.harness.waiting import receive_event

    clock, waits, finishes, closed, agents = [0.0], [], [], [], []
    monkeypatch.setattr(runner.time, "monotonic", lambda: clock[0])
    prepared = tmp_path / "input.json"
    prepared.write_text('{"content":["fixed historical context"]}')

    class Sim:
        client = True

        def __init__(self, *args):
            self.observation = observation(1)
            if stop != "no_deadline":
                self.observation["episode_deadline_monotonic"] = 20.0

        def check_alive(self):
            if stop == "simulator_failure" and clock[0] > 0:
                raise RuntimeError("simulator exited during backoff")

        def move(self, args):
            pytest.fail("An unavailable provider must not replay or advance actions")

        def finish(self, reason):
            finishes.append((reason, self.observation["step_id"]))
            return runner.empty_result(reason)

        def close(self):
            closed.append("sim")

    class Agent:
        client = SimpleNamespace(thread_id="same-thread")
        last_context_refresh = last_decision_timing = None
        last_recovery = ()

        def start(self, context):
            agents.append(self)

        def decide(self, turn):
            assert json.loads(turn.observation)["step_id"] == 1
            raise AgentRecoverableError("short retries exhausted",
                                        provider="codex", code="opaqueBadRequest")

        def close(self):
            closed.append("agent")

    def wait(delay):
        waits.append(delay)
        clock[0] = 20
        # Use the real active health callback, rather than raising directly.
        receive_event(queue.Queue())

    monkeypatch.setattr(runner, "Simulator", Sim)
    monkeypatch.setattr(runner, "create_agent", lambda *args: Agent())
    monkeypatch.setattr(runner, "wait_with_health", wait)
    result = runner.run_episode(
        {"model": "test", "effort": "medium", "codex_bin": "test",
         "episode_wall_time_s": 20,
         "prepared_inputs": {"classify_objects": str(prepared)}},
        {"case": {"task": "classify_objects"}}, tmp_path / "run", 0,
    )
    reason = "wall_time_budget" if stop == "deadline" else "runtime_error"
    assert result["reason"] == reason
    assert result["decisions"] == 0 and result["native_success"] is None
    assert not result["valid_for_success_rate"]
    assert finishes == [(reason, 1)] and closed == ["agent", "sim"]
    assert len(agents) == 1
    assert waits == ([] if stop == "no_deadline" else [60])
