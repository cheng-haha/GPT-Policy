import json
from contextlib import nullcontext
from copy import deepcopy
from unittest.mock import Mock

import pytest

from gpt_policy.harness.errors import AgentProtocolError
from gpt_policy.input import TextPart
from gpt_policy.input.request import RunInput
from gpt_policy.motion.coordination import TrajectoryIKError
from gpt_policy.runtime.runner import compact_execution_result, run_loop
from gpt_policy.settings import runtime_config
from gpt_policy.interrupts import HomeInterrupted


@pytest.mark.parametrize("mode", ["execute", "rejected", "ik_failure", "done", "give_up"])
def test_task_stops_at_100_decisions_including_rejected_tools(mode):
    runtime = runtime_config({})
    robot, cameras, video, agent, executor, recorder = [Mock() for _ in range(6)]
    terminal = mode in {"done", "give_up"}
    decisions = [{"name": "move_to", "arguments": {}} for _ in range(100)]
    if terminal:
        decisions[-1] = {"name": mode, "arguments": {"summary": "finished"}}
    agent.decide.side_effect = decisions
    executor.is_terminal.side_effect = lambda action: action in {"done", "give_up"}
    executor.execute.return_value = {}
    if mode == "rejected":
        executor.execute.side_effect = ValueError("bad target")
    elif mode == "ik_failure":
        executor.execute.side_effect = TrajectoryIKError(
            segment=0, sample=1, status=-9, status_name="JOINT_LIMIT",
            translation_error_m=.03, rotation_error_rad=.1,
        )
    robot.return_home.return_value = {"home": {"settle": {"settled": True}}}
    observe = Mock(return_value="observation")

    status = run_loop(runtime, RunInput("task", "model"), robot, cameras, video,
                      agent, executor, recorder, observe)

    assert agent.decide.call_count == 100
    assert video.snapshot.call_count == 100
    assert executor.execute.call_count == (99 if terminal else 100)
    assert [call.args[-1] for call in observe.call_args_list] == list(range(100))
    robot.return_home.assert_called_once()
    events = [call.args for call in recorder.write.call_args_list]
    assert sum(event == "decision_timing" for event, _ in events) == 100
    tool_timings = [data for event, data in events if event == "tool_timing"]
    assert len(tool_timings) == executor.execute.call_count
    assert all(data["status"] == ("failed" if mode in {"rejected", "ik_failure"} else "completed")
               for data in tool_timings)
    assert events[-1][0] == "home_timing" and events[-1][1]["status"] == "completed"
    events = [(event, data) for event, data in events if not event.endswith("_timing")]
    assert sum(event == "model_decision" for event, _ in events) == 100
    if terminal:
        assert status == ("give_up" if mode == "give_up" else "completed")
        assert not any(event == "budget_exhausted" for event, _ in events)
        assert events[-2][0] == "terminal"
        assert events[-1][0] == "return_home"
    else:
        assert status == "budget_exhausted"
        assert events[-2] == ("budget_exhausted", {"decisions_used": 100, "max_decisions": 100})
        assert events[-1][0] == "return_home"
        assert events[-1][1]["trigger"] == "budget_exhausted"
        assert not any(event == "terminal" for event, _ in events)


@pytest.mark.parametrize("outcome", ["interrupt", "unsettled", "error"])
def test_budget_home_faults_are_propagated_without_another_decision(outcome):
    robot, cameras, video, agent, executor, recorder = [Mock() for _ in range(6)]
    agent.decide.return_value = {"name": "move_to", "arguments": {}}
    executor.is_terminal.return_value = False
    executor.execute.return_value = {}
    if outcome == "unsettled":
        robot.return_home.return_value = {"home": {"settle": {"settled": False}}}
    else:
        robot.return_home.side_effect = KeyboardInterrupt if outcome == "interrupt" else RuntimeError("home failed")
    with pytest.raises(HomeInterrupted if outcome == "interrupt" else RuntimeError):
        run_loop(runtime_config({"runtime": {"max_decisions": 1}}), RunInput("task", "model"),
                 robot, cameras, video, agent, executor, recorder, Mock(return_value="observation"))
    agent.decide.assert_called_once()
    executor.execute.assert_called_once()
    robot.return_home.assert_called_once()
    timing = recorder.write.call_args.args
    assert timing[0] == "home_timing"
    assert timing[1]["status"] == ("interrupted" if outcome == "interrupt" else "failed")
    assert timing[1]["trigger"] == "budget_exhausted"


@pytest.mark.parametrize("terminal", ["done", "give_up"])
def test_loop_keeps_first_turn_content_live_observations_and_trace_semantics(terminal):
    runtime = runtime_config({"runtime": {"right_interface": ""}})
    run_input = RunInput("task", "model", (TextPart("static instruction"),))
    robot, cameras, video, agent, executor, recorder = [Mock() for _ in range(6)]
    states = [{"sample": 0}, {"sample": 1}]
    frames = [{"left": object()}, {"left": object()}]
    robot.state.side_effect = states
    video.snapshot.side_effect = frames
    cameras.describe.return_value = []
    agent.decide.side_effect = [
        {"name": "move_to", "arguments": {"note": "move"}},
        {"name": terminal, "arguments": {"summary": "finished"}},
    ]
    executor.is_terminal.side_effect = [False, True]
    executor.execute.return_value = {"ok": True, "nested": [{"_trace": "large", "value": 1}]}
    robot.return_home.return_value = {"home": "complete"}
    observe = Mock(side_effect=["observation-0", "observation-1"])

    run_loop(runtime, run_input, robot, cameras, video, agent, executor, recorder, observe)

    first, second = [call.args[0] for call in agent.decide.call_args_list]
    assert first.content is run_input.content and second.content is None
    assert first.images is frames[0] and second.images is frames[1]
    assert first.observation == "observation-0" and second.observation == "observation-1"
    assert observe.call_args_list[0].args[-2:] == (None, 0)
    assert json.loads(observe.call_args_list[1].args[-2]) == {
        "tool": "move_to", "result": {"ok": True, "nested": [{"value": 1}]},
    }
    assert observe.call_args_list[1].args[-1] == 1
    executor.execute.assert_called_once()
    robot.return_home.assert_called_once()
    events = [(call.args[0], call.args[1]) for call in recorder.write.call_args_list]
    execution = next(payload for kind, payload in events if kind == "execution_result")
    assert execution["result"]["nested"][0]["_trace"] == "large"
    assert [kind for kind, _ in events if not kind.endswith("_timing")] == [
        "model_decision", "execution_result", "model_decision", "terminal", "return_home",
    ]


def test_invalid_agent_output_never_calls_tools_or_returns_home():
    runtime = runtime_config({})
    robot, cameras, video, agent, executor, recorder = [Mock() for _ in range(6)]
    agent.decide.side_effect = AgentProtocolError("bad decision")
    with pytest.raises(AgentProtocolError):
        run_loop(runtime, RunInput("task", "model"), robot, cameras, video,
                 agent, executor, recorder, Mock(return_value="observation"))
    executor.execute.assert_not_called()
    executor.is_terminal.assert_not_called()
    robot.return_home.assert_not_called()
    events = [call.args for call in recorder.write.call_args_list]
    assert [event for event, _ in events] == ["decision_timing"]


def test_interrupt_during_normal_home_is_not_restarted():
    robot, cameras, video, agent, executor, recorder = [Mock() for _ in range(6)]
    agent.decide.return_value = {"name": "done", "arguments": {}}
    executor.is_terminal.return_value = True
    robot.return_home.side_effect = KeyboardInterrupt
    with pytest.raises(HomeInterrupted):
        run_loop(runtime_config({}), RunInput("task", "model"), robot, cameras,
                 video, agent, executor, recorder, Mock(return_value="observation"))
    robot.return_home.assert_called_once()
    timing = recorder.write.call_args.args
    assert timing[0] == "home_timing" and timing[1]["status"] == "interrupted"


def test_timing_separates_preparation_recording_decision_action_and_home(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr("gpt_policy.runtime.runner.time.perf_counter", lambda: clock[0])

    def take(seconds, value):
        def call(*args, **kwargs):
            clock[0] += seconds
            return value
        return call

    robot, cameras, video, agent, executor, recorder, display = [Mock() for _ in range(7)]
    video.snapshot.side_effect = take(1, {})
    robot.state.side_effect = take(2, {})
    cameras.describe.side_effect = take(3, [])
    observe = Mock(side_effect=take(4, "observation"))
    recorder.observation.side_effect = take(5, None)
    decisions = iter([{"name": "move_to", "arguments": {}}, {"name": "done", "arguments": {}}])

    def decide(turn):
        clock[0] += 6
        return next(decisions)

    agent.decide.side_effect = decide
    agent.last_decision_timing = {"client_decide_s": 6.0}
    executor.is_terminal.side_effect = [False, True]
    executor.execute.side_effect = take(7, {})
    robot.return_home.side_effect = take(8, {"home": {"settle": {"settled": True}}})
    display.waiting.side_effect = lambda *args: nullcontext()
    run_loop(runtime_config({}), RunInput("task", "model"), robot, cameras, video,
             agent, executor, recorder, observe, display=display)

    events = [call.args for call in recorder.write.call_args_list]
    assert [data for event, data in events if event == "decision_timing"] == [
        {"step": step, "observation_prepare_s": 10.0, "observation_record_s": 5.0,
         "agent_decide_s": 6.0, "provider": {"client_decide_s": 6.0}} for step in range(2)
    ]
    assert next(data for event, data in events if event == "tool_timing") == {
        "step": 0, "name": "move_to", "execute_s": 7.0, "status": "completed",
    }
    assert next(data for event, data in events if event == "home_timing") == {
        "step": 1, "trigger": "done", "home_s": 8.0, "status": "completed",
    }


@pytest.mark.parametrize("terminal", ["done", "give_up"])
@pytest.mark.parametrize("bimanual", [False, True])
def test_unsettled_home_is_recorded_and_never_reported_as_complete(terminal, bimanual, capsys):
    robot, cameras, video, agent, executor, recorder = [Mock() for _ in range(6)]
    agent.decide.return_value = {"name": terminal, "arguments": {"summary": "finished"}}
    executor.is_terminal.return_value = True
    failed = {"home": {"settle": {"settled": False, "max_joint_residual_rad": .0155}}}
    result = {"arms": {"left": {"home": {"settle": {"settled": True}}}, "right": failed}} if bimanual else failed
    robot.return_home.return_value = result
    with pytest.raises(RuntimeError, match="right" if bimanual else "arm"):
        run_loop(runtime_config({}), RunInput("task", "model"), robot, cameras,
                 video, agent, executor, recorder, Mock(return_value="observation"))
    robot.return_home.assert_called_once()
    events = [call.args for call in recorder.write.call_args_list]
    assert next(payload for event, payload in events if event == "return_home")["result"] is result
    assert next(payload for event, payload in events if event == "terminal")["name"] == terminal
    assert [event for event, _ in events].index("terminal") < [event for event, _ in events].index("return_home")
    assert "\nfinished\n" not in capsys.readouterr().out


def test_rejected_tool_is_reported_to_same_session_without_replaying_static_input():
    runtime = runtime_config({})
    robot, cameras, video, agent, executor, recorder = [Mock() for _ in range(6)]
    agent.decide.side_effect = [
        {"name": "move_to", "arguments": {}}, {"name": "give_up", "arguments": {}},
    ]
    executor.is_terminal.side_effect = [False, True]
    executor.execute.side_effect = ValueError("missing target")
    robot.return_home.return_value = {}
    observe = Mock(return_value="observation")
    run_loop(runtime, RunInput("task", "model"), robot, cameras, video,
             agent, executor, recorder, observe)
    assert json.loads(observe.call_args.args[-2]) == {
        "tool": "move_to", "error": "tool_rejected: missing target",
    }
    assert agent.decide.call_count == 2
    executor.execute.assert_called_once()


def _robot_state(sample):
    return {
        "joint_positions_rad": [sample] * 6,
        "joint_velocities_rad_s": [0.02] * 6,
        "joint_torques_nm": [0.1] * 6,
        "tcp_xyzrpy": [0.2, 0.0, 0.1, 0.0, 0.0, 0.0],
        "tcp_xyzquat": [0.2, 0.0, 0.1, 0.0, 0.0, 0.0, 1.0],
        "sdk_eef_xyzrpy": [0.2, 0.0, 0.2, 0.0, 0.0, 0.0],
        "gripper_position_m": 0.04,
        "gripper_normalized": 0.45,
        "gripper_command_normalized": 0.44,
        "gripper_velocity_m_s": 0.0,
        "gripper_torque_nm": 0.8,
        "gravity_compensation": True,
        "timestamp_s": sample,
        "temperature_rotor_c": [74.0] * 7,
        "temperature_mos_c": [42.0] * 7,
        "temperature_limit_c": 80.0,
        "temperature_over_limit": None,
    }


@pytest.mark.parametrize("bimanual", [False, True])
@pytest.mark.parametrize("action", ["move_to", "move_eef_chunk", "set_gripper"])
def test_loop_compacts_robot_results_but_keeps_live_state_feedback_and_full_log(bimanual, action):
    runtime = runtime_config({"runtime": {"right_interface": "can3" if bimanual else ""}})
    robot, cameras, video, agent, executor, recorder = [Mock() for _ in range(6)]

    def state(sample):
        if not bimanual:
            return _robot_state(sample)
        return {
            "arms": {"left": _robot_state(sample), "right": _robot_state(sample + 10)},
            "interfaces": {"left": "can1", "right": "can3"},
        }

    result = state(1)
    if action == "set_gripper":
        feedback = {
            "gripper_target_normalized": 0.0,
            "gripper_active_command_normalized": 0.44,
            "gripper_measured_normalized": 0.45,
            "gripper_residual_normalized": 0.45,
            "gripper_torque_nm": 0.8,
            "motion": {"sdk_contact": True, "sent_samples": 81},
        }
    else:
        feedback = {
            "target_tcp_xyzrpy": [0.2, 0.0, 0.102, 0.0, 0.0, 0.0],
            "measured_tcp_xyzrpy": _robot_state(1)["tcp_xyzrpy"],
            "tcp_error_xyz_m": [0.0, 0.0, 0.002],
            "tcp_translation_error_m": 0.002,
            "tcp_rotation_error_rad": 0.01,
            "joint_residual_rad": [0.001] * 6,
            "settle": {"settled": False, "max_velocity_rad_s": 0.02},
        }
        result["trajectory"] = {
            "planned_duration_s": 1.25,
            "note": "already in the model decision",
            "_trace": {"joint_waypoints_rad": [[0.0] * 6]},
        }
    result["execution_feedback"] = {"left": feedback} if bimanual else feedback
    original = deepcopy(result)
    robot.state.side_effect = [state(0), state(2)]
    robot.return_home.return_value = {}
    cameras.describe.return_value = []
    video.snapshot.return_value = {}
    executor.execute.return_value = result
    executor.is_terminal.side_effect = [False, True]
    agent.decide.side_effect = [
        {"name": action, "arguments": {"note": "act"}},
        {"name": "done", "arguments": {"summary": "complete"}},
    ]

    run_loop(runtime, RunInput("task", "model"), robot, cameras, video, agent, executor, recorder)

    observation = json.loads(agent.decide.call_args_list[1].args[0].observation)
    previous = observation["previous_result"]
    expected = {"execution_feedback": original["execution_feedback"]}
    if bimanual:
        expected["state_timestamps_s"] = {"left": 1, "right": 11}
    else:
        expected["timestamp_s"] = 1
    if action != "set_gripper":
        expected["trajectory"] = {"planned_duration_s": 1.25}
    assert previous == {"tool": action, "result": expected}
    latest = observation["state"]["left"] if bimanual else observation["state"]
    assert latest["joint_pos"] == [2] * 6
    assert latest["joint_vel"] == [0.02] * 6
    assert latest["tcp_pose_xyzquat"] == _robot_state(2)["tcp_xyzquat"]
    assert latest["gripper_normalized"] == 0.45
    assert "execution_feedback" not in observation["extra"]
    recorded = next(
        call.args[1]["result"] for call in recorder.write.call_args_list
        if call.args[0] == "execution_result"
    )
    assert recorded == original and result == original
    for call in agent.decide.call_args_list:
        for key in ("temperature_rotor_c", "temperature_mos_c", "temperature_limit_c", "temperature_over_limit"):
            assert key not in call.args[0].observation
    recorded_state = recorder.observation.call_args.args[2]
    recorded_arm = recorded_state["arms"]["left"] if bimanual else recorded_state
    assert recorded_arm["temperature_rotor_c"] == [74.0] * 7
    assert recorded_arm["temperature_limit_c"] == 80.0


@pytest.mark.parametrize("bimanual", [False, True])
def test_trajectory_summary_preserves_waypoints_holds_and_synchronized_timing(bimanual):
    points = [
        [0.2, 0.0, 0.1, 0.0, 0.0, 0.0],
        [0.2, 0.0, 0.3, 0.0, 0.0, 0.0],
        [0.2, 0.0, 0.3, 0.0, 0.0, 0.0],
        [0.4, 0.0, 0.3, 0.0, 0.0, 0.0],
    ]
    segments = [
        {
            "segment": index, "duration_s": duration,
            "endpoint_fk_translation_error_m": 0.001,
            "endpoint_fk_rotation_error_rad": 0.01,
        }
        for index, duration in enumerate([1.0, 0.5, 2.0], start=1)
    ]
    plan = {
        "planned_duration_s": 3.62,
        "ik_execution_tolerance_m_rad": [0.002, 0.017],
        "gripper_during_motion": "held",
        "note": "lift before crossing",
        "ik": "implementation detail",
        "_trace": {
            "model_tcp_points_xyzrpy": points,
            "segments": [
                {**segment, "samples": 100, "independent_duration_s": 0.4,
                 "peak_joint_velocity_rad_s": [1.0] * 6}
                for segment in segments
            ],
            "tcp_samples_xyzrpy": [[0.0] * 6] * 300,
            "joint_waypoints_rad": [[0.0] * 6] * 300,
            "relative_times_s": [index / 100 for index in range(300)],
        },
    }
    trajectory = plan
    if bimanual:
        right_plan = deepcopy(plan)
        right_plan["_trace"]["model_tcp_points_xyzrpy"] = [
            [pose[0], -0.1, *pose[2:]] for pose in points
        ]
        trajectory = {
            "planned_duration_s": 3.62,
            "concurrent": True,
            "segment_synchronized": True,
            "common_segment_durations_s": [1.0, 0.5, 2.0],
            "coordinated_start_delay_s": 0.1,
            "arms": {"left": plan, "right": right_plan},
        }
    result = {
        "trajectory": trajectory,
        "execution_feedback": {"measured_tcp_xyzrpy": [0.399, 0.0, 0.299, 0.0, 0.0, 0.0]},
    }
    original = deepcopy(result)

    compact = compact_execution_result(result)

    summary = compact["trajectory"]
    if bimanual:
        assert summary["concurrent"] and summary["segment_synchronized"]
        assert summary["common_segment_durations_s"] == [1.0, 0.5, 2.0]
        assert summary["coordinated_start_delay_s"] == 0.1
        assert summary["arms"]["right"]["planned_tcp_points_xyzrpy"] == (
            right_plan["_trace"]["model_tcp_points_xyzrpy"]
        )
        summary = summary["arms"]["left"]
    assert summary == {
        "planned_duration_s": 3.62,
        "ik_execution_tolerance_m_rad": [0.002, 0.017],
        "gripper_during_motion": "held",
        "planned_tcp_points_xyzrpy": points,
        "segments": segments,
    }
    assert compact["execution_feedback"] == original["execution_feedback"]
    assert result == original


@pytest.mark.parametrize("metric_available", [False, True])
def test_localization_results_preserve_rays_quality_and_metric_availability(metric_available):
    result = {
        "camera": "left", "pixel_xy": [320, 240],
        "ray_origin_base_xyz": [0.0, 0.0, 0.1],
        "ray_direction_base_xyz": [1.0, 0.0, 0.0],
        "metric_position_available": metric_available,
        "base_xyz" if metric_available else "triangulation_candidate_base_xyz": [0.2, 0.0, 0.1],
        "triangulation": {"ray_residual_m": 0.001, "positive_depth": True},
    }
    assert compact_execution_result(result) == result


def test_result_without_feedback_retains_its_state():
    result = _robot_state(1)
    assert compact_execution_result(result) == result


@pytest.mark.parametrize('bimanual', [False, True])
@pytest.mark.parametrize('with_feedback', [False, True])
def test_model_observation_omits_nested_thermal_flags_without_changing_raw_feedback(bimanual, with_feedback):
    from gpt_policy.harness.protocol import observation

    arm = _robot_state(1)
    arm['temperature_over_limit'] = {
        'over_limit': {'temp_rotor': [{'motor_index': 2, 'value_c': 83.375}]},
        'limit_c': 80.0,
    }
    state = {'arms': {'left': deepcopy(arm), 'right': deepcopy(arm)}} if bimanual else deepcopy(arm)
    result = deepcopy(state)
    if with_feedback:
        result['execution_feedback'] = {'left': {
            'tcp_translation_error_m': .0123,
            'diagnostics': [{'snapshot': deepcopy(arm)}],
        }}
    original = deepcopy(result)
    model = observation('task', state, [], json.dumps({'tool':'state', 'result':compact_execution_result(result)}))
    for key in ('temperature_rotor_c', 'temperature_mos_c', 'temperature_limit_c',
                'temperature_over_limit', 'temp_rotor', 'limit_c', 'value_c'):
        assert key not in model
    assert '83.375' not in model
    assert result == original
    assert json.loads(model)['state']
    if with_feedback:
        assert json.loads(model)['previous_result']['result']['execution_feedback']['left']['tcp_translation_error_m'] == .0123


def test_ik_failure_preserves_rejected_motion_and_diagnostics_in_next_observation():
    runtime = runtime_config({})
    robot, cameras, video, agent, executor, recorder = [Mock() for _ in range(6)]
    arguments = {"target": {"pose_xyzquat": [0.7, 0.0, 0.2, 0.0, 0.0, 0.0, 1.0]}, "note": "reach"}
    error = TrajectoryIKError(
        segment=0, sample=12, status=-9, status_name="JOINT_LIMIT",
        translation_error_m=0.03, rotation_error_rad=0.1, arm="left",
    )
    agent.decide.side_effect = [
        {"name": "move_to", "arguments": arguments}, {"name": "give_up", "arguments": {}},
    ]
    executor.is_terminal.side_effect = [False, True]
    executor.execute.side_effect = error
    robot.return_home.return_value = {}
    observe = Mock(return_value="observation")

    run_loop(runtime, RunInput("task", "model"), robot, cameras, video, agent, executor, recorder, observe)

    assert json.loads(observe.call_args.args[-2]) == {
        "tool": "move_to", "error": "motion_not_executed",
        "requested_motion": arguments, **error.details,
    }
    executor.execute.assert_called_once()
