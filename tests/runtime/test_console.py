from copy import deepcopy
from io import StringIO
import json
from unittest.mock import Mock

import pytest
from rich.cells import cell_len
from rich.console import Console

from gpt_policy.runtime.console import RunConsole
from gpt_policy.runtime.runner import run_loop
from gpt_policy.input.request import RunInput
from gpt_policy.interrupts import HomeInterrupted
from gpt_policy.settings import runtime_config


def renderer(width=88, **options):
    output = StringIO()
    console = Console(file=output, width=width, _environ={"TERM": "xterm-256color"}, **options)
    return RunConsole(console), output


def plain_words(text):
    # Ignore wrapping/indentation, but retain every printed character.
    return "".join(text.translate(str.maketrans("", "", "│├└─")).split())


@pytest.mark.parametrize("width", [40, 80, 120])
def test_full_multilingual_note_pose_and_literal_markup_survive_wrapping(width):
    ui, output = renderer(width)
    note = "尚未确认抓牢；" + "保持当前夹爪姿态，仅抬升观察水果是否随手移动。" * 12 + "[red]do not hide this[/red]"
    args = {"target": {"left": None, "right": {"pose_xyzquat": [0.29, 0.155, -0.025, 0, 0.963558, 0, 0.267499]}},
            "positions": None, "note": note, "summary": None}
    original = deepcopy(args)
    ui.decision(9, "move_to", args)
    rendered = output.getvalue()
    compact = plain_words(rendered)
    assert plain_words(note) in compact
    for value in ("STEP 009", "MOVE TO", "RIGHT", "REQUESTED", "x=0.29", "z=-0.025", "qy=0.963558", "qw=0.267499"):
        assert plain_words(value) in compact
    assert "null" not in rendered and "…" not in rendered and "\x1b" not in rendered
    assert all(cell_len(line) <= width for line in rendered.splitlines())
    assert args == original


def test_waypoints_keep_both_arms_indices_and_zero_gripper_targets():
    ui, output = renderer()
    pose = {"pose_xyzquat": [0.25, 0.052, 0.31, 0, 1, 0, 0]}
    args = {"poses": [{"left": pose, "right": None}, {"left": None, "right": pose}],
            "positions": {"left": 0, "right": 0.62}, "reference_step": 0, "note": "two waypoints"}
    ui.decision(3, "move_eef_chunk", args)
    text = output.getvalue()
    assert "LEFT/RIGHT" in text
    assert text.index("Waypoint 1") < text.index("Waypoint 2")
    assert text.count("Quaternion (xyzw)") == 2
    assert "Left  0" in text and "Right  0.62" in text and "Reference step  0" in text


def test_every_diagnostic_and_nonempty_field_is_expanded_without_mutation():
    ui, output = renderer()
    payload = {
        "error": "motion_not_executed", "reason": "ik_joint_limit", "arm": "right",
        "segment": 1, "sample": 108, "sdk_status": -9, "sdk_status_name": "E_EXCEED_JOINT_LIMIT",
        "best_translation_error_m": 0.002778453602500335,
        "best_rotation_error_rad": 0.020392162716488776,
        "requested_motion": {"positions": {"right": 0.62}, "note": "keep full requested motion"},
        "custom_diagnostic": {"settled": False, "samples": list(range(31)), "extra": {"value": "LAST-FIELD"}},
        "_wire": {"duplicate": "DO-NOT-PRINT-DUPLICATE"}, "unused": None,
    }
    original = deepcopy(payload)
    ui.error(3, "move_eef_chunk", payload)
    text = output.getvalue()
    compact = plain_words(text)
    for value in ("NOT EXECUTED", "ik_joint_limit", "right", "108", "-9", "E_EXCEED_JOINT_LIMIT",
                  "0.002778453602500335", "0.020392162716488776", "keep full requested motion",
                  "false", "LAST-FIELD", json.dumps(list(range(31)))):
        assert plain_words(value) in compact
    assert "DO-NOT-PRINT-DUPLICATE" not in text and "unused" not in text
    assert payload == original


def test_piped_output_has_no_spinner_or_ansi_and_does_not_claim_success():
    ui, output = renderer()
    with ui.waiting("Waiting for model"):
        pass
    ui.decision(0, "set_gripper", {"gripper": 0, "note": "grip not yet verified"})
    ui.returned(0, "set_gripper")
    text = output.getvalue()
    assert "Waiting for model..." in text
    assert "RESULT RECORDED" in text and "SUCCESS" not in text
    assert "\x1b" not in text


def test_no_color_disables_animation_even_on_a_terminal():
    ui, output = renderer(force_terminal=True, no_color=True)
    with ui.waiting("Waiting for model"):
        pass
    assert "\x1b" not in output.getvalue()


def test_waiting_animation_is_closed_on_ctrl_c():
    ui, _ = renderer(force_terminal=True)
    with pytest.raises(KeyboardInterrupt):
        with ui.waiting("Waiting for model"):
            assert ui.console._live_stack
            raise KeyboardInterrupt
    assert ui.console._live_stack == []


def test_console_does_not_change_recorded_decision_or_next_model_input():
    ui, output = renderer()
    robot, cameras, video, agent, executor, recorder = [Mock() for _ in range(6)]
    decision = {"name": "move_to", "arguments": {"target": None, "note": "[red]literal[/red]"}, "_wire": {"full": "kept"}}
    original = deepcopy(decision)
    agent.decide.side_effect = [decision, {"name": "give_up", "arguments": {"reason": "finish"}}]
    executor.is_terminal.side_effect = [False, True]
    executor.execute.side_effect = ValueError("invalid target")
    robot.return_home.return_value = {"home": "complete"}
    observe = Mock(return_value="observation")
    run_loop(runtime_config({}), RunInput("task", "model"), robot, cameras, video,
             agent, executor, recorder, observe, display=ui)
    records = [call.args for call in recorder.write.call_args_list if not call.args[0].endswith("_timing")]
    assert records[0] == ("model_decision", {"step": 0, "decision": original})
    assert decision == original
    assert json.loads(observe.call_args.args[-2]) == {"tool": "move_to", "error": "tool_rejected: invalid target"}
    assert "REJECTED" in output.getvalue() and "SUCCESS" not in output.getvalue()


def test_interrupt_while_printing_home_feedback_does_not_restart_homing():
    robot, cameras, video, agent, executor, recorder = [Mock() for _ in range(6)]
    ui, _ = renderer()
    ui.home = Mock(side_effect=KeyboardInterrupt)
    agent.decide.return_value = {"name": "done", "arguments": {"summary": "finished"}}
    executor.is_terminal.return_value = True
    robot.return_home.return_value = {"home": {"settle": {"settled": True}}}
    with pytest.raises(HomeInterrupted):
        run_loop(runtime_config({}), RunInput("task", "model"), robot, cameras, video,
                 agent, executor, recorder, Mock(return_value="observation"), display=ui)
    robot.return_home.assert_called_once()
    assert any(call.args[0] == "terminal" and call.args[1]["name"] == "done"
               for call in recorder.write.call_args_list)


def test_completed_task_and_failed_finalization_are_shown_separately():
    ui, output = renderer()
    ui.finished("failed", "run_success", task_status="completed", error="left home timed out")
    text = output.getvalue()
    assert "TASK · SUCCESS" in text
    assert "FINALIZATION · FAILED" in text
    assert "left home timed out" in text
    assert "TASK · FAILED" not in text
    assert "run_success" in text


def test_failed_task_shows_its_primary_cause():
    ui, output = renderer()
    ui.finished("failed", "run_failed", error="right J2: temp_rotor 80°C >= 80°C")
    assert "TASK · FAILED" in output.getvalue()
    assert "right J2: temp_rotor 80°C >= 80°C" in output.getvalue()
