"""Keep decision inputs small without losing physical evidence or raw logs."""

from copy import deepcopy
import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator, ValidationError

from gpt_policy.harness.protocol import observation, output_schema, tool_schemas
from gpt_policy.harness.models import AgentContext
from gpt_policy.harness.validation import parse_decision


@pytest.mark.parametrize("arms", [("left",), ("left", "right")])
def test_native_schema_accepts_only_relevant_tool_arguments(arms):
    context = AgentContext("", tool_schemas(6, arms), output_schema(6, arms))
    pose = {"pose_xyzquat": [.3, .1, .2, 0, 1, 0, 0]}
    target = pose if len(arms) == 1 else {"left": pose, "right": None}
    examples = {
        "move_to": {"target": target, "note": "接近已定位目标。"},
        "move_eef_chunk": {"poses": [target, target], "note": "沿已确认的自由空间路径移动。"},
        "check_path": {"poses": [target], "note": "检查替代路径。"},
        "set_gripper": {**({"gripper": .5} if len(arms) == 1 else {
            "positions": {"left": .5, "right": None}}), "note": "对准后闭合。"},
        "locate_point": {"camera": "left", "pixel_xy": [320, 240],
                         "reference_step": None, "reference_pixel_xy": None, "note": "记录特征射线。"},
        "done": {"summary": "完成", "hindsight": "已观察确认。"},
        "give_up": {"reason": "目标不可达", "hindsight": "已尝试替代路径。"},
    }
    for name, arguments in examples.items():
        selection = {"name": name, "arguments": arguments}
        Draft202012Validator(context.output_schema).validate(selection)
        assert parse_decision(selection, context)["arguments"] == arguments
    with pytest.raises(ValidationError):
        Draft202012Validator(context.output_schema).validate({
            "name": "move_to", "arguments": {"target": None, "note": "invalid"},
        })
    # A valid argument shape for another tool must still fail host validation.
    from gpt_policy.harness.errors import AgentProtocolError
    with pytest.raises(AgentProtocolError):
        parse_decision({"name": "move_to", "arguments": examples["done"]}, context)


def test_recorded_yam_observation_keeps_tracking_but_omits_routine_temperatures():
    row = json.loads(Path("tests/fixtures/yam_model_observation.json").read_text())
    original = deepcopy(row)
    old = json.loads(row["input_json"])
    text = observation(old["instruction"], row["state"], row["cameras"],
                       json.dumps(old["previous_result"]), row["step"])
    assert len(text) < .65 * len(row["input_json"])
    current = json.loads(text)
    right = current["state"]["right"]
    for key in ("joint_pos", "joint_vel", "joint_torque", "tcp_pose_xyzquat"):
        assert right[key] == pytest.approx(old["state"]["right"][key], abs=5.1e-7)
    for key in ("temperature_rotor_c", "temperature_mos_c", "temperature_limit_c"):
        assert key in old["state"]["right"]
        assert key not in text  # Includes any nested previous-result snapshots.
    feedback = current["previous_result"]["result"]["execution_feedback"]["right"]
    # This actual step was stable but missed the TCP by more than 1 cm.
    assert feedback["settle"]["settled"] is True
    assert feedback["tcp_translation_error_m"] > .01
    assert feedback["target_tcp_xyzrpy"] == pytest.approx(
        old["previous_result"]["result"]["execution_feedback"]["right"]["target_tcp_xyzrpy"], abs=5.1e-7)
    assert [x["name"] for x in current["images"]] == ["left", "right", "top"]
    assert row == original


@pytest.mark.parametrize("bimanual", [False, True])
def test_full_previous_results_cannot_reintroduce_temperature_telemetry(bimanual):
    row = json.loads(Path("tests/fixtures/yam_model_observation.json").read_text())
    state = row["state"] if bimanual else row["state"]["arms"]["right"]
    previous = {"tool": "state", "result": {"snapshots": [deepcopy(state)]}}
    original = deepcopy(previous)
    text = observation("task", state, [], json.dumps(previous))
    for key in ("temperature_rotor_c", "temperature_mos_c", "temperature_limit_c"):
        assert key in json.dumps(previous)
        assert key not in text
    result = json.loads(text)["previous_result"]["result"]["snapshots"][0]
    arm = result["arms"]["right"] if bimanual else result
    source = state["arms"]["right"] if bimanual else state
    assert arm["joint_torques_nm"] == pytest.approx(source["joint_torques_nm"], abs=5.1e-7)
    assert previous == original


def test_temperature_fault_reason_and_trigger_are_not_hidden():
    row = json.loads(Path("tests/fixtures/yam_model_observation.json").read_text())
    fault = {"error": "motion_fault", "reason": "motor_overtemperature",
             "side": "right", "motor": "J2", "value": 80, "limit": 80}
    result = json.loads(observation("task", row["state"], [], json.dumps(fault)))
    assert result["previous_result"] == fault


def test_failed_settling_diagnostics_and_localization_flags_survive_compaction():
    row = json.loads(Path("tests/fixtures/yam_model_observation.json").read_text())
    previous = json.loads(row["input_json"])["previous_result"]
    settle = previous["result"]["execution_feedback"]["right"]["settle"]
    settle["settled"] = False
    original = deepcopy(previous)
    text = observation("task", row["state"], row["cameras"], json.dumps(previous), 23)
    retained = json.loads(text)["previous_result"]["result"]["execution_feedback"]["right"]["settle"]
    assert retained.keys() == settle.keys()
    assert retained["target_joint_positions_rad"] == pytest.approx(settle["target_joint_positions_rad"], abs=5.1e-7)
    assert previous == original
    located = {"tool": "locate_point", "result": {"metric_position_available": False,
               "triangulation_candidate_base_xyz": [.123456789, 0, 0],
               "reason": "insufficient parallax"}}
    result = json.loads(observation("task", row["state"], [], json.dumps(located)))["previous_result"]
    assert result["result"]["metric_position_available"] is False
    assert result["result"]["reason"] == located["result"]["reason"]


@pytest.mark.parametrize("bimanual", [False, True])
def test_progress_logging_does_not_change_the_baseline_model_observation(bimanual):
    row = json.loads(Path("tests/fixtures/yam_model_observation.json").read_text())
    previous = json.loads(row["input_json"])["previous_result"]
    result = previous["result"]
    if not bimanual:
        result["execution_feedback"] = result["execution_feedback"]["right"]
    expected = observation("task", row["state"], row["cameras"], json.dumps(previous), 23)
    feedback = result["execution_feedback"]
    if bimanual:
        feedback = feedback["right"]
    feedback["motion_progress"] = {
        "requested_delta_xyz_m": [0, 0, -.01],
        "achieved_delta_xyz_m": [0, 0, -.001],
        "remaining_delta_xyz_m": [0, 0, -.009],
    }
    raw = deepcopy(previous)
    assert observation("task", row["state"], row["cameras"], json.dumps(previous), 23) == expected
    assert previous == raw
