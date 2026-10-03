import json

import pytest

from gpt_policy.harness.errors import AgentProtocolError
from gpt_policy.harness.validation import parse_decision
from gpt_policy.harness.models import AgentContext
from gpt_policy.harness.protocol import output_schema, tool_schemas


@pytest.mark.parametrize("encoding", ["native", "json", "fenced", "legacy"])
def test_structured_decision_is_normalized(selection, context, encoding):
    value = selection
    if encoding == "legacy":
        value = {**selection, "arguments": json.dumps(selection["arguments"])}
    if encoding == "json":
        value = json.dumps(selection)
    if encoding == "fenced":
        value = "```json\n" + json.dumps(selection) + "\n```"
    result = parse_decision(value, context)
    assert result["arguments"] == selection["arguments"]
    assert result["name"] == "done"


@pytest.mark.parametrize("mutate", [
    lambda x: {**x, "name": "run_shell_command"},
    lambda x: {**x, "arguments": {}},
    lambda x: {**x, "arguments": {**x["arguments"], "summary": 42}},
    lambda x: {**x, "arguments": {**x["arguments"], "gripper": float("nan")}},
    lambda x: {**x, "arguments": {**x["arguments"], "gripper": 0.5}},
    lambda x: {**x, "unrequested": "field"},
    lambda x: "prefix " + json.dumps(x),
    lambda x: json.dumps([x, x]),
])
def test_invalid_decision_never_crosses_adapter_boundary(selection, context, mutate):
    with pytest.raises(AgentProtocolError):
        parse_decision(mutate(selection), context)


def test_tool_specific_requirements_are_checked(selection, context):
    selection["name"] = "move_to"
    selection["arguments"]["summary"] = None
    # A terminal argument shape is not a valid movement.
    with pytest.raises(AgentProtocolError):
        parse_decision(selection, context)


def test_bimanual_gripper_requires_both_arm_fields():
    arms = ("left", "right")
    context = AgentContext("", tool_schemas(6, arms), output_schema(6, arms))
    arguments = {"positions": {"left": 0.2, "right": None}, "note": "hold right"}
    selection = {"name": "set_gripper", "arguments": arguments}
    assert parse_decision(selection, context)["arguments"]["positions"]["left"] == 0.2
    del arguments["positions"]["right"]
    with pytest.raises(AgentProtocolError):
        parse_decision(selection, context)
