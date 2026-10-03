import pytest

from gpt_policy.harness.models import AgentContext
from gpt_policy.harness.protocol import instructions, output_schema, tool_schemas


@pytest.fixture
def context():
    return AgentContext(instructions("X5", "can1", 6), tool_schemas(6), output_schema(6))


@pytest.fixture
def selection(context):
    arguments = {"summary": "complete", "hindsight": "verified"}
    return {"name": "done", "arguments": arguments}
