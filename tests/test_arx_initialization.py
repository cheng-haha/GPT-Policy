"""Native constructor-failure regression, opt-in and virtual CAN only."""

import json
import os
import subprocess
import sys

import pytest


@pytest.mark.skipif(not os.environ.get("GPT_TEST_VIRTUAL_CAN"), reason="requires isolated virtual CAN")
def test_failed_native_constructor_can_retry_and_exit_cleanly():
    interface = os.environ["GPT_TEST_VIRTUAL_CAN"]
    links = json.loads(subprocess.check_output(["ip", "-details", "-json", "link", "show"], text=True))
    selected = next(link for link in links if link["ifname"] == interface)
    assert selected.get("linkinfo", {}).get("info_kind") == "vcan", "Never test on physical CAN"
    assert all(link["ifname"] == "lo" or link.get("linkinfo", {}).get("info_kind") == "vcan"
               for link in links), "Use an isolated network namespace"
    code = """
import sys
from gpt_policy.hardware.robot import _load_sdk
sdk = _load_sdk()
for attempt in range(2):
    try:
        sdk.Arx5JointController('X5', sys.argv[1])
    except RuntimeError as error:
        assert 'None of the motors are initialized' in str(error), str(error)
    else:
        raise AssertionError('Empty virtual CAN must not initialize motors')
print('NATIVE_FAILURE_CLEANUP_OK', flush=True)
"""
    result = subprocess.run([sys.executable, "-c", code, interface], capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "NATIVE_FAILURE_CLEANUP_OK" in result.stdout
    assert "invalid pointer" not in result.stderr
