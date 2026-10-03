import sys
import time

import pytest

from gpt_policy.harness.errors import AgentProtocolError, AgentTimeoutError
from gpt_policy.harness.process import JsonProcess


def test_real_subprocess_echo_and_idempotent_cleanup(tmp_path):
    process = JsonProcess([sys.executable, "-u", "-c",
                           "import sys; [print(line.strip(), flush=True) for line in sys.stdin]"], tmp_path)
    try:
        process.send({"text": "中文", "id": 1})
        assert process.receive(time.monotonic() + 3) == {"text": "中文", "id": 1}
        process.send({"id": 2})
        assert process.receive(time.monotonic() + 3) == {"id": 2}
    finally:
        process.close()
        process.close()
    assert process.process.poll() is not None
    assert all(not reader.is_alive() for reader in process._readers)


@pytest.mark.parametrize("program", ["pass", "print('not-json')", "print('[]')"])
def test_eof_and_bad_json_fail_without_waiting_forever(tmp_path, program):
    process = JsonProcess([sys.executable, "-u", "-c", program], tmp_path)
    try:
        with pytest.raises(AgentProtocolError):
            process.receive(time.monotonic() + 3)
    finally:
        process.close()


def test_silent_child_times_out_and_is_reaped(tmp_path):
    process = JsonProcess([sys.executable, "-c", "import time; time.sleep(30)"], tmp_path)
    try:
        with pytest.raises(AgentTimeoutError):
            process.receive(time.monotonic() + 0.05)
    finally:
        process.close()
    assert process.process.poll() is not None


def test_large_image_write_also_obeys_deadline(tmp_path):
    process = JsonProcess([sys.executable, "-c", "import time; time.sleep(30)"], tmp_path)
    try:
        with pytest.raises(AgentTimeoutError):
            process.send({"image": "x" * 1_000_000}, time.monotonic() + 0.05)
    finally:
        process.close()
