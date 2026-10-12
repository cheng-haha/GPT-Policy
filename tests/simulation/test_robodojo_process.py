"""Real child process/socket teardown without importing Isaac."""

import json
import socket
import subprocess
import sys

import pytest

from gpt_policy.simulation.robodojo import runtime


@pytest.mark.parametrize("mode", ["clean", "startup_failure"])
@pytest.mark.parametrize("arrival_stop", [True, False])
@pytest.mark.parametrize("backend", ["bounded_dls", "native"])
@pytest.mark.parametrize("distribution", [False, True])
def test_real_process_and_reserved_socket_cleanup(tmp_path, monkeypatch, mode, arrival_stop, backend, distribution):
    root = tmp_path / "native"
    root.mkdir()
    original_popen = subprocess.Popen
    children = []
    native_ids = []

    def launch(command, **kwargs):
        native_id = kwargs["env"]["ROBODOJO_RUN_ID"]
        assert native_id.startswith("gpt-policy-")
        assert native_id != "inherited-fixed"
        assert native_id not in native_ids
        native_ids.append(native_id)
        out = command[command.index("--output") + 1]
        control_file = command[command.index("--control-file") + 1]
        assert json.loads(runtime.Path(control_file).read_text()) == {
            "early_arrival": True, "stop_on_arrival": arrival_stop,
            "ik_backend": backend, "max_joint_increment_rad": 0.15,
            "chunk_feedback": True, "pass_through": arrival_stop,
            "distribute_closed_gripper_moves": distribution,
        }
        assert (
            json.loads((runtime.Path(out) / "launch.json").read_text())["native_run_id"]
            == native_id
        )
        fd = int(command[command.index("--listen-fd") + 1])
        code = """
import json, socket, sys
from pathlib import Path
from gpt_policy.simulation.robodojo.protocol import receive_packet, send_packet, VERSION
output, fd, mode = Path(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
if mode == "startup_failure":
    raise SystemExit(2)
with socket.socket(fileno=fd) as listener:
    (output/"ready.json").write_text("{}")
    connection, _ = listener.accept()
    with connection:
        while True:
            try:
                request = receive_packet(connection)
            except EOFError:
                break
            send_packet(connection, {"version": VERSION, "request_id": request["request_id"],
                                     "ok": True, "result": {"step_id": 0}})
"""
        process = original_popen(
            [sys.executable, "-c", code, out, str(fd), mode], **kwargs
        )
        children.append(process)
        return process

    monkeypatch.setattr(runtime.subprocess, "Popen", launch)
    monkeypatch.setattr(
        runtime,
        "gpu_inventory",
        lambda gpu: {"gpu": gpu, "uuid": "test", "processes": []},
    )
    config = {
        "robodojo_root": str(root),
        "simulation_python": sys.executable,
        "startup_timeout_s": 5,
        "early_arrival": True,
        "stop_on_arrival": arrival_stop,
        "pass_through": arrival_stop,
        "ik_backend": backend, "max_joint_increment_rad": .15, "chunk_feedback": True,
        "simulation_environment": {"ROBODOJO_RUN_ID": "inherited-fixed"},
        "distribute_closed_gripper_moves": distribution,
    }
    out = tmp_path / "sim"
    if mode == "clean":
        with runtime.Simulator(config, {"task": "test"}, out, 93847) as sim:
            assert sim.observation["step_id"] == 0
            assert children[0].poll() is None
            with runtime.Simulator(
                config, {"task": "test"}, tmp_path / "sim2", 93848
            ) as second:
                assert second.port != sim.port
                assert len(set(native_ids)) == 2
    else:
        with pytest.raises(RuntimeError, match="Simulator exited"):
            runtime.Simulator(config, {"task": "test"}, out, 93847)
    assert len(children) == (2 if mode == "clean" else 1)
    assert all(child.poll() is not None for child in children)
    cleanup = json.loads((out / "cleanup.json").read_text())
    assert cleanup["process_exited"]
    port = json.loads((out / "launch.json").read_text())["port"]
    # A new connection must fail: the inherited listener is no longer open.
    with socket.socket() as client:
        client.settimeout(0.2)
        assert client.connect_ex(("127.0.0.1", port)) != 0


def test_busy_gpu_refuses_launch_and_releases_lease(tmp_path, monkeypatch):
    monkeypatch.setattr(
        runtime,
        "gpu_inventory",
        lambda gpu: {
            "gpu": gpu,
            "uuid": "busy",
            "processes": [{"uuid": "busy", "pid": 123}],
        },
    )
    monkeypatch.setattr(
        runtime.subprocess, "Popen", lambda *a, **kw: pytest.fail("Must not launch")
    )
    root = tmp_path / "native"
    root.mkdir()
    config = {"robodojo_root": str(root), "simulation_python": sys.executable}
    for attempt in range(2):
        with pytest.raises(RuntimeError, match="compute processes"):
            runtime.Simulator(config, {"task": "test"}, tmp_path / str(attempt), 93847)
