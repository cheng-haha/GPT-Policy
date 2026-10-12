"""Dedicated simulator process and single-owner no-retry RPC lifecycle."""

import fcntl
import os
import signal
import socket
import subprocess
import time
import uuid
from pathlib import Path

from .arrival import (
    early_arrival_enabled,
    pass_through_enabled,
    stop_on_arrival_enabled,
)
from .budget import budget_config
from .execution import execution_config
from .io import write_json
from .protocol import RPCClient


def gpu_inventory(gpu):
    uuid = subprocess.check_output(
        [
            "nvidia-smi",
            "-i",
            str(gpu),
            "--query-gpu=uuid",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        timeout=10,
    ).strip()
    rows = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        timeout=10,
    ).splitlines()
    processes = [
        {"uuid": parts[0].strip(), "pid": int(parts[1].strip())}
        for row in rows
        if len(parts := row.split(",")) == 2
    ]
    return {"gpu": gpu, "uuid": uuid, "processes": processes}


class Simulator:
    def __init__(self, config, case, output, gpu):
        budgets = budget_config(config)
        self.early_arrival = early_arrival_enabled(config)
        arrival_stop = stop_on_arrival_enabled(config)
        self.config, self.output = config, Path(output)
        self.process = self.client = self.log = self.gpu_lock = None
        self.output.mkdir(parents=True, exist_ok=False)
        write_json(self.output / "case.json", case)
        write_json(self.output / "budget-config.json", budgets)
        write_json(self.output / "control-config.json", {
            **execution_config(config),
            "early_arrival": self.early_arrival, "stop_on_arrival": arrival_stop,
            "pass_through": pass_through_enabled(config),
        })
        root = Path(config["robodojo_root"]).resolve()
        project = Path(__file__).resolve().parents[4]
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        paths = [
            project / "src",
            root,
            root / "XPolicyLab",
            root / "third_party/curobo",
        ]
        paths += sorted((root / "third_party/IsaacLab/source").glob("*"))
        env["PYTHONPATH"] = os.pathsep.join(map(str, paths))
        env.update(config.get("simulation_environment", {}))
        # Native second-resolution defaults collide for concurrent identical tasks.
        env["ROBODOJO_RUN_ID"] = "gpt-policy-" + uuid.uuid4().hex
        env["XDG_RUNTIME_DIR"] = str(self.output / "xdg")
        Path(env["XDG_RUNTIME_DIR"]).mkdir(mode=0o700)
        # Parent reserves the port until the child inherits it; no bind race.
        listener = socket.socket()
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        self.port = listener.getsockname()[1]
        self.log = (self.output / "server.log").open("x")
        command = [
            config["simulation_python"],
            "-u",
            "-m",
            "gpt_policy.simulation.robodojo.server",
            "--task",
            case["task"],
            "--control-file",
            str(self.output / "control-config.json"),
            "--budget-file",
            str(self.output / "budget-config.json"),
            "--case-file",
            str(self.output / "case.json"),
            "--output",
            str(self.output),
            "--listen-fd",
            str(listener.fileno()),
            "--device",
            "cuda:0",
            "--render-gpu",
            str(config.get("render_gpu_indices", {}).get(str(gpu), gpu)),
        ]
        write_json(
            self.output / "launch.json",
            {
                "command": command,
                "gpu": gpu,
                "port": self.port,
                "pythonpath": env["PYTHONPATH"],
                "cwd": str(root),
                "native_run_id": env["ROBODOJO_RUN_ID"],
                "pid": None,
            },
        )
        try:
            self.gpu_lock = Path(f"/tmp/gpt-policy-gpu-{os.getuid()}-{gpu}.lock").open(  # noqa: SIM115 - owned until close()
                "a"
            )
            try:
                fcntl.flock(self.gpu_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise RuntimeError(
                    f"GPU {gpu} already leased by another GPT-Policy simulator"
                ) from error
            inventory = gpu_inventory(gpu)
            write_json(self.output / "gpu_before.json", inventory)
            busy = [
                p["pid"]
                for p in inventory["processes"]
                if p["uuid"] == inventory["uuid"]
            ]
            if busy:
                raise RuntimeError(f"GPU {gpu} already has compute processes: {busy}")
            self.process = subprocess.Popen(
                command,
                cwd=root,
                env=env,
                stdout=self.log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                pass_fds=(listener.fileno(),),
            )
        except BaseException:
            self.close()
            raise
        finally:
            listener.close()
        write_json(
            self.output / "pid.json", {"pid": self.process.pid, "port": self.port}
        )
        try:
            deadline = time.monotonic() + config.get("startup_timeout_s", 300)
            while not (self.output / "ready.json").exists():
                self.check_alive()
                if time.monotonic() > deadline:
                    raise TimeoutError(
                        "Native initialization deadline exceeded; inspect server.log"
                    )
                time.sleep(0.25)
            inventory = gpu_inventory(gpu)
            write_json(self.output / "gpu_ready.json", inventory)
            own_devices = {
                p["uuid"]
                for p in inventory["processes"]
                if p["pid"] == self.process.pid
            }
            if own_devices and own_devices != {inventory["uuid"]}:
                raise RuntimeError(f"Simulator escaped assigned GPU: {own_devices}")
            self.client = RPCClient("127.0.0.1", self.port, timeout=600)
            self.observation = self.client.request("reset")
        except BaseException:
            self.close()
            raise

    def check_alive(self):
        if self.process is None or self.process.poll() is not None:
            raise RuntimeError(
                f"Simulator exited; inspect {self.output / 'server.log'}"
            )
        if self.client:
            self.client.check_alive()

    def move(self, arguments):
        from .control import validate_path

        validate_path(arguments, self.observation, early_arrival=self.early_arrival)
        result = self.client.request(
            "move",
            observation_id=self.observation["observation_id"],
            arguments=arguments,
        )
        self.observation = result["observation"]
        return result

    def finish(self, reason):
        return self.client.request(
            "finish", observation_id=self.observation["observation_id"], reason=reason
        )

    def close(self):
        if self.client:
            self.client.close()
            self.client = None
        if self.process:
            try:
                self.process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(self.process.pid, signal.SIGTERM)
                try:
                    self.process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(self.process.pid, signal.SIGKILL)
                    self.process.wait(timeout=10)
            write_json(
                self.output / "cleanup.json",
                {
                    "pid": self.process.pid,
                    "exit_code": self.process.returncode,
                    "process_exited": self.process.poll() is not None,
                },
            )
            self.process = None
        if self.log:
            self.log.close()
            self.log = None

        if self.gpu_lock:
            self.gpu_lock.close()
            self.gpu_lock = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
