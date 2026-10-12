"""Isaac-only child entry. The parent owns the single native episode."""

import argparse
import importlib
import socket
import traceback
from pathlib import Path


class NoPolicyConnection:
    def __init__(self, **kwargs):
        pass

    def call(self, func_name, **kwargs):
        if func_name != "reset":
            raise RuntimeError("Autonomous policy loop disabled")

    def close(self):
        pass


def main():
    import json

    from isaaclab.app import AppLauncher

    from .io import write_json

    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case-file", type=Path, required=True)
    parser.add_argument("--budget-file", type=Path, required=True)
    parser.add_argument("--control-file", type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--listen-fd", type=int)
    mode.add_argument("--replay-from", type=Path)
    mode.add_argument("--transport-probe", type=Path)
    parser.add_argument("--render-gpu", type=int, required=True)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.headless, args.enable_cameras = True, True
    case = json.loads(args.case_file.read_text())
    control = json.loads(args.control_file.read_text()) if args.control_file else {}
    from .execution import execution_config
    execution = execution_config(control)
    from env.global_configs import BENCHMARK, ROOT_DIR

    registry = importlib.import_module(f"task.{BENCHMARK}.task_registry")
    import yaml

    task_path = registry.task_config_path(
        str(Path(ROOT_DIR) / "task" / BENCHMARK / "config"), args.task
    )
    values = yaml.safe_load(Path(task_path).read_text())
    monitor = bool(values.get("Articulation"))
    if monitor:
        from src.eval_client.physx_warning_monitor import get_monitor

        get_monitor().start(enabled=True)
    from env.camera_manager.capture.render_sync import (
        add_zero_delay_kit_args,
        zero_delay_active,
    )

    # CUDA_VISIBLE_DEVICES remaps CUDA ordinals, but Vulkan retains physical indices.
    # Kit's explicit final argument overrides AppLauncher's CUDA-based active_gpu=0.
    args.kit_args = (args.kit_args or "") + f" --/renderer/activeGpu={args.render_gpu}"
    add_zero_delay_kit_args(args)
    app = AppLauncher(args, multi_gpu=False).app
    env = session = None
    try:
        import isaaclab
        from env.global_configs import ENV_CONFIG_PATH
        from omegaconf import OmegaConf
        from src.eval_client import eval_env
        from utils.load_file import load_yaml
        from utils.pipeline_utils import process_config, process_randomization

        from .environment import NativeSession
        from .protocol import VERSION, receive_packet, send_packet

        root = Path(ENV_CONFIG_PATH)
        evaluation = load_yaml(str(root / "arx_x5.yml"))
        evaluation.update(
            task_name=args.task,
            num_envs=1,
            device_id=0,
            eval_batch=False,
            policy_name="Pi_05",
            additional_info="gpt_policy",
            seed=case["eval_seed"],
            physx_monitor_enabled=monitor,
        )
        values = {
            key: load_yaml(str(root / key / (evaluation["config"][key] + ".yml")))
            for key in ("sim", "scene", "camera", "robot")
        }
        values.update(
            eval_cfg=evaluation,
            deploy_cfg={"port": 1, "policy_name": "Pi_05"},
            task_env=load_yaml(task_path),
        )
        cfg = OmegaConf.create(values)
        cfg.sim.scene.num_envs = 1
        cfg = process_randomization(cfg)
        cfg, _ = process_config(cfg, task_name=args.task)
        cfg.eval_cfg.eval_num = 1
        cfg.camera.default_frequency = cfg.eval_cfg.observation.collect_freq
        cfg.sim.seed = [0]
        for robot in cfg.robot.robots:
            robot.need_planner = execution["ik_backend"] == "native" and robot.get("type", "target") == "target"
        original = eval_env.WsModelClient
        try:
            eval_env.WsModelClient = NoPolicyConnection
            env = eval_env.create_eval_env(cfg, app)
        finally:
            eval_env.WsModelClient = original
        if not zero_delay_active():
            raise RuntimeError("Native zero-delay renderer settings not active")
        import carb

        render_gpu = carb.settings.get_settings().get("/renderer/activeGpu")
        if render_gpu != args.render_gpu:
            raise RuntimeError(
                f"Renderer selected GPU {render_gpu}, expected {args.render_gpu}"
            )
        write_json(
            args.output / "resolved_config.json",
            OmegaConf.to_container(cfg, resolve=True),
        )
        write_json(
            args.output / "imports.json",
            {
                "robodojo_root": str(Path(ROOT_DIR).resolve()),
                "eval_env": eval_env.__file__,
                "isaaclab": isaaclab.__file__,
                "zero_delay_active": True,
                "native_auxiliary_output": str(Path(env.save_dir).resolve()),
                "render_gpu_index": render_gpu,
                "rgb_path": "native RGB array to Pillow; no XPolicyLab byte transport or BGR conversion",
            },
        )
        session = NativeSession(
            env, args.output, args.task, case,
            budget=json.loads(args.budget_file.read_text()),
            control=control,
        )
        if args.transport_probe is not None:
            from .transport_probe import run_transport_probe

            run_transport_probe(session, args.transport_probe)
            return
        if args.replay_from is not None:
            from .replay import replay_recorded_episode

            replay_recorded_episode(session, args.replay_from)
            return
        with socket.socket(fileno=args.listen_fd) as listener:
            write_json(args.output / "ready.json", {"port": listener.getsockname()[1]})
            connection, _ = listener.accept()
            with connection:
                connection.settimeout(None)  # Waiting for model never steps physics.
                while True:
                    request = receive_packet(connection)
                    response = {
                        "version": VERSION,
                        "request_id": request.get("request_id"),
                    }
                    try:
                        if request.get("version") != VERSION:
                            raise ValueError("RPC protocol version mismatch")
                        result = session.dispatch(
                            request["op"], request.get("args", {})
                        )
                        response.update(ok=True, result=result)
                    except Exception as error:  # noqa: BLE001 - preserve terminal evidence across native/provider failures
                        session.poisoned = True
                        traceback.print_exc()
                        response.update(
                            ok=False, error=f"{type(error).__name__}: {error}"
                        )
                    send_packet(connection, response)
                    if request["op"] == "finish":
                        break
    except (EOFError, ConnectionError):
        if session:
            session.finish("controller_disconnect")
    except BaseException as error:
        write_json(
            args.output / "server_error.json",
            {"type": type(error).__name__, "error": str(error)},
        )
        traceback.print_exc()
        raise
    finally:
        if session:
            session.finish("server_close")
        if env:
            env.close()
        app.close()


if __name__ == "__main__":
    main()
