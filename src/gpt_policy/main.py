"""Command-line setup for the selected robot and agent."""

from __future__ import annotations

import argparse
import getpass
import json
import os
import shutil
import tempfile
from dataclasses import replace
from datetime import datetime
from pathlib import Path

from .hardware.camera import CameraSet
from .hardware.camera_warmup import warm_up_cameras
from .hardware.bimanual import BimanualRobot
from .hardware.robot import ArxRobot
from .hardware.motion_control import MotionFault
from .harness.config import AGENT_NAMES, agent_config, named_agent_config, save_claude_key
from .harness.factory import create_agent, preflight_agent
from .harness.models import AgentContext
from .harness.input_content import FIRST_TURN_RESERVE_CHARS, validate_input_size
from .harness.prompts import robot_instructions
from .harness.protocol import instructions, observation, output_schema, tool_schemas
from .harness.task_prompts import control_prompt_profile
from .harness.video_selector import CodexVideoSelector
from .harness.task_name import DEFAULT_TASK_NAME_MODEL, select_task_request
from .harness.usage import attach_usage, collect_usage, set_usage_run_root, usage_phase, usage_summary
from .harness.waiting import monitor_health
from .paths import generated_var_path
from .input import ImagePart, resolve_run_input
from .input.references import image_path, instruction_media, instruction_mode
from .input.request import normalize_request, save_request, save_task_request
from .input.preparation import has_video_input, prepare_input_videos
from .input.video import FfmpegVideoExtractor, VideoProcessingConfig
from .input.demonstration import demonstration_instruction, save_input
from .recording.trace import RunRecorder
from .recording.video import RunVideo
from .recording.state import StateRecorder
from .settings import camera_defaults, load_settings, runtime_config, settings_path
from .tools import ToolExecutor, load_tool_catalog
from .runtime.runner import require_home_settled, run_loop, without_trace as _without_trace
from .runtime.console import RunConsole
from .interrupts import HomeInterrupted, Interrupts, defer_interrupts
from .vision.perception import PixelLocalizer


REQUEST_DIRECTORY = Path(__file__).resolve().parents[2] / "request_json"


def parse_args(default_agent: str | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Control ARX or YAM arms using a machine profile, selected agent, and mixed input JSON"
    )
    parser.add_argument(
        "instruction",
        nargs="?",
        help='任务文字，可直接包含本地图片或示范路径，例如 gpt-policy "参考 ./target.png，摆放积木"',
    )
    parser.add_argument(
        "--input-json",
        "--input",
        dest="input_json",
        type=Path,
        help="混合文字/图片/视频输入 JSON；媒体路径相对于该 JSON 文件解析",
    )
    parser.add_argument(
        "--image",
        type=Path,
        action="append",
        metavar="PATH",
        help="加入本地参考图片，可重复使用；路径相对于当前工作目录解析",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="主运行配置 JSON；也可通过 GPT_POLICY_CONFIG 或项目默认配置提供",
    )
    parser.add_argument(
        "--set-claude-key",
        action="store_true",
        help="隐藏输入并保存 Claude 网关 API Key，然后退出",
    )
    parser.add_argument("--machine", help="选择 configs/machines/<name>.json 中的本地机器配置")
    parser.add_argument("--agent", choices=AGENT_NAMES, default=default_agent, help="覆盖本次运行的 agent 配置")
    parser.add_argument("--check", action="store_true", help="只检查配置，不打开相机、CAN 或模型会话")
    parser.add_argument("--demo", type=Path, help="Historical demo.json, recorded run directory, or video")
    parser.add_argument("--demo-mode", choices=("video", "video+action"), help="Override the mode in task text or JSON (default: video)")
    parser.add_argument("--prepare-only", type=Path, metavar="DIRECTORY", help="Save portable input and exit before opening hardware")
    return parser.parse_args()


def main(default_agent: str | None = None) -> None:
    args = parse_args(default_agent)
    with Interrupts() as interrupts:
        _run(args, interrupts)


def claude_main() -> None:
    main("claude")


def kimi_main() -> None:
    main("kimi")


@collect_usage()
def _run(args, interrupts) -> None:
    if getattr(args, "set_claude_key", False):
        key = getpass.getpass("Claude API Key: ")
        filename = save_claude_key(key)
        print(f"Claude API Key 已保存到 {filename}")
        return
    machine = getattr(args, "machine", None)
    if machine and args.config:
        raise ValueError("Choose --machine or --config, not both")
    config_path = settings_path(args.config, machine) if machine else settings_path(args.config)
    settings = load_settings(config_path)
    if selected_agent := getattr(args, "agent", None):
        settings["agent"] = selected_agent
    runtime = runtime_config(settings, legacy=args, base_dir=config_path.parent)
    if getattr(args, "check", False):
        from .preflight import check_configuration
        print(json.dumps(check_configuration(settings, config_path), ensure_ascii=False, indent=2))
        return
    agent_settings = agent_config(settings, config_path.parent)
    runs_root = generated_var_path("runs") / {"codex": "gpt", "claude": "claude", "kimi": "kimi"}[settings.get("agent", "codex")]
    set_usage_run_root(runs_root)
    display = RunConsole()
    instruction = getattr(args, "instruction", None)
    input_json = getattr(args, "input_json", None)
    inline_images = ()
    if input_json is None and not getattr(args, "demo", None):
        args.demo, input_json, inline_images = instruction_media(instruction)
    if instruction is None and input_json is None and getattr(args, "demo", None):
        instruction = demonstration_instruction(args.demo)
    if instruction is None and input_json is None:
        input_json = runtime.input_json
    run_input = resolve_run_input(
        instruction,
        input_json,
        agent_settings.model,
    )
    mode = getattr(args, "demo_mode", None) or instruction_mode(getattr(args, "instruction", None))
    run_input = normalize_request(run_input, getattr(args, "demo", None), mode)
    flagged_images = tuple(image_path(path) for path in (getattr(args, "image", None) or ()))
    images = tuple(dict.fromkeys((*inline_images, *flagged_images)))
    if images:
        run_input = replace(run_input, content=run_input.content + tuple(
            ImagePart(path, label=path.name) for path in images
        ))
    request_path = Path(input_json).expanduser().resolve() if input_json is not None else None
    if getattr(args, "prepare_only", None):
        destination = args.prepare_only.expanduser().resolve()
        if destination.exists():
            raise ValueError(f"Prepared input directory already exists: {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        temp = Path(tempfile.mkdtemp(prefix=".input-", dir=destination.parent))
        try:
            attach_usage(temp)
            if request_path is None:
                request_path = save_task_request(run_input, REQUEST_DIRECTORY, destination.name)
            display.message(f"Request: {request_path}")
            save_request(run_input, temp / "request.json")
            run_input = _prepare_inputs(run_input, args, temp, settings, config_path, agent_settings, display)
            save_input(run_input, temp)
            os.rename(temp, destination)
            attach_usage(destination)
        finally:
            if temp.exists():
                shutil.rmtree(temp)
        display.message(f"Prepared input saved: {destination / 'input.json'}")
        return
    preflight_agent(agent_settings)
    camera_values = runtime.camera_overrides or camera_defaults(settings)
    task_name = runtime.task_name
    if request_path is not None:
        if task_name == "task":
            task_name = request_path.stem
    else:
        codex_settings = (agent_settings if agent_settings.type == "codex" else
                          named_agent_config("codex", Path(settings.get("agent_config_dir", config_path.parent))))
        with display.waiting(f"Finding task request ({codex_settings.task_name_model or DEFAULT_TASK_NAME_MODEL})"):
            selected_name, request_path, run_input = select_task_request(run_input, REQUEST_DIRECTORY, codex_settings, mode)
        if task_name == "task":
            task_name = selected_name
        if request_path is None:
            request_path = save_task_request(run_input, REQUEST_DIRECTORY, task_name)
    run_name = f"{datetime.now():%Y%m%d-%H%M%S-%f}-{task_name}"
    display.message(f"Request: {request_path}")
    record_dir = runtime.record_dir or runs_root / run_name
    recorder = RunRecorder(
        record_dir,
        {
            "instruction": run_input.instruction,
            "task_name": task_name,
            "max_decisions": runtime.max_decisions,
            "model": run_input.model,
            "agent": {"profile": settings.get("agent"), "type": agent_settings.type, "model": run_input.model},
            "input": run_input.record(),
            "request_json": str(request_path),
            "robot_model": runtime.robot_model,
            "interface": runtime.interface,
            "right_interface": runtime.right_interface,
            "trajectory_hz": runtime.trajectory_hz,
            "camera_args": camera_values,
            "settings": settings,
        },
    )
    robot = None
    cameras = None
    video = None
    agent = None
    tool_executor = None
    state_recording = None
    task_started = False
    status = "completed"
    failure: str | None = None
    motion_fault = None
    try:
        display.header(run_input.instruction, run_input.model, settings.get("machine", runtime.robot_model), record_dir, runtime.max_decisions)
        run_input = _prepare_inputs(run_input, args, record_dir, settings, config_path, agent_settings, display, recorder)
        camera_specs = parse_camera_specs(camera_values)
        if settings.get("camera_backend", "v4l2") == "realsense":
            from .hardware.realsense import RealSenseCameraSet
            cameras = RealSenseCameraSet(camera_specs, runtime.camera_width, runtime.camera_height)
        else:
            cameras = CameraSet(camera_specs, runtime.camera_width, runtime.camera_height)
            if controls := settings.get("camera_controls"):
                from .hardware.d405_controls import configure_d405_cameras
                applied = configure_d405_cameras(camera_specs, **controls)
                recorder.write("camera_controls_applied", {"cameras": applied})
        recorder.write("camera_warmup_started", {"cameras": cameras.describe()})
        with display.waiting("Warming up cameras; waiting for exposure and white balance to stabilize"):
            warmup = warm_up_cameras(cameras)
        recorder.write("camera_warmup_completed", warmup)
        display.message("Cameras ready. Starting recording.")
        recording_settings = settings.get("recording", {})
        video = RunVideo(
            cameras, record_dir, int(recording_settings.get("fps", 10)),
            int(recording_settings.get("jpeg_quality", 65)),
        )
        recorder.set_recording(video.details)
        backend = settings.get("backend", "arx")
        if backend == "yam":
            robot = _open_yam(runtime, settings)
            arms = ("left", "right") if runtime.right_interface else ("left",)
            interface_text = f"left={runtime.interface}, right={runtime.right_interface}"
        elif backend != "arx":
            raise ValueError(f"Unknown robot backend: {backend}")
        elif runtime.right_interface:
            robot = BimanualRobot(
                runtime.robot_model,
                runtime.interface,
                runtime.right_interface,
                runtime.gripper_open_readout,
                runtime.trajectory_hz,
                settings,
            )
            arms = ("left", "right")
            interface_text = f"left={runtime.interface}, right={runtime.right_interface}"
        else:
            robot = ArxRobot(
                runtime.robot_model,
                runtime.interface,
                runtime.gripper_open_readout,
                runtime.trajectory_hz,
                settings,
            )
            arms = ("left",)
            interface_text = runtime.interface
        if recording_settings.get("state_hz"):
            state_recording = StateRecorder(robot, record_dir, float(recording_settings["state_hz"]))
        localizer = PixelLocalizer(settings)
        tool_catalog = load_tool_catalog(settings)
        tool_executor = ToolExecutor(tool_catalog, arms, robot, localizer)
        agent = create_agent(
            agent_settings,
            run_input.model,
            runtime.convert_camera_images_to_jpeg,
            runtime.camera_jpeg_quality,
        )
        schema = output_schema(robot.dof, arms, tool_catalog)
        function_catalog = tool_schemas(robot.dof, arms, tool_catalog)
        base_instructions = instructions(
            runtime.robot_model, interface_text, robot.dof, arms, settings, tool_catalog,
            task_instruction=run_input.instruction,
        )
        base_instructions = robot_instructions(base_instructions, agent_settings.type)
        base_instructions += (
            f"\nThis task allows at most {runtime.max_decisions} decisions; observation.extra.env_step starts at 0. "
            "Each tool selection counts, including rejected arguments or IK failures. "
            "When the budget is exhausted, the host stops model calls, returns home and saves the recording."
        )
        recorder.write(
            "protocol",
            {
                "output_schema": schema,
                "tool_catalog": function_catalog,
                "base_instructions": base_instructions,
                "control_prompt_profile": control_prompt_profile(run_input.instruction) or "default",
                "cameras": cameras.describe(),
            },
        )
        if run_input.manifest is not None:
            recorder.write("input_manifest", run_input.manifest.record())
        with monitor_health(state_recording.check if state_recording is not None else None):
            agent.start(AgentContext(base_instructions, function_catalog, schema))
        task_started = True
        if state_recording is not None:
            status = run_loop(runtime, run_input, robot, cameras, video, agent, tool_executor, recorder, observation,
                              health_check=state_recording.check, display=display)
        else:
            status = run_loop(runtime, run_input, robot, cameras, video, agent, tool_executor, recorder, observation, display=display)
    except KeyboardInterrupt as exc:
        status = "interrupted"
        recorder.write("interrupted", {"trigger": "ctrl_c"})
        try:
            if isinstance(exc, HomeInterrupted) or interrupts.count > 1:
                recorder.write("return_home_cancelled", {"trigger": "ctrl_c"})
                display.message("Return home cancelled. Saving recordings and closing devices.", "yellow")
            elif robot is not None:
                display.message("Task interrupted. Stopping motion and returning home; press Ctrl+C again to cancel homing.", "yellow")
                with defer_interrupts():
                    robot.cancel()
                robot.resume()
                recorder.write("return_home_started", {"trigger": "ctrl_c"})
                result = robot.return_home()
                recorder.write("return_home", {"trigger": "ctrl_c", "result": result})
                display.home(_without_trace(result))
                require_home_settled(result)
                display.message("Home complete. Saving video and logs.")
            else:
                recorder.write("return_home_skipped", {"reason": "robot_not_initialized"})
                display.message("Initialization interrupted. Saving recordings.", "yellow")
        except KeyboardInterrupt:
            recorder.write("return_home_cancelled", {"trigger": "ctrl_c"})
            display.message("Return home cancelled. Saving recordings and closing devices.", "yellow")
        except Exception as home_error:
            status, failure = "failed", repr(home_error)
            if isinstance(home_error, MotionFault):
                motion_fault = home_error
                recorder.write("motion_fault", home_error.details)
            recorder.write("return_home_error", {"trigger": "ctrl_c", "error": failure})
            display.event("RETURN HOME", "FAILED", {"error": str(home_error)}, "red")
    except Exception as exc:
        status = "failed"
        failure = repr(exc)
        if isinstance(exc, MotionFault):
            motion_fault = exc
            recorder.write("motion_fault", exc.details)
            display.event("ROBOT", "PROTECTIVE STOP", exc.details, "red")
        recorder.write("run_error", {"error": failure})
        raise
    finally:
        # Further Ctrl+C must not abandon recording finalization or a live
        # command worker. Hardware emergency stop remains independent.
        interrupts.ignoring = True
        if status not in {"completed", "give_up"} and robot is not None and hasattr(robot, "cancel"):
            try:
                robot.cancel()
            except Exception as exc:
                status, failure = "failed", failure or repr(exc)
                recorder.write("cleanup_error", {"resource": "robot_cancel", "error": repr(exc)})
        if video is not None:
            try:
                recorder.set_recording(video.stop())
            except Exception as exc:
                recording_error = repr(exc)
                recorder.set_recording({**video.details, "error": recording_error})
                recorder.write("recording_error", {"error": recording_error})
                if failure is None:
                    status = "failed"
                    failure = recording_error
        if state_recording is not None:
            try:
                recorder.write("state_recording", state_recording.stop())
            except Exception as exc:
                status, failure = "failed", failure or repr(exc)
                recorder.write("recording_error", {"error": repr(exc)})
        for name, resource in (("cameras", cameras), ("agent", agent), ("robot", robot)):
            try:
                if resource is not None and hasattr(resource, "close"):
                    resource.close()
            except Exception as exc:
                status, failure = "failed", failure or repr(exc)
                recorder.write("cleanup_error", {"resource": name, "error": repr(exc)})
        recorder.write("execution_finished", {"status": status, "error": failure})
        if task_started:
            # No motion or recording worker should wait on a human's verdict.
            # Ctrl+C here skips rating only; cleanup has already completed.
            try:
                interrupts.ignoring = False
                outcome = display.task_result()
            except (EOFError, KeyboardInterrupt):
                outcome = None
            finally:
                interrupts.ignoring = True
            if outcome is not None:
                recorder.write("human_evaluation", {"outcome": outcome, "source": "terminal"})
            else:
                recorder.write("human_evaluation_skipped", {"reason": "no_terminal_or_cancelled"})
        final_directory = recorder.close(status, failure)
        display.finished(status, final_directory,
                         task_status=getattr(recorder, "task_status", None), error=failure)
        display.usage(usage_summary())
    if status == "failed":
        raise RuntimeError(failure)


@usage_phase("demonstration")
def _prepare_inputs(run_input, args, directory, settings, config_path, agent_settings, display, recorder=None):
    if not has_video_input(run_input):
        if agent_settings.type == "codex":
            validate_input_size(run_input.content, run_input.instruction, reserve_chars=FIRST_TURN_RESERVE_CHARS)
        save_input(run_input, directory)
        return run_input
    processing = VideoProcessingConfig(**settings.get("video_input", {}))
    codex_settings = (agent_settings if agent_settings.type == "codex" else
                      named_agent_config("codex", Path(settings.get("agent_config_dir", config_path.parent))))
    selector = CodexVideoSelector(codex_settings, processing)
    extractor = FfmpegVideoExtractor(processing)
    with display.waiting("Preparing demonstration context"):
        run_input, results = prepare_input_videos(run_input, directory / "input-videos", selector, extractor)
        for result in results:
            metadata = result.record(relative_to=directory)
            if recorder is not None:
                recorder.write("video_preprocessed", metadata)
            keyframes = metadata.get("keyframes", len(metadata.get("selection", {}).get("selected", [])))
            display.message(
                f"Demonstration ready ({metadata['mode']}): {keyframes} keyframes, "
                f"{metadata.get('state_keyframes', 0)} state frames, "
                f"{metadata.get('output_action_samples', 0)} action samples"
            )
            if metadata.get("numeric_data_omitted"):
                display.message(
                    "WARNING: video mode omitted available demonstration states/actions. "
                    "Use --demo-mode video+action to include them.", "yellow",
                )
    # Offline replay/debugging uses the exact first-turn image bytes.
    if agent_settings.type == "codex":
        validate_input_size(run_input.content, run_input.instruction, reserve_chars=FIRST_TURN_RESERVE_CHARS)
    save_input(run_input, directory)
    return run_input


def _open_yam(runtime, settings):
    from .hardware.yam import YamRobot

    interfaces = {"left": runtime.interface}
    if runtime.right_interface:
        interfaces["right"] = runtime.right_interface
    arms = {}
    try:
        for side, interface in interfaces.items():
            arms[side] = YamRobot(interface, settings, side)
        return BimanualRobot.from_arms(arms, interfaces, settings) if len(arms) == 2 else arms["left"]
    except BaseException:
        for arm in arms.values():
            arm.close()
        raise


def parse_camera_specs(values: list[str] | None) -> list[tuple[str, str]]:
    if not values:
        return []
    specs = []
    for value in values:
        if "=" not in value:
            raise ValueError(f"相机参数必须是 NAME=/dev/videoN: {value}")
        name, path = value.split("=", 1)
        specs.append((name or path.rsplit("/", 1)[-1], path))
    return specs


if __name__ == "__main__":
    main()
