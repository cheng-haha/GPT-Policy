"""GPT-Policy-shaped tool and observation protocol for the standalone bridge."""

from __future__ import annotations

import json
from typing import Any

from ..tools.catalog import ToolCatalog, load_tool_catalog
from .task_prompts import PLUG_CONTROL, PLUG_DONE, control_prompt_profile


def _scene_safety_notes(settings: dict[str, Any] | None = None) -> str:
    """Render hardware-specific facts that cannot be inferred from camera images."""
    scene = (settings or {}).get("scene", {})
    if not isinstance(scene, dict):
        raise ValueError("scene 必须是对象")
    notes = scene.get("safety_notes", [])
    if not isinstance(notes, list) or any(
        not isinstance(note, str) or not note.strip() for note in notes
    ):
        raise ValueError("scene.safety_notes 必须是非空字符串数组")
    if not notes:
        return ""
    lines = "\n".join(f"- {note.strip()}" for note in notes)
    return f"""Fixed scene and hidden obstacles (persistent physical facts):
{lines}
- An obstacle absent from every camera view is not absent from the scene. Keep all controlled arms' links, wrists, grippers and held objects clear of these obstacles throughout every waypoint and path.
"""


def _robot_calibration_notes(arms: tuple[str, ...], settings: dict[str, Any] | None = None) -> str:
    if (settings or {}).get("backend") == "robodojo":
        sim = (settings or {}).get("robodojo", {})
        cameras = sim.get("cameras", ["cam_head", "cam_left_wrist", "cam_right_wrist"])
        return f"""Robot and calibration conventions:
- RoboDojo controls {len(arms)} simulated X5 arm(s); each arm has its own base_link frame.
- All GPT-Policy poses are absolute TCP poses in the selected arm base_link frame, in metres.
- GPT-Policy quaternion order is [qx,qy,qz,qw]. RoboDojo's source ee_pose is [x,y,z,qw,qx,qy,qz]; the host performs this conversion.
- TCP is the calibrated fingertip TCP: its +z axis points toward the fingertips and +y is the gripper opening axis.
- The host performs all world/base/TCP conversion, IK validation, interpolation, and gripper normalization. Do not emit simulator-world poses.
Camera calibration and views:
- Available simulated cameras: {", ".join(str(name) for name in cameras)}.
- cam_head is fixed; wrist cameras move with their corresponding simulated arm.
- RoboDojo intrinsics and camera-to-world extrinsics are captured from the simulator and converted by the host to base-from-camera rays.
- locate_point returns a calibrated ray in an arm base_link frame. A single RGB image still does not establish metric depth.
- Camera matrices are retained in the host observation and run record; use the named camera/frame semantics rather than guessing pixel-to-metre scale.
"""
    camera_lines = [
        "- left/right: D405 RGB cameras mounted on the corresponding wrists; they move with the end effectors.",
        "- top: fixed overhead D405 RGB camera with stored extrinsics to both arm base_link frames.",
        "- Images are RGB, not depth maps. Pixel coordinates are not metric base-frame coordinates.",
        "- locate_point uses stored intrinsics/distortion for all three cameras, wrist extrinsics relative to link6, and top-to-base extrinsics.",
        "- A single RGB image has no depth. Triangulation requires the same stationary feature in two wrist views with sufficient parallax.",
    ]
    if (settings or {}).get("backend") == "yam":
        camera_lines = [line.replace("relative to link6", "relative to grasp_site") for line in camera_lines]
    tcp_note = "TCP is the midpoint between the inner fingertips, not the SDK eef_link origin; the host converts between them using project calibration."
    if (settings or {}).get("backend") == "yam":
        tcp_note = "TCP uses the calibrated grasp_site reference consistent with I2RT ac096928, at [0,0,-0.1347] m in the flange frame. Do not substitute another version's fingertip origin."
    runtime = (settings or {}).get("runtime", {})
    model = runtime.get("robot_model", "X5")
    model = "ARX X5" if model == "X5" else model
    left_can = runtime.get("interface", "can1")
    right_can = runtime.get("right_interface", "can3")
    return f"""Robot and calibration conventions:
- {model} has 6 revolute joints per arm; left uses {left_can}, right uses {right_can}, when enabled.
- SDK joint order is J1..J6, in radians. Do not convert to degrees or reinterpret encoder values as a 0/180-degree mode.
- The host seeds IK with live joint_pos. Output TCP targets, not joint angles.
- Each arm has its own base_link: +x forward, +y left, +z up. Reported and commanded TCP poses use that arm's frame.
- {tcp_note}
- Tool +z points from wrist to fingertips; tool +y is the jaw opening axis. pose_xyzrpy=[x,y,z,roll,pitch,yaw] reports absolute TCP pose in metres/radians, with angles ordered roll, pitch, yaw.
- Command pose_xyzquat=[x,y,z,qx,qy,qz,qw], with position in metres and a unit quaternion mapping TCP tool axes into that arm's base_link, not relative to its starting orientation.
  For example, xyzw=[1,0,0,0] points tool +z along base -z and tool +x along base +x. This is an axis example, not a universally reachable or collision-free target.
- The host densifies each straight-line/SLERP segment and solves IK at each sample. It preserves target poses and order, changing timing only; actual IK may have the small residuals explicitly allowed by configuration.
- Bimanual waypoint arrival times are synchronized by slowing the faster arm. A null side holds its previously submitted TCP/gripper targets for that segment.
Camera calibration and views:
{chr(10).join(camera_lines)}
"""


def output_schema(
    dof: int,
    arms: tuple[str, ...] = ("left",),
    catalog: ToolCatalog | None = None,
) -> dict[str, Any]:
    """Return GPT-Policy's outer Codex selection schema."""
    return (catalog or load_tool_catalog()).output_schema(dof, arms)


def tool_schemas(
    dof: int,
    arms: tuple[str, ...] = ("left",),
    catalog: ToolCatalog | None = None,
) -> list[dict[str, Any]]:
    """Return the OpenAI function catalog injected into the Codex thread."""
    return (catalog or load_tool_catalog()).function_schemas(dof, arms)


def instructions(
    model: str,
    interface: str,
    dof: int,
    arms: tuple[str, ...] = ("left",),
    settings: dict[str, Any] | None = None,
    catalog: ToolCatalog | None = None,
    *,
    task_instruction: str = "",
) -> str:
    """Describe the standalone tool protocol to Codex."""
    active_catalog = catalog or load_tool_catalog(settings)
    arm_text = ", ".join(arms)
    multi_text = "Bimanual parameters use left/right for the two arms; null holds that side's submitted targets." if len(arms) == 2 else "Only one arm is controlled."
    calibration = _robot_calibration_notes(arms, settings)
    scene_safety = _scene_safety_notes(settings)
    robot_label = "YAM" if (settings or {}).get("backend") == "yam" else "ARX"
    plug_profile = control_prompt_profile(task_instruction)
    tool_text = active_catalog.prompt_catalog(dof, arms, overrides={"done": PLUG_DONE} if plug_profile else None)
    common = f"""You are a {robot_label} robot controller using the native Codex harness. You control {model} on {interface}, arms: {arm_text}, with {dof} joints per arm.

{calibration}
{scene_safety}

Return exactly one tool selection per turn: {{"name": "...", "arguments": {{...}}}}. No Markdown or text outside this object. The host executes the full motion segment, then supplies a fresh observation.

Tools:
{tool_text}

State joint_pos/tcp_pose are live joint and calibrated TCP feedback; joint_vel is measured near the same observation time. Gripper commands use normalized opening from 0 to 1.
{multi_text}

"""
    if plug_profile:
        return common + PLUG_CONTROL
    return common + """Orientation retention and reachability: For position-only adjustments, keep the last acknowledged target quaternion in the complete pose_xyzquat. Never replace a held orientation with load-induced measured drift; investigate drift first. After IK rejection, preserve required tool-axis directions and compare approach positions, heights and permitted axial rotations. Use check_path for uncertain alternatives; do not tilt the gripper merely to make IK pass. The configured workspace is not a measured reachability boundary; an in-range point may be unreachable at the required orientation. IK acceptance does not certify clearance between camera housings, arms or objects.

Every motion tool requires a note: 1-2 short Chinese sentences, usually 20-60 characters, stating current evidence and the next purpose. Keep failure causes when relevant; do not repeat coordinate arrays, history, tool mechanics or general rules. Use only the selected tool's arguments. Follow its schema for omitted and nullable fields. Camera images are supplied through observation.images and actual image inputs.

Efficient motion: Prefer move_eef_chunk for clear, contact-free paths that need no intermediate observation. Do not split them into many decisions or repeatedly check already-established reachability. Stop to observe at contact, gripper changes, occlusion or tracking anomalies; efficiency never justifies skipping verification.

Grasp sequence: Approach and align, inspect fresh images and state, then call set_gripper separately. Before closure, consider wrist close-ups, the top overview, measured TCP target error and joint_vel together. A submitted target does not prove arrival. Target-versus-measured errors are reported after each action. settled means joint stability, not precise TCP arrival or task success. Observation numbers use 6 decimal places; logs retain full precision.
Check fingertips, wrists, camera housings, forearms and table clearance in fresh views. Do not advance contact without verified stabilization and clearance.
previous_result.trajectory.planned_tcp_points_xyzrpy is the planned path including its start, not a measured trajectory. Judge execution from execution_feedback and fresh state.

Release verification: Before releasing or regrasping, verify that the intended surface or another hand reliably supports the entire object. Observe after release and before withdrawing. A fully-open command does not prove that a wide object detached. If the object stays fixed relative to the fingers or the destination still appears empty, keep it reliably supported by the intended surface or container while disengaging the fingers. Do not lift a still-trapped object away and report completion.

Persistence: One failed action, tool rejection, missed grasp, occluded target or uncertain result does not establish impossibility. Diagnose from fresh images, measured state and previous_result, then try safe alternatives in viewpoint, approach, grasp, orientation, path or step size and verify each result. Call done only when the physical goal is established. Do not call give_up while reasonable safe strategies remain. Consider meaningfully different recoveries; give_up requires evidence that the task cannot be completed or further attempts would violate safety constraints. In reason, list attempted strategies and the evidence preventing further progress.
"""


def observation(
    instruction: str,
    state: dict[str, Any],
    cameras: list[dict[str, Any]],
    previous: str | None = None,
    step: int = 0,
    metadata: dict[str, Any] | None = None,
) -> str:
    """Format the GPT-Policy-style JSON observation sent to Codex."""
    payload: dict[str, Any] = {
        "instruction": instruction,
        "images": [{key: value for key, value in camera.items() if key not in {"serial", "device"}}
                   for camera in cameras],
        "state": _state_payload(state),
        "extra": {
            "env_step": step,
            "interface": state.get("interface"),
            "interfaces": state.get("interfaces"),
        },
    }
    if metadata:
        payload["extra"].update(metadata)
    if previous is not None:
        previous_payload = _without_temperature_telemetry(json.loads(previous))
        if isinstance(previous_payload, dict):
            result = previous_payload.get("result", previous_payload)
            if isinstance(result, dict) and isinstance(result.get("execution_feedback"), dict):
                result["execution_feedback"] = _feedback_payload(result["execution_feedback"])
        payload["previous_result"] = previous_payload
    return json.dumps(_model_numbers(payload), ensure_ascii=False, separators=(",", ":"))


def _without_temperature_telemetry(value: Any) -> Any:
    """Omit temperature telemetry and threshold flags from model results.

    Older/full results can contain nested arm snapshots. Copy only the model
    view; hardware checks and raw recordings retain the original telemetry.
    """
    if isinstance(value, dict):
        return {key: _without_temperature_telemetry(item) for key, item in value.items()
                if key not in {"temperature_rotor_c", "temperature_mos_c", "temperature_limit_c",
                               "temperature_over_limit"}}
    if isinstance(value, list):
        return [_without_temperature_telemetry(item) for item in value]
    return value


def _feedback_payload(feedback: dict[str, Any]) -> dict[str, Any]:
    """Omit normal settling bookkeeping; retain every failed-settle diagnostic."""
    result = dict(feedback)
    # Keep the pre-experiment model input; derived progress stays in raw logs.
    result.pop("motion_progress", None)
    for side in ("left", "right"):
        if isinstance(result.get(side), dict):
            result[side] = _feedback_payload(result[side])
    settle = result.get("settle")
    if isinstance(settle, dict) and settle.get("settled") is True:
        result["settle"] = {key: value for key, value in settle.items() if key not in {
            "method", "samples", "required_samples", "required_window_s", "window_s",
            "target_joint_positions_rad",
        }}
    return result


def _model_numbers(value: Any) -> Any:
    """Round model feedback to 1e-6 (<= 0.5 micrometre/microradian error).

    Commands, raw states and recorded execution results never pass through here.
    """
    if isinstance(value, dict):
        return {key: _model_numbers(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_model_numbers(item) for item in value]
    return round(value, 6) if isinstance(value, float) else value


def _state_payload(state: dict[str, Any]) -> dict[str, Any]:
    if "arms" in state:
        payload: dict[str, Any] = {side: _single_state_payload(value) for side, value in state["arms"].items()}
        for key in ("backend", "robot_model", "calibration", "simulator_raw"):
            if key in state:
                payload[key] = state[key]
        return payload
    return _single_state_payload(state)


def _single_state_payload(state: dict[str, Any]) -> dict[str, Any]:
    return {
        "joint_pos": state["joint_positions_rad"],
        "joint_vel": state["joint_velocities_rad_s"],
        "joint_torque": state["joint_torques_nm"],
        "tcp_pose_xyzrpy": state["tcp_xyzrpy"],
        "tcp_pose_xyzquat": state.get("tcp_xyzquat"),
        "gripper": state["gripper_position_m"],
        "gripper_normalized": state.get("gripper_normalized"),
        "gripper_command_normalized": state.get("gripper_command_normalized"),
        "gripper_vel": state["gripper_velocity_m_s"],
        "gripper_torque": state["gripper_torque_nm"],
        "gravity_compensation": state.get("gravity_compensation"),
    }
