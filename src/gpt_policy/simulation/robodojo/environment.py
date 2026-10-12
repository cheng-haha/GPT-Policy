"""One native episode. Physics advances only inside acknowledged take_action."""

import time
import uuid
from pathlib import Path

import numpy as np
from PIL import Image

from ...recording.h264 import H264Mp4Writer
from .arrival import (
    ARRIVAL_LIMITS,
    ArrivalCheck,
    early_arrival_enabled,
    pass_through_enabled,
    stop_on_arrival_enabled,
)
from .budget import EpisodeBudget, budget_config
from .control import ARMS, measured_tracking, native_targets, validate_path
from .evaluation import (
    native_process_score_100,
    native_stop_reason,
    outcome,
    register_native_evaluation,
)
from .execution import IKRejected, TrackingGuard, execution_config
from .io import append_json, require, sha256, write_json
from .kinematics import DualKinematics
from .proprioception import gripper_measurement
from .target_schedule import waypoint_targets
from .transforms import transform, xyzw_pose

CAMERAS = ("cam_high", "cam_left_wrist", "cam_right_wrist")


class NativeSession:
    def __init__(self, env, output, task, case, *, budget=None, control=None):
        self.budget_config = budget_config(budget or {})
        self.early_arrival = early_arrival_enabled(control or {})
        self.stop_on_arrival = stop_on_arrival_enabled(control or {})
        self.pass_through = pass_through_enabled(control or {})
        self.execution = execution_config(control or {})
        self.last_gripper_command = None
        self.budget = None
        self.env, self.output, self.task, self.case = env, Path(output), task, case
        self.output.mkdir(parents=True, exist_ok=True)
        write_json(self.output / "waypoint-control.json", {
            **self.execution,
            "early_arrival": self.early_arrival,
            "stop_on_arrival": self.stop_on_arrival,
            "pass_through": self.pass_through,
            "limits": dict(ARRIVAL_LIMITS) if self.early_arrival else None,
            "default_mode": "hold",
            "gripper_changes": "fixed_controls",
        })
        self.episode_id = None
        self.step_id = 0
        self.poisoned = False
        self.finished = False
        self.reason = None
        self.writers = {}
        write_json(
            self.output / "video-format.json",
            {
                "codec": "h264",
                "encoder": "libx264",
                "pixel_format": "yuv420p",
                "container": "mp4",
                "faststart": True,
                "input": "native_rgb",
                "lossless_evidence": "observations/*/*.png",
                "published_after": "successful_encoder_close",
            },
        )
        self.obs = None
        env._stream_vision = lambda *args, **kwargs: None

    def reset(self):
        require(self.episode_id is None, "No reset/replay within an episode")
        layout = self.env.seed_manager.seed_info[self.case["layout_id"]]["scene_layout"]
        require(
            sha256(layout) == self.case["layout_sha256"],
            "Native layout identity mismatch",
        )
        self.env.reset(seed=[self.case["layout_id"]])
        self.budget = EpisodeBudget(self.env, self.budget_config)
        write_json(self.output / "episode-budget.json", self.budget.evidence)
        write_json(
            self.output / "native_evaluation.json", register_native_evaluation(self.env)
        )
        # Audit actual USD layers on the host only; never attach to model observations.
        import omni.usd

        layers = []
        for layer in omni.usd.get_context().get_stage().GetUsedLayers():
            path = Path(layer.realPath) if layer.realPath else None
            layers.append(
                {
                    "identifier": layer.identifier,
                    "path": str(path) if path else None,
                    "sha256": sha256(path) if path and path.is_file() else None,
                }
            )
        write_json(self.output / "loaded_layers.json", layers)
        self.episode_id = uuid.uuid4().hex
        self.kinematics = DualKinematics(self.env, self.execution)
        write_json(self.output / "fk_validation.json", self.kinematics.check())
        from .tool_geometry import calibration, verify_native

        self.calibrations = {
            arm: calibration(robot.urdf_path)
            for arm, robot in self.kinematics.robots.items()
        }
        write_json(
            self.output / "tool_geometry.json",
            verify_native(self.kinematics, self.calibrations),
        )
        self.home = None
        return self.observe()

    def observe(self):
        # This pinned get_obs calls render_for_capture; renders do not step physics.
        before = int(self.env.take_action_cnt[0])
        raw = self.env.get_obs()
        require(
            int(self.env.take_action_cnt[0]) == before, "Observation advanced controls"
        )
        aliases = {
            "cam_high": ("cam_high", "cam_head", "head_camera", "top_camera"),
            "cam_left_wrist": ("cam_left_wrist", "left_camera"),
            "cam_right_wrist": ("cam_right_wrist", "right_camera"),
        }
        images, sources = {}, {}
        for key, names in aliases.items():
            source = next((n for n in names if n in raw["vision"]), None)
            require(source is not None, "Missing native view " + key)
            rgb = np.asarray(raw["vision"][source]["color"])
            require(
                rgb.dtype == np.uint8 and rgb.ndim == 3 and rgb.shape[2] in (3, 4),
                "Expected native uint8 RGB(A)",
            )
            images[key] = np.ascontiguousarray(rgb[:, :, :3])
            sources[key] = source
        state = raw["state"]
        arms = {}
        for arm in ARMS:
            ee = np.asarray(state[f"{arm}_ee_pose"], float)
            joints = np.asarray(state[f"{arm}_arm_joint_state"], float)
            grip = np.asarray(state[f"{arm}_ee_joint_state"], float)
            require(
                joints.shape == (6,) and np.isfinite(joints).all(), "Invalid joints"
            )
            arms[arm] = {
                "joint_pos": joints.tolist(),
                # Keep the legacy target proxy for replay/arrival compatibility.
                "gripper": float(grip.reshape(-1)[0]),
                "gripper_source": "native_control_target_proxy",
                **gripper_measurement(self.kinematics.manager, self.kinematics.robots[arm]),
                "link6_pose_xyzquat": xyzw_pose(transform(ee[:3], ee[3:])),
                "environment_from_base": self.kinematics.root(arm).tolist(),
            }
        if self.home is None:
            self.home = arms
        from .camera_geometry import capture_calibration

        calibration = capture_calibration(self.env, sources, images)
        packet = {
            "episode_id": self.episode_id,
            "step_id": self.step_id,
            "observation_id": f"{self.episode_id}:{self.step_id}",
            "instruction": raw["instruction"],
            "arms": arms,
            "initial_robot_state": self.home,
            "remaining_controls": None if self.budget.extended else max(0, int(self.env.step_lim) - self.step_id),
            "remaining_wall_time_s": self.budget.remaining_wall_time(),
            "episode_deadline_monotonic": self.budget.deadline,
            "native_terminal": bool(self.env.end_flag[0]),
            "native_stop_reason": native_stop_reason(self.env),
            "camera_calibration": calibration,
            "images": images,
            "frame": "environment_origin",
            "site": "link6",
            "quaternion": "xyzw",
        }
        packet["tool_geometry"] = {
            arm: {
                k: v
                for k, v in values.items()
                if k
                in (
                    "link6_from_tool",
                    "site",
                    "orientation",
                    "physical_inner_contact_point_certified",
                )
            }
            for arm, values in self.calibrations.items()
        }
        self.obs = packet
        directory = self.output / "observations" / f"{self.step_id:06d}"
        directory.mkdir(parents=True, exist_ok=False)
        write_json(
            directory / "state.json", {k: v for k, v in packet.items() if k != "images"}
        )
        for camera, rgb in images.items():
            Image.fromarray(rgb).save(directory / f"{camera}.png")
            if camera not in self.writers:
                self.writers[camera] = H264Mp4Writer(
                    self.output / f"{camera}.mp4",
                    rgb.shape[1],
                    rgb.shape[0],
                    round(self.env.obs_manager.collect_freq),
                )
            self.writers[camera].append(rgb)
        append_json(
            self.output / "video-frames.jsonl",
            {
                "frame_index": self.step_id,
                "native_control_step": self.step_id,
                "simulation_time_s": self.step_id / self.env.obs_manager.collect_freq,
                "captured_at_s": time.time(),
                "image_paths": {k: str(directory / f"{k}.png") for k in images},
            },
        )
        # Host-only ledger: never add score/state-machine diagnostics to packet.
        append_json(
            self.output / "native-process-scores.jsonl",
            {
                "native_control_step": self.step_id,
                "native_complete": bool(self.env.end_flag[0]),
                "native_layout_valid": 0 not in getattr(self.env, "unstable_envs", set()),
                "score_100": native_process_score_100(self.env, self.poisoned),
                "support_checked": bool(self.env.support_checked[0])
                if hasattr(self.env, "support_checked") else None,
                "support_actions_remaining": len(self.env.support_arm_action[0])
                if hasattr(self.env, "support_arm_action") else None,
            },
        )
        return packet

    def move(self, arguments):
        require(not self.finished and native_stop_reason(self.env) is None, "Episode finished or invalid")
        points = validate_path(arguments, self.obs, early_arrival=self.early_arrival)
        start = self.step_id
        last_executed = None
        rejection = None
        rejection_details = None
        waypoint_results = []
        tracking_guard = TrackingGuard() if self.execution["chunk_feedback"] else None
        for index, point in enumerate(points):
            if self.budget.expired():
                break
            # Fresh measured pose must still satisfy the bound at each path segment.
            try:
                validate_path(
                    dict(
                        arguments,
                        waypoints=[point],
                        observation_id=self.obs["observation_id"],
                    ),
                    self.obs,
                    early_arrival=self.early_arrival,
                )
            except ValueError as error:
                rejection = str(error)
                break
            point_start = self.step_id
            grippers = {arm: point[arm]["gripper"] for arm in ARMS}
            guard = (
                "disabled" if not self.early_arrival
                else "hold" if point.get("mode", "hold") == "hold"
                else "initial_or_changed_gripper" if grippers != self.last_gripper_command
                else None
            )
            arrival = ArrivalCheck(point, self.obs, self.env.obs_manager.collect_freq) if guard is None else None
            # Only an interior free-motion point may pass through. The endpoint
            # and any boundary into hold/gripper change keep their old behavior.
            following = points[index + 1] if index + 1 < len(points) else None
            pass_through = (
                self.pass_through and arrival is not None and following is not None
                and following.get("mode", "hold") == "move"
                and all(following[arm]["gripper"] == grippers[arm] for arm in ARMS)
            )
            schedule, distributed_arms = waypoint_targets(
                point, self.obs, self.last_gripper_command,
                enabled=self.execution.get("distribute_closed_gripper_moves", False),
            )
            stop = "requested_controls"
            for control_target in schedule:
                if self.budget.expired():
                    stop = "wall_time_budget"
                    break
                if distributed_arms:
                    try:
                        validate_path(
                            dict(arguments, waypoints=[control_target],
                                 observation_id=self.obs["observation_id"]),
                            self.obs, early_arrival=self.early_arrival,
                        )
                    except ValueError as error:
                        rejection = str(error)
                        stop = "path_rejected"
                        break
                try:
                    proposal = self.kinematics.target(native_targets(control_target))
                except IKRejected as error:
                    rejection = str(error)
                    rejection_details = {
                        **error.details,
                        "waypoint_index": index,
                        "acknowledged_controls_before_rejection": self.step_id - start,
                        "waypoint_acknowledged_controls": self.step_id - point_start,
                        "rejected_proposal_controls": 0,
                        "unexecuted_suffix_discarded": True,
                    }
                    append_json(self.output / "ik-rejections.jsonl", {
                        "step": self.step_id, "observation_id": self.obs["observation_id"],
                        "message": rejection, **rejection_details,
                    })
                    stop = "ik_rejected"
                    break
                action = np.asarray(proposal["action"], np.float32)
                require(
                    action.shape == (14,) and np.isfinite(action).all(),
                    "Invalid joint action",
                )
                command = {
                    key: value
                    for arm, offset in (("left", 0), ("right", 7))
                    for key, value in (
                        (f"{arm}_arm_joint_state", action[offset : offset + 6]),
                        (f"{arm}_ee_joint_state", action[offset + 6 : offset + 7]),
                    )
                }
                before = int(self.env.take_action_cnt[0])
                self.env.take_action(command)
                require(
                    int(self.env.take_action_cnt[0]) == before + 1,
                    "ACK must equal one native control",
                )
                if self.execution["ik_backend"] == "native":
                    self.kinematics.acknowledged(action)
                self.step_id += 1
                append_json(
                    self.output / "actions.jsonl",
                    {
                        "step": self.step_id,
                        "command": action,
                        "requested": point,
                        **({"applied_target": control_target, "distributed_arms": distributed_arms}
                           if distributed_arms else {}),
                        "ik": proposal["diagnostics"],
                        "observation_id": arguments["observation_id"],
                    },
                )
                self.observe()
                self.last_gripper_command = grippers
                last_executed = point
                if reason := native_stop_reason(self.env):
                    stop = reason
                    self.finish(reason)
                    break
                if tracking_guard is not None and tracking_guard.update(control_target, self.obs, action):
                    stop = "tracking_deviation"
                    rejection = (
                        "Tracking exceeded path/joint bounds for two consecutive controls; "
                        "unexecuted chunk suffix discarded. Replan from returned robot state and RGB."
                    )
                    break
                if arrival is not None:
                    stable = arrival.update(self.obs)
                    if (self.stop_on_arrival and not self.budget.expired()
                            and self.step_id - point_start < point["steps"]):
                        if pass_through and arrival.pose_arrived:
                            stop = "pass_through_arrival"
                            break
                        if stable:
                            stop = "stable_arrival"
                            break
            if self.early_arrival:
                executed = self.step_id - point_start
                record = {
                    "observation_id": arguments["observation_id"],
                    "waypoint_index": index,
                    "mode": point.get("mode", "hold"),
                    **({"distributed_arms": distributed_arms} if distributed_arms else {}),
                    "start_step": point_start,
                    "end_step": self.step_id,
                    "requested_controls": point["steps"],
                    "executed_controls": executed,
                    "saved_controls": point["steps"] - executed if stop in ("stable_arrival", "pass_through_arrival") else 0,
                    "stop_reason": stop,
                    "early_completion_blocked_by": guard,
                    "stable_samples": arrival.stable_samples if arrival else 0,
                    "arrival": arrival.feedback if arrival else None,
                }
                append_json(self.output / "waypoint-execution.jsonl", record)
                waypoint_results.append({k: v for k, v in record.items() if k not in ("arrival", "observation_id")})
            if rejection or native_stop_reason(self.env):
                break
        return {
            "start_step": start,
            "end_step": self.step_id,
            "acknowledged_controls": self.step_id - start,
            "tracking": measured_tracking(last_executed, self.obs)
            if last_executed
            else {},
            "rejection": rejection,
            "rejection_details": rejection_details,
            "wall_time_budget_exhausted": self.budget.expired(),
            **({"waypoint_results": waypoint_results} if self.early_arrival else {}),
            "observation": self.obs,
        }

    def finish(self, reason):
        if self.reason is None:
            self.reason = reason
        self.finished = True
        encoding_errors = []
        for writer in self.writers.values():
            try:
                writer.close()
            except (RuntimeError, OSError) as error:
                encoding_errors.append(error)
        if encoding_errors:
            raise RuntimeError(
                "Simulation video finalization failed"
            ) from encoding_errors[0]
        if self.episode_id is None:
            result = {
                "complete": False,
                "valid_for_success_rate": False,
                "native_success": None,
                "native_score": None,
                "status": "initialization_error",
                "reason": self.reason,
            }
        else:
            result = outcome(self.env, self.step_id, self.reason, self.poisoned)
        result.update(case=self.case, task=self.task, episode_id=self.episode_id)
        write_json(self.output / "native_result.json", result)
        if result["complete"] and self.task == "classify_objects":
            from .classify_audit import write_classify_terminal_audit

            write_classify_terminal_audit(self.env, self.output)
        else:
            write_json(
                self.output / "terminal_conditions.json",
                {
                    "audit_available": False,
                    "reason": "native_episode_incomplete"
                    if not result["complete"]
                    else "task_not_supported",
                    "task": self.task,
                    "native_control_steps": self.step_id,
                },
            )
        return result

    def dispatch(self, op, args):
        if op == "reset":
            return self.reset()
        require(not self.poisoned, "Uncertain simulator state")
        require(
            args.pop("observation_id", None) == self.obs["observation_id"],
            "Stale RPC identity",
        )
        if op == "observe":
            return self.obs  # No waiting/render/physics advance.
        if op == "move":
            return self.move(args["arguments"])
        if op == "finish":
            return self.finish(args["reason"])
        raise ValueError("Unsupported operation: " + op)
