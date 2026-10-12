"""Thin native episode coordinator using GPT-Policy AgentSession and recording."""

import json
import time
from io import BytesIO
from itertools import count
from pathlib import Path

import jsonschema
from PIL import Image

from ...hardware.camera import CapturedImage
from ...harness.errors import (
    AgentOverloadedError,
    AgentRecoverableError,
    AgentUsageLimitError,
)
from ...harness.factory import create_agent
from ...harness.models import AgentTurn
from ...harness.usage import collect_usage, usage_summary
from ...harness.waiting import monitor_health, wait_with_health
from ...input import resolve_run_input
from ...input.manifest import ImagePart, content_records
from ...recording.trace import RunRecorder
from .arrival import early_arrival_enabled, pass_through_enabled
from .budget import EpisodeWallTimeExceeded, budget_config
from .config import decision_limit, simulation_agent_config
from .io import sha256, write_json
from .perception import locate
from .quota import recover_quota
from .runtime import Simulator
from .tools import agent_context, catalog


def model_observation(observation, previous=None):
    # Explicit allowlist: terminal audits/layout/object truth cannot enter policy context.
    keys = (
        "observation_id",
        "instruction",
        "step_id",
        "arms",
        "initial_robot_state",
        "remaining_controls",
        "remaining_wall_time_s",
        "camera_calibration",
        "frame",
        "site",
        "quaternion",
        "tool_geometry",
    )
    packet = {key: observation[key] for key in keys if key in observation}
    packet["previous_result"] = previous
    packet["image_source"] = "CURRENT SIMULATION"
    return json.dumps(packet, ensure_ascii=False, separators=(",", ":"))


def image_inputs(observation):
    result = {}
    for name, rgb in observation["images"].items():
        data = BytesIO()
        Image.fromarray(rgb).save(data, format="PNG")
        result[name] = CapturedImage(
            name,
            data.getvalue(),
            "image/png",
            rgb.shape[1],
            rgb.shape[0],
            time.time(),
            rgb_data=rgb.tobytes(),
            source_timestamp_s=float(observation["step_id"]),
            source_clock="native_control_index",
        )
    if set(result) != {"cam_high", "cam_left_wrist", "cam_right_wrist"}:
        raise ValueError("Exactly three live camera images required")
    return result


def empty_result(reason, error=None):
    return {
        "complete": False,
        "valid_for_success_rate": False,
        "native_success": None,
        "native_score": None,
        "status": reason,
        "reason": reason,
        "error": error,
    }


def native_accounting(native):
    """Keep confirmed native accounting without promoting an infrastructure outcome."""
    return {
        key: native[key]
        for key in (
            "native_control_steps", "native_step_limit", "evaluation_budget",
            "task", "episode_id",
        )
        if key in native
    }


@collect_usage()
def run_episode(config, evidence, output, gpu):
    budget = decision_limit(config)
    episode_budget = budget_config(config)
    early_arrival = early_arrival_enabled(config)
    agent_cfg = simulation_agent_config(config)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "manifest.json", evidence)
    write_json(output / "run_config.json", config)
    recorder = RunRecorder(
        output / "policy",
        {
            "case": evidence["case"],
            "model": config["model"],
            "effort": config["effort"],
            "decision_budget": budget,
        },
    )
    sim = agent = None
    result = empty_result("initialization_error")
    reason = "initialization_error"
    decisions = 0
    provider_recoveries = 0
    quota_recoveries = 0
    try:
        prepared = config.get("prepared_inputs", {}).get(evidence["case"]["task"])
        if not prepared:
            raise ValueError("Prepared expert context missing for this task")
        run_input = resolve_run_input(None, Path(prepared), config["model"])
        write_json(
            output / "model-input-manifest.json",
            {
                "content": content_records(run_input.content),
                "historical_images": [
                    {"path": str(x.path), "sha256": sha256(x.path)}
                    for x in run_input.content
                    if isinstance(x, ImagePart)
                ],
                "source_success": {
                    "success": True,
                    "source": "dataset_owner_confirmation",
                },
                "online_success_source": "native_environment",
            },
        )
        sim = Simulator(config, evidence["case"], output / "sim", gpu)
        agent = create_agent(agent_cfg, config["model"])
        context = agent_context(
            evidence["case"]["task"], budget,
            extended_controls=episode_budget["control_step_limit"] is None,
            early_arrival=early_arrival,
            chunk_feedback=config.get("chunk_feedback", False),
            pass_through=pass_through_enabled(config),
            native_ik_increment=(config.get("max_joint_increment_rad", 0.05)
                                 if config.get("ik_backend") == "native" else None),
        )
        write_json(
            output / "agent_context.json",
            {
                "decision_only": True,
                "instructions": context.instructions,
                "tools": context.tools,
                "output_schema": context.output_schema,
            },
        )
        agent.start(context)
        previous = None
        history = {}
        schemas = {
            tool["function"]["name"]: tool["function"]["parameters"]
            for tool in catalog(early_arrival=early_arrival).function_schemas(6, ("left", "right"))
        }
        deadline = sim.observation.get("episode_deadline_monotonic")

        def check_health():
            sim.check_alive()
            if deadline is not None and time.monotonic() >= deadline:
                raise EpisodeWallTimeExceeded("Episode wall-time budget exhausted")

        for decision_index in count():
            obs = sim.observation
            stop_reason = obs.get("native_stop_reason") or (
                "native_terminal" if obs["native_terminal"] else None
            )
            if stop_reason:
                reason = stop_reason
                break
            if budget is not None and decision_index >= budget:
                reason = "decision_budget"
                break
            check_health()
            history[obs["step_id"]] = {k: v for k, v in obs.items() if k != "images"}
            images = image_inputs(obs)
            text = model_observation(obs, previous)
            recorder.observation(
                decision_index, text, history[obs["step_id"]], [], images
            )
            turn = AgentTurn(
                text, images, run_input.content if decision_index == 0 else None
            )
            started = time.monotonic()
            quota_retries = 0
            with monitor_health(check_health):
                while True:
                    try:
                        selection = agent.decide(turn)
                        break
                    except AgentUsageLimitError as error:
                        # The failed model call returned no executable selection.
                        # Keep this simulator, observation and decision index unchanged.
                        if error.code != "usageLimitExceeded" or quota_retries >= 3:
                            raise
                        recovered = recover_quota(config, health_check=check_health)
                        recorder.write("quota_recovery", {
                            "decision": decision_index, "native_step": obs["step_id"],
                            "observation_id": obs["observation_id"],
                            "availability_restored": recovered, "action_replayed": False,
                        })
                        if not recovered:
                            raise
                        quota_retries += 1
                        quota_recoveries += 1
                        wait_with_health(5)
                    except AgentRecoverableError as error:
                        # Short native-thread retries exhausted without an action.
                        # Keep this pending turn and scene until the original deadline.
                        if deadline is None:
                            raise
                        check_health()
                        recorder.write("provider_wait", {
                            "decision": decision_index, "native_step": obs["step_id"],
                            "observation_id": obs["observation_id"],
                            "thread_id": getattr(agent, "thread_id", None),
                            "code": error.code, "error": str(error),
                            "delay_s": 60, "action_replayed": False,
                            "remaining_wall_time_s": max(0, deadline - time.monotonic()),
                        })
                        wait_with_health(60)
                    except AgentOverloadedError as error:
                        recorder.write(
                            "provider_overloaded",
                            {"decision": decision_index, "error": str(error)},
                        )
                        if time.monotonic() - started > 600:
                            raise
                        wait_with_health(15)
                    finally:
                        for recovery in getattr(agent, "last_recovery", ()):
                            recorder.write("provider_recovery", {
                                "decision": decision_index,
                                "native_step": obs["step_id"],
                                "observation_id": obs["observation_id"],
                                **recovery,
                            })
                            provider_recoveries += 1
            decisions = decision_index + 1
            name, args = selection["name"], selection["arguments"]
            recorder.write(
                "model_decision",
                {
                    "step": decision_index,
                    "decision": selection["_wire"],
                    "name": name,
                    "arguments": args,
                    "native_step": obs["step_id"],
                    "elapsed_s": time.monotonic() - started,
                    "thread_id": agent.client.thread_id,
                    "context_refresh": agent.last_context_refresh,
                    "timing": agent.last_decision_timing,
                },
            )
            check_health()
            try:
                if name not in schemas:
                    raise ValueError("Unknown tool")
                jsonschema.validate(args, schemas[name])
                if args["observation_id"] != obs["observation_id"]:
                    raise ValueError("Stale observation identity")
                if name == "move_eef_chunk":
                    receipt = sim.move(args)
                    previous = {k: v for k, v in receipt.items() if k != "observation"}
                elif name == "locate_point":
                    previous = locate(args, history)
                elif name == "done":
                    previous = {
                        "declaration_recorded": True,
                        "native_episode_still_running": True,
                        "instruction": "Complete native placement, release and home conditions using bounded actions.",
                    }
                elif name == "give_up":
                    # A difficulty report cannot terminate or reset the episode.
                    # Feedback uses only public observations, never native object truth.
                    previous = {
                        "declaration_recorded": True,
                        "termination_rejected": True,
                        "native_episode_still_running": True,
                        "instruction": (
                            "The episode continues from the same state. Reassess the "
                            "current three views, measured robot state and prior execution "
                            "feedback. Identify the visible failure, change the approach, "
                            "and attempt a bounded corrective action using the current "
                            "observation_id. Use locate_point if visible geometry is "
                            "uncertain; do not repeat an unchanged failing path or bypass "
                            "action limits. Complete the public placement, release and "
                            "home conditions within remaining_wall_time_s."
                        ),
                    }
            except (ValueError, KeyError, jsonschema.ValidationError) as error:
                previous = {
                    "tool_rejected": str(error),
                    "native_step": sim.observation["step_id"],
                }
            recorder.write(
                "execution_result",
                {"step": decision_index, "name": name, "result": previous},
            )
            write_json(
                output / "progress.json",
                {
                    "decisions": decisions,
                    "native_step": sim.observation["step_id"],
                    "thread_id": agent.client.thread_id,
                },
            )
            print(
                json.dumps(
                    {
                        "decision": decisions,
                        "native_step": sim.observation["step_id"],
                        "tool": name,
                        "feedback": previous,
                    }
                ),
                flush=True,
            )
        if sim.observation["native_terminal"]:
            reason = "native_terminal"
        result = sim.finish(reason)
    except EpisodeWallTimeExceeded:
        reason = "wall_time_budget"
        result = empty_result(reason)
        try:
            result = sim.finish(reason)
        except Exception as finish_error:  # noqa: BLE001 - preserve timeout evidence if cleanup fails
            recorder.write("cleanup_error", {"error": str(finish_error)})
    except AgentUsageLimitError as error:
        reason = "quota_error"
        result = empty_result(reason, str(error))
        if sim and sim.client:
            try:
                result.update(native_accounting(sim.finish(reason)))
            except Exception as finish_error:  # noqa: BLE001 - preserve quota stop even if transport failed
                recorder.write("cleanup_error", {"error": str(finish_error)})
    except BaseException as error:  # noqa: BLE001 - finalize evidence and child processes on interruption
        reason = (
            "interrupted" if isinstance(error, KeyboardInterrupt) else "runtime_error"
        )
        result = empty_result(reason, f"{type(error).__name__}: {error}")
        if sim and sim.client:
            try:
                result.update(native_accounting(sim.finish(reason)))
            except Exception as finish_error:  # noqa: BLE001 - transport may already be closed
                recorder.write("cleanup_error", {"error": str(finish_error)})
        recorder.write("runtime_error", {"error": result["error"]})
    finally:
        for resource, instance in (("agent", agent), ("simulator", sim)):
            if instance:
                try:
                    instance.close()
                except Exception as close_error:  # noqa: BLE001 - always close other resources and retain evidence
                    recorder.write(
                        "cleanup_error",
                        {"resource": resource, "error": str(close_error)},
                    )
                    result.setdefault("cleanup_errors", []).append(
                        {"resource": resource, "error": str(close_error)}
                    )
                    if result.get("valid_for_success_rate"):
                        result.update(
                            valid_for_success_rate=False,
                            native_success=None,
                            native_score=None,
                            status="cleanup_error",
                        )
        result.update(
            decisions=decisions,
            provider_recoveries=provider_recoveries,
            quota_recoveries=quota_recoveries,
            usage=usage_summary(),
            case=evidence["case"],
            historical_expert_success=True,
            model=config["model"],
            effort=config["effort"],
        )
        write_json(output / "result.json", result)
        recorder.write("simulation_finished", result)
        write_json(recorder.root / "transcript.json", recorder.transcript)
        # Hardware RunRecorder.close adds a human label and renames the run.
        # Simulation uses the authoritative native result above instead.
        recorder._stream.close()
    return result
