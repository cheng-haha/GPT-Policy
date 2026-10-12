#!/usr/bin/env python3
"""Prepare and validate a RoboDojo demonstration from an explicit JSON request."""

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]


def save(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class ContextClient:
    """Use the existing bounded reconnect logic for read-only vision requests."""

    def __init__(self, model, executable, effort, convert=False, quality=85):
        from functools import partial

        from gpt_policy.harness.config import AgentConfig
        from gpt_policy.harness.providers.codex import CodexSession

        self.factory = partial(
            CodexSession,
            AgentConfig(
                model=model, executable=executable, effort=effort, decision_only=True
            ),
            model,
            convert,
            quality,
        )
        self.session = self.factory()
        self.instructions = None

    def start_thread(self, instructions):
        self.instructions = instructions

    def decide(self, observation, schema, images=None, content=None):
        from gpt_policy.harness.errors import AgentOverloadedError, AgentTimeoutError
        from gpt_policy.harness.models import AgentContext, AgentTurn
        from gpt_policy.harness.waiting import monitor_health, wait_with_health

        # Every selector/review/smoke call is a single read-only request. A timed
        # out request can be repeated verbatim without replaying any robot action.
        for attempt in range(3):
            started = time.monotonic()
            print(
                json.dumps(
                    {
                        "event": "context_request_started",
                        "attempt": attempt + 1,
                        "request": observation[:180],
                    }
                ),
                flush=True,
            )
            try:
                with monitor_health(None, deadline=started + 600):
                    self.session.start(AgentContext(self.instructions, [], schema))
                    while True:
                        try:
                            decision = self.session.decide(
                                AgentTurn(observation, images, content)
                            )
                            print(
                                json.dumps(
                                    {
                                        "event": "context_request_completed",
                                        "elapsed_s": time.monotonic() - started,
                                    }
                                ),
                                flush=True,
                            )
                            return decision
                        except AgentOverloadedError:
                            wait_with_health(15)
            except AgentTimeoutError:
                self.session.close()
                print(
                    json.dumps(
                        {
                            "event": "context_request_timeout",
                            "attempt": attempt + 1,
                            "retry": attempt < 2,
                        }
                    ),
                    flush=True,
                )
                if attempt == 2:
                    raise
                time.sleep(15)
                self.session = self.factory()

    def close(self):
        self.session.close()


def validate(output):
    import numpy as np

    from gpt_policy.input.lerobot_demo import smoke

    result = smoke(output / "prepared/input.json")
    source = json.loads((output / "source.json").read_text())
    demo = json.loads((output / "reviewed.json").read_text())
    frames = demo["keyframes"]
    assert 2 <= len(frames) <= 16
    assert result["images"] == 3 * len(frames)
    assert all(set(f["images"]) == {"top", "left_wrist", "right_wrist"} for f in frames)
    assert frames[0]["video_frame_index"] == 0
    assert frames[-1]["video_frame_index"] == source["episode"]["length"] - 1
    assert source["rotation_conversion"]["output"] == "xyzw"
    assert source["rotation_conversion"]["frame_or_site_transform_applied"] is False
    assert digest(output / "raw_endposes.jsonl") == source["raw_endposes"]["sha256"]
    for frame in frames:
        for arm in ("left", "right"):
            pose = frame["state"][arm]["endpose_xyz_xyzw_gripper"]
            assert len(pose) == 8 and np.isfinite(pose).all()
            assert abs(np.linalg.norm(pose[3:7]) - 1) < 1e-6
    audit = json.loads((output / "frame_audit.json").read_text())
    assert audit["matches_native_link6_in_environment_frame"] is True
    return dict(
        result,
        keyframes=len(frames),
        action_references=source["counts"]["output_action_samples"],
        input_sha256=digest(output / "prepared/input.json"),
        frame_audit_sha256=digest(output / "frame_audit.json"),
    )


def worker(task, *, dataset, output, frame_audit, episode=0,
           instruction_override=None, relabel_reason=None, status_directory=None,
           codex_bin="codex"):
    from gpt_policy.harness.config import AgentConfig
    from gpt_policy.harness.errors import AgentUsageLimitError
    from gpt_policy.harness.usage import (
        attach_usage,
        collect_usage,
        usage_phase,
        usage_summary,
    )
    from gpt_policy.harness.video_selector import CodexVideoSelector
    from gpt_policy.input.lerobot_demo import model_smoke, prepare
    from gpt_policy.input.video import VideoProcessingConfig

    output = Path(output)
    dataset = Path(dataset)
    frame_audit = Path(frame_audit)
    status_directory = Path(status_directory) if status_directory else output.parent / (output.name + "-status")
    status_directory.mkdir(parents=True, exist_ok=True)
    status = {"task": task, "dataset": str(dataset), "episode": episode,
              "output": str(output), "started_at_s": time.time()}
    status["preparation_source"] = {
        "commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT, text=True
        ).strip(),
        "status": subprocess.check_output(
            ["git", "status", "--short"], cwd=PROJECT, text=True
        ).strip(),
        "script_sha256": digest(__file__),
    }
    try:
        with_usage = collect_usage()

        @with_usage
        @usage_phase("context_preparation")
        def compile_context():
            usage_dir = status_directory / "usage" / f"{task}-{time.time_ns()}"
            attach_usage(usage_dir)
            status["usage_directory"] = str(usage_dir)
            cfg = AgentConfig(
                model="gpt-6-astra",
                executable=codex_bin,
                effort="medium",
                decision_only=True,
            )
            if not (output / "prepared/input.json").is_file():
                processing = VideoProcessingConfig()
                selector = CodexVideoSelector(
                    cfg, processing, client_factory=ContextClient
                )
                prepare(
                    dataset,
                    episode,
                    output,
                    selector,
                    processing,
                    confirmed_expert=True,
                    frame_audit=frame_audit,
                    instruction_override=instruction_override,
                    relabel_reason=relabel_reason,
                )
            check_request_identity(output, dataset, episode, instruction_override, relabel_reason)
            verified = validate(output)
            if not (output / "model_smoke.json").exists():
                inspection = model_smoke(
                    output / "prepared/input.json", cfg, client_factory=ContextClient
                )
                save(output / "model_smoke.json", inspection)
            save(output / "context_verification.json", verified)
            save(output / "preparation_usage.json", usage_summary())
            return verified

        status.update(status="prepared", verification=compile_context())
    except AgentUsageLimitError as error:
        status.update(status="quota_error", error=str(error))
    except Exception as error:  # noqa: BLE001 - preserve per-task failure and continue independent tasks
        status.update(status="error", error=f"{type(error).__name__}: {error}")
        import traceback

        traceback.print_exc()
    status["ended_at_s"] = time.time()
    save(status_directory / f"{task}.status.json", status)
    print(json.dumps(status, ensure_ascii=False), flush=True)
    return status


def check_request_identity(output, dataset, episode, instruction, reason):
    """Never reuse a completed package for a different source or relabel request."""
    demo = json.loads((output / "reviewed.json").read_text())
    source = demo["source"]
    relabel = source.get("instruction_relabel") or {}
    if (Path(source["dataset"]).resolve() != Path(dataset).resolve()
            or source["episode_index"] != episode
            or relabel.get("replacement_instruction") != instruction
            or relabel.get("reason") != reason):
        raise ValueError("Existing context does not match the requested source/relabel identity")


def main():
    from gpt_policy.simulation.robodojo.tasks import TASKS

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--spec", type=Path, required=True,
        help="JSON with task, dataset, output, frame_audit and optional worker settings",
    )
    args = parser.parse_args()
    spec = json.loads(args.spec.read_text())
    required = {"task", "dataset", "output", "frame_audit"}
    optional = {"episode", "instruction_override", "relabel_reason", "status_directory", "codex_bin"}
    if not isinstance(spec, dict) or not required <= spec.keys():
        parser.error("Spec requires task, dataset, output and frame_audit")
    if spec.keys() - required - optional:
        parser.error("Unknown preparation spec fields")
    if spec["task"] not in TASKS:
        parser.error("Spec must name a supported task")
    if any(not isinstance(spec[key], str) or not spec[key].strip()
           for key in ("dataset", "output", "frame_audit")):
        parser.error("Dataset, output and frame_audit must be nonempty paths")
    if not Path(spec["dataset"]).is_dir() or not Path(spec["frame_audit"]).is_file():
        parser.error("Dataset directory and frame audit must exist")
    episode = spec.get("episode", 0)
    if type(episode) is not int or episode < 0:
        parser.error("Episode must be a nonnegative integer")
    result = worker(**spec)
    if result["status"] != "prepared":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
