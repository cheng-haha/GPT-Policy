"""Offline replay of previously executed joint commands. No policy connection."""

import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

from .environment import CAMERAS
from .evaluation import native_process_score_100
from .io import append_json, require, sha256, write_json


def recorded_actions(source):
    result = json.loads((source / "result.json").read_text())
    path = source / "sim/actions.jsonl"
    with path.open() as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    require(len(rows) == result["native_control_steps"], "Recorded control count mismatch")
    for step, row in enumerate(rows, 1):
        require(row["step"] == step, "Recorded commands must be contiguous")
        action = np.asarray(row["command"], dtype=np.float32)
        require(action.shape == (14,) and np.isfinite(action).all(), "Invalid recorded joints")
        require(np.all((action[[6, 13]] >= 0) & (action[[6, 13]] <= 1)),
                "Invalid recorded gripper command")
    return result, rows


def compare_observation(observation, source, step):
    directory = source / "sim/observations" / f"{step:06d}"
    original = json.loads((directory / "state.json").read_text())
    positions, joints, grippers = [], [], []
    for arm in ("left", "right"):
        a, b = observation["arms"][arm], original["arms"][arm]
        positions.append(float(np.linalg.norm(
            np.asarray(a["link6_pose_xyzquat"][:3]) - b["link6_pose_xyzquat"][:3])))
        joints.append(float(np.max(np.abs(np.asarray(a["joint_pos"]) - b["joint_pos"]))))
        grippers.append(abs(a["gripper"] - b["gripper"]))
    images = {}
    for camera in CAMERAS:
        with Image.open(directory / f"{camera}.png") as im:
            reference = np.asarray(im.convert("RGB"))
        replay = observation["images"][camera]
        require(replay.shape == reference.shape, "Replay camera dimensions changed")
        difference = np.abs(replay.astype(np.int16) - reference.astype(np.int16))
        images[camera] = {
            "mae_255": float(difference.mean()),
            "changed_pixel_fraction": float(np.any(difference, axis=-1).mean()),
            "exact": bool(np.array_equal(replay, reference)),
        }
    return {
        "step": step, "max_link6_position_error_m": max(positions),
        "max_joint_error_rad": max(joints), "max_gripper_error": max(grippers),
        "images": images,
    }


def replay_recorded_episode(session, source):
    source = Path(source).resolve()
    require(source != session.output.parent.resolve(), "Replay cannot overwrite source")
    original, rows = recorded_actions(source)
    require(session.case == original["case"], "Replay case identity changed")
    before = {str(p): sha256(p) for p in (
        source / "result.json", source / "sim/actions.jsonl",
        source / "sim/resolved_config.json", source / "sim/case.json",
    )}
    expected_config = json.loads((source / "sim/resolved_config.json").read_text())
    actual_config = json.loads((session.output / "resolved_config.json").read_text())
    require(actual_config == expected_config, "Resolved replay configuration differs")
    write_json(session.output / "replay-provenance.json", {
        "source": source, "source_hashes": before, "planned_controls": len(rows),
        "model_calls": 0, "command_mode": "original_float32_joint_commands",
        "score_kind": "offline_replay_reconstruction",
        "original_evaluation_budget": original.get("evaluation_budget"),
    })
    started = time.time()
    session.reset()
    requested_controls = min(len(rows), int(session.env.step_lim))
    old_layers = json.loads((source / "sim/loaded_layers.json").read_text())
    new_layers = json.loads((session.output / "loaded_layers.json").read_text())
    def layer_hashes(items):
        return {v["path"]: v["sha256"] for v in items if v.get("path")}
    require(layer_hashes(old_layers) == layer_hashes(new_layers), "Replay USD assets changed")
    comparisons = []

    def record():
        comparison = compare_observation(session.obs, source, session.step_id)
        comparisons.append(comparison)
        append_json(session.output / "replay-comparison.jsonl", comparison)
        rm = session.env.reward_manager
        append_json(session.output / "replay-score-state.jsonl", {
            "step": session.step_id,
            "score_100": native_process_score_100(session.env),
            "completed_count": rm.score_completed_count[0],
            "score_meta": rm.score_meta[0],
            "score_achieved": rm.score_achieved[0],
            "final_completed_count": rm.final_score_completed_count[0],
        })
        if session.step_id % 25 == 0 or session.step_id == requested_controls:
            write_json(session.output / "replay-progress.json", {
                "step": session.step_id, "total": requested_controls,
                "elapsed_s": time.time() - started,
                "score_100": native_process_score_100(session.env),
                "last_comparison": comparison,
            })

    record()
    for row in rows:
        if session.env.end_flag[0]:
            break  # Preserve native termination; never force past it.
        require(not session.budget.expired(), "Replay wall deadline exceeded")
        action = np.asarray(row["command"], dtype=np.float32)
        command = {
            key: value
            for arm, offset in (("left", 0), ("right", 7))
            for key, value in (
                (f"{arm}_arm_joint_state", action[offset:offset + 6]),
                (f"{arm}_ee_joint_state", action[offset + 6:offset + 7]),
            )
        }
        before_step = int(session.env.take_action_cnt[0])
        session.env.take_action(command)
        require(int(session.env.take_action_cnt[0]) == before_step + 1,
                "Replay control ACK mismatch")
        session.step_id += 1
        append_json(session.output / "actions.jsonl", row)
        session.observe()
        record()
    native = session.finish("offline_replay_end")
    unchanged = all(sha256(Path(path)) == digest for path, digest in before.items())
    require(unchanged, "Source artifacts changed during replay")
    complete_replay = session.step_id == len(rows)
    write_json(session.output / "replay-result.json", {
        "source": source, "case": session.case,
        "score_kind": "offline_replay_reconstruction",
        "reconstructed_score_100": native_process_score_100(session.env),
        "controls_applied": session.step_id, "controls_expected": len(rows),
        "controls_requested": requested_controls,
        "replayed_requested_prefix": session.step_id == requested_controls,
        "replay_evaluation_budget": session.budget.evidence,
        "replayed_all_controls": complete_replay, "native_result": native,
        "source_artifacts_unchanged": unchanged, "model_calls": 0,
        "elapsed_s": time.time() - started,
        "max_link6_position_error_m": max(c["max_link6_position_error_m"] for c in comparisons),
        "max_joint_error_rad": max(c["max_joint_error_rad"] for c in comparisons),
        "max_gripper_error": max(c["max_gripper_error"] for c in comparisons),
        "all_rgb_exact": all(v["exact"] for c in comparisons for v in c["images"].values()),
        "image_mae_255": {
            camera: {
                "mean": float(np.mean([c["images"][camera]["mae_255"] for c in comparisons])),
                "max": max(c["images"][camera]["mae_255"] for c in comparisons),
                "final": comparisons[-1]["images"][camera]["mae_255"],
            } for camera in CAMERAS
        },
        "exact_original_score_claim": False,
    })
