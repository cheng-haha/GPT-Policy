"""Offline, model-free transport probe; never part of policy evaluation."""

import copy
import json
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from .control import ARMS
from .evaluation import native_stop_reason
from .io import append_json, require, sha256, write_json
from .replay import compare_observation, recorded_actions


def transport_points(observation, target, variant, previous_grippers, arm="left"):
    """Change only the selected arm's target schedule, with equal requested controls."""
    require(variant in ("direct", "distributed"), "Unknown transport variant")
    require(arm in ARMS, "Unknown arm")
    require(target.get("mode") == "move", "Probe requires a move, not contact/hold")
    require(type(target["steps"]) is int and 2 <= target["steps"] <= 5,
            "Probe needs 2..5 requested controls")
    require(all(target[a]["gripper"] == previous_grippers[a] for a in ARMS),
            "Do not redistribute gripper transitions")
    if variant == "direct":
        return [copy.deepcopy(target)]
    start = np.asarray(observation["arms"][arm]["link6_pose_xyzquat"], float)
    end = np.asarray(target[arm]["pose_xyzquat"], float)
    rotations = Slerp([0, 1], Rotation.from_quat([start[3:], end[3:]]))
    points = []
    for fraction in np.linspace(0, 1, target["steps"] + 1)[1:]:
        point = copy.deepcopy(target)
        point["steps"] = 1
        point[arm]["pose_xyzquat"] = [
            *((1 - fraction) * start[:3] + fraction * end[:3]).tolist(),
            *rotations(fraction).as_quat().tolist(),
        ]
        points.append(point)
    # Keep the exact endpoint representation, including quaternion sign.
    points[-1][arm] = copy.deepcopy(target[arm])
    return points


def replay_prefix_control(session, row):
    """Execute one immutable, previously acknowledged joint command; no new IK."""
    require(native_stop_reason(session.env) is None, "Native replay terminal")
    require(not session.budget.expired(), "Replay wall deadline exceeded")
    action = np.asarray(row["command"], np.float32)
    command = {
        key: value
        for arm, offset in (("left", 0), ("right", 7))
        for key, value in (
            (f"{arm}_arm_joint_state", action[offset:offset + 6]),
            (f"{arm}_ee_joint_state", action[offset + 6:offset + 7]),
        )
    }
    before = int(session.env.take_action_cnt[0])
    session.env.take_action(command)
    require(int(session.env.take_action_cnt[0]) == before + 1, "Replay ACK mismatch")
    session.kinematics.acknowledged(action)
    session.last_gripper_command = {
        arm: row["requested"][arm]["gripper"] for arm in ARMS
    }
    session.step_id += 1
    append_json(session.output / "actions.jsonl", {**row, "offline_prefix_replay": True})
    session.observe()


def run_transport_probe(session, specification):
    spec = json.loads(Path(specification).read_text())
    source = Path(spec["source"]).resolve()
    require(source != session.output.parent.resolve(), "Cannot overwrite source")
    original, rows = recorded_actions(source)
    require(session.case == original["case"], "Probe case changed")
    prefix = spec["prefix_controls"]
    require(type(prefix) is int and 0 < prefix < len(rows), "Invalid prefix")
    target = rows[prefix]["requested"]
    require(all(row["requested"] == target
                for row in rows[prefix:prefix + target["steps"]]),
            "Source target did not execute the requested matched controls")
    paths = ["result.json", "sim/actions.jsonl", "sim/case.json",
             "sim/resolved_config.json", "sim/control-config.json", "sim/loaded_layers.json"]
    hashes = {name: sha256(source / name) for name in paths}
    for name in ("resolved_config.json", "control-config.json"):
        expected = json.loads((source / "sim" / name).read_text())
        if name == "control-config.json" and spec["variant"] == "executor_distributed":
            expected["distribute_closed_gripper_moves"] = True
        require(expected == json.loads((session.output / name).read_text()), name + " changed")
    started = time.time()
    session.reset()
    def layer_hashes(path):
        return {v["path"]: v["sha256"] for v in json.loads(path.read_text())
                if v.get("path")}
    require(layer_hashes(source / "sim/loaded_layers.json")
            == layer_hashes(session.output / "loaded_layers.json"), "USD assets changed")
    write_json(session.output / "probe-provenance.json", {
        **spec, "source_hashes": hashes, "model_calls": 0,
        "valid_for_success_rate": False, "score_kind": "local_transport_engineering",
        "prefix_mode": "immutable_previous_float32_joint_commands",
        "intervention_mode": "NativeSession.move with all current guards",
        "other_arm": "same final target repeated in both variants",
    })
    checkpoints = []
    for step in range(prefix + 1):
        if step:
            replay_prefix_control(session, rows[step - 1])
        if step % 50 == 0 or step == prefix:
            comparison = compare_observation(session.obs, source, step)
            checkpoints.append(comparison)
            append_json(session.output / "prefix-comparison.jsonl", comparison)
            write_json(session.output / "probe-progress.json", {
                "phase": "prefix", "step": step, "prefix_controls": prefix,
                "elapsed_s": time.time() - started, "comparison": comparison,
            })
    original_state = json.loads(
        (source / "sim/observations" / f"{prefix:06d}" / "state.json").read_text())
    aperture_error = max(abs(session.obs["arms"][a]["measured_gripper"]
                             - original_state["arms"][a]["measured_gripper"]) for a in ARMS)
    last = checkpoints[-1]
    numerical_gate = (
        last["max_link6_position_error_m"] <= 0.001
        and last["max_joint_error_rad"] <= 0.005
        and aperture_error <= 0.02
        and all(v["mae_255"] <= 2 for v in last["images"].values())
    )
    write_json(session.output / "prefix-gate.json", {
        "passed": numerical_gate, "comparison": last,
        "max_physical_aperture_error": aperture_error,
        "visual_grasp_verification_required": True,
        "limits": {"position_m": 0.001, "joint_rad": 0.005,
                   "physical_aperture": 0.02, "full_frame_rgb_mae_255": 2},
    })
    receipts = []
    if numerical_gate:
        variant = "direct" if spec["variant"] == "executor_distributed" else spec["variant"]
        points = transport_points(session.obs, target, variant,
                                  session.last_gripper_command, spec.get("arm", "left"))
        write_json(session.output / "intervention.json", {"waypoints": points})
        chunks = [("transport", points)]
        hold = {**copy.deepcopy(target), "mode": "hold", "steps": 5}
        chunks += [("settle", [hold, copy.deepcopy(hold)])]
        for label, points in chunks:
            receipt = session.move({
                "observation_id": session.obs["observation_id"],
                "note": "Offline paired transport probe: " + label,
                "waypoints": points,
            })
            receipts.append({"label": label, **{k: v for k, v in receipt.items()
                                               if k != "observation"}})
            write_json(session.output / "probe-receipts.json", receipts)
            if receipt["rejection"] or session.finished:
                break
    native = session.finish("offline_transport_probe_end")
    unchanged = all(sha256(source / name) == digest for name, digest in hashes.items())
    require(unchanged, "Source artifacts changed")
    write_json(session.output / "probe-result.json", {
        **spec, "prefix_gate_passed": numerical_gate,
        "visual_grasp_verification_required": True,
        "intervention_controls": session.step_id - prefix,
        "receipts": receipts, "native_result": native,
        "source_artifacts_unchanged": unchanged,
        "model_calls": 0, "valid_for_success_rate": False,
        "elapsed_s": time.time() - started,
    })
