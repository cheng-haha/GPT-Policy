"""Offline frame audit against native X5 URDF; no simulator, policy or motion.

Run with PYTHONPATH=src in an environment with pyarrow, scipy and PyYAML.
The supplied native checkout/assets are evidence, never modified.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET

import numpy as np
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation
import yaml

from gpt_policy.input.lerobot_demo import endpose_sequence, numeric_samples, validate_rows


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pose_matrix(position, wxyz):
    result = np.eye(4)
    result[:3, 3] = position
    result[:3, :3] = Rotation.from_quat(np.asarray(wxyz)[[1, 2, 3, 0]]).as_matrix()
    return result


class UrdfFK:
    def __init__(self, path, names, base, tip):
        self.names = names
        edges = {j.find("child").get("link"): j for j in ET.parse(path).getroot().findall("joint")}
        self.chain = []
        while tip != base:
            joint = edges[tip]
            origin = joint.find("origin")
            matrix = np.eye(4)
            if origin is not None:
                matrix[:3, 3] = np.fromstring(origin.get("xyz", "0 0 0"), sep=" ")
                matrix[:3, :3] = Rotation.from_euler("xyz", np.fromstring(
                    origin.get("rpy", "0 0 0"), sep=" ")).as_matrix()
            axis = joint.find("axis")
            axis = np.fromstring(axis.get("xyz"), sep=" ") if axis is not None else np.array([1., 0, 0])
            self.chain.insert(0, (joint.get("name"), joint.get("type"), matrix, axis))
            tip = joint.find("parent").get("link")
        if {n for n, kind, _, _ in self.chain if kind != "fixed"} != set(names):
            raise ValueError("URDF chain does not match the configured joint order")

    def matrix(self, joints):
        result = np.eye(4)
        values = dict(zip(self.names, joints))
        for name, kind, origin, axis in self.chain:
            motion = np.eye(4)
            if kind in ("revolute", "continuous"):
                motion[:3, :3] = Rotation.from_rotvec(axis * values[name]).as_matrix()
            elif kind != "fixed":
                raise ValueError(f"Unsupported joint type: {kind}")
            result = result @ origin @ motion
        return result


def errors(predicted, measured):
    return [float(np.linalg.norm(predicted[:3, 3] - measured[:3, 3])),
            float(Rotation.from_matrix(predicted[:3, :3].T @ measured[:3, :3]).magnitude())]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--robodojo", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    info_path = args.dataset / "meta/info.json"
    info = json.loads(info_path.read_text())
    semantics = info["starvla_metadata"]
    if semantics["rotation_6d_convention"] != "first_two_rotation_matrix_columns_column_major":
        raise ValueError("Unsupported rot6d layout")
    episodes = [row for file in sorted((args.dataset/"meta/episodes").glob("chunk-*/*.parquet"))
                for row in pq.read_table(file).to_pylist() if row["episode_index"] == args.episode]
    if len(episodes) != 1:
        raise ValueError("Episode must exist exactly once")
    episode = episodes[0]
    data = args.dataset / info["data_path"].format(
        chunk_index=episode["data/chunk_index"], file_index=episode["data/file_index"])
    columns = ["episode_index", "frame_index", "index", "timestamp", "observation.state",
               "action", "observation.states.eef", "actions.eef"]
    rows = pq.read_table(data, filters=[("episode_index", "=", args.episode)], columns=columns).to_pylist()
    validate_rows(rows, args.episode, episode)
    converted = endpose_sequence(rows)
    numeric_samples(rows, endposes=converted)  # Validate the derived next-state labels, too.
    assets = args.robodojo / "Assets/Robots/x5"
    config_path = assets / "robot_config.yml"
    config = yaml.safe_load(config_path.read_text())
    if config["ee_link"] != "link6":
        raise ValueError("This audit expects the native link6 endpoint")
    urdf = assets / config["urdf_path"]
    fk = UrdfFK(urdf, config["arm_joints_name"], config["base_link"], config["ee_link"])
    roots_path = args.robodojo / "env_cfg/robot/dual_x5.yml"
    roots = yaml.safe_load(roots_path.read_text())["robots"]
    calibration = json.loads(args.calibration.read_text())["calibration"]
    link6_from_tcp = np.array(calibration["link6_from_tcp"])
    report = {
        "scope": "offline robot-only FK audit; no physics, native success check or hardware calibration",
        "dataset": str(args.dataset.resolve()), "episode": args.episode, "frames": len(rows),
        "native_commit": subprocess.check_output(["git", "-C", str(args.robodojo), "rev-parse", "HEAD"], text=True).strip(),
        "source_metadata_frame": semantics["coordinate_system"],
        "thresholds": {"position_m": .002, "rotation_rad": .01},
        "files": {str(p.resolve()): sha256(p) for p in (info_path, data, urdf, config_path,
            roots_path, args.calibration, args.robodojo/"env/robot_manager/robot_manager.py")},
        "gpt_policy_arx": {
            "reference": "per-arm base_link, calibrated fingertip TCP, quaternion xyzw",
            "link6_from_tcp": link6_from_tcp.tolist(),
            "translation_norm_m": float(np.linalg.norm(link6_from_tcp[:3, 3])),
            "rotation_angle_rad": float(Rotation.from_matrix(link6_from_tcp[:3, :3]).magnitude()),
            "applied_to_dataset": False,
            "note": "hardware calibration is not evidence that the simulation fingertips have the same offset",
        }, "arms": {},
    }
    max_roundtrip_error = 0.
    for a, (arm, j, p) in enumerate((("left", 0, 0), ("right", 7, 10))):
        base = pose_matrix(roots[a]["default_root_pos"], roots[a]["default_root_rot"])
        native_errors, base_errors, inferred_roots, measured_poses, local_poses = [], [], [], [], []
        for i, row in enumerate(rows):
            local = fk.matrix(row["observation.state"][j:j+6])
            observed = np.eye(4)
            observed[:3, 3] = converted[i, a, :3]
            observed[:3, :3] = Rotation.from_quat(converted[i, a, 3:7]).as_matrix()
            source_columns = np.array(row["observation.states.eef"][p+3:p+9]).reshape(2, 3).T
            max_roundtrip_error = max(max_roundtrip_error, float(np.max(np.abs(observed[:3, :2]-source_columns))))
            native_errors.append(errors(base @ local, observed))
            base_errors.append(errors(local, observed))
            inferred_roots.append(observed @ np.linalg.inv(local))
            measured_poses.append(observed)
            local_poses.append(local)
        fitted_errors = [errors(inferred_roots[0] @ local, observed)
                         for local, observed in zip(local_poses, measured_poses)]
        maximum = np.max(native_errors, axis=0)
        report["arms"][arm] = {
            "configured_environment_from_base": base.tolist(),
            "link6_fk_max_position_error_m": float(maximum[0]),
            "link6_fk_max_rotation_error_rad": float(maximum[1]),
            "link6_fk_matches": bool(maximum[0] < .002 and maximum[1] < .01),
            "incorrect_per_arm_base_interpretation_median_errors_m_rad": np.median(base_errors, axis=0).tolist(),
            "environment_from_base_inferred_first_frame": inferred_roots[0].tolist(),
            "first_frame_fit_max_errors_m_rad": np.max(fitted_errors, axis=0).tolist(),
            "min_adjacent_quaternion_dot": float(np.min(np.sum(converted[1:,a,3:7]*converted[:-1,a,3:7], axis=1)))
                if len(rows) > 1 else None,
        }
    report["rotation_roundtrip_max_matrix_element_error"] = max_roundtrip_error
    report["matches_native_link6_in_environment_frame"] = all(x["link6_fk_matches"] for x in report["arms"].values())
    report["direct_gpt_policy_tcp_equivalence"] = False
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(report, indent=2))
    if not report["matches_native_link6_in_environment_frame"] or max_roundtrip_error > 1e-3:
        raise SystemExit("Frame audit did not pass")


if __name__ == "__main__":
    main()
