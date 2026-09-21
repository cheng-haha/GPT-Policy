#!/usr/bin/env python3
"""Validate RoboDojo base-frame TCP commands with a fixed low side-push path.

This is a kinematic/action-interface check. It does not move Isaac Sim. It
proves that base-frame GPT poses become RoboDojo world-frame poses and recover
the same base-frame pose when read back through the adapter.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from gpt_policy.robodojo import RoboDojoAdapter, RoboDojoCalibration


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--calibration",
        type=Path,
        default=Path("configs/examples/robodojo_calibration.json"),
    )
    args = parser.parse_args()
    manifest = json.loads(args.calibration.read_text())
    calibration = RoboDojoCalibration.from_manifest(manifest)
    adapter = RoboDojoAdapter(calibration, ("left", "right"))

    # Low, horizontal, left-arm side push in the left arm base frame.
    path = [
        [0.10, 0.20, 0.11, 0.0, 0.0, 0.0, 1.0],
        [0.16, 0.20, 0.11, 0.0, 0.0, 0.0, 1.0],
        [0.22, 0.20, 0.11, 0.0, 0.0, 0.0, 1.0],
    ]
    errors = []
    for index, target in enumerate(path):
        action = adapter.action({"target": {"left": target, "right": None}})
        frame = {"state": {
            "left_arm_joint_state": [0.0] * 6,
            "left_ee_joint_state": [1.0],
            "left_ee_pose": action["left_ee_pose"],
            "right_arm_joint_state": [0.0] * 6,
            "right_ee_joint_state": [1.0],
            "right_ee_pose": [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        }}
        recovered = np.asarray(adapter.state(frame)["arms"]["left"]["tcp_xyzquat"])
        error = float(np.max(np.abs(recovered - np.asarray(target))))
        errors.append(error)
        print(f"waypoint {index}: max base-frame round-trip error = {error:.3e}")
    maximum = max(errors)
    print(f"PASS: fixed low side-push path, maximum error = {maximum:.3e}")
    return 0 if maximum <= 1e-8 else 1


if __name__ == "__main__":
    raise SystemExit(main())
