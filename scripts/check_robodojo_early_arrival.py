#!/usr/bin/env python3
"""Model-free native check of opt-in completion; never an evaluation attempt."""

import argparse
from pathlib import Path

from gpt_policy.simulation.robodojo.config import check, load_config
from gpt_policy.simulation.robodojo.io import write_json
from gpt_policy.simulation.robodojo.runtime import Simulator


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()
    config = load_config(args.config)
    if config.get("early_arrival") is not True:
        raise ValueError("This check requires early_arrival=true")
    evidence = check(config, "classify_objects", 0, 0)
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "manifest.json", evidence)
    receipts = []
    with Simulator(config, evidence["case"], args.output / "sim", args.gpu) as sim:
        targets = {
            arm: {"pose_xyzquat": list(sim.observation["arms"][arm]["link6_pose_xyzquat"]),
                  "gripper": 1.0} for arm in ("left", "right")
        }
        for label, mode, opening in [
            ("initial", "hold", 1.0), ("settle", "hold", 1.0),
            ("arrived", "move", 1.0), ("explicit_wait", "hold", 1.0),
            ("close", "move", 0.0), ("open", "move", 1.0),
        ]:
            point = dict(steps=5, mode=mode, **{
                arm: {**target, "gripper": opening} for arm, target in targets.items()
            })
            result = sim.move({
                "observation_id": sim.observation["observation_id"],
                "note": "Engineering check at initial robot pose: " + label,
                "waypoints": [point],
            })
            receipts.append({"label": label, **{
                k: v for k, v in result.items() if k != "observation"
            }})
        native = sim.finish("engineering_early_arrival_check")
    write_json(args.output / "receipts.json", receipts)
    write_json(args.output / "result.json", native)
    for row in receipts:
        assert row["rejection"] is None, row
        if row["label"] != "arrived":
            assert row["acknowledged_controls"] == 5, row
    arrived = receipts[2]
    if config.get("stop_on_arrival", True):
        assert 2 <= arrived["acknowledged_controls"] < 5, arrived
        assert arrived["waypoint_results"][0]["stop_reason"] == "stable_arrival"
    else:
        assert arrived["acknowledged_controls"] == 5, arrived
        assert arrived["waypoint_results"][0]["stop_reason"] == "requested_controls"
        assert arrived["waypoint_results"][0]["stable_samples"] >= 2, arrived
    write_json(args.output / "check.json", {
        "passed": True, "model_called": False, "valid_for_success_rate": False,
        "stop_on_arrival": config.get("stop_on_arrival", True),
        "saved_controls": arrived["waypoint_results"][0]["saved_controls"],
        "acknowledged_controls": sum(r["acknowledged_controls"] for r in receipts),
    })


if __name__ == "__main__":
    main()
