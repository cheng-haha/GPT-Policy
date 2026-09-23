#!/usr/bin/env python3
"""Exercise one bounded EE action in RoboDojo without a model service."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--layout-id", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    robo = root / "third_party/RoboDojo"
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    sys.path[:0] = [str(root / "src"), str(robo), str(root / "third_party/XPolicyLab")]
    os.chdir(robo)

    from isaaclab.app import AppLauncher
    from env.camera_manager.capture.render_sync import zero_delay_kit_args
    app = AppLauncher(headless=True, enable_cameras=True, device=f"cuda:{args.gpu}",
                      kit_args=zero_delay_kit_args()).app
    environment = None
    report = {"task": "organize_table", "layout_id": args.layout_id, "gpu": args.gpu}
    try:
        from omegaconf import OmegaConf
        from utils.load_file import load_yaml
        from utils.pipeline_utils import process_config, process_randomization
        from src.eval_client import eval_env
        from gpt_policy.robodojo import RoboDojoAdapter, RoboDojoCalibration
        from gpt_policy.robodojo.model import GPTPolicyModel
        from gpt_policy.robodojo.adapter import _jsonable

        class NoPolicy:
            def __init__(self, **kwargs):
                pass

            def call(self, **kwargs):
                return None

        eval_env.WsModelClient = NoPolicy
        cfg = load_yaml(str(robo / "env_cfg/gpt_policy_x5.yml"))
        cfg.update(task_name="organize_table", num_envs=1, seed=0,
                   policy_name="GPT_Policy", eval_num=1)
        config = OmegaConf.create({
            key: load_yaml(str(robo / "env_cfg" / key / f"{cfg['config'][key]}.yml"))
            for key in ("sim", "scene", "camera", "robot")
        } | {"task_env": load_yaml(str(robo / "task/RoboDojo/config/organize_table.yml")),
             "eval_cfg": cfg, "deploy_cfg": {"port": 1}})
        config, _ = process_config(process_randomization(config), "organize_table")
        config.sim.scene.num_envs = 1
        config.sim.seed = [0]
        config.camera.default_frequency = 25
        environment = eval_env.create_eval_env(config, app)
        environment.reset(seed=[args.layout_id])
        environment.run_reward()
        before = environment.get_obs()
        calibration = RoboDojoCalibration.from_observation(
            json.loads((root / "configs/examples/robodojo_calibration.json").read_text()), before)
        model = GPTPolicyModel.__new__(GPTPolicyModel)
        model.adapter = RoboDojoAdapter(calibration, ("left", "right"))
        model.arms = ("left", "right")
        model.step = 1
        model._observation_history = {0: {"state": before["state"]}}
        model._max_tracking_error_m = 0.03
        model._max_tracking_error_rad = 0.20
        model._reachability_bounds = {"x": [-0.05, 0.65], "y": [-0.55, 0.55],
                                      "z": [None, 0.45]}
        model._max_reachability_step_m = 0.05
        model._max_reachability_rotation_rad = 0.35
        model.latest_frame = before
        action = {}
        for arm in model.arms:
            pose = list(before["state"][f"{arm}_ee_pose"])
            if arm == "left":
                pose[0] += 0.02
            action[f"{arm}_ee_pose"] = pose
            action[f"{arm}_ee_joint_state"] = list(before["state"][f"{arm}_ee_joint_state"])
        report["reachability_rejection"] = model._validate_reachability([action])
        if report["reachability_rejection"] is not None:
            raise RuntimeError("Bounded diagnostic action was rejected")
        model._last_commanded_world = {arm: action[f"{arm}_ee_pose"] for arm in model.arms}
        environment.take_action(action)
        after = environment.get_obs()
        report["ik_feedback"] = _jsonable(after.get("ik_feedback"))
        report["execution_feedback"] = _jsonable(model._execution_feedback(after, after.get("ik_feedback")))
        before_pose = np.asarray(before["state"]["left_ee_pose"], dtype=float)
        after_pose = np.asarray(after["state"]["left_ee_pose"], dtype=float)
        report["measured_translation_m"] = float(np.linalg.norm(after_pose[:3] - before_pose[:3]))
        left = report["execution_feedback"]["arms"]["left"]
        report["passed"] = (report["ik_feedback"]["arms"]["left"]["status"] == "Success"
                            and report["measured_translation_m"] > 0.002
                            and left["tracking_checked"] and not left["stalled"])
        output.write_text(json.dumps(_jsonable(report), indent=2) + "\n")
        print(json.dumps({"passed": report["passed"],
                          "measured_translation_m": report["measured_translation_m"],
                          "ik_status": report["ik_feedback"]["status"],
                          "tracking_status": left["status"]}), flush=True)
        return 0 if report["passed"] else 1
    except Exception as error:
        import traceback
        report["error"] = repr(error)
        output.write_text(json.dumps(report, default=str, indent=2) + "\n")
        traceback.print_exc()
        return 1
    finally:
        if environment is not None:
            environment.close()
        app.close()


if __name__ == "__main__":
    raise SystemExit(main())
