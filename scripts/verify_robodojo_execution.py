#!/usr/bin/env python3
"""Replay recorded TCP commands through the real evaluator, without a VLM.

Only the policy transport is stubbed. IK, control interpolation, physics,
observations, and camera calibration use the production execution path.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--legacy", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    robo = root / "third_party/RoboDojo"
    sys.path[:0] = [str(root / "src"), str(robo), str(root / "third_party/XPolicyLab")]
    os.chdir(robo)
    from isaaclab.app import AppLauncher
    from env.camera_manager.capture.render_sync import zero_delay_kit_args

    app = AppLauncher(headless=True, enable_cameras=True, device="cuda:0", kit_args=zero_delay_kit_args()).app
    environment = None
    report = {"legacy": args.legacy, "seed": args.seed, "steps": []}
    try:
        import torch
        from PIL import Image
        from omegaconf import OmegaConf
        from utils.load_file import load_yaml
        from utils.pipeline_utils import process_config, process_randomization
        from src.eval_client import eval_env
        from gpt_policy.robodojo import RoboDojoAdapter, RoboDojoCalibration
        from gpt_policy.robodojo.model import GPTPolicyModel
        from gpt_policy.robodojo.calibration import quat_wxyz_to_matrix

        class NoPolicy:
            def __init__(self, **kwargs):
                pass

            def call(self, **kwargs):
                return None

        eval_env.WsModelClient = NoPolicy
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        cfg = load_yaml(str(robo / "env_cfg/gpt_policy_x5.yml"))
        cfg.update(task_name="push_T", num_envs=1, seed=0, policy_name="GPT_Policy", eval_num=1)
        if args.legacy:
            cfg.pop("action_execution", None)
        config = OmegaConf.create({
            key: load_yaml(str(robo / "env_cfg" / key / f"{cfg['config'][key]}.yml"))
            for key in ("sim", "scene", "camera", "robot")
        } | {"task_env": load_yaml(str(robo / "task/RoboDojo/config/push_T.yml")),
             "eval_cfg": cfg, "deploy_cfg": {"port": 1}})
        config, _ = process_config(process_randomization(config), "push_T")
        config.sim.scene.num_envs = 1
        config.sim.seed = [args.seed]
        config.camera.default_frequency = 25
        environment = eval_env.create_eval_env(config, app)
        environment.reset(seed=[args.seed])
        environment.run_reward()
        manifest = json.loads((root / "configs/examples/robodojo_calibration.json").read_text())
        model = GPTPolicyModel.__new__(GPTPolicyModel)
        model.arms = ("left", "right")
        # These are the exact two base-frame TCP requests from the failed run.
        targets = [
            {"right": [0.283, -0.012, 0.1565, 0, 0.707107, 0, 0.707107]},
            {"left": [0.32, 0.13, 0.1565, 0, 0.707107, 0, 0.707107]},
        ]
        frame = environment.get_obs()
        for index, target in enumerate(targets):
            calibration = RoboDojoCalibration.from_observation(manifest, frame)
            model.adapter = RoboDojoAdapter(calibration, model.arms)
            action = model.adapter.action({"target": target})
            for arm in model.arms:
                action.setdefault(f"{arm}_ee_pose", frame["state"][f"{arm}_ee_pose"])
                action[f"{arm}_ee_joint_state"] = [1.0]
            model._last_commanded_world = {arm: action[f"{arm}_ee_pose"] for arm in model.arms}
            environment.take_action(action)
            frame = environment.get_obs()
            model.adapter.calibration = RoboDojoCalibration.from_observation(manifest, frame)
            feedback = model._execution_feedback(frame)
            step = {"target": target, "feedback": feedback, "ik": frame.get("ik_feedback"),
                    "state": frame["state"], "camera_errors": {}}
            for camera_index, camera_name in enumerate(environment.camera_manager.camera_names[0]):
                sensor = environment.camera_manager.cameras[0][camera_index]
                positions, quaternions = sensor._prim_view.get_world_poses(usd=False)
                expected = np.eye(4)
                expected[:3, 3] = positions[0].cpu().numpy()
                expected[:3, :3] = quat_wxyz_to_matrix(quaternions[0].cpu().numpy())
                actual = frame["vision"][camera_name]["extrinsic_matrix"]
                step["camera_errors"][camera_name] = float(np.max(np.abs(expected - actual)))
                Image.fromarray(frame["vision"][camera_name]["color"][..., :3]).save(output / f"step{index}_{camera_name}.png")
            report["steps"].append(step)
            from gpt_policy.robodojo.adapter import _jsonable
            (output / "report.json").write_text(json.dumps(_jsonable(report), indent=2) + "\n")
            print("EXECUTION", index, json.dumps(feedback), flush=True)
        report["passed"] = all(
            arm["translation_error_m"] < 0.002 and arm["rotation_error_rad"] < 0.01
            for step in report["steps"] for arm in step["feedback"]["arms"].values()
        ) and all(e < 2e-6 for step in report["steps"] for e in step["camera_errors"].values())
        (output / "report.json").write_text(json.dumps(_jsonable(report), indent=2) + "\n")
        print("EXECUTION PASSED:", report["passed"], flush=True)
        return 0 if report["passed"] else 1
    finally:
        if environment is not None:
            environment.close()
        app.close()


if __name__ == "__main__":
    raise SystemExit(main())
