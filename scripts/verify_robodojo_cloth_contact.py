#!/usr/bin/env python3
"""Probe the recorded fold-clothes approach through real IK and physics.

This is a contact diagnostic, not a full folding evaluation. No VLM is called.
"""
import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--height", type=float, default=0.025)
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
    report = {"height_m": args.height, "steps": []}
    try:
        import torch
        from PIL import Image
        from omegaconf import OmegaConf
        from utils.load_file import load_yaml
        from utils.pipeline_utils import process_config, process_randomization
        from src.eval_client import eval_env
        from gpt_policy.robodojo import RoboDojoAdapter, RoboDojoCalibration
        from gpt_policy.robodojo.adapter import _jsonable
        from gpt_policy.robodojo.model import GPTPolicyModel

        class NoPolicy:
            def __init__(self, **kwargs):
                pass

            def call(self, **kwargs):
                return None

        eval_env.WsModelClient = NoPolicy
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        cfg = load_yaml(str(robo / "env_cfg/gpt_policy_x5.yml"))
        cfg.update(task_name="fold_clothes", num_envs=1, seed=0, policy_name="GPT_Policy", eval_num=1)
        config = OmegaConf.create({
            key: load_yaml(str(robo / "env_cfg" / key / f"{cfg['config'][key]}.yml"))
            for key in ("sim", "scene", "camera", "robot")
        } | {"task_env": load_yaml(str(robo / "task/RoboDojo/config/fold_clothes.yml")),
             "eval_cfg": cfg, "deploy_cfg": {"port": 1}})
        config, _ = process_config(process_randomization(config), "fold_clothes")
        config.sim.scene.num_envs = 1
        config.sim.seed = [0]
        config.camera.default_frequency = 25
        environment = eval_env.create_eval_env(config, app)
        environment.reset(seed=[0])
        environment.run_reward()
        manifest = json.loads((root / "configs/examples/robodojo_calibration.json").read_text())
        model = GPTPolicyModel.__new__(GPTPolicyModel)
        model.arms = ("left", "right")
        model.robot_model = "X5"
        model.task_name = "fold_clothes"
        model._reachability_bounds = {"x": [-0.05, 0.65], "y": [-0.55, 0.55], "z": [None, 0.45]}
        model._max_reachability_step_m = 0.30
        # This legacy contact probe intentionally uses larger diagnostic
        # rotations than the 0.35 rad policy action contract.
        model._max_reachability_rotation_rad = np.pi
        model._max_tracking_error_m = 0.03
        model._max_tracking_error_rad = 0.20
        model._observation_history = {}
        layout = environment.reward_manager.func_parser.layout_manager
        garment = layout.get_scene_object(0, layout.get_instance_name(label="target", env_idx=0))
        initial_points = garment.sample_mesh_vertices()[0].copy()
        targets = [
            ([0.2427, -0.0805, 0.2365, 0, 0.707107, 0, 0.707107], 1.0),
            ([0.2427, -0.0805, 0.2365, 0, 1, 0, 0], 1.0),
            ([0.2427, -0.1305, 0.2365, 0, 1, 0, 0], 1.0),
            ([0.14, -0.052, 0.09, 0, 1, 0, 0], 1.0),
            ([0.148, -0.052, args.height, 0, 1, 0, 0], 1.0),
            ([0.148, -0.052, args.height, 0, 1, 0, 0], 0.0),
            ([0.148, -0.052, 0.08, 0, 1, 0, 0], 0.0),
        ]
        frame = environment.get_obs()
        for index, (pose, grip) in enumerate(targets):
            model.step = index + 1
            model._observation_history[index] = {"state": frame["state"]}
            model.adapter = RoboDojoAdapter(RoboDojoCalibration.from_observation(manifest, frame))
            model.latest_frame = frame | {"task_name": "fold_clothes"}
            action = model.adapter.action({"target": {"left": {"pose_xyzquat": pose}}})
            for arm in model.arms:
                action.setdefault(f"{arm}_ee_pose", frame["state"][f"{arm}_ee_pose"])
                action[f"{arm}_ee_joint_state"] = [grip] if arm == "left" else frame["state"][f"{arm}_ee_joint_state"]
            rejection = model._validate_reachability([action])
            if rejection:
                report["rejected"] = rejection
                break
            model._last_commanded_world = {arm: action[f"{arm}_ee_pose"] for arm in model.arms}
            environment.take_action(action)
            frame = environment.get_obs()
            feedback = model._execution_feedback(frame)
            points = garment.sample_mesh_vertices()[0]
            step = {"target_tcp": pose, "grip": grip, "feedback": feedback,
                    "ik": frame.get("ik_feedback"), "cloth_max_z_m": float(points[:, 2].max()),
                    "cloth_max_vertex_lift_m": float((points[:, 2] - initial_points[:, 2]).max())}
            report["steps"].append(step)
            for name, camera in frame.get("vision", {}).items():
                if camera.get("color") is not None:
                    Image.fromarray(camera["color"][..., :3]).save(output / f"step{index}_{name}.png")
            (output / "report.json").write_text(json.dumps(_jsonable(report), indent=2))
            print("CONTACT", index, json.dumps(_jsonable(step)), flush=True)
        report["approach_executed"] = len(report["steps"]) == len(targets) and all(
            step["feedback"]["arms"]["left"]["translation_error_m"] < 0.005 for step in report["steps"])
        (output / "report.json").write_text(json.dumps(_jsonable(report), indent=2))
        return 0 if report["approach_executed"] else 1
    except Exception as exc:
        import traceback
        report["error"] = repr(exc)
        (output / "report.json").write_text(json.dumps(report, default=str, indent=2))
        traceback.print_exc()
        return 1
    finally:
        if environment is not None:
            environment.close()
        app.close()


if __name__ == "__main__":
    raise SystemExit(main())
