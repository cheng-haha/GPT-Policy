#!/usr/bin/env python3
"""Validate calibration against live RoboDojo geometry and rendered markers.

Diagnostic-only scene markers and simulator ground truth never enter policy
observations. Results and RGB evidence are saved without starting a VLM.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import traceback
from pathlib import Path

import numpy as np


def pose_matrix(pose):
    from gpt_policy.robodojo.calibration import quat_wxyz_to_matrix

    matrix = np.eye(4)
    matrix[:3, :3] = quat_wxyz_to_matrix(pose[3:])
    matrix[:3, 3] = pose[:3]
    return matrix


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-cfg", choices=("gpt_policy_x5", "gpt_policy_franka"), default="gpt_policy_x5")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[1]
    robodojo = root / "third_party" / "RoboDojo"
    sys.path[:0] = [str(root / "src"), str(robodojo)]
    os.chdir(robodojo)
    seeds = [int(value) for value in args.seeds.split(",")]
    report = {"status": "RUNNING", "env_cfg": args.env_cfg, "seeds": seeds, "checks": [], "cameras": [], "robots": [], "frames": [], "weak_baseline_diagnostics": []}

    def save_report():
        (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")

    def check(name, error, tolerance, **context):
        passed = bool(np.isfinite(error) and error <= tolerance)
        report["checks"].append({"name": name, "error": float(error), "tolerance": tolerance, "passed": passed, **context})
        if not passed:
            print(f"FAIL {name}: {error:.6g} > {tolerance:.6g} {context}", flush=True)

    save_report()
    from env.camera_manager.capture.render_sync import zero_delay_kit_args
    from isaaclab.app import AppLauncher

    app = AppLauncher(headless=True, enable_cameras=True, device=f"cuda:{args.gpu}", kit_args=zero_delay_kit_args()).app
    environment = None
    try:
        import omni.usd
        import torch
        from omegaconf import OmegaConf
        from PIL import Image
        from pxr import Gf, Sdf, UsdGeom, UsdShade
        from scipy.ndimage import center_of_mass, label
        from scipy.optimize import linear_sum_assignment
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False

        from env.observation_manager.obs_manager import ObsManager
        from task.RoboDojo.task_registry import load_task_class
        from utils.load_file import load_yaml
        from utils.pipeline_utils import process_config, process_randomization

        from gpt_policy.robodojo import RoboDojoAdapter, RoboDojoCalibration
        from gpt_policy.vision.perception import _triangulate

        env_cfg = load_yaml(str(robodojo / "env_cfg" / f"{args.env_cfg}.yml"))
        env_cfg.update(task_name="push_T", num_envs=1, seed=seeds[0])
        config = OmegaConf.create({
            key: load_yaml(str(robodojo / "env_cfg" / key / f"{env_cfg['config'][key]}.yml"))
            for key in ("sim", "scene", "camera", "robot")
        } | {
            "task_env": load_yaml(str(robodojo / "task/RoboDojo/config/push_T.yml")),
            "eval_cfg": env_cfg,
        })
        config, _ = process_config(process_randomization(config), "push_T")
        config.sim.scene.num_envs = 1
        config.sim.seed = [seeds[0]]
        config.sim.device = f"cuda:{args.gpu}"
        config.camera.default_frequency = 25
        _, task_class = load_task_class("push_T")
        environment = task_class(config, app)
        manifest_name = "robodojo_calibration.json" if args.env_cfg == "gpt_policy_x5" else "robodojo_franka_calibration.json"
        manifest = json.loads((root / "configs/examples" / manifest_name).read_text())
        arms = tuple(manifest["world_from_base"])
        samples = [(0.0, 0.0, 0.0), (0.12, 0.08, -0.06), (-0.12, -0.06, 0.08), (0.06, -0.10, 0.10), (-0.06, 0.10, -0.08)]
        colors = [(1, 0, 1), (0, 1, 1), (1, 1, 0)]
        local_points = np.array([[-0.045, -0.035, -0.32, 1], [0.045, -0.035, -0.32, 1], [0, 0, -0.36, 1], [-0.045, 0.035, -0.40, 1], [0.045, 0.035, -0.40, 1]])

        for seed in seeds:
            print(f"SCENE {args.env_cfg} seed={seed}: resetting", flush=True)
            environment.reset(seed=[seed])
            environment.scene_manager.apply_saved_poses(env_idx_list=[0])
            manager = environment.camera_manager
            observer = ObsManager(config.eval_cfg.observation, 1, environment.dt, "push_T", {}, [seed])
            observer.initialize(environment)
            observer.render_for_capture()
            initial_frame = observer.get_obs([0])[0]
            stage = omni.usd.get_context().get_stage()
            stage.RemovePrim("/World/CalibrationMarkers")
            marker_points = {}
            history = {}
            for camera_index, camera_name in enumerate(manager.camera_names[0]):
                camera = manager.cameras[0][camera_index]
                positions, orientations = camera._prim_view.get_world_poses(usd=False)
                world_from_usd = pose_matrix(np.r_[positions[0].detach().cpu().numpy(), orientations[0].detach().cpu().numpy()])
                points = (world_from_usd @ local_points.T).T[:, :3]
                marker_points[camera_name] = points
                material = UsdShade.Material.Define(stage, f"/World/CalibrationMarkers/Material{camera_index}")
                shader = UsdShade.Shader.Define(stage, f"{material.GetPath()}/Shader")
                shader.CreateIdAttr("UsdPreviewSurface")
                color = colors[camera_index % len(colors)]
                shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*[5.0 * value for value in color]))
                shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0, 0, 0))
                material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
                for point_index, point in enumerate(points):
                    sphere = UsdGeom.Sphere.Define(stage, f"/World/CalibrationMarkers/{camera_name}_{point_index}")
                    sphere.CreateRadiusAttr(0.004)
                    UsdGeom.Xformable(sphere).AddTranslateOp().Set(Gf.Vec3d(*point))
                    UsdShade.MaterialBindingAPI.Apply(sphere.GetPrim()).Bind(material)
                report["cameras"].append({"seed": seed, "name": camera_name, "sensor_prim": str(camera.prim.GetPath()), "initial_intrinsics": np.asarray(initial_frame["vision"][camera_name]["intrinsic_matrix"]).tolist()})

            for _ in range(5):
                environment.render()

            defaults = [key.data.default_joint_pos.clone() for key in environment.robot_manager.robot_key]
            for sample_index, offsets in enumerate(samples):
                for robot_index, (robot, key) in enumerate(zip(environment.robot_manager.robot_list, environment.robot_manager.robot_key)):
                    positions = defaults[robot_index].clone()
                    for joint_index, offset in zip(robot.arm_joint_indices[:3], offsets):
                        positions[:, joint_index] += offset * (-1 if robot_index else 1)
                    key.write_joint_state_to_sim(positions, torch.zeros_like(positions))
                    key.set_joint_position_target(positions)
                for _ in range(20):
                    environment.sim_step(render=False)
                observer.render_for_capture()
                frame = observer.get_obs([0])[0]
                calibration = RoboDojoCalibration.from_observation(manifest, frame)
                adapter = RoboDojoAdapter(calibration, arms)
                state = adapter.state(frame)
                context = {"seed": seed, "sample": sample_index}
                for arm, robot in zip(arms, environment.robot_manager.robot_list):
                    actual_base = pose_matrix(environment.robot_manager.get_link_pose(robot, robot.base_link, [0], is_relative=False)[0])
                    check("base_transform_max_abs", np.max(np.abs(actual_base - calibration.world_from_base[arm])), 2e-5, arm=arm, **context)
                    tcp = state["arms"][arm]["tcp_xyzquat"]
                    source_link = pose_matrix(environment.robot_manager.get_link_pose(robot, robot.ee_link_name, [0], is_relative=False)[0])
                    source_offset = np.array([robot.gripper_bias, 0, 0]) if robot.robot_name == "x5" else np.array([0, 0, robot.gripper_bias])
                    expected_tcp = source_link[:3, 3] + source_link[:3, :3] @ source_offset
                    actual_tcp = calibration.world_from_base[arm] @ np.array([*tcp[:3], 1])
                    check("tcp_grasp_center_error_m", np.linalg.norm(actual_tcp[:3] - expected_tcp), 1e-6, arm=arm, **context)
                    from gpt_policy.robodojo.calibration import quat_wxyz_to_matrix
                    tcp_rotation = calibration.world_from_base[arm][:3, :3] @ quat_wxyz_to_matrix([tcp[6], *tcp[3:6]])
                    extension_axis = source_link[:3, 0 if robot.robot_name == "x5" else 2]
                    check("tcp_forward_axis_error", np.linalg.norm(tcp_rotation[:, 2] - extension_axis), 1e-6, arm=arm, **context)
                    target = {arm: tcp} if len(arms) > 1 else tcp
                    action = adapter.action({"target": target})
                    pose_key = f"{arm}_ee_pose" if len(arms) > 1 else "ee_pose"
                    check("control_pose_round_trip", np.max(np.abs(pose_matrix(action[pose_key]) - pose_matrix(frame["state"][pose_key]))), 1e-9, arm=arm, **context)
                    if sample_index == 0:
                        report["robots"].append({"seed": seed, "arm": arm, "base_link": robot.base_link, "source_ee_link": robot.ee_link_name, "configured_gripper_bias": robot.gripper_bias})

                for camera_index, camera_name in enumerate(manager.camera_names[0]):
                    camera = manager.cameras[0][camera_index]
                    data = frame["vision"][camera_name]
                    points = marker_points[camera_name]
                    camera_context = context | {"camera": camera_name}
                    positions, orientations = camera._prim_view.get_world_poses(usd=False)
                    native_pose = pose_matrix(np.r_[positions[0].detach().cpu().numpy(), orientations[0].detach().cpu().numpy()])
                    check("sensor_extrinsics_vs_fabric", np.max(np.abs(native_pose - data["extrinsic_matrix"])), 2e-6, **camera_context)
                    native_intrinsics = camera.get_intrinsics_matrix().detach().cpu().numpy()
                    view_points = (np.diag([1, -1, -1, 1]) @ np.linalg.inv(native_pose) @ np.c_[points, np.ones(len(points))].T)[:3]
                    native_projection = native_intrinsics @ view_points
                    sdk_pixels = (native_projection[:2] / native_projection[2]).T
                    pixels = []
                    for point, sdk_pixel in zip(points, sdk_pixels):
                        base_point = np.linalg.solve(calibration.world_from_base[arms[0]], [*point, 1])[:3]
                        optical_point = np.linalg.solve(calibration.base_from_camera(camera_name, arms[0]), [*base_point, 1])[:3]
                        projection = calibration.intrinsics[camera_name] @ optical_point
                        pixel = projection[:2] / projection[2]
                        pixels.append(pixel)
                        ray = calibration.camera_ray_in_base(camera_name, sdk_pixel, arms[0])
                        delta = base_point - ray["ray_origin_base_xyz"]
                        check("sdk_projection_error_px", np.linalg.norm(pixel - sdk_pixel), 0.001, **camera_context)
                        check("sdk_ray_error_m", np.linalg.norm(np.cross(delta, ray["ray_direction_base_xyz"])), 2e-6, **camera_context)
                    pixels = np.asarray(pixels)
                    image = np.asarray(data["color"])[..., :3]
                    Image.fromarray(image).save(output / f"seed{seed}_pose{sample_index}_{camera_name}.png")
                    color = np.array(colors[camera_index % len(colors)], dtype=bool)
                    signal = np.min(image[..., color], axis=2).astype(float)
                    background = np.max(image[..., ~color], axis=2).astype(float)
                    mask = (signal > 25) & (signal > 4 * (background + 1)) & (background < 80)
                    components, count = label(mask)
                    centers = []
                    for component in range(1, count + 1):
                        area = np.count_nonzero(components == component)
                        if 2 <= area <= 500:
                            vertical, horizontal = center_of_mass(signal * mask, components, component)
                            centers.append([horizontal + 0.5, vertical + 0.5])
                    measured = {}
                    if centers:
                        distances = np.linalg.norm(pixels[:, None, :] - np.asarray(centers)[None, :, :], axis=2)
                        rows, columns = linear_sum_assignment(distances)
                        for row, column in zip(rows, columns):
                            error = distances[row, column]
                            check("rendered_marker_error_px", error, 1.25, marker=int(row), **camera_context)
                            if error <= 1.25:
                                measured[int(row)] = np.asarray(centers[column])
                    check("rendered_marker_missing_count", len(points) - len(measured), 0, **camera_context)
                    report["frames"].append(camera_context | {
                        "intrinsics": np.asarray(data["intrinsic_matrix"]).tolist(),
                        "world_from_usd_camera": np.asarray(data["extrinsic_matrix"]).tolist(),
                        "world_from_base": {arm: transform.tolist() for arm, transform in calibration.world_from_base.items()},
                        "marker_world_xyz": points.tolist(), "projected_pixels": pixels.tolist(),
                        "detected_pixels": {str(index): pixel.tolist() for index, pixel in measured.items()},
                    })
                    previous = history.get(camera_name)
                    if previous is not None:
                        for point_index, point in enumerate(points):
                            current_ray = calibration.camera_ray_in_base(camera_name, sdk_pixels[point_index], arms[0])
                            prior_ray = previous["calibration"].camera_ray_in_base(camera_name, previous["sdk_pixels"][point_index], arms[0])
                            reconstructed, _, quality = _triangulate(np.array(current_ray["ray_origin_base_xyz"]), np.array(current_ray["ray_direction_base_xyz"]), np.array(prior_ray["ray_origin_base_xyz"]), np.array(prior_ray["ray_direction_base_xyz"]))
                            if quality["camera_baseline_m"] > 0.005 and quality["parallax_angle_deg"] > 2:
                                target = np.linalg.solve(calibration.world_from_base[arms[0]], [*point, 1])[:3]
                                check("temporal_triangulation_sdk_error_m", np.linalg.norm(reconstructed - target), 2e-5, **camera_context)
                                if point_index in measured and point_index in previous["measured"]:
                                    current_ray = calibration.camera_ray_in_base(camera_name, measured[point_index], arms[0])
                                    prior_ray = previous["calibration"].camera_ray_in_base(camera_name, previous["measured"][point_index], arms[0])
                                    reconstructed, _, _ = _triangulate(np.array(current_ray["ray_origin_base_xyz"]), np.array(current_ray["ray_direction_base_xyz"]), np.array(prior_ray["ray_origin_base_xyz"]), np.array(prior_ray["ray_direction_base_xyz"]))
                                    error = float(np.linalg.norm(reconstructed - target))
                                    if quality["parallax_angle_deg"] >= 5.0:
                                        check("temporal_triangulation_rendered_error_m", error, 0.005, parallax_deg=quality["parallax_angle_deg"], **camera_context)
                                    else:
                                        report["weak_baseline_diagnostics"].append({"error_m": error, "parallax_deg": quality["parallax_angle_deg"], **camera_context})
                    history[camera_name] = {"calibration": calibration, "sdk_pixels": sdk_pixels, "measured": measured}
                save_report()
                print(f"SCENE seed={seed} pose={sample_index}: {len(report['checks'])} checks", flush=True)
            for arm, robot, key in zip(arms, environment.robot_manager.robot_list, environment.robot_manager.robot_key):
                observer.render_for_capture()
                frame = observer.get_obs([0])[0]
                adapter.calibration = RoboDojoCalibration.from_observation(manifest, frame)
                tcp = adapter.state(frame)["arms"][arm]["tcp_xyzquat"]
                target_tcp = list(tcp)
                target_tcp[0] += 0.01
                action = adapter.action({"target": {arm: target_tcp} if len(arms) > 1 else target_tcp})
                source_key = f"{arm}_ee_pose" if len(arms) > 1 else "ee_pose"
                solution = environment.robot_manager.solve_ik(action[source_key], 0, robot)
                check("tcp_ik_solve_failure", int(solution["status"] != "Success"), 0, seed=seed, arm=arm)
                if solution["status"] == "Success":
                    positions = key.data.joint_pos.clone()
                    positions[:, robot.arm_joint_indices] = torch.as_tensor(solution["joint_value"], device=positions.device)
                    key.set_joint_position_target(positions)
                    for _ in range(250):
                        environment.sim_step(render=False)
                    observer.render_for_capture()
                    actual = adapter.state(observer.get_obs([0])[0])["arms"][arm]["tcp_xyzquat"]
                    check("tcp_ik_execution_position_error_m", np.linalg.norm(np.array(actual[:3]) - target_tcp[:3]), 0.005, seed=seed, arm=arm)
                    actual_rotation = quat_wxyz_to_matrix([actual[6], *actual[3:6]])
                    target_rotation = quat_wxyz_to_matrix([target_tcp[6], *target_tcp[3:6]])
                    angle = np.arccos(np.clip((np.trace(actual_rotation.T @ target_rotation) - 1) / 2, -1, 1))
                    check("tcp_ik_execution_rotation_error_rad", angle, 0.03, seed=seed, arm=arm)
                save_report()
        failures = [item for item in report["checks"] if not item["passed"]]
        report["status"] = "FAIL" if failures else "PASS"
        report["failed_checks"] = len(failures)
        report["total_checks"] = len(report["checks"])
        print(f"{report['status']}: {report['total_checks']} checks, {len(failures)} failures; {output / 'report.json'}", flush=True)
        save_report()
        if failures:
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(1)
        return 0
    except BaseException as error:
        report["status"] = "ERROR"
        report["error"] = traceback.format_exc()
        save_report()
        logging.getLogger(__name__).exception("Scene calibration verification failed")
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(130 if isinstance(error, KeyboardInterrupt) else 1)
    finally:
        if environment is not None:
            environment.close()
        app.close()


if __name__ == "__main__":
    raise SystemExit(main())
