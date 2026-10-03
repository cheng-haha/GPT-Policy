"""Hardware-free configuration validation for every deployment target."""

from pathlib import Path

import numpy as np

from .geometry.frames import FrameCalibration
from .harness.config import agent_config
from .settings import runtime_config
from .tools.catalog import load_tool_catalog
from .vision.perception import PixelLocalizer


def check_configuration(settings, path, *, resolved_agent=None):
    runtime = runtime_config(settings, base_dir=path.parent)
    backend = settings.get("backend", "arx")
    if backend not in {"arx", "yam"}:
        raise ValueError(f"Unknown backend: {backend}")
    if runtime.right_interface == runtime.interface:
        raise ValueError("Two arms cannot share one CAN interface")
    if not np.isfinite(runtime.trajectory_hz):
        raise ValueError("trajectory_hz must be finite")
    frames = FrameCalibration(settings)
    localizer = PixelLocalizer(settings)
    agent = resolved_agent if resolved_agent is not None else agent_config(settings, path.parent)
    arms = ("left", "right") if runtime.right_interface else ("left",)
    catalog = load_tool_catalog(settings)
    catalog.function_schemas(6, arms)
    cameras = settings.get("cameras", {})
    for name, camera in cameras.items():
        if name not in localizer.intrinsics:
            raise ValueError(f"Camera {name} has no intrinsics")
        if settings.get("camera_backend") == "realsense" and not camera.get("serial"):
            raise ValueError(f"Camera {name} has no RealSense serial")
    transforms = [frames.sdk_eef_from_tcp, *frames.tcp_from_camera.values()]
    transforms.extend(t for group in frames.fixed_camera_transforms.values() for t in group.values())
    for matrix in transforms:
        if not np.allclose(matrix[3], [0, 0, 0, 1]) or not np.allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=1e-5):
            raise ValueError("Calibration is not a rigid transform")
    return {"ok": True, "hardware_opened": False, "machine": settings.get("machine"),
            "backend": backend, "model": runtime.robot_model, "config": str(path),
            "interfaces": [runtime.interface, runtime.right_interface], "cameras": cameras,
            "agent": agent.type, "agent_profile": settings.get("agent"), "agent_model": agent.model,
            **({"live_image_window": agent.live_image_window,
                "live_window_enabled": agent.live_image_window is not None} if agent.type == "codex" else {}),
            "top_camera_bases": list(frames.fixed_camera_arms("top")),
            "max_decisions": runtime.max_decisions,
            "recording": settings.get("recording", {})}
