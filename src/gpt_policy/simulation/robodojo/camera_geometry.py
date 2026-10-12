"""Camera-only calibration, captured with the cached RGB observation.

Never reads object poses, depth, segmentation, rewards or task geometry.
"""

import numpy as np

from .transforms import transform


def array(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=float)


def capture_calibration(env, camera_sources, images):
    # This pinned RoboDojo evaluation uses CPU physics with USD pose updates.
    # A Fabric-only run needs a verified Fabric pose reader before enabling this.
    if getattr(env, "use_fabric", False):
        raise ValueError(
            "locate_point optical pose capture requires USD pose updates (use_fabric=false)"
        )
    manager = env.camera_manager
    origin = array(env.robot_manager.scene.env_origins[0])
    result = {}
    for name, source in camera_sources.items():
        index = manager.camera_names[0].index(source)
        camera = manager.cameras[0][index]
        if camera.get_lens_distortion_model() != "pinhole":
            raise ValueError(
                "locate_point currently requires a calibrated pinhole camera"
            )
        # Use the actual optical Camera, not its parent mount or a template K.
        # fx/fy can differ. ROS optical axes are +X right, +Y down, +Z forward.
        intrinsic = array(camera.get_intrinsics_matrix())
        position, orientation = camera.get_world_pose(camera_axes="ros")
        camera_to_environment = transform(array(position) - origin, array(orientation))
        height, width = images[name].shape[:2]
        if tuple(camera.get_resolution()) != (width, height):
            raise ValueError("RGB/calibration resolution mismatch")
        if intrinsic.shape != (3, 3) or not np.isfinite(intrinsic).all():
            raise ValueError("Invalid camera intrinsics")
        result[name] = {
            "intrinsic": intrinsic.tolist(),
            "resolution": [width, height],
            "camera_to_environment": camera_to_environment.tolist(),
            "distortion_model": "pinhole",
            "frame": "environment_origin",
            "optical_axes": "x_right_y_down_z_forward",
            "source": "Isaac Camera optical pose and intrinsics at the RGB observation; camera data only",
        }
    return result
