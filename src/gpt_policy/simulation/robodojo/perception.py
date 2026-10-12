"""Native camera adapter around GPT-Policy's existing triangulation math."""

import numpy as np

from ...vision.perception import _triangulate


def ray(reference, history):
    record = history.get(reference["step"])
    if record is None:
        raise ValueError("Unknown recorded observation step")
    camera = record["camera_calibration"].get(reference["camera"])
    if camera is None:
        raise ValueError("Unknown camera")
    pixel = np.asarray(reference["pixel_xy"], float)
    width, height = camera["resolution"]
    if (
        pixel.shape != (2,)
        or not np.isfinite(pixel).all()
        or not (0 <= pixel[0] < width and 0 <= pixel[1] < height)
    ):
        raise ValueError("Pixel must be within the full-resolution RGB")
    matrix = np.asarray(camera["intrinsic"])
    transform = np.asarray(camera["camera_to_environment"])
    direction = transform[:3, :3] @ np.linalg.solve(matrix, np.r_[pixel, 1])
    return transform[:3, 3], direction / np.linalg.norm(direction)


def locate(arguments, history):
    origin, direction = ray(arguments["first"], history)
    result = {
        "frame": "environment_origin",
        "metric_position_available": False,
        "ray_origin": origin.tolist(),
        "ray_direction": direction.tolist(),
    }
    if arguments["second"] is None:
        return result
    point, residual, quality = _triangulate(
        origin, direction, *ray(arguments["second"], history)
    )
    reasons = []
    if quality["parallax_angle_deg"] < 5:
        reasons.append("parallax_too_small")
    if quality["condition_number"] > 500:
        reasons.append("ill_conditioned")
    if residual > 0.01:
        reasons.append("ray_residual_too_large")
    if any(x <= 0 for x in quality["ray_depths_m"]):
        reasons.append("intersection_behind_camera")
    result.update(quality=quality, residual_m=residual, rejection_reasons=reasons)
    if not reasons:
        result.update(metric_position_available=True, position=point.tolist())
    return result
