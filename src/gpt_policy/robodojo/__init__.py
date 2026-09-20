"""RoboDojo-to-GPT-Policy protocol adapters.

The adapter deliberately keeps simulator-native data alongside the normalized
GPT-Policy view. This makes the model-facing observation equivalent to the
real-robot observation while preserving enough information to diagnose frame,
camera, and action-conversion mistakes.
"""

from .calibration import RoboDojoCalibration
from .adapter import RoboDojoAdapter

__all__ = ["RoboDojoAdapter", "RoboDojoCalibration"]
