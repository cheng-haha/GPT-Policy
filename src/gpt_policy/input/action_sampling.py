"""Offline sampling of historical feedback and commands, never motion planning."""

from __future__ import annotations

from collections import deque
import math

import numpy as np


JOINTS = ("joint_target_rad", "joint_measured_rad")
POSES = ("eef_target_xyz_xyzw", "eef_measured_xyz_xyzw")
GRIPPERS = ("gripper_command", "gripper_measured", "gripper_reference")


def _band(value):
    return 0 if value <= .1 else 2 if value >= .9 else 1


def gripper_events(samples, fields=GRIPPERS, threshold=.08):
    """Both sides of transitions, including gradual closures and reopenings."""
    indices = set()
    for side in ("left", "right"):
        for field in fields:
            anchor = None
            previous = None
            for i, sample in enumerate(samples):
                value = sample.get(side, {}).get(field)
                if value is None:
                    anchor = previous = None
                    continue
                if not math.isfinite(value):
                    raise ValueError("Non-finite gripper sample")
                if previous is not None and (_band(value) != _band(previous) or abs(value - anchor) >= threshold):
                    indices.update((i - 1, i))
                    anchor = value
                if anchor is None:
                    anchor = value
                previous = value
    return indices


def event_times(samples):
    """Gripper transitions and moving→still encoder windows; ignore motor dq."""
    indices = gripper_events(samples)
    for side in ("left", "right"):
        window = deque()
        moving = False
        for i, sample in enumerate(samples):
            q = sample.get(side, {}).get("joint_measured_rad")
            if q is None:
                continue
            t = sample["t_s"]
            if window and t - window[-1][0] > .15:
                window.clear()
                moving = False
            window.append((t, q))
            while len(window) > 2 and window[1][0] <= t - .3:
                window.popleft()
            if t - window[0][0] < .3:
                continue
            still = np.max(np.ptp(np.asarray([x[1] for x in window]), axis=0)) <= .002
            if moving and still:
                indices.add(i)
            moving = not still
    return [samples[i]["t_s"] for i in sorted(indices)]


def compress_samples(samples, *, preserve_indices=()):
    """GPT-Policy-style context sampling: endpoints, 1 Hz and gripper commands.

    This is a sparse historical example, not an error-bounded control trajectory.
    Full input samples are saved separately before sampling.
    """
    count = len(samples)
    if any(type(i) is not int or not 0 <= i < count for i in preserve_indices):
        raise ValueError("Preserved sample indices must be integers within this segment")
    preserved = set(preserve_indices)
    times = [s["t_s"] for s in samples]
    if any(not math.isfinite(t) for t in times) or any(b < a for a, b in zip(times, times[1:])):
        raise ValueError("Action video timestamps must be finite and nondecreasing")
    time_field = "t_s"
    if any(b == a for a, b in zip(times, times[1:])):
        # Legacy controls can share one camera frame. Use the independent
        # recording clock for sampling, without altering either timestamp.
        times = [s.get("recording_t_s") for s in samples]
        if (any(isinstance(t, bool) or not isinstance(t, (int, float)) or not math.isfinite(t) for t in times)
                or any(b <= a for a, b in zip(times, times[1:]))):
            raise ValueError("Repeated video timestamps require finite, strictly increasing recording_t_s")
        time_field = "recording_t_s"
    info = {"method": "periodic_and_gripper_events", "sample_period_s": 1.0,
            "time_field": time_field, "input_samples": count}
    if preserved:
        info.update(method="periodic_and_gripper_events_with_reviewed_anchors",
                    preserved_indices=sorted(preserved))
    if count < 3:
        return samples, {**info, "output_samples": count}
    keep = preserved | {0, count - 1} | gripper_events(samples, ("gripper_command", "gripper_reference"), .1)
    anchor = 0
    previous_channels = None
    for i, sample in enumerate(samples):
        # Preserve loss/recovery boundaries, never fill missing feedback with zeros.
        channels = {(side, key) for side in ("left", "right")
                    for key, value in sample.get(side, {}).items() if value is not None}
        if previous_channels is not None and channels != previous_channels:
            keep.update((i - 1, i))
        previous_channels = channels
        if times[i] - times[anchor] >= 1.0:
            keep.add(i)
            anchor = i
    return [samples[i] for i in sorted(keep)], {**info, "output_samples": len(keep)}
