from __future__ import annotations

import unittest

import numpy as np

from gpt_policy.geometry.poses import rotation_distance
from gpt_policy.motion.trajectory import (
    MotionLimits,
    derivatives,
    retime_path_segment,
    sample_count_for_segment,
    sample_pose_segment,
)


class PathTimingTest(unittest.TestCase):
    def setUp(self) -> None:
        self.limits = MotionLimits(
            output_hz=100.0,
            cartesian_step_m=0.005,
            cartesian_step_rad=0.035,
            tcp_velocity_m_s=0.08,
            tcp_angular_velocity_rad_s=0.5,
            joint_velocity_rad_s=np.full(6, 1.0),
            joint_acceleration_rad_s2=np.full(6, 2.0),
            joint_jerk_rad_s3=np.full(6, 12.0),
        )

    def test_sample_density_is_geometric_and_temporal(self) -> None:
        self.assertEqual(sample_count_for_segment(0.08, 0.0, self.limits), 100)
        self.assertEqual(sample_count_for_segment(0.0, 0.7, self.limits), 140)

    def test_retiming_only_changes_time(self) -> None:
        fractions = np.linspace(0.0, 1.0, 101)
        joints = np.stack(
            [
                fractions,
                fractions**2,
                np.sin(fractions),
                np.zeros_like(fractions),
                np.zeros_like(fractions),
                np.zeros_like(fractions),
            ],
            axis=1,
        )
        unchanged = joints.copy()
        result = retime_path_segment(fractions, joints, 0.08, 0.2, self.limits)

        np.testing.assert_array_equal(joints, unchanged)
        self.assertEqual(result.times_s[0], 0.0)
        self.assertGreater(result.duration_s, 0.0)
        self.assertTrue(np.all(np.diff(result.times_s) > 0.0))

        velocity, acceleration, jerk = derivatives(joints, result.times_s)
        self.assertLessEqual(float(np.max(np.abs(velocity))), 1.0 + 1e-9)
        self.assertLessEqual(float(np.max(np.abs(acceleration))), 2.0 + 1e-9)
        self.assertLessEqual(float(np.max(np.abs(jerk))), 12.0 + 1e-9)

    def test_pose_sampling_preserves_line_and_exact_endpoints(self) -> None:
        start = np.array([0.2, -0.1, 0.3, 0.1, -0.2, 0.3])
        end = np.array([0.28, 0.04, 0.22, -0.3, 0.25, -0.15])
        fractions, poses = sample_pose_segment(start, end, self.limits)

        np.testing.assert_array_equal(poses[0], start)
        np.testing.assert_array_equal(poses[-1], end)
        expected_positions = start[:3] + fractions[:, None] * (end[:3] - start[:3])
        np.testing.assert_allclose(poses[:, :3], expected_positions, atol=1e-14)
        self.assertLess(rotation_distance(poses[-1, 3:], end[3:]), 1e-7)

    def test_high_frequency_noise_stretches_instead_of_rejecting(self) -> None:
        fractions = np.linspace(0.0, 1.0, 101)
        noisy = fractions + 0.01 * np.sin(40.0 * np.pi * fractions)
        joints = np.zeros((fractions.size, 6))
        joints[:, 0] = noisy
        baseline = np.zeros_like(joints)
        baseline[:, 0] = fractions
        baseline_result = retime_path_segment(
            fractions, baseline, 0.08, 0.0, self.limits
        )
        result = retime_path_segment(fractions, joints, 0.08, 0.0, self.limits)

        self.assertGreater(result.duration_s, baseline_result.duration_s)
        self.assertLessEqual(float(np.max(result.peak_jerk_rad_s3)), 12.0 + 1e-8)


if __name__ == "__main__":
    unittest.main()
