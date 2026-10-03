"""The insert-chapai step 68 command lost 35 degrees during quaternion conversion."""
import numpy as np
import pytest

from gpt_policy.geometry.poses import (
    quaternion_to_rpy, rpy_to_quaternion, rpy_to_matrix, matrix_to_rpy, interpolate_pose,
)


def same_rotation(a, b, atol=1e-8):
    a, b = np.asarray(a), np.asarray(b)
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    assert min(np.linalg.norm(a - b), np.linalg.norm(a + b)) < atol


def test_recorded_insert_step_68_keeps_the_requested_orientation():
    requested = [.212631, .67438, -.212631, .67438]
    same_rotation(requested, rpy_to_quaternion(quaternion_to_rpy(requested)))


@pytest.mark.parametrize("pitch", [np.pi / 2, -np.pi / 2, np.pi / 2 - 1e-7, -np.pi / 2 + 1e-7])
@pytest.mark.parametrize("roll,yaw", [(.6, -.4), (-2.1, 2.7), (0., .61)])
def test_rotation_round_trip_at_and_near_both_poles(pitch, roll, yaw):
    original = rpy_to_quaternion([roll, pitch, yaw])
    for sign in (1, -1):
        same_rotation(original, rpy_to_quaternion(quaternion_to_rpy(sign * np.array(original))))
    np.testing.assert_allclose(rpy_to_matrix(matrix_to_rpy(rpy_to_matrix([roll, pitch, yaw]))),
                               rpy_to_matrix([roll, pitch, yaw]), atol=1e-8)


def test_interpolation_through_vertical_pitch_keeps_roll_yaw_relationship():
    start = np.r_[.3, .1, .1, .3, np.pi / 2 - .1, -.31]
    end = np.r_[.3, .1, .1, .3, np.pi / 2 + .1, -.31]
    for fraction in np.linspace(0, 1, 21):
        pose = interpolate_pose(start, end, fraction)
        expected = rpy_to_quaternion([.3, np.pi / 2 - .1 + .2 * fraction, -.31])
        same_rotation(expected, rpy_to_quaternion(pose[3:]))
