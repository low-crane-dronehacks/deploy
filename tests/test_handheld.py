"""Pure tests for the handheld mode's frame conversion and velocity estimate."""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from policynav import handheld as H, obs as O                                  # noqa: E402


def _yaw_q_xyzw(yaw):
    return (0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2))


def test_identity_ros_pose_is_north_facing_east_in_ned():
    # ROS: body x (forward) along ENU x = EAST, level  ->  NED yaw +90 deg, level
    p, q = H.ros_pose_to_ned((1.0, 2.0, 3.0), (0.0, 0.0, 0.0, 1.0))
    assert np.allclose(p, (2.0, 1.0, -3.0))
    roll, pitch, yaw = O.rpy_from_R(O.quat_to_mat(*q))
    assert abs(roll) < 1e-9 and abs(pitch) < 1e-9 and abs(yaw - math.pi / 2) < 1e-9


def test_ros_yaw_left_is_ned_yaw_right_handed():
    # ROS yaw +90 (body x -> ENU north) is NED yaw 0 (facing north)
    _, q = H.ros_pose_to_ned((0, 0, 0), _yaw_q_xyzw(math.pi / 2))
    _, _, yaw = O.rpy_from_R(O.quat_to_mat(*q))
    assert abs(yaw) < 1e-9


def test_nose_down_pitch_maps_to_ned_pitch_down():
    # ROS FLU pitch +0.2 about body y (left) = nose DOWN; NED/FRD pitch is nose-UP positive -> -0.2
    s, c = math.sin(0.1), math.cos(0.1)
    _, q = H.ros_pose_to_ned((0, 0, 0), (0.0, s, 0.0, c))
    _, pitch, _ = O.rpy_from_R(O.quat_to_mat(*q))
    assert abs(pitch + 0.2) < 1e-9


def test_quat_round_trip():
    rng = np.random.default_rng(0)
    for _ in range(20):
        q = rng.normal(size=4); q /= np.linalg.norm(q)
        R = H.quat_xyzw_to_mat(*q)
        w, x, y, z = H.mat_to_quat_wxyz(R)
        assert np.allclose(O.quat_to_mat(w, x, y, z), R, atol=1e-9)


def test_velocity_estimator_converges_and_resets_on_gap():
    ve = H.VelocityEstimator(tau_s=0.1)
    for k in range(60):
        v = ve.update(0.05 * k, (1.0 * 0.05 * k, 0.0, 0.0))
    assert abs(v[0] - 1.0) < 1e-3 and abs(v[1]) < 1e-9
    v = ve.update(10.0, (5.0, 0.0, 0.0))                 # 7 s gap: no velocity spike
    assert np.allclose(v, 0.0)
