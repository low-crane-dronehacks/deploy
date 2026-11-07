"""Handheld / data-collection helpers. PURE: numpy only, no ROS (tested in tests/test_handheld.py).

The handheld mode runs the camera + the policy while a person carries the drone (or just the camera) through
the course. It NEVER talks to PX4's inputs. State can come from:
  px4   a Pixhawk on USB/UART (props off): PX4's own estimate, exactly what the flight node uses
  odom  a nav_msgs/Odometry pose (cuVSLAM's /visual_slam/tracking/odometry: ROS frames, odom ENU-ish, body FLU)
        -> NED / FRD, velocity by differencing the pose (the message carries no twist)
  none  depth only: zero velocity, level attitude (shows what the policy makes of the depth alone)
"""
from __future__ import annotations

import math

import numpy as np

# ROS (ENU world, FLU body) -> PX4 (NED world, FRD body): R_ned_frd = A @ R_enu_flu @ B
_A = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])
_B = np.diag([1.0, -1.0, -1.0])


def quat_xyzw_to_mat(x, y, z, w) -> np.ndarray:
    n = math.sqrt(x * x + y * y + z * z + w * w) or 1.0
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def mat_to_quat_wxyz(R: np.ndarray):
    """Rotation matrix -> (w, x, y, z), w >= 0."""
    t = np.trace(R)
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        q = (0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s)
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        q = ((R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s)
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        q = ((R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s)
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        q = ((R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s)
    q = np.asarray(q, float)
    return tuple(float(c) for c in (q if q[0] >= 0 else -q))


def ros_pose_to_ned(pos_enu, q_xyzw_flu):
    """ROS odometry pose (ENU position, FLU-body quaternion x,y,z,w) -> (NED position, FRD->NED quaternion w,x,y,z)."""
    R = _A @ quat_xyzw_to_mat(*q_xyzw_flu) @ _B
    p = _A @ np.asarray(pos_enu, float)
    return tuple(float(c) for c in p), mat_to_quat_wxyz(R)


class VelocityEstimator:
    """Velocity from timestamped positions: finite difference + first-order low-pass (tau_s). Resets on a gap."""

    def __init__(self, tau_s: float = 0.15, max_gap_s: float = 0.5):
        self.tau, self.max_gap = tau_s, max_gap_s
        self.t = self.p = None
        self.v = np.zeros(3)

    def update(self, t: float, pos) -> np.ndarray:
        pos = np.asarray(pos, float)
        if self.t is not None and 1e-4 < t - self.t <= self.max_gap:
            raw = (pos - self.p) / (t - self.t)
            a = (t - self.t) / (self.tau + (t - self.t))
            self.v = self.v + a * (raw - self.v)
        elif self.t is not None and t - self.t > self.max_gap:
            self.v = np.zeros(3)
        self.t, self.p = t, pos
        return self.v.copy()


def _quat_wxyz_to_mat(w, x, y, z):
    return quat_xyzw_to_mat(x, y, z, w)


def ned_to_enu_twist(vel_ned, yawspeed_ned):
    """NED velocity + NED yaw rate (clockwise from above, positive) -> ENU velocity + ENU yaw rate (CCW positive)."""
    return (float(vel_ned[1]), float(vel_ned[0]), float(-vel_ned[2])), float(-yawspeed_ned)


def cap_norm(v, vmax):
    n = math.sqrt(sum(c * c for c in v))
    return tuple(c * vmax / n for c in v) if n > vmax else tuple(v)


def body_flu_to_ned(v_flu, q_ned_frd_wxyz):
    """Body FLU velocity (cuVSLAM twist) -> NED world velocity."""
    R = _quat_wxyz_to_mat(*q_ned_frd_wxyz)
    return tuple(float(c) for c in R @ np.array([v_flu[0], -v_flu[1], -v_flu[2]], float))
