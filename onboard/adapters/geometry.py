"""Frame conversions for the adapters. PURE (numpy)."""
from __future__ import annotations

import math

import numpy as np

# optical (x right, y down, z forward) -> camera rig (x forward, y left, z up): the columns are
# the optical axes expressed in the rig frame
R_RIG_OPTICAL = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
R_ENU_NED = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])
R_FRD_FLU = np.diag([1.0, -1.0, -1.0])


def rot_x(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], float)


def rot_y(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], float)


def rot_z(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], float)


def mount_rotation(pitch_deg, yaw_deg, roll_deg):
    """The simulator's camera mount (generator/sim_cameras.cam_mount_rotation): yaw about +Z,
    then pitch about +Y (positive = nose down), then roll about the look axis; body FLU."""
    return rot_z(math.radians(yaw_deg)) @ rot_y(math.radians(pitch_deg)) @ rot_x(math.radians(roll_deg))


def camera_optical_pose(cam, eye="center"):
    """(R_body_optical, t_body) of a configured camera: eye = center | left | right. Mirrors
    generator/sim_cameras.stereo_offset (the eyes sit on the rig's local left axis)."""
    pitch, yaw, roll = (list(cam.get("orientation_deg", [0.0, 0.0, 0.0])) + [0.0, 0.0, 0.0])[:3]
    Rm = mount_rotation(pitch, yaw, roll)
    t = np.array(cam["position_m"], float)
    if eye in ("left", "right"):
        half = float(cam.get("baseline_mm", 0.0)) / 2000.0
        t = t + (1.0 if eye == "left" else -1.0) * half * (Rm @ np.array([0.0, 1.0, 0.0]))
    elif eye != "center":
        raise ValueError(f"eye must be center | left | right, not {eye!r}")
    return Rm @ R_RIG_OPTICAL, t


def mat_to_quat(R):
    """Rotation matrix -> (x, y, z, w)."""
    R = np.asarray(R, float)
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        w, x, y, z = 0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w, x, y, z = (R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w, x, y, z = (R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w, x, y, z = (R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s
    n = math.sqrt(w * w + x * x + y * y + z * z)
    return (x / n, y / n, z / n, w / n)


def quat_to_mat(w, x, y, z):
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def px4_to_ros(pos_ned, q_wxyz_frd_to_ned, vel_ned=None):
    """PX4 local NED position + FRD->NED attitude -> ROS ENU position + FLU->ENU (x, y, z, w)."""
    p = R_ENU_NED @ np.asarray(pos_ned, float)
    R = R_ENU_NED @ quat_to_mat(*q_wxyz_frd_to_ned) @ R_FRD_FLU
    v = None if vel_ned is None else R_ENU_NED @ np.asarray(vel_ned, float)
    return p, mat_to_quat(R), v
