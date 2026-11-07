"""Frame math for feeding PX4 external-vision odometry from ROS sources. PURE.

ROS (REP-103/105): world ENU, or an arbitrary-heading x-forward / y-left /
z-up map such as cuVSLAM's start frame; body FLU; Odometry twist in the
child (body) frame by default.
PX4 VehicleOdometry: world NED (pose_frame 1) or FRD-world (pose_frame 2:
x-forward at init, constant unknown heading offset from north); body FRD.

Quaternions are (w, x, y, z) in this module. ROS messages carry (x, y, z, w);
convert at the edge with quat_xyzw_to_wxyz(). Attitude quaternions rotate
BODY vectors into the WORLD frame, in both conventions.

No ROS imports: this runs and is unit-tested on any Python.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

NAN = float("nan")

POSE_FRAME_NED = 1
POSE_FRAME_FRD = 2
VELOCITY_FRAME_NED = 1
VELOCITY_FRAME_FRD = 2
VELOCITY_FRAME_BODY_FRD = 3

WORLD_ENU = "enu"   # north-aligned ROS world
WORLD_FLU = "flu"   # arbitrary-heading x-forward/y-left/z-up map (cuVSLAM, most VIO)
WORLDS = (WORLD_ENU, WORLD_FLU)

_S = math.sqrt(0.5)
# rotation taking ENU-expressed vectors to NED-expressed vectors (180 deg about (1,1,0)/sqrt2)
Q_ENU_TO_NED = (0.0, _S, _S, 0.0)
# 180 deg about x: FLU<->FRD for a body, and x-fwd/y-left/z-up <-> FRD for a world
Q_X180 = (0.0, 1.0, 0.0, 0.0)


# ---------------------------------------------------------------- quaternions
def qmul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return (w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2)


def qconj(q):
    w, x, y, z = q
    return (w, -x, -y, -z)


def qnormalize(q):
    """Unit quaternion with w >= 0; (nan, nan, nan, nan) when not finite / zero."""
    if not all(math.isfinite(c) for c in q):
        return (NAN, NAN, NAN, NAN)
    n = math.sqrt(sum(c * c for c in q))
    if n < 1e-12:
        return (NAN, NAN, NAN, NAN)
    w, x, y, z = (c / n for c in q)
    if w < 0.0:
        w, x, y, z = -w, -x, -y, -z
    return (w, x, y, z)


def qrotate(q, v):
    """Rotate vector v by unit quaternion q (q * v * q^-1)."""
    w, x, y, z = q
    vx, vy, vz = v
    # t = 2 * cross(q.xyz, v)
    tx = 2.0 * (y * vz - z * vy)
    ty = 2.0 * (z * vx - x * vz)
    tz = 2.0 * (x * vy - y * vx)
    # v' = v + w*t + cross(q.xyz, t)
    return (vx + w * tx + (y * tz - z * ty),
            vy + w * ty + (z * tx - x * tz),
            vz + w * tz + (x * ty - y * tx))


def quat_xyzw_to_wxyz(x, y, z, w):
    return (float(w), float(x), float(y), float(z))


def quat_from_yaw(yaw):
    """Rotation about +z by yaw (radians)."""
    return (math.cos(0.5 * yaw), 0.0, 0.0, math.sin(0.5 * yaw))


def yaw_of(q):
    """ZYX yaw (rad) of a (w,x,y,z) quaternion."""
    w, x, y, z = q
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def tilt_of(q):
    """Angle (rad) between the body z axis and the world z axis."""
    bz = qrotate(q, (0.0, 0.0, 1.0))
    c = max(-1.0, min(1.0, bz[2]))
    return math.acos(c)


# ------------------------------------------------------------ frame mapping
def _check_world(world):
    if world not in WORLDS:
        raise ValueError(f"world must be one of {WORLDS}, got {world!r}")


def world_vec(v, world):
    """World-frame vector from ROS convention to the matching PX4 world frame.
    enu -> NED: (n, e, d) = (y, x, -z).  flu-map -> FRD-world: (x, -y, -z)."""
    _check_world(world)
    x, y, z = (float(c) for c in v)
    if world == WORLD_ENU:
        return (y, x, -z)
    return (x, -y, -z)


def body_vec(v):
    """Body vector FLU -> FRD."""
    x, y, z = (float(c) for c in v)
    return (x, -y, -z)


def pose_frame_for(world):
    _check_world(world)
    return POSE_FRAME_NED if world == WORLD_ENU else POSE_FRAME_FRD


def velocity_frame_for(world, in_body):
    if in_body:
        return VELOCITY_FRAME_BODY_FRD
    _check_world(world)
    return VELOCITY_FRAME_NED if world == WORLD_ENU else VELOCITY_FRAME_FRD


def attitude_to_px4(q_wxyz, world):
    """Body-FLU->ROS-world attitude  ->  body-FRD->PX4-world attitude.
    q_px4 = q_W (x) q_ros (x) q_B with q_B: FRD->FLU (x180), q_W: ROS world -> PX4 world."""
    _check_world(world)
    q = qnormalize(tuple(float(c) for c in q_wxyz))
    if not math.isfinite(q[0]):
        return (NAN, NAN, NAN, NAN)
    q_w = Q_ENU_TO_NED if world == WORLD_ENU else Q_X180
    return qnormalize(qmul(qmul(q_w, q), Q_X180))


def attitude_from_px4(q_px4_wxyz, world):
    """Inverse of attitude_to_px4 (for estimated-state consumers and round-trip tests)."""
    _check_world(world)
    q_w = Q_ENU_TO_NED if world == WORLD_ENU else Q_X180
    q = qnormalize(tuple(float(c) for c in q_px4_wxyz))
    if not math.isfinite(q[0]):
        return (NAN, NAN, NAN, NAN)
    return qnormalize(qmul(qmul(qconj(q_w), q), qconj(Q_X180)))


# --------------------------------------------------------------- covariance
def _diag(cov, i):
    """Diagonal element i of a row-major 6x6 covariance; 0.0 when absent."""
    if cov is None:
        return 0.0
    try:
        v = float(cov[i * 6 + i])
    except (IndexError, TypeError, ValueError):
        return 0.0
    return v if math.isfinite(v) and v >= 0.0 else 0.0


def position_variance(pose_cov, world):
    """Diagonal of the ROS pose covariance mapped to the PX4 world axis order."""
    _check_world(world)
    vx, vy, vz = _diag(pose_cov, 0), _diag(pose_cov, 1), _diag(pose_cov, 2)
    return (vy, vx, vz) if world == WORLD_ENU else (vx, vy, vz)


def orientation_variance(pose_cov):
    """Roll/pitch/yaw variances; axis sign flips do not change a variance."""
    return (_diag(pose_cov, 3), _diag(pose_cov, 4), _diag(pose_cov, 5))


def velocity_variance(twist_cov, world, in_body):
    vx, vy, vz = _diag(twist_cov, 0), _diag(twist_cov, 1), _diag(twist_cov, 2)
    if in_body or world == WORLD_FLU:
        return (vx, vy, vz)
    return (vy, vx, vz)


def apply_floor(var, floor):
    return tuple(max(float(a), float(b)) for a, b in zip(var, floor))


# -------------------------------------------------------------- timestamps
def stamp_us(sec, nanosec):
    """ROS builtin_interfaces/Time -> integer microseconds."""
    return int(sec) * 1_000_000 + int(nanosec) // 1000


def finite3(v):
    return v is not None and len(v) == 3 and all(math.isfinite(float(c)) for c in v)


# ------------------------------------------------------------ the message
@dataclass
class PX4Odometry:
    """Field-for-field image of px4_msgs/VehicleOdometry (no ROS types)."""
    timestamp: int
    timestamp_sample: int
    pose_frame: int
    position: tuple
    q: tuple
    velocity_frame: int
    velocity: tuple
    angular_velocity: tuple
    position_variance: tuple
    orientation_variance: tuple
    velocity_variance: tuple
    reset_counter: int = 0
    quality: int = 0

    def as_dict(self):
        return {k: (list(v) if isinstance(v, tuple) else v) for k, v in self.__dict__.items()}


def align_to_enu(position, q_wxyz, velocity, pose_cov, twist_cov, world, in_body, heading):
    """Apply a declared heading calibration; never infer north from simulator pose."""
    if world != WORLD_FLU or not math.isfinite(heading):
        raise ValueError("a finite heading reference applies only to a startup-local FLU frame")
    c, s = math.cos(heading), math.sin(heading)

    def vector(v):
        if v is None:
            return None
        x, y, z = v
        return (c*x-s*y, s*x+c*y, z)

    def covariance(values, *, euler_pose=False):
        if values is None:
            return None
        if len(values) != 36:
            raise ValueError("pose/twist covariance needs 36 elements")
        # Euler pose: fixed world yaw adds to yaw; roll/pitch coordinates
        # remain unchanged. Twist angular rates are vectors and DO rotate.
        # Preserve full cross terms before selecting PX4's diagonal.
        r = ((c,-s,0.), (s,c,0.), (0.,0.,1.))
        a = [[(float(i == j) if euler_pose and i >= 3 else r[i%3][j%3])
              if i//3 == j//3 else 0. for j in range(6)] for i in range(6)]
        return [sum(a[i][k]*values[k*6+l]*a[j][l] for k in range(6) for l in range(6)
                    if a[i][k] and a[j][l]) for i in range(6) for j in range(6)]

    yaw = (math.cos(heading/2), 0., 0., math.sin(heading/2))
    return (vector(position), qmul(yaw, q_wxyz), velocity if in_body else vector(velocity),
            covariance(pose_cov, euler_pose=True), twist_cov if in_body else covariance(twist_cov))


def odometry_to_px4(*, stamp_sample_us, now_us, position, q_wxyz, world,
                    velocity=None, velocity_in_body=True, angular_velocity=None,
                    pose_cov=None, twist_cov=None, reset_counter=0, quality=0,
                    position_floor=(0.0, 0.0, 0.0), orientation_floor=(0.0, 0.0, 0.0),
                    velocity_floor=(0.0, 0.0, 0.0), heading_enu_rad=None):
    """Convert one ROS-convention odometry sample into a PX4Odometry.

    position / q_wxyz: ROS world pose (world = 'enu' or 'flu').
    velocity: body FLU linear velocity (velocity_in_body=True, the REP-105
      Odometry default) or ROS-world velocity (False). None -> NaN (unknown).
    angular_velocity: body FLU rates. None -> NaN.
    pose_cov / twist_cov: row-major 6x6 or None. Missing -> 0 (PX4 then uses
      its EKF2_EV*_NOISE lower bounds); floors are applied after mapping.
    heading_enu_rad: surveyed heading of the source origin in ENU. An unknown
      startup heading remains FRD; a finite reference rotates pose and covariance.
    """
    _check_world(world)
    if heading_enu_rad is not None:
        position, q_wxyz, velocity, pose_cov, twist_cov = align_to_enu(
            position, q_wxyz, velocity, pose_cov, twist_cov, world, velocity_in_body, heading_enu_rad)
        world = WORLD_ENU
    pos = world_vec(position, world) if finite3(position) else (NAN, NAN, NAN)
    q = attitude_to_px4(q_wxyz, world)
    if finite3(velocity):
        vel = body_vec(velocity) if velocity_in_body else world_vec(velocity, world)
        vel_frame = velocity_frame_for(world, velocity_in_body)
    else:
        vel, vel_frame = (NAN, NAN, NAN), velocity_frame_for(world, velocity_in_body)
    ang = body_vec(angular_velocity) if finite3(angular_velocity) else (NAN, NAN, NAN)
    return PX4Odometry(
        timestamp=int(now_us),
        timestamp_sample=int(stamp_sample_us),
        pose_frame=pose_frame_for(world),
        position=pos,
        q=q,
        velocity_frame=vel_frame,
        velocity=vel,
        angular_velocity=ang,
        position_variance=apply_floor(position_variance(pose_cov, world), position_floor),
        orientation_variance=apply_floor(orientation_variance(pose_cov), orientation_floor),
        velocity_variance=apply_floor(velocity_variance(twist_cov, world, velocity_in_body),
                                      velocity_floor),
        reset_counter=int(reset_counter) & 0xFF,
        quality=int(quality),
    )


def initial_reset_counter(pid=None, now_s=None):
    """A non-zero initial reset generation for a newly started vision source."""
    import os
    import time
    pid = os.getpid() if pid is None else pid
    now_s = time.time() if now_s is None else now_s
    return 1 + (int(pid) * 7 + int(now_s)) % 254
