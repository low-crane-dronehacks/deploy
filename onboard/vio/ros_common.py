"""Shared ROS 2 plumbing for the onboard nodes (QoS that matches PX4's
uXRCE-DDS endpoints, message filling, clock). Everything that can be pure is
NOT here — see frames.py / validation.py."""
from __future__ import annotations

import math

from rclpy.qos import (DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy,
                       qos_profile_sensor_data)

# QoS compatible with the pinned native PX4 interface. Validate compatibility
# against the installed endpoints when changing firmware or middleware.
PX4_QOS = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                     durability=DurabilityPolicy.TRANSIENT_LOCAL,
                     history=HistoryPolicy.KEEP_LAST, depth=1)

# For simulator / VIO topics: a BEST_EFFORT subscriber matches reliable and
# best-effort publishers alike.
SENSOR_QOS = qos_profile_sensor_data


def now_us(node):
    """Node clock in integer microseconds (honours use_sim_time)."""
    return node.get_clock().now().nanoseconds // 1000


def stamp_of(header):
    from .frames import stamp_us
    return stamp_us(header.stamp.sec, header.stamp.nanosec)


def _f3(v):
    return [float(c) for c in v]


def fill_vehicle_odometry(msg, odo):
    """Copy a frames.PX4Odometry into a px4_msgs/VehicleOdometry message."""
    msg.timestamp = int(odo.timestamp)
    msg.timestamp_sample = int(odo.timestamp_sample)
    msg.pose_frame = int(odo.pose_frame)
    msg.position = _f3(odo.position)
    msg.q = [float(c) for c in odo.q]
    msg.velocity_frame = int(odo.velocity_frame)
    msg.velocity = _f3(odo.velocity)
    msg.angular_velocity = _f3(odo.angular_velocity)
    msg.position_variance = _f3(odo.position_variance)
    msg.orientation_variance = _f3(odo.orientation_variance)
    msg.velocity_variance = _f3(odo.velocity_variance)
    msg.reset_counter = int(odo.reset_counter) & 0xFF
    msg.quality = int(max(-128, min(127, odo.quality)))
    return msg


def declare(node, name, default):
    """Declare + read a parameter. Numeric parameters are dynamically typed so
    `-p duration_s:=150` and `:=150.0` both work (rclpy otherwise refuses an
    INTEGER for a DOUBLE default and the node dies at startup)."""
    from rcl_interfaces.msg import ParameterDescriptor
    dyn = isinstance(default, (int, float)) and not isinstance(default, bool)
    node.declare_parameter(name, default, ParameterDescriptor(dynamic_typing=dyn))
    v = node.get_parameter(name).value
    if isinstance(default, float) and isinstance(v, int) and not isinstance(v, bool):
        v = float(v)
    return v


def finite_or_nan(x):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return math.nan
    return x if math.isfinite(x) else math.nan
