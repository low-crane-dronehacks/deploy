#!/usr/bin/env python3
"""SIMULATOR and AIRCRAFT adapter: estimator health from PX4's OWN estimate, for flights where
no external estimator (VIO) reports one - e.g. outdoors on GNSS + compass.

Publishes std_msgs/Bool on `health_topic` (default /refnav/px4_estimator_ok) at 5 Hz: true only
while PX4's latest vehicle_local_position is fresh and says position and velocity are valid
(xy_valid, z_valid, v_xy_valid, v_z_valid). Fails closed: nothing heard, stale or any flag false =
false. Heading is the mission's own readiness check (refnav/node.py: with a compass PX4 withholds
heading_good_for_control until it has re-aligned in flight). The mission treats this exactly like
a VIO bridge's health (reference/refnav/core.py Health)."""
from __future__ import annotations

import json
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Bool

PX4_QOS = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                     history=HistoryPolicy.KEEP_LAST, depth=1)
FLAGS = ("xy_valid", "z_valid", "v_xy_valid", "v_z_valid")
VERTICAL_FLAGS = ("z_valid", "v_z_valid")


def estimate_ok(flags, age_s, stale_s=0.5, vertical_only=False):
    """PURE: all validity flags set and the sample younger than stale_s."""
    required = VERTICAL_FLAGS if vertical_only else FLAGS
    return flags is not None and age_s is not None and age_s <= stale_s and all(bool(flags.get(f)) for f in required)


class Px4Health(Node):
    def __init__(self):
        super().__init__("refnav_px4_health")
        from px4_msgs.msg import VehicleLocalPosition
        self.stale_s = float(self.declare_parameter("stale_s", 0.5).value)
        self.vertical_only = bool(self.declare_parameter("vertical_only", False).value)
        topic = self.declare_parameter("px4_topic", "/fmu/out/vehicle_local_position_v1").value
        self.pub = self.create_publisher(Bool, self.declare_parameter("health_topic", "/refnav/px4_estimator_ok").value,
                                         10)
        self.flags, self.t = None, None
        self.received, self.sample_stamp, self.sample_wall, self.reported = 0, None, None, None
        self.create_subscription(VehicleLocalPosition, topic, self._on, PX4_QOS)
        self.create_timer(0.2, self._tick)

    def _now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def _on(self, m):
        self.flags = {f: getattr(m, f) for f in FLAGS}
        self.t = self._now()
        self.received += 1
        self.sample_stamp, self.sample_wall = int(m.timestamp), time.monotonic()

    def _tick(self):
        now = self._now()
        age = None if self.t is None else now - self.t
        healthy = bool(estimate_ok(self.flags, age, self.stale_s, self.vertical_only))
        self.pub.publish(Bool(data=healthy))
        if healthy != self.reported:
            self.get_logger().info('[PX4_HEALTH] ' + json.dumps({
                'schema': 'flyx-px4-health/1', 'healthy': healthy, 'vertical_only': self.vertical_only,
                'received': self.received,
                'sample_stamp_us': self.sample_stamp, 'flags': self.flags, 'age_s': age,
                'node_clock_s': now, 'stale_s': self.stale_s,
                'sample_wall_age_s': None if self.sample_wall is None else time.monotonic() - self.sample_wall,
            }, sort_keys=True))
            self.reported = healthy


def main():
    rclpy.init()
    rclpy.spin(Px4Health())


if __name__ == "__main__":
    main()
