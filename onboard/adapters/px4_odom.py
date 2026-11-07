#!/usr/bin/env python3
"""SIMULATOR and AIRCRAFT adapter: PX4's own estimate (/fmu/out/vehicle_odometry, NED/FRD) as
nav_msgs/Odometry (ENU/FLU, frame odom -> base_link) plus the TF odom -> base_link. This is the
pose a 3-D mapper, RViz or a recorder consumes; it is the estimate the vehicle flies on, never
simulator truth.

Stamp: the node clock at receipt (simulation time under use_sim_time, system time on the
aircraft). PX4 publishes its estimate as it computes it, so receipt time is within one transport
delay of the sample time; the alternative (PX4's own timestamp) is on PX4's clock and would need
a clock conversion that differs between the simulator and the aircraft.
"""
from __future__ import annotations

import os
import sys

import rclpy
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from tf2_ros import TransformBroadcaster

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from adapters.geometry import px4_to_ros  # noqa: E402

PX4_QOS = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                     history=HistoryPolicy.KEEP_LAST, depth=1)


class Px4Odom(Node):
    def __init__(self):
        super().__init__("refnav_px4_odom")
        from px4_msgs.msg import VehicleOdometry
        self.odom_frame = self.declare_parameter("odom_frame", "odom").value
        self.base_frame = self.declare_parameter("base_frame", "base_link").value
        topic = self.declare_parameter("px4_topic", "/fmu/out/vehicle_odometry").value
        self.pub = self.create_publisher(Odometry, self.declare_parameter("odom_topic", "/refnav/odom").value, 10)
        self.tf = TransformBroadcaster(self)
        self.n = 0
        self.create_subscription(VehicleOdometry, topic, self._on, PX4_QOS)

    def _on(self, m):
        if m.pose_frame != 1:                       # POSE_FRAME_NED only (an FRD-labelled frame has no heading)
            return
        vel = list(m.velocity) if m.velocity_frame == 1 else None      # VELOCITY_FRAME_NED
        p, q, v = px4_to_ros(list(m.position), list(m.q), vel)
        stamp = self.get_clock().now().to_msg()
        o = Odometry()
        o.header.stamp, o.header.frame_id, o.child_frame_id = stamp, self.odom_frame, self.base_frame
        o.pose.pose.position.x, o.pose.pose.position.y, o.pose.pose.position.z = (float(c) for c in p)
        o.pose.pose.orientation.x, o.pose.pose.orientation.y, o.pose.pose.orientation.z, o.pose.pose.orientation.w = q
        if v is not None:
            # twist in the child (body) frame per REP-105: rotate the world velocity into FLU
            from adapters.geometry import quat_to_mat
            Rwb = quat_to_mat(q[3], q[0], q[1], q[2])
            vb = Rwb.T @ v
            o.twist.twist.linear.x, o.twist.twist.linear.y, o.twist.twist.linear.z = (float(c) for c in vb)
        self.pub.publish(o)
        t = TransformStamped()
        t.header = o.header
        t.child_frame_id = self.base_frame
        t.transform.translation.x, t.transform.translation.y, t.transform.translation.z = (float(c) for c in p)
        t.transform.rotation = o.pose.pose.orientation
        self.tf.sendTransform(t)
        self.n += 1


def main():
    rclpy.init(args=sys.argv)
    n = Px4Odom()
    try:
        rclpy.spin(n)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass


if __name__ == "__main__":
    main()
