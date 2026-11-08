#!/usr/bin/env python3
"""SIMULATOR and AIRCRAFT adapter for the maze autonomy stack: PX4's own estimate
(/fmu/out/vehicle_local_position_v1 + /fmu/out/vehicle_attitude, the topics policynav and
isam2_depth_node use) as /odometry/local in PX4's NED / FRD convention, plus TF
odom_ned -> base_link_frd.

The maze stack (src/: RRT*, path transformer, MPPI, mux) works in NED like PX4's TrajectorySetpoint.
px4_odom.py publishes the same estimate in ROS ENU / FLU under the frames odom -> base_link; this
adapter uses different frame names (odom_ned, base_link_frd) so the two TF trees never collide.

Twist is in the child (body FRD) frame, as nav_msgs/Odometry defines it. Stamp: node clock at
receipt (see px4_odom.py for why).
"""
from __future__ import annotations

import sys

import rclpy
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from tf2_ros import TransformBroadcaster

PX4_QOS = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                     history=HistoryPolicy.KEEP_LAST, depth=1)


def rotate_inverse(qw, qx, qy, qz, v):
    """R(q)^T v for a unit quaternion (world <- body), i.e. world vector into the body frame."""
    # v' = q* v q, expanded
    x, y, z = v
    # t = 2 * cross(-q_vec, v)
    tx = 2.0 * (-qy * z + qz * y)
    ty = 2.0 * (-qz * x + qx * z)
    tz = 2.0 * (-qx * y + qy * x)
    return (x + qw * tx + (-qy * tz + qz * ty),
            y + qw * ty + (-qz * tx + qx * tz),
            z + qw * tz + (-qx * ty + qy * tx))


class Px4LocalOdomNed(Node):
    def __init__(self):
        super().__init__("px4_local_odom_ned")
        from px4_msgs.msg import VehicleAttitude, VehicleLocalPosition
        self.odom_frame = self.declare_parameter("odom_frame", "odom_ned").value
        self.base_frame = self.declare_parameter("base_frame", "base_link_frd").value
        lpos = self.declare_parameter("lpos_topic", "/fmu/out/vehicle_local_position_v1").value
        att = self.declare_parameter("attitude_topic", "/fmu/out/vehicle_attitude").value
        out = self.declare_parameter("odom_topic", "/odometry/local").value
        self.pub = self.create_publisher(Odometry, out, qos_profile_sensor_data)
        self.tf = TransformBroadcaster(self)
        self.q = None  # latest attitude (w, x, y, z), FRD -> NED
        self.create_subscription(VehicleAttitude, att, self._on_att, PX4_QOS)
        self.create_subscription(VehicleLocalPosition, lpos, self._on_lpos, PX4_QOS)
        self.get_logger().info(f"{lpos} + {att} -> {out} ({self.odom_frame} -> {self.base_frame}, NED/FRD)")

    def _on_att(self, m):
        self.q = tuple(float(c) for c in m.q)

    def _on_lpos(self, m):
        # Publish only complete estimates: a NaN twist would propagate into MPPI's commands.
        if self.q is None or not (m.xy_valid and m.z_valid and m.v_xy_valid and m.v_z_valid):
            return
        qw, qx, qy, qz = self.q
        o = Odometry()
        o.header.stamp = self.get_clock().now().to_msg()
        o.header.frame_id, o.child_frame_id = self.odom_frame, self.base_frame
        o.pose.pose.position.x, o.pose.pose.position.y, o.pose.pose.position.z = float(m.x), float(m.y), float(m.z)
        o.pose.pose.orientation.w, o.pose.pose.orientation.x = qw, qx
        o.pose.pose.orientation.y, o.pose.pose.orientation.z = qy, qz
        o.pose.covariance[0] = o.pose.covariance[7] = float(m.eph) ** 2
        o.pose.covariance[14] = float(m.epv) ** 2
        vb = rotate_inverse(qw, qx, qy, qz, [float(m.vx), float(m.vy), float(m.vz)])
        o.twist.twist.linear.x, o.twist.twist.linear.y, o.twist.twist.linear.z = vb
        self.pub.publish(o)

        t = TransformStamped()
        t.header = o.header
        t.child_frame_id = self.base_frame
        t.transform.translation.x = o.pose.pose.position.x
        t.transform.translation.y = o.pose.pose.position.y
        t.transform.translation.z = o.pose.pose.position.z
        t.transform.rotation = o.pose.pose.orientation
        self.tf.sendTransform(t)


def main():
    rclpy.init(args=sys.argv)
    n = Px4LocalOdomNed()
    try:
        rclpy.spin(n)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass


if __name__ == "__main__":
    main()
