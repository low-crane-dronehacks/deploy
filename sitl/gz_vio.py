#!/usr/bin/env python3
"""Gazebo truth -> a stand-in for cuVSLAM: /visual_slam/tracking/odometry (nav_msgs/Odometry), what skybeta's
px4_odom_bridge, flight_interface and our policy node read on the drone.

Matches what cuVSLAM gives on the drone (skybeta CLAUDE.md, cuVSLAM grounding):
  frame     odom = base_link's pose when this node first sees the vehicle (startup-aligned, arbitrary heading, no
            gravity alignment), ENU/FLU; child base_link. Restarting the node re-zeroes it, like a stack restart.
  twist     body-frame (FLU) linear velocity, a finite difference over `vel_window_s` (cuVSLAM differentiates its
            pose cache, so it is smoothed and lags); angular from the same window.
  rate      `rate` Hz (30), stamped with the capture time and published `latency_ms` later (camera -> cuVSLAM output;
            the drone measures ~60-70 ms to the bridge, EKF2_EV_DELAY 60-70).
  errors    white noise (sigma_p m, sigma_a rad) + random-walk drift in position (drift_p m/sqrt(s)) and yaw
            (drift_yaw rad/sqrt(s)). Tilt does not drift: the error that matters for EKF2 is heading.
  outage    outage_at_s / outage_s: stop publishing for that long (tracking loss: dark room, blank wall).

Wall clock (use_sim_time false), as on the drone: skybeta's bridge stamps with the ROS clock and PX4's uXRCE time sync
maps it, so the sim must run at ~1x real time.
"""
import collections
import math
import sys
import threading

import numpy as np
import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time



def quat_to_mat(w, x, y, z):
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def mat_to_quat(R):
    """Rotation matrix -> (w, x, y, z), w >= 0."""
    t = np.trace(R)
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        q = np.array([0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s])
    else:
        i = int(np.argmax(np.diag(R)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = math.sqrt(max(1e-12, 1.0 + R[i, i] - R[j, j] - R[k, k])) * 2
        q = np.zeros(4)
        q[0] = (R[k, j] - R[j, k]) / s
        q[1 + i] = 0.25 * s
        q[1 + j] = (R[j, i] + R[i, j]) / s
        q[1 + k] = (R[k, i] + R[i, k]) / s
    return q if q[0] >= 0 else -q


def rotvec_to_mat(v):
    a = float(np.linalg.norm(v))
    if a < 1e-12:
        return np.eye(3)
    k = np.asarray(v) / a
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(a) * K + (1 - math.cos(a)) * K @ K


def mat_to_rotvec(R):
    c = max(-1.0, min(1.0, (np.trace(R) - 1) / 2))
    a = math.acos(c)
    if a < 1e-9:
        return np.zeros(3)
    return a / (2 * math.sin(a)) * np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])


def rot_z(yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


class VioModel:
    """Truth (world ENU, body FLU) -> odom-frame pose with startup alignment, drift and noise. No ROS, testable."""

    def __init__(self, sigma_p=0.005, sigma_a=0.002, drift_p=0.002, drift_yaw=0.0005, seed=1):
        self.sigma_p, self.sigma_a, self.drift_p, self.drift_yaw = sigma_p, sigma_a, drift_p, drift_yaw
        self.rng = np.random.default_rng(seed)
        self.origin = None                  # (p0, R0) of base_link at startup
        self.bias_p, self.bias_yaw = np.zeros(3), 0.0
        self.last_t = None

    def odom(self, t, p_w, R_w):
        """Pose of base_link in the startup frame, with drift: (p, R). t in seconds."""
        if self.origin is None:
            self.origin = (p_w.copy(), R_w.copy())
        if self.last_t is not None:
            dt = max(0.0, t - self.last_t)
            self.bias_p = self.bias_p + self.rng.normal(0, self.drift_p * math.sqrt(dt), 3)
            self.bias_yaw += float(self.rng.normal(0, self.drift_yaw * math.sqrt(dt)))
        self.last_t = t
        p0, R0 = self.origin
        p = R0.T @ (p_w - p0)
        R = R0.T @ R_w
        Rb = rot_z(self.bias_yaw)           # the odom frame slowly rotates about gravity (heading drift)
        p = Rb @ p + self.bias_p + self.rng.normal(0, self.sigma_p, 3)
        R = rotvec_to_mat(self.rng.normal(0, self.sigma_a, 3)) @ Rb @ R
        return p, R


def velocity_body(hist):
    """Body-frame (of the newest sample) linear and angular velocity from the oldest -> newest sample in hist."""
    (t0, p0, R0), (t1, p1, R1) = hist[0], hist[-1]
    dt = t1 - t0
    if dt <= 1e-6:
        return np.zeros(3), np.zeros(3)
    v = R1.T @ (p1 - p0) / dt
    w = mat_to_rotvec(R0.T @ R1) / dt
    return v, w


class GzVio(Node):
    def __init__(self):
        super().__init__('sim_vio')
        dp = lambda n, v: self.declare_parameter(n, v).value     # noqa: E731
        self.model = dp('model', 'x500_skybeta_0')
        topic = dp('gz_topic', '/world/dronehacks/dynamic_pose/info')
        self.base_z = float(dp('base_link_z', 0.24))             # x500_base: base_link at (0, 0, .24) in the model
        self.rate = float(dp('rate', 30.0))
        self.latency = float(dp('latency_ms', 65.0)) / 1000.0
        self.vel_window = float(dp('vel_window_s', 0.1))
        self.outage_at, self.outage_s = float(dp('outage_at_s', -1.0)), float(dp('outage_s', 0.0))
        self.vio = VioModel(dp('sigma_p', 0.005), dp('sigma_a', 0.002), dp('drift_p', 0.002), dp('drift_yaw', 0.0005))
        self.var_p, self.var_a = max(dp('sigma_p', 0.005), 0.01) ** 2, max(dp('sigma_a', 0.002), 0.005) ** 2
        self.pub = self.create_publisher(Odometry, dp('odom_topic', '/visual_slam/tracking/odometry'),
                                         qos_profile_sensor_data)
        self.lock = threading.Lock()
        self.queue = collections.deque()        # (publish_at, Odometry)
        self.hist = collections.deque()         # truth (t, p, R) within the velocity window
        self.t_start = None
        self.last_sample = -1e9
        self.n_pub = 0
        from gz.msgs10.pose_v_pb2 import Pose_V          # Gazebo's Python bindings: only in the rig image
        from gz.transport13 import Node as GzNode
        self.gz = GzNode()
        if not self.gz.subscribe(Pose_V, topic, self._on_pose):
            raise RuntimeError(f'cannot subscribe to gz topic {topic}')
        self.create_timer(0.002, self._flush)
        self.create_timer(5.0, self._report)
        self.get_logger().info(f'{topic}[{self.model}] -> odometry at {self.rate:g} Hz, latency {self.latency * 1e3:.0f} ms'
                               + (f', outage at {self.outage_at:g} s for {self.outage_s:g} s' if self.outage_at >= 0 else ''))

    def _now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def _on_pose(self, msg):                    # gz-transport thread
        for e in msg.pose:
            if e.name == self.model:
                break
        else:
            return
        t = self._now()
        o, q = e.position, e.orientation
        R_w = quat_to_mat(q.w, q.x, q.y, q.z)
        p_w = np.array([o.x, o.y, o.z]) + R_w @ np.array([0.0, 0.0, self.base_z])
        self.hist.append((t, p_w, R_w))
        while len(self.hist) > 2 and t - self.hist[0][0] > self.vel_window:
            self.hist.popleft()
        if t - self.last_sample < 1.0 / self.rate - 1e-3:
            return
        self.last_sample = t
        if self.t_start is None:
            self.t_start = t
        el = t - self.t_start
        if self.outage_at >= 0 and self.outage_at <= el < self.outage_at + self.outage_s:
            return
        p, R = self.vio.odom(t, p_w, R_w)
        v_b, w_b = velocity_body(self.hist)
        m = Odometry()
        m.header.stamp = Time(seconds=t).to_msg()
        m.header.frame_id, m.child_frame_id = 'odom', 'base_link'
        m.pose.pose.position.x, m.pose.pose.position.y, m.pose.pose.position.z = (float(c) for c in p)
        qw, qx, qy, qz = mat_to_quat(R)
        m.pose.pose.orientation.w, m.pose.pose.orientation.x = float(qw), float(qx)
        m.pose.pose.orientation.y, m.pose.pose.orientation.z = float(qy), float(qz)
        m.twist.twist.linear.x, m.twist.twist.linear.y, m.twist.twist.linear.z = (float(c) for c in v_b)
        m.twist.twist.angular.x, m.twist.twist.angular.y, m.twist.twist.angular.z = (float(c) for c in w_b)
        cov = [0.0] * 36
        for i in range(3):
            cov[i * 7] = self.var_p
            cov[(i + 3) * 7] = self.var_a
        m.pose.covariance = cov
        m.twist.covariance = [c * 4.0 for c in cov]
        with self.lock:
            self.queue.append((t + self.latency, m))

    def _flush(self):
        now = self._now()
        with self.lock:
            while self.queue and self.queue[0][0] <= now:
                self.pub.publish(self.queue.popleft()[1])
                self.n_pub += 1

    def _report(self):
        self.get_logger().info(f'{self.n_pub / 5.0:.1f} Hz published' + ('' if self.n_pub else
                               f' -- no pose for model {self.model!r} yet'))
        self.n_pub = 0


def main():
    rclpy.init(args=sys.argv)
    rclpy.spin(GzVio())


if __name__ == '__main__':
    main()
