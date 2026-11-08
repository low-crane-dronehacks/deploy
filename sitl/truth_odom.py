#!/usr/bin/env python3
"""Gazebo's true pose as /visual_slam/tracking/odometry, in cuVSLAM's convention (sitl/run_native.sh EV_SOURCE=truth_odom).

Lets the whole real-drone pipeline run with a perfect odometry source: onboard/vio/ev_bridge.py -> PX4, and skybeta-style
team nodes reading the same topic, exactly as with cuVSLAM, but without vision errors. If a flight works with
truth_odom and fails with cuvslam, the problem is the vision, not the start procedure / interface / team node / PX4.

Convention (onboard/slam/cuvslam_lib_node.py): frame odom = base_link at start (FLU, z up), child base_link;
pose = base_link relative to its start pose; twist in the body (child) frame; pose covariance set (sigma_p, sigma_a).
Stamp = node clock (use_sim_time:=true, as the bridge expects capture stamps on the ROS clock).
Reads the pose like sitl/ev_bridge.py: `gz topic -e .../dynamic_pose/info --json-output` (honours GZ_PARTITION).
"""
import json
import math
import subprocess
import sys
import threading

import numpy as np
import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from scipy.spatial.transform import Rotation


class TruthOdom(Node):
    def __init__(self):
        super().__init__('truth_odom')
        dp = lambda n, v: self.declare_parameter(n, v).value
        self.model = dp('model', 'x500_stereo_0')
        self.gz_topic = dp('gz_topic', '/world/dronehacks/dynamic_pose/info')
        self.sigma_p, self.sigma_a, rate = float(dp('sigma_p', 0.005)), float(dp('sigma_a', 0.002)), float(dp('rate', 30.0))
        self.rng = np.random.default_rng(2)
        self.pose = None                     # (p ENU, R world<-body FLU), latest from gz
        self.T0inv = None
        self.prev = None
        self.lock = threading.Lock()
        self.pub = self.create_publisher(Odometry, dp('topic', '/visual_slam/tracking/odometry'), 10)
        threading.Thread(target=self._read_gz, daemon=True).start()
        self.create_timer(1.0 / rate, self._tick)
        self.get_logger().info(f'Gazebo truth of {self.model} -> /visual_slam/tracking/odometry (cuVSLAM convention), '
                               f'noise {self.sigma_p} m / {self.sigma_a} rad')

    def _read_gz(self):
        p = subprocess.Popen(['gz', 'topic', '-e', '-t', self.gz_topic, '--json-output'], stdout=subprocess.PIPE, text=True)
        for line in p.stdout:
            try:
                for e in json.loads(line).get('pose', []):
                    if e.get('name') == self.model:
                        t, o = e.get('position', {}), e.get('orientation', {})
                        R = Rotation.from_quat([o.get('x', 0.0), o.get('y', 0.0), o.get('z', 0.0), o.get('w', 1.0)]).as_matrix()
                        with self.lock:
                            self.pose = (np.array([t.get('x', 0.0), t.get('y', 0.0), t.get('z', 0.0)]), R)
            except ValueError:
                pass

    def _tick(self):
        with self.lock:
            if self.pose is None:
                return
            p, R = self.pose
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = R, p
        if self.T0inv is None:
            self.T0inv = np.linalg.inv(T)
        O = self.T0inv @ T                                   # base_link(t) in base_link(0), FLU
        O[:3, 3] += self.rng.normal(0, self.sigma_p, 3)
        O[:3, :3] = Rotation.from_rotvec(self.rng.normal(0, self.sigma_a, 3)).as_matrix() @ O[:3, :3]
        now = self.get_clock().now()
        t = now.nanoseconds * 1e-9
        m = Odometry()
        m.header.stamp, m.header.frame_id, m.child_frame_id = now.to_msg(), 'odom', 'base_link'
        m.pose.pose.position.x, m.pose.pose.position.y, m.pose.pose.position.z = (float(v) for v in O[:3, 3])
        q = Rotation.from_matrix(O[:3, :3]).as_quat()
        (m.pose.pose.orientation.x, m.pose.pose.orientation.y,
         m.pose.pose.orientation.z, m.pose.pose.orientation.w) = (float(v) for v in q)
        cov = np.zeros((6, 6))
        cov[:3, :3] = np.eye(3) * max(self.sigma_p, 1e-3) ** 2
        cov[3:, 3:] = np.eye(3) * max(self.sigma_a, 1e-3) ** 2
        m.pose.covariance = [float(v) for v in cov.ravel()]
        if self.prev is not None and t > self.prev[0]:      # body-frame twist by finite difference, as cuvslam_lib_node
            dt = t - self.prev[0]
            Op = self.prev[1]
            v_body = O[:3, :3].T @ (O[:3, 3] - Op[:3, 3]) / dt
            w_body = Rotation.from_matrix(Op[:3, :3].T @ O[:3, :3]).as_rotvec() / dt
            m.twist.twist.linear.x, m.twist.twist.linear.y, m.twist.twist.linear.z = (float(v) for v in v_body)
            m.twist.twist.angular.x, m.twist.twist.angular.y, m.twist.twist.angular.z = (float(v) for v in w_body)
        if self.prev is None or t > self.prev[0]:
            self.prev = (t, O)
            self.pub.publish(m)


def main():
    rclpy.init(args=sys.argv)
    node = TruthOdom()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
