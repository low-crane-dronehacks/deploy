#!/usr/bin/env python3
"""Gazebo depth camera -> what our stereo nodes (vpi_stereo / stereo_depth) publish on the drone, so the policy runs
unchanged with `skybeta.launch.py stereo_backend:=none`:

  /policynav/depth/image        32FC1 metres, frame sim_depth_optical, NaN = invalid (stereo_depth's convention)
  /policynav/depth/camera_info  pinhole K from the model's FOV (640x400, 74.7 deg -> fx ~420, as on the drone)
  /tf_static                    base_link -> sim_camera_link (CAMERA_X/Y/Z/ROLL/PITCH/YAW, FLU) -> sim_depth_optical

Gazebo's depth camera returns +inf where nothing is within `far`; that stays +inf (the policy reads a simulator's
+inf as far, AGENTS.md). Readings closer than `min_range_m` (what stereo cannot match at max_disparity 96: ~0.33 m)
become NaN, which the policy reads as near. Optional stereo-like noise: sigma = noise_k * z^2, dropout fraction.
"""
import math
import sys

import numpy as np
import rclpy
from geometry_msgs.msg import TransformStamped
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image
from tf2_ros import StaticTransformBroadcaster



def quat_from_rpy(r, p, y):
    """(x, y, z, w) of Rz(y) Ry(p) Rx(r), the convention of SDF poses and static_transform_publisher."""
    cr, sr, cp, sp, cy, sy = math.cos(r / 2), math.sin(r / 2), math.cos(p / 2), math.sin(p / 2), math.cos(y / 2), math.sin(y / 2)
    return (sr * cp * cy - cr * sp * sy, cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy, cr * cp * cy + sr * sp * sy)


def intrinsics(width, height, hfov):
    fx = width / (2.0 * math.tan(hfov / 2.0))
    return fx, fx, width / 2.0, height / 2.0


def degrade(z, rng, min_range, noise_k, dropout):
    """Stereo-like degradation of a float32 depth image (metres); returns a new array."""
    z = z.astype(np.float32, copy=True)
    finite = np.isfinite(z)
    if noise_k > 0:
        z[finite] += (rng.standard_normal(int(finite.sum())) * noise_k * z[finite] ** 2).astype(np.float32)
    z[finite & (z < min_range)] = np.nan
    if dropout > 0:
        z[rng.random(z.shape) < dropout] = np.nan
    return z


class GzDepth(Node):
    def __init__(self):
        super().__init__('sim_depth')
        dp = lambda n, v: self.declare_parameter(n, v).value     # noqa: E731
        topic = dp('gz_topic', '/skybeta_sim/depth')
        self.hfov = float(dp('hfov', 1.30376))
        self.min_range = float(dp('min_range_m', 0.33))
        self.noise_k, self.dropout = float(dp('noise_k', 0.0)), float(dp('dropout', 0.0))
        mount = [float(dp(k, 0.0)) for k in ('camera_x', 'camera_y', 'camera_z', 'camera_roll', 'camera_pitch', 'camera_yaw')]
        self.frame = 'sim_depth_optical'
        self.rng = np.random.default_rng(2)
        self.pub = self.create_publisher(Image, '/policynav/depth/image', qos_profile_sensor_data)
        self.pub_info = self.create_publisher(CameraInfo, '/policynav/depth/camera_info', qos_profile_sensor_data)
        self.tf = StaticTransformBroadcaster(self)
        self.tf.sendTransform([self._tf('base_link', 'sim_camera_link', mount[:3], quat_from_rpy(*mount[3:])),
                               # camera link (x forward, FLU) -> optical (z forward, x right, y down)
                               self._tf('sim_camera_link', self.frame, (0, 0, 0), quat_from_rpy(-math.pi / 2, 0, -math.pi / 2))])
        self.n = 0
        from gz.msgs10.image_pb2 import Image as GzImage  # Gazebo's Python bindings: only in the rig image
        from gz.transport13 import Node as GzNode
        self.gz = GzNode()
        if not self.gz.subscribe(GzImage, topic, self._on_image):
            raise RuntimeError(f'cannot subscribe to gz topic {topic}')
        self.create_timer(5.0, self._report)
        self.get_logger().info(f'{topic} -> /policynav/depth/image; mount FLU xyz={mount[:3]} rpy={mount[3:]}')

    def _tf(self, parent, child, xyz, q):
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id, t.child_frame_id = parent, child
        t.transform.translation.x, t.transform.translation.y, t.transform.translation.z = (float(c) for c in xyz)
        t.transform.rotation.x, t.transform.rotation.y, t.transform.rotation.z, t.transform.rotation.w = q
        return t

    def _on_image(self, msg):                   # gz-transport thread
        w, h = msg.width, msg.height
        if len(msg.data) != w * h * 4:
            self.get_logger().error(f'expected R_FLOAT32 {w}x{h}, got {len(msg.data)} bytes', throttle_duration_sec=5.0)
            return
        z = degrade(np.frombuffer(msg.data, np.float32).reshape(h, w), self.rng, self.min_range, self.noise_k, self.dropout)
        stamp = self.get_clock().now().to_msg()
        out = Image()
        out.header.stamp, out.header.frame_id = stamp, self.frame
        out.height, out.width, out.encoding, out.is_bigendian, out.step = h, w, '32FC1', 0, 4 * w
        out.data = z.tobytes()
        fx, fy, cx, cy = intrinsics(w, h, self.hfov)
        ci = CameraInfo()
        ci.header = out.header
        ci.height, ci.width, ci.distortion_model = h, w, 'plumb_bob'
        ci.d = [0.0] * 5
        ci.k = [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]
        ci.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        ci.p = [fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0]
        self.pub.publish(out)
        self.pub_info.publish(ci)
        self.n += 1

    def _report(self):
        self.get_logger().info(f'{self.n / 5.0:.1f} Hz depth' + ('' if self.n else ' -- no frames from gz yet'))
        self.n = 0


def main():
    rclpy.init(args=sys.argv)
    rclpy.spin(GzDepth())


if __name__ == '__main__':
    main()
