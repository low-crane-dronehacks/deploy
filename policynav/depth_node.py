#!/usr/bin/env python3
"""policynav.depth_node: stereo depth from the OAK's raw mono pair (skybeta publishes no depth topic).

  in   /oak/left/image_raw, /oak/right/image_raw (+ camera_info): skybeta's VIO feed, read only
  out  /policynav/depth/image        sensor_msgs/Image 32FC1, metres, NaN = invalid (rectified left camera)
       /policynav/depth/camera_info  rectified intrinsics at the output size
       TF  <left frame> -> <left frame>_rect  (the rectification rotation, static)
       /policynav/depth/stats        JSON: matcher, ms per frame, valid fraction, dropped pairs

VPI census+SGM on CUDA by default (backend:=sgbm for OpenCV). Matching runs at an average rate_hz (default 13.5 Hz, decimated by camera stamps);
pairs arriving while busy are dropped, never queued.
"""
from __future__ import annotations

import json
import math
import sys
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String

from . import stereo as S


def _img_to_np(m: Image) -> np.ndarray:
    ch = {'mono8': 1, '8uc1': 1, 'bgr8': 3, 'rgb8': 3, 'mono16': 1, '16uc1': 1}.get(m.encoding.lower())
    if ch is None:
        raise ValueError(f'unsupported image encoding {m.encoding!r}')
    dt = np.uint16 if m.encoding.lower() in ('mono16', '16uc1') else np.uint8
    a = np.frombuffer(m.data, dt).reshape(m.height, m.step // np.dtype(dt).itemsize)
    a = a[:, :m.width * ch].reshape(m.height, m.width, ch) if ch > 1 else a[:, :m.width]
    return S.to_gray8(a)


def _mat_to_quat_xyzw(R):
    w = math.sqrt(max(0.0, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    x = math.copysign(math.sqrt(max(0.0, 1 + R[0, 0] - R[1, 1] - R[2, 2])) / 2, R[2, 1] - R[1, 2])
    y = math.copysign(math.sqrt(max(0.0, 1 - R[0, 0] + R[1, 1] - R[2, 2])) / 2, R[0, 2] - R[2, 0])
    z = math.copysign(math.sqrt(max(0.0, 1 - R[0, 0] - R[1, 1] + R[2, 2])) / 2, R[1, 0] - R[0, 1])
    return x, y, z, w


class DepthNode(Node):
    def __init__(self):
        super().__init__('policynav_depth')
        dp = lambda n, d: self.declare_parameter(n, d).value      # noqa: E731
        self.p = S.StereoParams(out_w=int(dp('out_w', 640)), out_h=int(dp('out_h', 400)),
                                max_disparity=int(dp('max_disparity', 96)), window=int(dp('window', 5)),
                                conf_threshold=int(dp('conf_threshold', 32767)),
                                integer_disparity=bool(dp('integer_disparity', False)))
        self.rate_hz = float(dp('rate_hz', 13.5))
        self.slop = float(dp('sync_slop_s', 0.005))
        ns_l, ns_r = dp('left_ns', '/oak/left'), dp('right_ns', '/oak/right')
        img_topic = dp('image_topic', 'image_mono')            # skybeta publishes image_mono (mono8) + image_raw (bgr8)
        backend = dp('backend', 'auto')
        if backend == 'external':                 # rectify only; src/vpi_stereo_node.cpp matches (VPI C API)
            self.matcher, self.backend = None, 'external'
        else:
            self.matcher, self.backend = S.make_matcher(self.p, backend)
        from tf2_ros import Buffer, TransformListener
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.info = {}
        self.maps = None
        self.last = {'l': None, 'r': None}
        self.gate = S.RateGate(self.rate_hz)
        self.n = {'pairs': 0, 'dropped': 0, 'unsynced': 0}
        self.ms, self.valid = [], []
        for side, ns in (('l', ns_l), ('r', ns_r)):
            self.create_subscription(CameraInfo, f'{ns}/camera_info', lambda m, s=side: self._on_info(s, m),
                                     qos_profile_sensor_data)
            self.create_subscription(Image, f'{ns}/{img_topic}', lambda m, s=side: self._on_img(s, m),
                                     qos_profile_sensor_data)
        self.pub = self.create_publisher(Image, '/policynav/depth/image', qos_profile_sensor_data)
        self.pub_rect = {s: self.create_publisher(Image, f'/policynav/stereo/{s}/image_rect', qos_profile_sensor_data)
                         for s in ('left', 'right')}
        self.pub_rect_info = self.create_publisher(CameraInfo, '/policynav/stereo/camera_info', qos_profile_sensor_data)
        self.pub_info = self.create_publisher(CameraInfo, '/policynav/depth/camera_info', qos_profile_sensor_data)
        self.pub_stats = self.create_publisher(String, '/policynav/depth/stats', 10)
        self.create_timer(2.0, self._stats)
        self.get_logger().info(f'stereo depth: {self.backend}, {self.p.out_w}x{self.p.out_h}, '
                               f'max disparity {self.p.max_disparity}, <= {self.rate_hz} Hz')

    def _on_info(self, side, m):
        if side in self.info or m.k[0] <= 0:
            return
        self.info[side] = m
        if len(self.info) == 2 and self.maps is None:
            self._setup()

    def _setup(self):
        L, R = self.info['l'], self.info['r']
        p = self.p
        if S.camera_info_is_rectified_pair(L.p, R.p, L.r, R.r):
            self.maps = [S.rect_maps(c.k, c.d, c.r, c.p, c.width, c.height, p.out_w, p.out_h, c.distortion_model)
                         for c in (L, R)]
            self.baseline = S.rect_baseline_m(R.p)
            S.validate_stereo_baseline(self.baseline)
            self.fx, self.fy, self.cx, self.cy = S.rectified_intrinsics(L.p, L.width, L.height, p.out_w, p.out_h)
            Rl = np.asarray(L.r, float).reshape(3, 3)
            how = 'camera_info rectification'
        else:
            # skybeta's driver: raw per-eye K, R = I, D = 0, no common P -> rectify from the TF between the eyes
            from rclpy.time import Time
            try:
                tf = self.tf_buffer.lookup_transform(L.header.frame_id, R.header.frame_id, Time())
            except Exception as e:
                self.get_logger().warning(f'waiting for TF {L.header.frame_id} -> {R.header.frame_id}: {e}',
                                          throttle_duration_sec=5.0)
                self.info.pop('l', None)                          # retry on the next camera_info
                return
            q, t = tf.transform.rotation, tf.transform.translation
            from .obs import quat_to_mat
            R_lr = quat_to_mat(q.w, q.x, q.y, q.z)
            dist = lambda c: list(c.d) if c.distortion_model in ('plumb_bob', 'rational_polynomial') else []  # noqa
            m1, m2, self.fx, self.cx, self.cy, self.baseline, Rl = S.stereo_rect_from_extrinsics(
                np.reshape(L.k, (3, 3)), dist(L), np.reshape(R.k, (3, 3)), dist(R), L.width, L.height,
                R_lr, (t.x, t.y, t.z), p.out_w, p.out_h, alpha=0.0)
            self.fy = self.fx
            S.validate_stereo_baseline(self.baseline)
            self.maps = [m1, m2]
            how = 'rectified from TF extrinsics'
        self.frame = L.header.frame_id + '_rect'
        from geometry_msgs.msg import TransformStamped
        from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster
        self.tf_b = StaticTransformBroadcaster(self)
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id, t.child_frame_id = L.header.frame_id, self.frame
        q = _mat_to_quat_xyzw(Rl.T)                                # rect axes expressed in the left camera frame
        t.transform.rotation.x, t.transform.rotation.y, t.transform.rotation.z, t.transform.rotation.w = q
        self.tf_b.sendTransform(t)
        if abs(self.baseline) < 0.02 or abs(self.baseline) > 0.2:
            self.get_logger().error(f'implausible baseline {self.baseline:.4f} m from right P: check camera_info')
        self.get_logger().info(f'{how}: fx {self.fx:.1f} at {self.p.out_w} px, baseline {self.baseline * 100:.2f} cm,'
                               f' min range {self.fx * abs(self.baseline) / self.p.max_disparity:.2f} m')

    def _on_img(self, side, m):
        self.last[side] = m
        l, r = self.last['l'], self.last['r']
        if self.maps is None or l is None or r is None:
            return
        tl = l.header.stamp.sec + 1e-9 * l.header.stamp.nanosec
        tr = r.header.stamp.sec + 1e-9 * r.header.stamp.nanosec
        if abs(tl - tr) > self.slop:
            self.n['unsynced'] += 1
            return
        self.last = {'l': None, 'r': None}
        if not self.gate.accept(tl):                  # by camera stamp: average exactly rate_hz
            self.n['dropped'] += 1
            return
        self._process(l, r)

    def _process(self, l, r):
        import cv2
        t0 = time.perf_counter()
        (mxl, myl), (mxr, myr) = self.maps
        L = cv2.remap(_img_to_np(l), mxl, myl, cv2.INTER_LINEAR)
        R = cv2.remap(_img_to_np(r), mxr, myr, cv2.INTER_LINEAR)
        if self.matcher is None:                  # external matcher: publish the rectified pair + its geometry
            for side, img in (('left', L), ('right', R)):
                m = Image()
                m.header.stamp, m.header.frame_id = l.header.stamp, self.frame      # one stamp for the pair
                m.height, m.width, m.encoding, m.is_bigendian, m.step = img.shape[0], img.shape[1], 'mono8', 0, img.shape[1]
                m.data = img.tobytes()
                self.pub_rect[side].publish(m)
            ci = CameraInfo()
            ci.header.stamp, ci.header.frame_id = l.header.stamp, self.frame
            ci.width, ci.height = L.shape[1], L.shape[0]
            ci.k = [self.fx, 0.0, self.cx, 0.0, self.fy, self.cy, 0.0, 0.0, 1.0]
            ci.p = [self.fx, 0.0, self.cx, -self.fx * abs(self.baseline), 0.0, self.fy, self.cy, 0.0, 0.0, 0.0, 1.0, 0.0]
            ci.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
            ci.distortion_model, ci.d = 'plumb_bob', [0.0] * 5
            self.pub_rect_info.publish(ci)
            self.ms.append(1e3 * (time.perf_counter() - t0))
            self.n['pairs'] += 1
            return
        disp, valid = self.matcher(L, R)
        z = S.disparity_to_depth(disp, self.fx, abs(self.baseline), valid=valid, min_m=self.p.min_depth_m,
                                 max_m=self.p.max_depth_m, integer=self.p.integer_disparity)
        self.ms.append(1e3 * (time.perf_counter() - t0))
        self.valid.append(float(np.isfinite(z).mean()))
        self.n['pairs'] += 1
        out = Image()
        out.header.stamp, out.header.frame_id = l.header.stamp, self.frame
        out.height, out.width, out.encoding, out.is_bigendian, out.step = z.shape[0], z.shape[1], '32FC1', 0, 4 * z.shape[1]
        out.data = z.astype(np.float32).tobytes()
        self.pub.publish(out)
        ci = CameraInfo()
        ci.header = out.header
        ci.width, ci.height = z.shape[1], z.shape[0]
        ci.k = [self.fx, 0.0, self.cx, 0.0, self.fy, self.cy, 0.0, 0.0, 1.0]
        ci.p = [self.fx, 0.0, self.cx, 0.0, 0.0, self.fy, self.cy, 0.0, 0.0, 0.0, 1.0, 0.0]
        ci.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        ci.distortion_model, ci.d = 'plumb_bob', [0.0] * 5
        self.pub_info.publish(ci)

    def _stats(self):
        ms, v = self.ms[-30:], self.valid[-30:]
        st = dict(self.n, backend=self.backend, calibrated=self.maps is not None,
                  ms=round(float(np.mean(ms)), 1) if ms else None, valid_frac=round(float(np.mean(v)), 3) if v else None)
        self.pub_stats.publish(String(data=json.dumps(st)))
        if self.maps is None:
            self.get_logger().info('waiting for both camera_info messages', throttle_duration_sec=10.0)


def main(argv=None):
    rclpy.init(args=list(argv) if argv is not None else sys.argv)
    node = DepthNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
