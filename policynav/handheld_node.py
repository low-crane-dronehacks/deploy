#!/usr/bin/env python3
"""policynav.handheld_node: camera + policy only, for a handheld dry run / data collection. NEVER commands PX4.

Carry the drone (props OFF) or the camera through the course; the policy runs at 10 Hz on the live depth and
reports what it would command. Nothing is published to /fmu/in/*; this node has no PX4 publishers at all.

  in   depth + CameraInfo (depth_topic), TF base_link <- depth frame (the launch publishes the mount)
       state_source px4:  /fmu/out/vehicle_local_position_v1, vehicle_attitude (+ distance_sensor)
                    odom: nav_msgs/Odometry (odom_topic, default cuVSLAM's /visual_slam/tracking/odometry)
                    none: zero velocity, level attitude
       /policynav/handheld/reset (std_msgs/Empty): clear the policy's memory (do it at the launch mark)
  out  /policynav/handheld/obs     sensor_msgs/Image 32FC1: the newest policy depth frame (128x80, 0..1, 0=invalid)
       /policynav/handheld/action  std_msgs/Float32MultiArray: action[4], vel_sp_ned[3], yawspeed
       /policynav/handheld/cmd     geometry_msgs/TwistStamped in base_link (FLU): the would-be command, for rviz
       /policynav/status           JSON
       log_dir/handheld_<stamp>.npz  every step (saved every 30 s and on exit): times, actions, setpoints, state,
                                     range and the policy's depth frames (float16)

memory: closed (default) keeps the recurrent state and feeds back the policy's own previous action, as in
flight (the person moving the drone stands in for the controller); fresh resets every tick (first-step view).

    ros2 launch launch/policynav.launch.py hardware:=handheld [state:=px4|odom|none] [record:=true]
"""
from __future__ import annotations

import json
import math
import os
import sys
import time

import numpy as np
import rclpy
from geometry_msgs.msg import TwistStamped
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Empty, Float32MultiArray, String

from . import handheld as H
from . import obs as O
from .node import decode_depth
from .policy_np import NumpyPolicy, resolve_policy

PX4_QOS = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                     history=HistoryPolicy.KEEP_LAST, depth=1)
LEVEL_Q = (1.0, 0.0, 0.0, 0.0)


class Handheld(Node):
    def __init__(self):
        super().__init__('policynav_handheld')
        dp = lambda name, default: self.declare_parameter(name, default).value     # noqa: E731
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        pf = dp('policy_file', 'policies/cur4_clean50_800.npz')
        self.policy = NumpyPolicy(resolve_policy(pf))
        self.meta = self.policy.meta
        self.n_act = self.meta['num_actions']
        self.v_max, self.yaw_max = self.meta['v_max'], self.meta['yaw_rate_max']
        self.state_source = dp('state_source', 'px4')
        self.range_source = dp('range_source', 'auto')           # auto | distance_sensor | altitude | fixed
        self.range_fixed = float(dp('range_fixed_m', 1.0))
        self.start_height = float(dp('start_height_m', 0.08))    # height of the body centre where the pose starts
        self.memory = dp('memory', 'closed')
        self.base_frame = dp('base_frame', 'base_link')
        self.depth_topic = dp('depth_topic', '/oak/stereo/image_raw')
        info_topic = dp('depth_info_topic', '/oak/stereo/camera_info')
        log_dir = dp('log_dir', os.path.join(root, 'logs', 'handheld'))
        os.makedirs(log_dir, exist_ok=True)
        self.log_path = os.path.join(log_dir, time.strftime('handheld_%Y%m%d_%H%M%S.npz'))

        from tf2_ros import Buffer, TransformListener
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.create_subscription(CameraInfo, info_topic, self._on_info, qos_profile_sensor_data)
        self.create_subscription(Image, self.depth_topic, self._on_depth, qos_profile_sensor_data)
        self.create_subscription(Empty, '/policynav/handheld/reset', lambda _m: self._reset('reset topic'), 10)
        self.pos = self.vel = None
        self.q = None
        self.home = None
        self.range_m = None
        self.vel_est = H.VelocityEstimator()
        if self.state_source == 'px4':
            from px4_msgs.msg import DistanceSensor, VehicleAttitude, VehicleLocalPosition
            self.create_subscription(VehicleLocalPosition, '/fmu/out/vehicle_local_position_v1', self._on_lpos, PX4_QOS)
            self.create_subscription(VehicleAttitude, '/fmu/out/vehicle_attitude', self._on_att, PX4_QOS)
            self.create_subscription(DistanceSensor, '/fmu/out/distance_sensor', self._on_range, PX4_QOS)
        elif self.state_source == 'odom':
            from nav_msgs.msg import Odometry
            self.create_subscription(Odometry, dp('odom_topic', '/visual_slam/tracking/odometry'), self._on_odom, 10)
        elif self.state_source != 'none':
            raise ValueError(f"state_source must be px4 | odom | none, got {self.state_source!r}")

        self.pub_obs = self.create_publisher(Image, '/policynav/handheld/obs', 5)
        self.pub_act = self.create_publisher(Float32MultiArray, '/policynav/handheld/action', 10)
        self.pub_cmd = self.create_publisher(TwistStamped, '/policynav/handheld/cmd', 10)
        self.pub_status = self.create_publisher(String, '/policynav/status', 10)

        self.K = self.reproj = None
        self.depth_img, self.depth_t = None, None
        self.stack = O.FrameStack(self.meta['frame_stack'])
        self.prev_action = np.zeros(self.n_act, np.float32)
        self.rows, self.frames, self.timing = [], [], []
        self.last = None
        self._reset('start')
        self.create_timer(0.1, self._control)
        self.create_timer(1.0, self._report)
        self.create_timer(30.0, self._save)
        self.get_logger().info(f"HANDHELD (never commands PX4): policy {self.meta['source']}, state {self.state_source}, "
                               f"range {self.range_source}, memory {self.memory}, depth {self.depth_topic}, "
                               f"log {self.log_path}")

    def _now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def _reset(self, why):
        self.policy.reset()
        self.stack.reset()
        self.prev_action = np.zeros(self.n_act, np.float32)
        self.home = self.pos
        self.get_logger().info(f'policy memory reset ({why})')

    # ── inputs ──
    def _on_lpos(self, m):
        if all(math.isfinite(c) for c in (m.x, m.y, m.z, m.vx, m.vy, m.vz)):
            self.pos, self.vel = (m.x, m.y, m.z), (m.vx, m.vy, m.vz)

    def _on_att(self, m):
        self.q = tuple(float(c) for c in m.q)

    def _on_range(self, m):
        if m.signal_quality != 0 and m.min_distance <= m.current_distance <= m.max_distance:
            self.range_m = float(m.current_distance)

    def _on_odom(self, m):
        p, o = m.pose.pose.position, m.pose.pose.orientation
        if not all(math.isfinite(c) for c in (p.x, p.y, p.z, o.x, o.y, o.z, o.w)):
            return
        self.pos, self.q = H.ros_pose_to_ned((p.x, p.y, p.z), (o.x, o.y, o.z, o.w))
        t = m.header.stamp.sec + 1e-9 * m.header.stamp.nanosec
        self.vel = tuple(self.vel_est.update(t, self.pos))

    def _on_info(self, m):
        if m.k[0] > 0 and self.K is None:
            self.K, self.src_hw = (m.k[0], m.k[4], m.k[2], m.k[5]), (m.height, m.width)

    def _on_depth(self, m):
        if self.K is None:
            return
        if self.reproj is None:
            from rclpy.time import Time
            try:
                tf = self.tf_buffer.lookup_transform(self.base_frame, m.header.frame_id, Time())
            except Exception as e:
                self.get_logger().warning(f'no TF {self.base_frame} <- {m.header.frame_id}: {e}', throttle_duration_sec=5.0)
                return
            r, t = tf.transform.rotation, tf.transform.translation
            H_, W_ = self.meta['depth_hw']
            self.reproj = O.DepthReprojector(self.K, self.src_hw, O.quat_to_mat(r.w, r.x, r.y, r.z), (t.x, t.y, t.z),
                                             out_hw=(H_, W_), hfov_deg=self.meta['hfov_deg'],
                                             vfov_deg=self.meta['vfov_deg'], far=self.meta['far_clip'], stride=2)
            self.get_logger().info(f'depth calibration: optical axis {self.reproj.axis_err_deg:.1f} deg off body x')
        try:
            self.depth_img = self.reproj(decode_depth(m))
            self.depth_t = self._now()
        except ValueError as e:
            self.get_logger().error(str(e), throttle_duration_sec=5.0)

    def _range(self):
        if self.range_source in ('auto', 'distance_sensor') and self.range_m is not None:
            return self.range_m, 'rangefinder'
        if self.range_source in ('auto', 'altitude') and self.pos is not None and self.home is not None:
            return float(np.clip(self.start_height - (self.pos[2] - self.home[2]), 0.04, self.meta['rf_max'])), 'altitude'
        if self.range_source in ('auto', 'altitude') and self.pos is not None and self.home is None:
            self.home = self.pos
        return float(np.clip(self.range_fixed, 0.04, self.meta['rf_max'])), 'fixed'

    # ── policy ──
    def _control(self):
        t = self._now()
        if self.depth_img is None or self.depth_t is None or t - self.depth_t > 0.5:
            return
        q = self.q if self.q is not None else LEVEL_Q
        vel = self.vel if self.vel is not None else (0.0, 0.0, 0.0)
        rng, rsrc = self._range()
        if self.memory == 'fresh':
            self.policy.reset()
            self.stack.reset()
            self.prev_action[:] = 0.0
        t0 = time.perf_counter()
        img = self.stack.push(self.depth_img)
        pro = O.proprio(vel, q, rng, self.prev_action, self.v_max)
        a = np.clip(self.policy(img, pro), -1.0, 1.0).astype(np.float32)
        self.timing.append(time.perf_counter() - t0)
        self.prev_action = a
        vel_sp, yawspeed = O.action_to_setpoint(a[:4], q, self.v_max, self.yaw_max)
        self.last = dict(t=t, a=a, vel_sp=vel_sp, yawspeed=yawspeed, range=rng, range_src=rsrc, q=q, vel=vel)
        self.rows.append(np.concatenate([[t], a[:4], vel_sp, [yawspeed], vel, q, [rng]]).astype(np.float64))
        self.frames.append(self.depth_img.astype(np.float16))
        self._publish(t, a, vel_sp, yawspeed, q)

    def _publish(self, t, a, vel_sp, yawspeed, q):
        stamp = self.get_clock().now().to_msg()
        H_, W_ = self.depth_img.shape
        im = Image()
        im.header.stamp, im.header.frame_id = stamp, self.base_frame
        im.height, im.width, im.encoding, im.is_bigendian, im.step = H_, W_, '32FC1', 0, 4 * W_
        im.data = self.depth_img.astype(np.float32).tobytes()
        self.pub_obs.publish(im)
        self.pub_act.publish(Float32MultiArray(data=[float(c) for c in list(a[:4]) + list(vel_sp) + [yawspeed]]))
        R = O.quat_to_mat(*q)
        v_frd = R.T @ np.asarray(vel_sp, float)
        tw = TwistStamped()
        tw.header.stamp, tw.header.frame_id = stamp, self.base_frame
        tw.twist.linear.x, tw.twist.linear.y, tw.twist.linear.z = float(v_frd[0]), float(-v_frd[1]), float(-v_frd[2])
        tw.twist.angular.z = float(-yawspeed)
        self.pub_cmd.publish(tw)

    def _report(self):
        if self.last is None:
            self.get_logger().info('waiting for depth' + ('' if self.K else ' camera_info') +
                                   ('' if self.reproj or not self.K else ' / TF'), throttle_duration_sec=5.0)
            return
        L = self.last
        a = L['a']
        tm = self.timing[-50:]
        st = {'mode': 'handheld', 'action': [round(float(c), 3) for c in a[:4]],
              'would_command': {'fwd_mps': round(float(a[0] * self.v_max[0]), 2),
                                'right_mps': round(float(a[1] * self.v_max[1]), 2),
                                'up_mps': round(float(-a[2] * self.v_max[2]), 2),
                                'yaw_dps': round(math.degrees(L['yawspeed']), 1)},
              'range_m': round(L['range'], 2), 'range_src': L['range_src'], 'state': self.state_source,
              'speed_mps': round(float(np.linalg.norm(L['vel'])), 2),
              'depth_valid_frac': round(float((self.depth_img > 0).mean()), 3),
              'policy_ms': round(1e3 * float(np.mean(tm)), 2) if tm else None, 'steps': len(self.rows)}
        w = st['would_command']
        self.get_logger().info(f"would command: fwd {w['fwd_mps']:+.2f}  right {w['right_mps']:+.2f}  "
                               f"up {w['up_mps']:+.2f} m/s  yaw {w['yaw_dps']:+.0f} deg/s | range {st['range_m']} m "
                               f"({st['range_src']}) | depth valid {st['depth_valid_frac']:.0%}")
        self.pub_status.publish(String(data=json.dumps(st)))

    def _save(self):
        if not self.rows:
            return
        np.savez_compressed(self.log_path, rows=np.stack(self.rows), frames=np.stack(self.frames),
                            columns=np.array(['t', 'a_fwd', 'a_right', 'a_down', 'a_yaw', 'sp_n', 'sp_e', 'sp_d',
                                              'yawspeed', 'v_n', 'v_e', 'v_d', 'qw', 'qx', 'qy', 'qz', 'range']),
                            policy=self.meta['source'], state_source=self.state_source, memory=self.memory)


def main(argv=None):
    rclpy.init(args=list(argv) if argv is not None else sys.argv)
    node = Handheld()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        node._save()
        node.get_logger().info(f'saved {len(node.rows)} steps to {node.log_path}')
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
