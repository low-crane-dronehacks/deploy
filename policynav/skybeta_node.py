#!/usr/bin/env python3
"""policynav.skybeta_node: fly the policy through the organizers' skybeta flight interface (the real drone).

The skybeta contract (~/skybeta/docs/competitor_guide.md on the drone):
  * we never arm and never publish /fmu/in/*: the pilot arms on RC and switches to Position, and the interface
    then requests Offboard ONCE, but only if our setpoints have been arriving (>= 1 Hz) for the last second.
  * we take off ourselves (a twist with an upward component); PX4 disarms if not airborne 10 s after ARMING.
  * velocity setpoints: /skybeta/setpoint/twist (TwistStamped, ENU m/s, expires after 1 s); |v| is clamped to
    1.5 m/s by the interface (we stay under it ourselves); yaw rate is NOT limited by the interface.
  * /skybeta/hold (Trigger) freezes at the last safe position; the pilot lands (there is no land command).

  in   depth (policynav/depth_node.py: /policynav/depth/image + camera_info), TF base_link <- depth frame
       /fmu/out/vehicle_attitude                                  PX4 FRD->NED attitude
       /fmu/out/vehicle_local_position                            PX4 NED velocity + dist_bottom
       /skybeta/state (JSON)                                         armed / offboard_active
  out  /skybeta/setpoint/twist at 10 Hz: zero until Offboard is active, then the policy's command
       /policynav/status (JSON)

Holds on invalid/stale PX4 inputs or depth, at max_flight_s, or after touchdown.
Stops publishing permanently on stale interface state, disarm, or Offboard loss.
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
from std_msgs.msg import String

from . import handheld as H
from . import obs as O
from .node import decode_depth
from .policy_np import NumpyPolicy, resolve_policy
from .px4_feedback import Px4Feedback
from .transport import LATEST, tf_listener

PX4_QOS = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, durability=DurabilityPolicy.VOLATILE,
                     history=HistoryPolicy.KEEP_LAST, depth=1)


class Skybeta(Node):
    def __init__(self):
        super().__init__('policynav_skybeta')
        dp = lambda n, d: self.declare_parameter(n, d).value      # noqa: E731
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        pf = dp('policy_file', 'policies/cur4_clean50_800.npz')
        self.policy = NumpyPolicy(resolve_policy(pf))
        self.meta = self.policy.meta
        self.n_act = self.meta['num_actions']
        self.v_max, self.yaw_max = self.meta['v_max'], self.meta['yaw_rate_max']
        self.speed_cap = float(dp('speed_cap', 1.45))             # stay under the interface's 1.5 m/s norm clamp
        self.yaw_cap = float(dp('yaw_rate_cap', 0.0)) or float(self.yaw_max)   # 0 = the policy's trained yaw limit; VO dislikes fast spins
        self.max_flight_s = float(dp('max_flight_s', 120.0))
        self.depth_stale_s, self.depth_lost_s = float(dp('depth_stale_s', 0.5)), float(dp('depth_lost_s', 2.0))
        self.feedback_lost_s = float(dp('feedback_lost_s', 0.5))
        self.state_lost_s = float(dp('state_lost_s', 0.5))
        self.feedback = Px4Feedback(self.feedback_lost_s)
        self.twist_frame = dp('twist_frame', 'odom')
        self.base_frame = dp('base_frame', 'base_link')
        mismatch = [round(float(v), 2) for v in self.v_max]
        if math.sqrt(sum(float(v) ** 2 for v in self.v_max)) > 1.5 + 1e-6:
            self.get_logger().warning(f'policy v_max {mismatch} exceeds the 1.5 m/s envelope: commands will be scaled '
                                      'down (the policy was trained faster than it can fly here)')

        from tf2_ros import Buffer
        self.tf_buffer = Buffer()
        # transport: /tf_static only (the depth frame chain is static); tf_static_only:=false restores /tf too
        self.tf_listener = tf_listener(self.tf_buffer, self, bool(dp('tf_static_only', True)))
        from px4_msgs.msg import VehicleAttitude, VehicleLocalPosition
        self.create_subscription(VehicleAttitude, dp('attitude_topic', '/fmu/out/vehicle_attitude'),
                                 self._on_att, PX4_QOS)
        self.create_subscription(VehicleLocalPosition, dp('lpos_topic', '/fmu/out/vehicle_local_position'),
                                 self._on_lpos, PX4_QOS)
        self.create_subscription(String, dp('state_topic', '/skybeta/state'), self._on_state, 10)
        self.create_subscription(CameraInfo, dp('depth_info_topic', '/policynav/depth/camera_info'), self._on_info,
                                 qos_profile_sensor_data)
        self.create_subscription(Image, dp('depth_topic', '/policynav/depth/image'), self._on_depth,
                                 LATEST)
        self.pub = self.create_publisher(TwistStamped, dp('twist_topic', '/skybeta/setpoint/twist'), 10)
        self.pub_status = self.create_publisher(String, '/policynav/status', 10)
        from std_srvs.srv import Trigger
        self.hold_cli = self.create_client(Trigger, dp('hold_service', '/skybeta/hold'))
        self._Trigger = Trigger

        self.state = {}
        self.state_t = None
        self.K = self.reproj = None
        self.depth_img, self.depth_t = None, None
        self.stack = O.FrameStack(self.meta['frame_stack'])
        self.prev_action = np.zeros(self.n_act, np.float32)
        self.phase, self.why = 'wait', 'waiting for Offboard (pilot arms, then Position)'
        self.t_fly = None
        self.airborne = False
        self.cmd = ((0.0, 0.0, 0.0), 0.0)
        self.timing = []
        self.create_timer(0.1, self._tick)
        self.create_timer(1.0, self._status)
        self.get_logger().info(f"SKYBETA: policy {self.meta['source']} (v_max {mismatch}), never arms; "
                               f"publishing zero twists until Offboard")

    def _now(self):
        return time.monotonic()

    # ── inputs ──
    def _on_state(self, m):
        try:
            state = json.loads(m.data)
        except ValueError:
            return
        if not isinstance(state, dict) or any(type(state.get(k)) is not bool for k in ('armed', 'offboard_active')):
            return
        self.state, self.state_t = state, self._now()

    def _on_att(self, m):
        self.feedback.on_att(m, self._now(), active=self.phase == 'fly')

    def _on_lpos(self, m):
        self.feedback.on_lpos(m, self._now(), active=self.phase == 'fly')

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
            Hh, Ww = self.meta['depth_hw']
            self.reproj = O.DepthReprojector(self.K, self.src_hw, O.quat_to_mat(r.w, r.x, r.y, r.z), (t.x, t.y, t.z),
                                             out_hw=(Hh, Ww), hfov_deg=self.meta['hfov_deg'],
                                             vfov_deg=self.meta['vfov_deg'], far=self.meta['far_clip'], stride=2)
            self.get_logger().info(f'depth calibration: optical axis {self.reproj.axis_err_deg:.1f} deg off body x')
        try:
            self.depth_img = self.reproj(decode_depth(m))
            self.depth_t = self._now()
        except ValueError as e:
            self.get_logger().error(str(e), throttle_duration_sec=5.0)

    def _range(self):
        if self.feedback.range_m is None:
            return None
        return float(np.clip(self.feedback.range_m, 0.04, self.meta['rf_max']))

    # ── control ──
    def _hold(self, why):
        self.phase, self.why = 'hold', why
        self.cmd = ((0.0, 0.0, 0.0), 0.0)
        if self.hold_cli.service_is_ready():
            self.hold_cli.call_async(self._Trigger.Request())
        self.get_logger().warning(f'HOLD: {why} (pilot lands)')

    def _tick(self):
        t = self._now()
        state_fresh = self.state_t is not None and 0 <= t - self.state_t <= self.state_lost_s
        offboard = state_fresh and self.state.get('offboard_active', False) and self.state.get('armed', False)
        # A held mission must also yield to pilot takeover; never revive an old command.
        if self.phase in ('fly', 'hold') and not offboard:
            self.phase, self.why = 'yielded', 'interface state lost, Offboard ended or disarmed'
            self.cmd = ((0.0, 0.0, 0.0), 0.0)
        feedback_error = self.feedback.error(t, self.feedback_lost_s)
        if self.phase == 'wait' and offboard:
            if feedback_error or self.depth_t is None or not 0 <= t - self.depth_t <= self.depth_stale_s:
                self._hold(feedback_error or 'Offboard started without fresh depth')
            else:
                self.phase, self.why, self.t_fly = 'fly', 'policy', t
                self.policy.reset()
                self.stack.reset()
                self.prev_action[:] = 0.0
                self.get_logger().info('Offboard active: policy flying')
        if self.phase == 'fly':
            if not offboard:
                self.phase, self.why = 'yielded', 'Offboard ended (pilot took over or disarmed)'
                self.get_logger().info(self.why)
            elif feedback_error:
                self._hold(feedback_error)
            elif self.depth_t is None or t - self.depth_t > self.depth_lost_s:
                self._hold('depth lost')
            elif t - self.t_fly > self.max_flight_s:
                self._hold(f'flight time > {self.max_flight_s:.0f} s')
            elif t - self.depth_t > self.depth_stale_s:
                self.cmd, self.why = ((0.0, 0.0, 0.0), 0.0), 'stale depth: zero velocity'
            else:
                self._step(t)
        self._publish()

    def _step(self, t):
        rng = self._range()
        self.airborne |= rng > 0.3
        if self.airborne and rng < 0.12 and np.linalg.norm(self.feedback.vel) < 0.3:
            self._hold('touched down')
            return
        t0 = time.perf_counter()
        img = self.stack.push(self.depth_img)
        pro = O.proprio(self.feedback.vel, self.feedback.q, rng, self.prev_action, self.meta['v_obs_scale'])
        a = np.clip(self.policy(img, pro), -1.0, 1.0).astype(np.float32)
        self.prev_action = a
        vel_ned, yawspeed = O.action_to_setpoint(a[:4], self.feedback.q, self.v_max, self.yaw_max)
        v_enu, wz = H.ned_to_enu_twist(vel_ned, yawspeed)
        self.cmd = (H.cap_norm(v_enu, self.speed_cap), float(np.clip(wz, -self.yaw_cap, self.yaw_cap)))
        self.timing.append(time.perf_counter() - t0)
        self.why = 'policy'

    def _publish(self):
        if self.phase in ('yielded',):
            return
        (vx, vy, vz), wz = self.cmd
        m = TwistStamped()
        m.header.stamp, m.header.frame_id = self.get_clock().now().to_msg(), self.twist_frame
        m.twist.linear.x, m.twist.linear.y, m.twist.linear.z, m.twist.angular.z = vx, vy, vz, wz
        self.pub.publish(m)

    def _status(self):
        tm = self.timing[-50:]
        (vx, vy, vz), wz = self.cmd
        st = {'mode': 'skybeta', 'phase': self.phase, 'why': self.why,
              'offboard_active': bool(self.state.get('offboard_active', False)), 'armed': self.state.get('armed'),
              'cmd_enu': [round(vx, 2), round(vy, 2), round(vz, 2)], 'yaw_rate': round(wz, 2),
              'range_m': None if self._range() is None else round(self._range(), 2), 'airborne': self.airborne,
              'feedback_error': self.feedback.error(self._now(), self.feedback_lost_s),
              'lpos_age_s': None if self.feedback.lpos_t is None else round(self._now() - self.feedback.lpos_t, 2),
              'attitude_age_s': None if self.feedback.att_t is None else round(self._now() - self.feedback.att_t, 2),
              'state_age_s': None if self.state_t is None else round(self._now() - self.state_t, 2),
              'depth_age_s': None if self.depth_t is None else round(self._now() - self.depth_t, 2),
              'depth_valid_frac': None if self.depth_img is None else round(float((self.depth_img > 0).mean()), 3),
              'policy_ms': round(1e3 * float(np.mean(tm)), 2) if tm else None}
        self.pub_status.publish(String(data=json.dumps(st)))
        if self.phase != 'fly':
            self.get_logger().info(f"{self.phase}: {self.why} | {json.dumps(st)}", throttle_duration_sec=5.0)


def main(argv=None):
    rclpy.init(args=list(argv) if argv is not None else sys.argv)
    node = Skybeta()
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
