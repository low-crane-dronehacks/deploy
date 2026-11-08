#!/usr/bin/env python3
"""SITL stand-in for skybeta's flight_interface (docs/SKYBETA.md "Contract"), so skybeta-style team nodes
(policynav/skybeta_node.py, onboard/examples/climb_forward_hold_skybeta.py) fly in sitl/run_native.sh unchanged.

Written from the documented contract only (skybeta's source is not in this repo); where the docs are silent the choice
is marked APPROX. It never arms and never changes mode except the one Offboard request below.

  in   /skybeta/setpoint/twist  geometry_msgs/TwistStamped, frame odom (cuVSLAM's frame, treated as ENU), m/s, rad/s;
                                expires after 1 s
       /skybeta/setpoint/pose   geometry_msgs/PoseStamped, odom/ENU; all-zero quaternion = do not command yaw;
                                altitude band 0.3-3.0 m (pose only)
       /skybeta/hold            std_srvs/Trigger: freeze where the vehicle is
  out  /skybeta/state           std_msgs/String JSON: armed, offboard_active, clamped, px4_link, nav_state, ...
       /fmu/in/offboard_control_mode + /fmu/in/trajectory_setpoint (20 Hz while setpoints are fresh or holding)
       /fmu/in/vehicle_command  DO_SET_MODE Offboard: ONCE per arming, when armed in Position mode and a setpoint
                                arrived within the last second (the documented precondition)
Envelope: |v| clamped to 1.5 m/s by scaling (reported in "clamped"); yaw rate NOT limited (as documented).
Frames: PX4's NED world must be the odom frame read as ENU (run_native.sh SKYBETA=1 gives the vision bridge heading 0),
so velocity maps by an axis swap: (n, e, d) = (y, x, -z), yawspeed_ned = -wz.
APPROX: an expired twist and /skybeta/hold both hold the current position until the next setpoint arrives.
"""
from __future__ import annotations

import json
import math
import sys

import rclpy
from geometry_msgs.msg import PoseStamped, TwistStamped
from px4_msgs.msg import OffboardControlMode, TrajectorySetpoint, VehicleCommand, VehicleLocalPosition, VehicleStatus
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String
from std_srvs.srv import Trigger

PX4_QOS = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                     history=HistoryPolicy.KEEP_LAST, depth=1)
NAN3 = [math.nan] * 3


def px4_topic(name, msg_type):
    v = int(getattr(msg_type, 'MESSAGE_VERSION', 0))
    return '/fmu/out/' + name + (f'_v{v}' if v else '')


class FlightInterfaceSim(Node):
    def __init__(self):
        super().__init__('flight_interface_sim')
        dp = lambda n, v: self.declare_parameter(n, v).value
        self.v_max, self.expiry = float(dp('max_speed', 1.5)), float(dp('setpoint_timeout_s', 1.0))
        self.alt_lo, self.alt_hi = float(dp('alt_min', 0.3)), float(dp('alt_max', 3.0))
        self.sp = None             # ('vel', (vn, ve, vd), yawspeed) | ('pos', (n, e, d), yaw)
        self.sp_t = -math.inf
        self.hold = None           # (n, e, d) while holding
        self.clamped = []
        self.status = self.lpos = None
        self.lpos_t = -math.inf
        self.requested = False     # Offboard requested during this arming
        self.pub_mode = self.create_publisher(OffboardControlMode, '/fmu/in/offboard_control_mode', PX4_QOS)
        self.pub_sp = self.create_publisher(TrajectorySetpoint, '/fmu/in/trajectory_setpoint', PX4_QOS)
        self.pub_cmd = self.create_publisher(VehicleCommand, '/fmu/in/vehicle_command', PX4_QOS)
        self.pub_state = self.create_publisher(String, '/skybeta/state', 10)
        self.create_subscription(VehicleStatus, px4_topic('vehicle_status', VehicleStatus), self._on_status, PX4_QOS)
        self.create_subscription(VehicleLocalPosition, px4_topic('vehicle_local_position', VehicleLocalPosition),
                                 self._on_lpos, PX4_QOS)
        self.create_subscription(TwistStamped, '/skybeta/setpoint/twist', self._on_twist, 10)
        self.create_subscription(PoseStamped, '/skybeta/setpoint/pose', self._on_pose, 10)
        self.create_service(Trigger, '/skybeta/hold', self._on_hold)
        self.create_timer(0.05, self._tick)
        self.create_timer(0.1, self._publish_state)
        self.get_logger().info('flight_interface_sim: /skybeta/setpoint/{twist,pose} -> PX4 Offboard (SITL stand-in; '
                               'never arms; requests Offboard once per arming when armed in Position mode)')

    def _now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def _on_status(self, m):
        if self.status is not None and self.status.arming_state == VehicleStatus.ARMING_STATE_ARMED \
                and m.arming_state != VehicleStatus.ARMING_STATE_ARMED:
            self.requested = False                                    # a new arming may request again
            self.hold = None
        self.status = m

    def _on_lpos(self, m):
        self.lpos, self.lpos_t = m, self._now()

    def _on_twist(self, m):
        v = (m.twist.linear.x, m.twist.linear.y, m.twist.linear.z)
        if not all(math.isfinite(c) for c in v):
            return
        n = math.sqrt(sum(c * c for c in v))
        self.clamped = ['velocity'] if n > self.v_max else []
        s = self.v_max / n if n > self.v_max else 1.0
        wz = m.twist.angular.z if math.isfinite(m.twist.angular.z) else 0.0
        self.sp = ('vel', (v[1] * s, v[0] * s, -v[2] * s), -wz)       # ENU -> NED
        self.sp_t, self.hold = self._now(), None

    def _on_pose(self, m):
        p, q = m.pose.position, m.pose.orientation
        if not all(math.isfinite(c) for c in (p.x, p.y, p.z)):
            return
        z = min(max(p.z, self.alt_lo), self.alt_hi)
        self.clamped = ['altitude'] if z != p.z else []
        if q.x == q.y == q.z == q.w == 0.0:
            yaw = math.nan                                             # all-zero quaternion: do not command yaw
        else:
            yaw_enu = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
            yaw = math.atan2(math.sin(math.pi / 2 - yaw_enu), math.cos(math.pi / 2 - yaw_enu))
        self.sp = ('pos', (p.y, p.x, -z), yaw)
        self.sp_t, self.hold = self._now(), None

    def _on_hold(self, req, res):
        if self.lpos is None:
            res.success, res.message = False, 'no PX4 position yet'
            return res
        self.hold = (self.lpos.x, self.lpos.y, self.lpos.z)
        res.success, res.message = True, 'holding at ({:.2f}, {:.2f}, {:.2f}) NED'.format(*self.hold)
        self.get_logger().info(res.message)
        return res

    def _armed(self):
        return self.status is not None and self.status.arming_state == VehicleStatus.ARMING_STATE_ARMED

    def _tick(self):
        t = self._now()
        fresh = self.sp is not None and t - self.sp_t <= self.expiry
        if not fresh and self.hold is None and self.sp is not None and self.lpos is not None \
                and self.status is not None and self.status.nav_state == VehicleStatus.NAVIGATION_STATE_OFFBOARD:
            self.hold = (self.lpos.x, self.lpos.y, self.lpos.z)        # APPROX: expired twist -> hold here
            self.get_logger().warning('setpoints expired: holding at ({:.2f}, {:.2f}, {:.2f})'.format(*self.hold))
        if self.hold is not None:
            self._send('pos', self.hold, math.nan, t)
        elif fresh:
            self._send(self.sp[0], self.sp[1], self.sp[2], t)
        else:
            return
        if (fresh and not self.requested and self._armed()
                and self.status.nav_state == VehicleStatus.NAVIGATION_STATE_POSCTL):
            self._command(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, 1.0, 6.0)   # PX4 custom main mode 6 = Offboard
            self.requested = True
            self.get_logger().info('armed in Position with fresh setpoints: Offboard requested (once)')

    def _send(self, kind, xyz, yaw_or_rate, t):
        us = self.get_clock().now().nanoseconds // 1000
        mode = OffboardControlMode()
        mode.timestamp = us
        mode.position, mode.velocity = kind == 'pos', kind == 'vel'
        self.pub_mode.publish(mode)
        sp = TrajectorySetpoint()
        sp.timestamp = us
        sp.position = [float(c) for c in xyz] if kind == 'pos' else NAN3
        sp.velocity = [float(c) for c in xyz] if kind == 'vel' else NAN3
        sp.acceleration = NAN3
        sp.yaw = float(yaw_or_rate) if kind == 'pos' else math.nan
        sp.yawspeed = float(yaw_or_rate) if kind == 'vel' else math.nan
        self.pub_sp.publish(sp)

    def _command(self, cmd, p1=0.0, p2=0.0):
        m = VehicleCommand()
        m.timestamp = self.get_clock().now().nanoseconds // 1000
        m.command, m.param1, m.param2 = cmd, float(p1), float(p2)
        m.target_system = m.target_component = m.source_system = m.source_component = 1
        m.from_external = True
        self.pub_cmd.publish(m)

    def _publish_state(self):
        s = self.status
        st = {'armed': self._armed(),
              'offboard_active': bool(s is not None and s.nav_state == VehicleStatus.NAVIGATION_STATE_OFFBOARD),
              'clamped': self.clamped, 'px4_link': self._now() - self.lpos_t < 1.0,
              'nav_state': int(s.nav_state) if s is not None else None, 'holding': self.hold is not None,
              'offboard_requested': self.requested, 'source': 'sitl/flight_interface_sim.py'}
        self.pub_state.publish(String(data=json.dumps(st)))


def main():
    rclpy.init(args=sys.argv)
    node = FlightInterfaceSim()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
