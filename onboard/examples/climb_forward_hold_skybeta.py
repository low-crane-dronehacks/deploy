#!/usr/bin/env python3
"""climb_forward_hold for the REAL drone: the same climb -> forward -> hold through the organizers' skybeta flight
interface (docs/SKYBETA.md). climb_forward_hold.py talks to PX4 directly, which is right for SITL / FlyX but is
forbidden on the drone: skybeta's flight_interface owns the Offboard heartbeat and the envelope, and the pilot arms.

    ros2 launch policynav skybeta_climb_forward.launch.py                    # DRY RUN: prints the plan, publishes nothing
    ros2 launch policynav skybeta_climb_forward.launch.py fly:=true          # flies once the PILOT arms + Position

Procedure (competitor_guide.md section 4): the pilot arms in Altitude, then switches to Position on the ground;
flight_interface then requests Offboard once, if our setpoints arrived within the last second. PX4 disarms if not
airborne 10 s after ARMING, so we climb the moment Offboard is active.

  WAIT      offboard_active false: stream zero twists at 10 Hz (keeps the interface's 1 s precondition, stays down)
  CLIMB     twist (ENU) toward start + height, speed ramping down near the target
  FORWARD   twist toward the point forward_m along the heading at Offboard start, same altitude
  HOLD      POSE setpoint at the position where FORWARD ended (all-zero quaternion = hold yaw), for hold_s
  DONE      keep publishing that pose (the vehicle hovers) and tell the pilot to land: there is no land command
  ABORT     Offboard taken away (pilot / failsafe) -> stop publishing. Odometry stale -> /skybeta/hold + stop.
CLIMB / FORWARD timeouts go to HOLD where the vehicle is. Never publishes /fmu/in/*, never arms.
Speeds stay under the interface's 1.5 m/s norm clamp; the twist expires after 1 s if this node stops.
The mission (SkybetaForwardHold) is pure Python, tested in tests/test_climb_forward_hold_skybeta.py.
"""
from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

Vec3 = Tuple[float, float, float]


@dataclass
class Params:
    fly: bool = False               # False = dry run: report the plan, publish nothing
    height: float = 1.0             # m above the Offboard-start position
    forward_m: float = 1.0          # m along the heading at Offboard start
    climb_speed: float = 0.5        # m/s max
    forward_speed: float = 0.5      # m/s max
    decel: float = 0.5              # m/s^2: speed = min(max, sqrt(2 decel d))
    min_speed: float = 0.08
    reach_tol: float = 0.08         # m
    hold_s: float = 5.0
    max_height_m: float = 2.0       # skybeta's own ceiling is 3.0 m above the odometry origin
    max_forward_m: float = 2.5
    climb_timeout_s: float = 20.0
    forward_timeout_s: float = 20.0
    odom_timeout_s: float = 0.5     # cuVSLAM runs at 30 Hz

    def validate(self):
        for k, v in vars(self).items():
            if k != 'fly' and not math.isfinite(float(v)):
                raise ValueError(f'{k} must be finite')
        if not 0.3 <= self.height <= self.max_height_m:
            raise ValueError(f'height must be in [0.3, {self.max_height_m}] m')
        if not 0.0 <= self.forward_m <= self.max_forward_m:
            raise ValueError(f'forward_m must be in [0, {self.max_forward_m}] m')
        if min(self.climb_speed, self.forward_speed, self.decel, self.min_speed, self.reach_tol) <= 0:
            raise ValueError('speeds, decel and reach_tol must be positive')
        if self.climb_speed > 1.4 or self.forward_speed > 1.4:
            raise ValueError('speeds must stay under the interface clamp (1.5 m/s): max 1.4')
        if self.hold_s < 0:
            raise ValueError('hold_s must be >= 0')


# the two helpers below are climb_forward_hold.py's, copied so this file stands alone once installed
def approach_speed(dist: float, v_max: float, decel: float, v_min: float) -> float:
    return max(v_min, min(v_max, math.sqrt(2.0 * decel * max(dist, 0.0))))


def velocity_toward(pos: Vec3, target: Vec3, v_max: float, decel: float, v_min: float) -> Tuple[Vec3, float]:
    d = [target[i] - pos[i] for i in range(3)]
    dist = math.sqrt(sum(c * c for c in d))
    if dist < 1e-9:
        return (0.0, 0.0, 0.0), 0.0
    s = min(approach_speed(dist, v_max, decel, v_min), dist / 0.05)
    return tuple(c / dist * s for c in d), dist


def enu_yaw(qx, qy, qz, qw) -> float:
    """Heading (rad, CCW from east) of a body-FLU -> ENU quaternion."""
    return math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))


@dataclass
class State:
    t: float
    pos: Optional[Vec3] = None      # ENU (odom frame), m
    yaw: float = math.nan           # ENU heading, rad
    odom_age: float = math.inf
    offboard: bool = False
    armed: bool = False


@dataclass
class Output:
    twist: Optional[Vec3] = None    # ENU m/s (publish /skybeta/setpoint/twist)
    pose: Optional[Vec3] = None     # ENU m  (publish /skybeta/setpoint/pose, yaw held)
    hold: bool = False              # call /skybeta/hold once
    events: List[str] = field(default_factory=list)


class SkybetaForwardHold:
    def __init__(self, p: Params):
        p.validate()
        self.p = p
        self.phase, self.why, self.t_phase = 'WAIT', 'waiting for the pilot (arm, then Position)', None
        self.start = self.climb_target = self.forward_target = self.hold_pos = None
        self.yaw = math.nan
        self._events: List[str] = []

    def _set(self, t, phase, why):
        self._events.append(f'{self.phase} -> {phase}: {why}')
        self.phase, self.why, self.t_phase = phase, why, t

    def plan(self, pos: Vec3, yaw: float):
        h, f = self.p.height, self.p.forward_m
        climb = (pos[0], pos[1], pos[2] + h)
        fwd = (climb[0] + f * math.cos(yaw), climb[1] + f * math.sin(yaw), climb[2])
        return climb, fwd

    def plan_text(self, pos: Vec3, yaw: float) -> str:
        c, f = self.plan(pos, yaw)
        return (f'from ENU ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}), heading {math.degrees(yaw):.0f} deg: climb to '
                f'z {c[2]:.2f} at <= {self.p.climb_speed} m/s, fly to ({f[0]:.2f}, {f[1]:.2f}) at <= '
                f'{self.p.forward_speed} m/s, hold {self.p.hold_s:.0f} s, then hover until the pilot lands')

    def step(self, s: State) -> Output:
        out = Output()
        p = self.p
        if self.phase == 'ABORT':
            return self._out(out)
        flying = self.phase in ('CLIMB', 'FORWARD', 'HOLD', 'DONE')
        if flying and not s.offboard:
            self._set(s.t, 'ABORT', 'Offboard ended (pilot took over / failsafe): stopped publishing')
            return self._out(out)
        if flying and (s.pos is None or s.odom_age > p.odom_timeout_s):
            self._set(s.t, 'ABORT', 'odometry stale: /skybeta/hold, stopped publishing')
            out.hold = True
            return self._out(out)
        if self.phase == 'WAIT':
            if not s.offboard:
                out.twist = (0.0, 0.0, 0.0)          # keeps the interface's "setpoint within 1 s" precondition
                return self._out(out)
            if s.pos is None or s.odom_age > p.odom_timeout_s or not math.isfinite(s.yaw):
                self._set(s.t, 'ABORT', 'Offboard started without fresh odometry: /skybeta/hold')
                out.hold = True
                return self._out(out)
            self.start, self.yaw = s.pos, s.yaw
            self.climb_target, self.forward_target = self.plan(s.pos, s.yaw)
            self._set(s.t, 'CLIMB', 'Offboard active: ' + self.plan_text(s.pos, s.yaw))
        if self.phase == 'CLIMB':
            v, d = velocity_toward(s.pos, self.climb_target, p.climb_speed, p.decel, p.min_speed)
            if d <= p.reach_tol:
                self._set(s.t, 'FORWARD', f'climb reached (z {s.pos[2]:.2f})')
            elif s.t - self.t_phase > p.climb_timeout_s:
                return self._begin_hold(s, out, 'climb timeout')
            else:
                out.twist = v
                return self._out(out)
        if self.phase == 'FORWARD':
            v, d = velocity_toward(s.pos, self.forward_target, p.forward_speed, p.decel, p.min_speed)
            if d <= p.reach_tol:
                return self._begin_hold(s, out, 'forward point reached')
            if s.t - self.t_phase > p.forward_timeout_s:
                return self._begin_hold(s, out, 'forward timeout')
            out.twist = v
            return self._out(out)
        if self.phase == 'HOLD' and s.t - self.t_phase >= p.hold_s:
            self._set(s.t, 'DONE', f'held {p.hold_s:.0f} s: PILOT, LAND NOW (hovering on the hold pose)')
        if self.phase in ('HOLD', 'DONE'):
            out.pose = self.hold_pos
        return self._out(out)

    def _begin_hold(self, s, out, why):
        self.hold_pos = s.pos
        self._set(s.t, 'HOLD', f'{why}: holding ({s.pos[0]:.2f}, {s.pos[1]:.2f}, {s.pos[2]:.2f})')
        out.pose = self.hold_pos
        return self._out(out)

    def _out(self, out):
        out.events, self._events = self._events, []
        return out


# ───────────────────────────── ROS 2 node (data in / out only) ─────────────────────────────
try:
    import rclpy
    from geometry_msgs.msg import PoseStamped, TwistStamped
    from nav_msgs.msg import Odometry
    from rclpy.node import Node
    from std_msgs.msg import String
    from std_srvs.srv import Trigger
except ImportError:                                  # the pure part stays importable without ROS (tests)
    Node = object


class ClimbForwardHoldSkybeta(Node):
    def __init__(self):
        super().__init__('climb_forward_hold_skybeta')
        p = Params()
        for k, v in vars(Params()).items():
            val = self.declare_parameter(k, v).value
            if isinstance(v, bool) and not isinstance(val, bool):
                val = str(val).strip().strip('\'"').lower() in ('1', 'true', 'yes')
            setattr(p, k, type(v)(val))
        self.mission = SkybetaForwardHold(p)
        self.p = p
        self.pos, self.yaw, self.odom_t = None, math.nan, None
        self.offboard = self.armed = False
        self.clamped = []
        self.create_subscription(Odometry, '/visual_slam/tracking/odometry', self._on_odom, 10)
        self.create_subscription(String, '/skybeta/state', self._on_state, 10)
        self.pub_twist = self.create_publisher(TwistStamped, '/skybeta/setpoint/twist', 10)
        self.pub_pose = self.create_publisher(PoseStamped, '/skybeta/setpoint/pose', 10)
        self.hold_cli = self.create_client(Trigger, '/skybeta/hold')
        self.planned = False
        self.create_timer(0.1, self._tick)
        self.get_logger().info('climb_forward_hold_skybeta: ' + ('FLY: flies once the pilot arms and switches to '
                               'Position' if p.fly else 'DRY RUN (fly:=false): publishes nothing'))

    def _now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def _on_odom(self, m):
        q, pp = m.pose.pose.orientation, m.pose.pose.position
        if all(math.isfinite(c) for c in (pp.x, pp.y, pp.z, q.x, q.y, q.z, q.w)):
            self.pos, self.yaw, self.odom_t = (pp.x, pp.y, pp.z), enu_yaw(q.x, q.y, q.z, q.w), self._now()

    def _on_state(self, m):
        try:
            st = json.loads(m.data)
        except ValueError:
            return
        self.offboard, self.armed = bool(st.get('offboard_active', False)), bool(st.get('armed', False))
        c = st.get('clamped') or []
        if c and c != self.clamped:
            self.get_logger().warning(f'flight_interface clamped the setpoint: {c}')
        self.clamped = c

    def _tick(self):
        t = self._now()
        if not self.p.fly:                                   # dry run: plan from live odometry, publish nothing
            if self.pos is not None and not self.planned:
                self.get_logger().info('DRY RUN would ' + self.mission.plan_text(self.pos, self.yaw))
                self.planned = True
            return
        s = State(t, self.pos, self.yaw, math.inf if self.odom_t is None else t - self.odom_t, self.offboard,
                  self.armed)
        out = self.mission.step(s)
        for e in out.events:
            self.get_logger().info(e)
        stamp = self.get_clock().now().to_msg()
        if out.twist is not None:
            m = TwistStamped()
            m.header.stamp, m.header.frame_id = stamp, 'odom'
            m.twist.linear.x, m.twist.linear.y, m.twist.linear.z = (float(c) for c in out.twist)
            self.pub_twist.publish(m)
        if out.pose is not None:
            m = PoseStamped()
            m.header.stamp, m.header.frame_id = stamp, 'odom'
            m.pose.position.x, m.pose.position.y, m.pose.position.z = (float(c) for c in out.pose)
            m.pose.orientation.w = 0.0                       # all-zero quaternion: do not command yaw
            self.pub_pose.publish(m)
        if out.hold and self.hold_cli.service_is_ready():
            self.hold_cli.call_async(Trigger.Request())


def main(argv=None):
    rclpy.init(args=list(argv) if argv is not None else sys.argv)
    node = ClimbForwardHoldSkybeta()
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
