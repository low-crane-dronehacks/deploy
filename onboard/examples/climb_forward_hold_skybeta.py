#!/usr/bin/env python3
"""Skybeta climb -> body-forward -> stop, with PX4 fused feedback and velocity commands only.

The organizer flight_interface owns all /fmu/in/* publications and Offboard activation.
This node never arms. The pilot arms in Altitude, switches to Position, and lands.

Feedback: configurable VehicleLocalPosition topic, fused NED velocity and heading. Relative
height and forward travel are integrated from MEASURED velocity using PX4 sample timestamps;
no VIO position, VIO orientation, or absolute position targets are used. Integration drifts.
Forward/right follow CURRENT heading; vertical stays gravity-aligned. Commands are rotated
into the interface's fixed ENU frame, not published with a misleading body frame_id.

WAIT streams zero velocity until Offboard. CLIMB goes up height metres; FORWARD travels
forward_m along the changing heading while correcting integrated height. HOLD and DONE
stream zero velocity indefinitely (not a fixed-position hold). Stage timeouts also stop.
Bad/stale/reset feedback aborts to zero velocity until pilot takeover. Offboard loss stops
publishing permanently. Dry run prints the plan and publishes nothing.
"""
from __future__ import annotations

import json
import math
import sys
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

Vec3 = Tuple[float, float, float]
ZERO = (0.0, 0.0, 0.0)


@dataclass
class Params:
    fly: bool = False
    height: float = 1.0
    forward_m: float = 1.0
    climb_speed: float = 0.5
    forward_speed: float = 0.5
    decel: float = 0.5
    min_speed: float = 0.08
    reach_tol: float = 0.08
    hold_s: float = 5.0
    max_height_m: float = 2.0
    max_forward_m: float = 2.5
    climb_timeout_s: float = 20.0
    forward_timeout_s: float = 20.0
    feedback_timeout_s: float = 0.5

    def validate(self):
        for k, v in vars(self).items():
            if k != 'fly' and not math.isfinite(float(v)):
                raise ValueError(f'{k} must be finite')
        if not 0.3 <= self.height <= self.max_height_m:
            raise ValueError(f'height must be in [0.3, {self.max_height_m}] m')
        if not 0.0 <= self.forward_m <= self.max_forward_m:
            raise ValueError(f'forward_m must be in [0, {self.max_forward_m}] m')
        if min(self.climb_speed, self.forward_speed, self.decel, self.min_speed, self.reach_tol,
               self.climb_timeout_s, self.forward_timeout_s, self.feedback_timeout_s) <= 0:
            raise ValueError('speeds, decel, tolerance and timeouts must be positive')
        if math.hypot(self.climb_speed, self.forward_speed) > 1.4:
            raise ValueError('combined climb/forward speed must be <= 1.4 m/s (interface limit 1.5)')
        if self.min_speed > min(self.climb_speed, self.forward_speed):
            raise ValueError('min_speed must not exceed the speed caps')
        if self.hold_s < 0:
            raise ValueError('hold_s must be >= 0')


def body_to_enu(forward: float, right: float, down: float, heading_ned: float) -> Vec3:
    """Yaw-aligned FRD velocity -> fixed ENU, using PX4 heading (clockwise from north)."""
    c, s = math.cos(heading_ned), math.sin(heading_ned)
    return forward * s + right * c, forward * c - right * s, -down


@dataclass
class Feedback:
    sample_t: float                         # PX4 sample time, seconds; only differences are used
    velocity_ned: Vec3
    heading_ned: float
    valid: bool = True
    resets: Tuple[int, int, int] = (0, 0, 0)  # horizontal velocity, vertical velocity, heading

    def usable(self):
        return (self.valid and self.sample_t > 0 and
                all(math.isfinite(v) for v in (*self.velocity_ned, self.heading_ned, self.sample_t)))

    def forward_speed(self):
        vn, ve, _ = self.velocity_ned
        return vn * math.cos(self.heading_ned) + ve * math.sin(self.heading_ned)


def feedback_from_px4(m) -> Feedback:
    return Feedback(float(m.timestamp_sample) * 1e-6, (float(m.vx), float(m.vy), float(m.vz)),
                    float(m.heading), bool(m.v_xy_valid and m.v_z_valid and m.heading_good_for_control),
                    (int(m.vxy_reset_counter), int(m.vz_reset_counter), int(m.heading_reset_counter)))


@dataclass
class State:
    t: float
    feedback: Optional[Feedback] = None
    feedback_age: float = math.inf
    offboard: bool = False
    armed: bool = False
    feedback_fault: bool = False


@dataclass
class Output:
    twist: Optional[Vec3] = None            # ENU m/s for /skybeta/setpoint/twist
    events: List[str] = field(default_factory=list)


class SkybetaForwardHold:
    def __init__(self, p: Params):
        p.validate()
        self.p = p
        self.phase, self.t_phase = 'WAIT', None
        self.height_m = self.forward_m = 0.0
        self.previous = None
        self.abort_zero = False
        self._events: List[str] = []

    def _set(self, t, phase, why):
        self._events.append(f'{self.phase} -> {phase}: {why}')
        self.phase, self.t_phase = phase, t

    def plan_text(self):
        return (f'PX4 fused velocity + heading: climb {self.p.height:.2f} m at <= {self.p.climb_speed} m/s; '
                f'body-forward {self.p.forward_m:.2f} m at <= {self.p.forward_speed} m/s; '
                f'zero velocity {self.p.hold_s:.0f} s, then keep zero velocity until the pilot lands. '
                'Distances integrate measured velocity; forward follows current heading.')

    def _abort(self, s, why, zero=True):
        self._set(s.t, 'ABORT', why)
        self.abort_zero = zero
        return self._out(Output(twist=ZERO if zero else None))

    def _speed(self, error, cap):
        if abs(error) <= self.p.reach_tol:
            return 0.0
        speed = max(self.p.min_speed, min(cap, math.sqrt(2.0 * self.p.decel * abs(error))))
        return math.copysign(min(speed, abs(error) / 0.1), error)

    def step(self, s: State) -> Output:
        p, fb = self.p, s.feedback
        if self.phase == 'ABORT':
            if not s.offboard or not s.armed:
                self.abort_zero = False
            return self._out(Output(twist=ZERO if self.abort_zero else None))
        flying = self.phase != 'WAIT'
        if flying and (not s.offboard or not s.armed):
            return self._abort(s, 'Offboard ended or disarmed: stopped publishing', zero=False)
        if self.phase == 'WAIT' and not s.offboard:
            return self._out(Output(twist=ZERO))
        if (not s.armed or fb is None or not fb.usable() or s.feedback_fault or
                not 0 <= s.feedback_age <= p.feedback_timeout_s):
            return self._abort(s, 'PX4 feedback invalid/stale/reset: zero velocity; PILOT TAKE OVER')
        if self.phase == 'WAIT':
            self.previous = fb
            self._set(s.t, 'CLIMB', 'Offboard active: ' + self.plan_text())
        else:
            dt = fb.sample_t - self.previous.sample_t
            if fb.resets != self.previous.resets or not 0 <= dt <= p.feedback_timeout_s:
                return self._abort(s, 'PX4 estimate reset or sample time discontinuity: zero velocity; PILOT TAKE OVER')
            if dt > 0:
                if self.phase in ('CLIMB', 'FORWARD'):
                    self.height_m -= 0.5 * (self.previous.velocity_ned[2] + fb.velocity_ned[2]) * dt
                if self.phase == 'FORWARD':
                    self.forward_m += 0.5 * (self.previous.forward_speed() + fb.forward_speed()) * dt
                self.previous = fb
        if self.phase == 'CLIMB':
            if self.height_m >= p.height - p.reach_tol:
                self._set(s.t, 'FORWARD', f'climb reached (integrated rise {self.height_m:.2f} m)')
            elif s.t - self.t_phase > p.climb_timeout_s:
                self._set(s.t, 'HOLD', 'climb timeout: zero velocity')
            else:
                return self._out(Output(twist=body_to_enu(0.0, 0.0,
                    -self._speed(p.height - self.height_m, p.climb_speed), fb.heading_ned)))
        if self.phase == 'FORWARD':
            if self.forward_m >= p.forward_m - p.reach_tol:
                self._set(s.t, 'HOLD', f'forward travel reached ({self.forward_m:.2f} m): zero velocity')
            elif s.t - self.t_phase > p.forward_timeout_s:
                self._set(s.t, 'HOLD', 'forward timeout: zero velocity')
            else:
                return self._out(Output(twist=body_to_enu(
                    self._speed(p.forward_m - self.forward_m, p.forward_speed), 0.0,
                    -self._speed(p.height - self.height_m, p.climb_speed), fb.heading_ned)))
        if self.phase == 'HOLD' and s.t - self.t_phase >= p.hold_s:
            self._set(s.t, 'DONE', 'PILOT, LAND NOW (continuing zero velocity)')
        return self._out(Output(twist=ZERO))

    def _out(self, out):
        out.events, self._events = self._events, []
        return out


# ROS boundary: PX4 feedback in, organizer velocity commands out. No /fmu/in publishers.
try:
    import rclpy
    from geometry_msgs.msg import TwistStamped
    from px4_msgs.msg import VehicleLocalPosition
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
    from std_msgs.msg import String
except ImportError:
    Node = object


class ClimbForwardHoldSkybeta(Node):
    def __init__(self):
        super().__init__('climb_forward_hold_skybeta')
        p = Params()
        for k, v in vars(p).items():
            val = self.declare_parameter(k, v).value
            if isinstance(v, bool) and not isinstance(val, bool):
                val = str(val).strip().lower() in ('1', 'true', 'yes')
            setattr(p, k, type(v)(val))
        self.mission, self.p = SkybetaForwardHold(p), p
        self.feedback = self.feedback_t = self.state_t = None
        self.feedback_fault = False
        self.offboard = self.armed = False
        self.planned = False
        self.clamped = []
        topic = self.declare_parameter('px4_feedback_topic', '/fmu/out/vehicle_local_position').value
        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, durability=DurabilityPolicy.VOLATILE,
                         history=HistoryPolicy.KEEP_LAST, depth=1)
        self.create_subscription(VehicleLocalPosition, topic, self._on_feedback, qos)
        self.create_subscription(String, '/skybeta/state', self._on_state, 10)
        self.pub_twist = self.create_publisher(TwistStamped, '/skybeta/setpoint/twist', 10) if p.fly else None
        self.create_timer(0.1, self._tick)
        self.get_logger().info('climb_forward_hold_skybeta: ' +
            ('FLY: waits for pilot arm + Position' if p.fly else 'DRY RUN: publishes nothing') +
            f'; PX4 fused feedback: {topic}')

    def _on_feedback(self, m):
        fb, now = feedback_from_px4(m), time.monotonic()
        previous = self.feedback
        active = self.mission.phase not in ('WAIT', 'ABORT')
        if active and (not fb.usable() or (previous is not None and
                (fb.resets != previous.resets or fb.sample_t < previous.sample_t or
                 fb.sample_t - previous.sample_t > self.p.feedback_timeout_s))):
            self.feedback_fault = True
        # A repeated PX4 sample must not keep feedback artificially fresh.
        if previous is not None and fb.sample_t == previous.sample_t and fb.usable() and fb.resets == previous.resets:
            return
        self.feedback, self.feedback_t = fb, now

    def _on_state(self, m):
        try:
            st = json.loads(m.data)
            if not isinstance(st, dict):
                return
        except ValueError:
            return
        self.offboard, self.armed = bool(st.get('offboard_active', False)), bool(st.get('armed', False))
        self.state_t = time.monotonic()
        c = st.get('clamped') or []
        if c and c != self.clamped:
            self.get_logger().warning(f'flight_interface clamped the setpoint: {c}')
        self.clamped = c

    def _tick(self):
        now = time.monotonic()
        age = math.inf if self.feedback_t is None else now - self.feedback_t
        if not self.p.fly:
            if not self.planned and self.feedback is not None and self.feedback.usable() and age <= self.p.feedback_timeout_s:
                self.get_logger().info('DRY RUN would ' + self.mission.plan_text())
                self.planned = True
            return
        state_fresh = self.state_t is not None and now - self.state_t <= self.p.feedback_timeout_s
        out = self.mission.step(State(now, self.feedback, age, self.offboard and state_fresh,
                                      self.armed and state_fresh, self.feedback_fault))
        for event in out.events:
            self.get_logger().info(event)
        if out.twist is not None:
            m = TwistStamped()
            m.header.stamp = self.get_clock().now().to_msg()
            m.header.frame_id = 'odom'  # interface expects fixed ENU; command has already been rotated
            m.twist.linear.x, m.twist.linear.y, m.twist.linear.z = (float(v) for v in out.twist)
            self.pub_twist.publish(m)


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
