#!/usr/bin/env python3
"""Climb, move forward with VELOCITY commands, then HOLD a position setpoint wherever the vehicle ended up, then land.

    python3 onboard/examples/climb_forward_hold.py                       # dry run (default): publishes NOTHING
    python3 onboard/examples/climb_forward_hold.py --ros-args -p arm:=true -p forward_m:=1.0   # flies (team approval!)

Sequence (PX4 Offboard, NED):
  WAIT      valid PX4 position + heading, not armed; the plan is fixed from here: climb straight up by `height`,
            then `forward_m` along the takeoff heading
  PRIME     >= 2 s of Offboard heartbeats (zero velocity) before asking for Offboard + arm
  ACTIVATE  wait for armed + Offboard
  CLIMB     velocity toward (start x, y, start z - height); speed ramps down near the target
  FORWARD   velocity toward the forward point at the same altitude, same ramp
  HOLD      POSITION setpoint at the position where FORWARD finished (not the planned point), for `hold_s`
  LAND      PX4 AUTO_LAND; done once landed and disarmed
Velocity phases aim at the 3D target point, so sideways / vertical drift is steered out, not just the along-track part.

Safety: telemetry older than 1 s, or Offboard / arming taken away (pilot, failsafe) -> stop publishing and yield
(ABORT; PX4's own Offboard-loss failsafe applies). Position estimate invalid while flying, or a phase timeout -> LAND.
`forward_m` and `height` are capped by `max_forward_m` / `max_height_m`. The mission logic (ForwardHoldMission) is
pure Python, tested in tests/test_climb_forward_hold.py; the ROS node only moves data in and out.
"""
from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

Vec3 = Tuple[float, float, float]


# ───────────────────────────── pure mission logic (no ROS) ─────────────────────────────

@dataclass
class Params:
    arm: bool = False               # False = dry run: report the plan, publish nothing
    height: float = 1.0             # m above the takeoff point
    forward_m: float = 1.0          # m along the takeoff heading
    climb_speed: float = 0.5        # m/s, maximum
    forward_speed: float = 0.5      # m/s, maximum
    decel: float = 0.5              # m/s^2: speed = min(max, sqrt(2 decel d)) so it arrives slowly
    min_speed: float = 0.08         # m/s floor so the last centimetres are covered
    reach_tol: float = 0.08         # m: phase target reached
    hold_s: float = 5.0             # s holding the position setpoint before landing
    max_height_m: float = 2.0       # caps (the arena is 3.05 m high, 6.1 m across)
    max_forward_m: float = 2.5
    prime_s: float = 2.0            # PX4 needs > 1 s of Offboard heartbeats before the switch
    activate_timeout_s: float = 8.0
    climb_timeout_s: float = 20.0
    forward_timeout_s: float = 20.0
    preflight_timeout_s: float = 45.0
    telemetry_timeout_s: float = 1.0

    def validate(self):
        for k, v in vars(self).items():
            if k != 'arm' and not math.isfinite(float(v)):
                raise ValueError(f'{k} must be finite')
        if not 0.3 <= self.height <= self.max_height_m:
            raise ValueError(f'height must be in [0.3, {self.max_height_m}] m')
        if not 0.0 <= self.forward_m <= self.max_forward_m:
            raise ValueError(f'forward_m must be in [0, {self.max_forward_m}] m')
        if min(self.climb_speed, self.forward_speed, self.decel, self.min_speed, self.reach_tol) <= 0:
            raise ValueError('speeds, decel and reach_tol must be positive')
        if self.climb_speed > 1.5 or self.forward_speed > 1.5:
            raise ValueError('speeds above 1.5 m/s are not allowed here')
        if self.hold_s < 0:
            raise ValueError('hold_s must be >= 0')


@dataclass
class Telemetry:
    t: float                         # s, node clock
    pos: Optional[Vec3] = None       # NED m
    heading: float = math.nan        # rad, NED yaw
    pos_ok: bool = False             # xy/z valid, finite, heading usable for control
    pos_age: float = math.inf        # s since the last position message
    status_age: float = math.inf     # s since the last status message
    armed: bool = False
    offboard: bool = False
    auto_land: bool = False          # nav_state == AUTO_LAND
    landed: Optional[bool] = None    # land detector (None = never received)
    landed_age: float = math.inf


@dataclass
class Output:
    mode: Optional[str] = None       # None = publish nothing | 'velocity' | 'position'
    velocity: Vec3 = (0.0, 0.0, 0.0)
    position: Vec3 = (math.nan, math.nan, math.nan)
    yaw: float = math.nan
    commands: List[str] = field(default_factory=list)   # 'offboard', 'arm', 'land'


def approach_speed(dist: float, v_max: float, decel: float, v_min: float) -> float:
    """Speed for a target `dist` away: capped at v_max, slowing so it could stop at the target, never below v_min."""
    return max(v_min, min(v_max, math.sqrt(2.0 * decel * max(dist, 0.0))))


def velocity_toward(pos: Vec3, target: Vec3, v_max: float, decel: float, v_min: float) -> Tuple[Vec3, float]:
    """-> (NED velocity aimed straight at target, distance)."""
    d = [target[i] - pos[i] for i in range(3)]
    dist = math.sqrt(sum(c * c for c in d))
    if dist < 1e-9:
        return (0.0, 0.0, 0.0), 0.0
    s = min(approach_speed(dist, v_max, decel, v_min), dist / 0.05)   # never ask for more than one tick's distance
    return tuple(c / dist * s for c in d), dist


class ForwardHoldMission:
    def __init__(self, p: Params):
        p.validate()
        self.p = p
        self.phase, self.why = 'WAIT', 'starting'
        self.t0 = self.t_phase = None
        self.events: List[str] = []
        self.start: Optional[Vec3] = None
        self.yaw = math.nan
        self.climb_target = self.forward_target = self.hold_pos = None
        self.last_land_cmd = -math.inf
        self.auto_land_seen = False
        self.ground_wait_t = None

    def _set(self, t, phase, why):
        self.events.append(f'{self.phase} -> {phase}: {why}')
        self.phase, self.why, self.t_phase = phase, why, t

    def plan_text(self) -> str:
        s, c, f = self.start, self.climb_target, self.forward_target
        return (f'start NED ({s[0]:.2f}, {s[1]:.2f}, {s[2]:.2f}), heading {math.degrees(self.yaw):.1f} deg; climb to '
                f'z {c[2]:.2f}, then fly to ({f[0]:.2f}, {f[1]:.2f}) at {self.p.forward_speed:.2f} m/s max, '
                f'hold {self.p.hold_s:.1f} s, land')

    def step(self, tel: Telemetry) -> Output:
        p, t = self.p, tel.t
        if self.t0 is None:
            self.t0 = self.t_phase = t
        if self.phase in ('DONE', 'ABORT'):
            return Output()

        # stale telemetry: before any request just wait; afterwards stop publishing and let PX4's failsafe act
        if tel.pos_age > p.telemetry_timeout_s or tel.status_age > p.telemetry_timeout_s:
            if self.phase in ('WAIT', 'PRIME'):
                if self.phase == 'PRIME':
                    self._set(t, 'WAIT', 'telemetry stale while priming')
                return self._preflight_timeout(t)
            self._set(t, 'ABORT', 'telemetry lost; stopped publishing')
            return Output()

        if self.phase == 'WAIT':
            if tel.armed or not tel.pos_ok:
                return self._preflight_timeout(t)
            x, y, z = tel.pos
            self.start, self.yaw = (x, y, z), tel.heading
            self.climb_target = (x, y, z - p.height)
            self.forward_target = (x + p.forward_m * math.cos(self.yaw), y + p.forward_m * math.sin(self.yaw),
                                   z - p.height)
            if not p.arm:
                self._set(t, 'DONE', 'dry run (arm:=false): would ' + self.plan_text())
                return Output()
            self._set(t, 'PRIME', 'position valid; ' + self.plan_text())

        if self.phase == 'PRIME':
            if tel.armed:
                self._set(t, 'ABORT', 'armed by someone else before our request; yielding')
                return Output()
            if not tel.pos_ok:
                self._set(t, 'WAIT', 'position / heading lost while priming')
                return Output()
            out = Output('velocity', (0.0, 0.0, 0.0), yaw=self.yaw)
            if t - self.t_phase >= p.prime_s:
                out.commands = ['offboard', 'arm']
                self._set(t, 'ACTIVATE', 'Offboard + arm requested')
            return out

        if self.phase == 'LAND':
            return self._land(tel)

        # from here we have asked for control: losing it means a pilot / failsafe took over
        if self.phase in ('CLIMB', 'FORWARD', 'HOLD') and not (tel.armed and tel.offboard):
            self._set(t, 'ABORT', 'Offboard / arming taken away; yielding to PX4 / pilot')
            return Output()
        if self.phase in ('CLIMB', 'FORWARD', 'HOLD') and not tel.pos_ok:
            return self._begin_land(t, 'position estimate invalid', None)

        if self.phase == 'ACTIVATE':
            out = Output('velocity', (0.0, 0.0, 0.0), yaw=self.yaw)
            if tel.armed and tel.offboard:
                self._set(t, 'CLIMB', 'armed in Offboard; climbing')
            elif t - self.t_phase > p.activate_timeout_s:
                if tel.armed:
                    return self._begin_land(t, 'Offboard timeout while armed', None)
                self._set(t, 'ABORT', 'arm / Offboard timeout')
                return Output()
            return out

        if self.phase == 'CLIMB':
            v, d = velocity_toward(tel.pos, self.climb_target, p.climb_speed, p.decel, p.min_speed)
            if d < p.reach_tol:
                self._set(t, 'FORWARD', f'reached height (z {tel.pos[2]:.2f})')
            elif t - self.t_phase > p.climb_timeout_s:
                return self._begin_land(t, 'climb timeout', tel.pos)
            else:
                return Output('velocity', v, yaw=self.yaw)

        if self.phase == 'FORWARD':
            v, d = velocity_toward(tel.pos, self.forward_target, p.forward_speed, p.decel, p.min_speed)
            if d < p.reach_tol:
                self.hold_pos = tuple(tel.pos)
                self._set(t, 'HOLD', 'forward point reached; holding at ({:.2f}, {:.2f}, {:.2f})'.format(*tel.pos))
            elif t - self.t_phase > p.forward_timeout_s:
                return self._begin_land(t, 'forward timeout', tel.pos)
            else:
                return Output('velocity', v, yaw=self.yaw)

        if self.phase == 'HOLD':
            if t - self.t_phase >= p.hold_s:
                return self._begin_land(t, f'held {p.hold_s:.1f} s', self.hold_pos)
            return Output('position', position=self.hold_pos, yaw=self.yaw)
        return Output()

    def _preflight_timeout(self, t) -> Output:
        if t - self.t0 >= self.p.preflight_timeout_s:
            self._set(t, 'ABORT', 'preflight timeout: no fresh valid position / heading; nothing was requested')
        return Output()

    def _begin_land(self, t, why, hold_xyz) -> Output:
        # hold_xyz: last trusted position to hold until PX4 is in AUTO_LAND (None = none trusted)
        self.hold_pos = tuple(hold_xyz) if hold_xyz is not None else None
        self.last_land_cmd = -math.inf
        self._set(t, 'LAND', why)
        return Output()

    def _land(self, tel: Telemetry) -> Output:
        t = tel.t
        if tel.armed and tel.auto_land:
            self.auto_land_seen = True
        if not tel.armed:
            if tel.landed is True and tel.landed_age <= 1.0:
                self._set(t, 'DONE', 'landed and disarmed')
            elif not tel.auto_land and not self.auto_land_seen:
                self._set(t, 'ABORT', 'disarmed without AUTO_LAND or ground confirmation; inspect the vehicle')
            else:
                self.ground_wait_t = self.ground_wait_t if self.ground_wait_t is not None else t
                if t - self.ground_wait_t >= 5.0:
                    self._set(t, 'ABORT', 'disarmed without a fresh ground report; inspect the vehicle')
            return Output()
        if tel.auto_land:
            return Output()                    # PX4 owns the descent
        if self.auto_land_seen:
            self._set(t, 'ABORT', 'mode changed after AUTO_LAND began; yielding')
            return Output()
        if not tel.offboard:
            self._set(t, 'ABORT', 'mode changed before landing; yielding to PX4 / pilot')
            return Output()
        # Hold the last trusted point while PX4 switches to AUTO_LAND. With no trusted point (position went invalid
        # before HOLD) stream nothing: a setpoint from a bad estimate is worse than PX4's own failsafe.
        out = Output('position', position=self.hold_pos, yaw=self.yaw) if self.hold_pos is not None else Output()
        if t - self.last_land_cmd > 1.0:
            out.commands = ['land']
            self.last_land_cmd = t
        return out


# ───────────────────────────── ROS 2 / PX4 node ─────────────────────────────

try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
    from px4_msgs.msg import (EstimatorStatusFlags, OffboardControlMode, TrajectorySetpoint, VehicleCommand,
                              VehicleLandDetected, VehicleLocalPosition, VehicleStatus)
    _ROS = True
except ImportError:       # the mission logic above stays importable (and testable) without ROS / px4_msgs
    Node = object
    _ROS = False


def px4_topic(name, message_type):
    """Versioned PX4 output topic for the installed px4_msgs (e.g. vehicle_local_position_v1), as takeoff_land.py."""
    version = int(getattr(message_type, 'MESSAGE_VERSION', 0))
    return '/fmu/out/' + name + (f'_v{version}' if version else '')


class ClimbForwardHold(Node):
    def __init__(self):
        super().__init__('climb_forward_hold')
        p = Params()
        for k, v in vars(Params()).items():
            val = self.declare_parameter(k, v).value
            if isinstance(v, bool) and not isinstance(val, bool):     # 'false' must not become True
                val = str(val).strip().strip('\'"').lower() in ('1', 'true', 'yes')
            setattr(p, k, type(v)(val))
        self.mission = ForwardHoldMission(p)

        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                         history=HistoryPolicy.KEEP_LAST, depth=1)
        self.pub_mode = self.create_publisher(OffboardControlMode, '/fmu/in/offboard_control_mode', qos)
        self.pub_sp = self.create_publisher(TrajectorySetpoint, '/fmu/in/trajectory_setpoint', qos)
        self.pub_cmd = self.create_publisher(VehicleCommand, '/fmu/in/vehicle_command', qos)
        self.create_subscription(VehicleLocalPosition, px4_topic('vehicle_local_position', VehicleLocalPosition),
                                 self._on_pos, qos)
        self.create_subscription(VehicleStatus, px4_topic('vehicle_status', VehicleStatus), self._on_status, qos)
        self.create_subscription(VehicleLandDetected, px4_topic('vehicle_land_detected', VehicleLandDetected),
                                 self._on_landed, qos)
        self.create_subscription(EstimatorStatusFlags, px4_topic('estimator_status_flags', EstimatorStatusFlags),
                                 self._on_flags, qos)
        self.pos = self.status = None
        self.pos_t = self.status_t = self.landed_t = -math.inf
        self.landed = None
        self.yaw_aligned, self.flags_t, self.flags_stamp = False, None, 0
        self.last_wait_log = -math.inf
        self.create_timer(0.05, self._tick)          # 20 Hz: setpoint stream + mission step
        self.get_logger().info(f'climb_forward_hold: arm={p.arm} height {p.height} m, forward {p.forward_m} m, '
                               f'hold {p.hold_s} s' + ('' if p.arm else ' (DRY RUN: publishes nothing)'))

    def _now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def _on_pos(self, m):
        self.pos, self.pos_t = m, self._now()

    def _on_status(self, m):
        self.status, self.status_t = m, self._now()

    def _on_landed(self, m):
        self.landed, self.landed_t = bool(m.landed), self._now()

    def _on_flags(self, m):
        self.yaw_aligned, self.flags_t, self.flags_stamp = bool(m.cs_yaw_align), self._now(), int(m.timestamp)

    def _pos_ok(self, now) -> bool:
        """Same rule as takeoff_land.py: valid xy / z, finite, and a heading PX4 considers usable (or freshly
        yaw-aligned on the ground)."""
        m = self.pos
        if m is None:
            return False
        age_us = int(m.timestamp) - self.flags_stamp
        aligned = (self.yaw_aligned and self.flags_t is not None and 0 <= now - self.flags_t <= 3.0
                   and self.flags_stamp > 0 and 0 <= age_us <= 3_000_000)
        return bool(m.xy_valid and m.z_valid and (m.heading_good_for_control or aligned)
                    and all(math.isfinite(c) for c in (m.x, m.y, m.z, m.heading)))

    def _tick(self):
        now = self._now()
        s = self.status
        tel = Telemetry(t=now, pos_age=now - self.pos_t, status_age=now - self.status_t,
                        landed=self.landed, landed_age=now - self.landed_t)
        if self.pos is not None:
            tel.pos = (float(self.pos.x), float(self.pos.y), float(self.pos.z))
            tel.heading = float(self.pos.heading)
            tel.pos_ok = self._pos_ok(now)
        if s is not None:
            tel.armed = s.arming_state == VehicleStatus.ARMING_STATE_ARMED
            tel.offboard = s.nav_state == VehicleStatus.NAVIGATION_STATE_OFFBOARD
            tel.auto_land = s.nav_state == VehicleStatus.NAVIGATION_STATE_AUTO_LAND
        if self.mission.phase == 'WAIT' and now - self.last_wait_log >= 5.0:
            self.get_logger().info('waiting for valid PX4 position + heading (not armed)')
            self.last_wait_log = now
        out = self.mission.step(tel)
        for e in self.mission.events:
            self.get_logger().info(e)
        self.mission.events.clear()
        if out.mode is not None:
            self._publish(out)
        for c in out.commands:
            if c == 'offboard':
                self._command(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, 1.0, 6.0)
            elif c == 'arm':
                self._command(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, 1.0)
            elif c == 'land':
                self._command(VehicleCommand.VEHICLE_CMD_NAV_LAND)

    def _publish(self, out: Output):
        us = self.get_clock().now().nanoseconds // 1000
        mode = OffboardControlMode()
        mode.timestamp = us
        mode.position = out.mode == 'position'
        mode.velocity = out.mode == 'velocity'
        self.pub_mode.publish(mode)
        sp = TrajectorySetpoint()
        sp.timestamp = us
        nan3 = [math.nan] * 3
        sp.position = [float(c) for c in out.position] if out.mode == 'position' else nan3
        sp.velocity = [float(c) for c in out.velocity] if out.mode == 'velocity' else nan3
        sp.acceleration = nan3
        sp.yaw = float(out.yaw)            # hold the takeoff heading
        sp.yawspeed = math.nan
        self.pub_sp.publish(sp)

    def _command(self, cmd, p1=0.0, p2=0.0):
        m = VehicleCommand()
        m.timestamp = self.get_clock().now().nanoseconds // 1000
        m.command, m.param1, m.param2 = cmd, float(p1), float(p2)
        m.target_system = m.target_component = m.source_system = m.source_component = 1
        m.from_external = True
        self.pub_cmd.publish(m)


def ros_arguments():
    # Same deployment convention as takeoff_land.py: an optional parameter file (e.g. use_sim_time in simulation).
    configured = os.environ.get('DRONE_PARAMS_FILE')
    parameters = Path(configured or '/etc/flyx/mission.yaml')
    args = list(sys.argv[1:])
    if parameters.is_file():
        return ['--ros-args', '--params-file', str(parameters), *args]
    if configured:
        raise FileNotFoundError('Configured ROS parameter file is missing: ' + str(parameters))
    return args


def main():
    if not _ROS:
        raise SystemExit('rclpy / px4_msgs not importable: source ROS 2 and the px4_msgs workspace first')
    rclpy.init(args=ros_arguments())
    node = ClimbForwardHold()
    try:
        while rclpy.ok() and node.mission.phase not in ('DONE', 'ABORT'):
            rclpy.spin_once(node, timeout_sec=0.05)
    except KeyboardInterrupt:
        node.get_logger().warning('interrupted; verify the vehicle state in QGroundControl')
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
    return 0 if node.mission.phase == 'DONE' else 1


if __name__ == '__main__':
    raise SystemExit(main())
