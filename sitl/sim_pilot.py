#!/usr/bin/env python3
"""Simulated safety pilot for SITL (sitl/run_native.sh SKYBETA=1): the skybeta start procedure (docs/SKYBETA.md,
DRONE_TESTING.md) done by a script instead of an RC transmitter.

    python3 sitl/sim_pilot.py               # DRY RUN: print the procedure, publish nothing
    python3 sitl/sim_pilot.py --check       # disarmed: stream sticks, switch Altitude -> Position, report. No arming.
    python3 sitl/sim_pilot.py --arm         # the procedure below, ARMS (team approval for that flight, AGENTS.md)

Procedure (--arm), as the pilot does on the RC:
  1. sticks centred, throttle LOW, streamed at 50 Hz as RC input (/fmu/in/manual_control_input, source RC; PX4 runs
     with COM_RC_IN_MODE 0 under SKYBETA=1). Wait for PX4: local position + heading valid, disarmed.
  2. Altitude mode, then arm.
  3. after --switch_after_s (default 2 s, "within a few seconds"): Position mode, throttle to centre (if Offboard is
     lost later, PX4 holds altitude instead of descending; the stick does not move again, so no RC override).
  4. hands off: skybeta's flight_interface (sitl/flight_interface_sim.py) requests Offboard once the team node's
     twists are streaming; the team node climbs. PX4 disarms by itself if not airborne 10 s after arming.
  5. the pilot lands: after --land_after_s in Offboard (default: wait for Ctrl-C), Land mode (APPROX: a real pilot
     lands by hand in Position mode), then wait for PX4's own disarm on touchdown.
PX4-direct stacks (--offboard, e.g. launch/maze_autonomy.launch.py arm:=true, whose control_mux streams setpoints but
never arms or changes mode): wait for PX4 ready AND the stack's /fmu/in/offboard_control_mode stream, switch to
Offboard, arm (the pilot / QGroundControl role for a PX4-direct companion), hands off, land as in step 5.
    python3 sitl/sim_pilot.py --offboard --check    # disarmed: Offboard accepted, then back to Hold. No arming.
    python3 sitl/sim_pilot.py --offboard --arm      # Offboard + ARM: the stack flies (team approval)
Ctrl-C at any point while armed = step 5. Commands are the VehicleCommand equivalents of the stick/switch actions.
"""
from __future__ import annotations

import argparse
import math
import sys
import time

try:                                            # the dry run (no flags) needs no ROS
    import rclpy
    from px4_msgs.msg import (ManualControlSetpoint, OffboardControlMode, VehicleCommand, VehicleLocalPosition,
                              VehicleStatus)
    from rclpy.node import Node
    from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
    PX4_QOS = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                         history=HistoryPolicy.KEEP_LAST, depth=1)
    STREAM_QOS = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, durability=DurabilityPolicy.VOLATILE,
                            history=HistoryPolicy.KEEP_LAST, depth=5)
    NAV = {VehicleStatus.NAVIGATION_STATE_ALTCTL: 'Altitude', VehicleStatus.NAVIGATION_STATE_POSCTL: 'Position',
           VehicleStatus.NAVIGATION_STATE_OFFBOARD: 'Offboard', VehicleStatus.NAVIGATION_STATE_AUTO_LAND: 'Land'}
except ImportError:
    rclpy = None
    Node = object
ALTCTL, POSCTL, AUTO, OFFBOARD = 2.0, 3.0, 4.0, 6.0   # PX4 custom main modes for VEHICLE_CMD_DO_SET_MODE
AUTO_LOITER = 3.0                                     # custom sub mode of AUTO = Hold



def px4_topic(name, msg_type):
    v = int(getattr(msg_type, 'MESSAGE_VERSION', 0))
    return '/fmu/out/' + name + (f'_v{v}' if v else '')


class Pilot(Node):
    def __init__(self):
        super().__init__('sim_pilot')
        self.throttle = -1.0
        self.status = self.lpos = None
        self.pub_rc = self.create_publisher(ManualControlSetpoint, '/fmu/in/manual_control_input', PX4_QOS)
        self.pub_cmd = self.create_publisher(VehicleCommand, '/fmu/in/vehicle_command', PX4_QOS)
        self.create_subscription(VehicleStatus, px4_topic('vehicle_status', VehicleStatus),
                                 lambda m: setattr(self, 'status', m), PX4_QOS)
        self.create_subscription(VehicleLocalPosition, px4_topic('vehicle_local_position', VehicleLocalPosition),
                                 lambda m: setattr(self, 'lpos', m), PX4_QOS)
        self.ocm_t = -1e9                                   # last /fmu/in/offboard_control_mode seen (the stack's stream)
        self.create_subscription(OffboardControlMode, '/fmu/in/offboard_control_mode',
                                 lambda m: setattr(self, 'ocm_t', time.monotonic()), STREAM_QOS)
        self.create_timer(0.02, self._sticks)
        self.last_nav = None

    def _sticks(self):
        us = self.get_clock().now().nanoseconds // 1000
        m = ManualControlSetpoint()
        m.timestamp = m.timestamp_sample = us
        m.valid, m.data_source = True, ManualControlSetpoint.SOURCE_RC
        m.roll = m.pitch = m.yaw = 0.0
        m.throttle = float(self.throttle)
        m.flaps = m.aux1 = m.aux2 = m.aux3 = m.aux4 = m.aux5 = m.aux6 = 0.0
        self.pub_rc.publish(m)

    def command(self, cmd, p1=0.0, p2=0.0, p3=0.0):
        m = VehicleCommand()
        m.timestamp = self.get_clock().now().nanoseconds // 1000
        m.command, m.param1, m.param2, m.param3 = cmd, float(p1), float(p2), float(p3)
        m.target_system = m.target_component = m.source_system = m.source_component = 1
        m.from_external = True
        self.pub_cmd.publish(m)

    def set_mode(self, main_mode):
        self.command(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, 1.0, main_mode)

    # ---- state
    def armed(self):
        return self.status is not None and self.status.arming_state == VehicleStatus.ARMING_STATE_ARMED

    def nav(self):
        return None if self.status is None else self.status.nav_state

    def nav_name(self):
        n = self.nav()
        return NAV.get(n, f'nav_state {n}')

    def ready(self):
        p = self.lpos
        return bool(p is not None and p.xy_valid and p.z_valid and p.heading_good_for_control and not self.armed())

    def log_mode_changes(self):
        if self.status is not None and (self.nav(), self.armed()) != self.last_nav:
            self.last_nav = (self.nav(), self.armed())
            alt = '' if self.lpos is None else f', altitude {-self.lpos.z:.2f} m'
            self.get_logger().info(f'PX4: {self.nav_name()}, {"ARMED" if self.armed() else "disarmed"}{alt}')

    def spin_for(self, s, until=None):
        t_end = time.monotonic() + s
        while time.monotonic() < t_end:
            rclpy.spin_once(self, timeout_sec=0.02)
            self.log_mode_changes()
            if until is not None and until():
                return True
        return False


def land_and_wait(p: Pilot):
    p.get_logger().info('pilot: landing (Land mode), waiting for PX4 to disarm on touchdown')
    p.throttle = 0.0
    p.command(VehicleCommand.VEHICLE_CMD_NAV_LAND)
    if not p.spin_for(90.0, until=lambda: not p.armed()):
        p.get_logger().error('still armed after 90 s: take over (QGroundControl) and check the vehicle')
        return 1
    p.get_logger().info('landed and disarmed')
    return 0


def offboard_procedure(p: Pilot, args):
    log = p.get_logger()
    log.info('pilot (PX4-direct): waiting for the stack\'s Offboard stream on /fmu/in/offboard_control_mode')
    if not p.spin_for(15.0, until=lambda: time.monotonic() - p.ocm_t < 0.5):
        log.error('no Offboard stream: is the stack running with arm:=true? Nothing switched, nothing armed')
        return 1
    p.set_mode(OFFBOARD)
    if not p.spin_for(5.0, until=lambda: p.nav() == VehicleStatus.NAVIGATION_STATE_OFFBOARD):
        log.error(f'PX4 refused Offboard (now {p.nav_name()}); nothing armed')
        return 1
    if args.check:
        p.command(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, 1.0, AUTO, AUTO_LOITER)      # back to Hold
        p.spin_for(2.0, until=lambda: p.nav() != VehicleStatus.NAVIGATION_STATE_OFFBOARD)
        log.info('check passed: the stack\'s stream is accepted and PX4 enters Offboard (not armed); back to '
                 f'{p.nav_name()}')
        return 0
    log.info('pilot: Offboard, arming; hands off (Ctrl-C = land)')
    p.command(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, 1.0)
    if not p.spin_for(5.0, until=p.armed):
        log.error('arming refused (see the px4 log); nothing armed')
        return 1
    p.spin_for(args.land_after_s if args.land_after_s > 0 else 1e9,
               until=lambda: not p.armed() or p.nav() != VehicleStatus.NAVIGATION_STATE_OFFBOARD)
    if p.armed() and p.nav() != VehicleStatus.NAVIGATION_STATE_OFFBOARD:
        log.warning(f'Offboard lost (now {p.nav_name()}): the stack stopped streaming or PX4 failed over')
    if p.armed():
        return land_and_wait(p)
    log.warning('disarmed during the flight')
    return 1


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--arm', action='store_true', help='run the procedure: ARMS the simulated vehicle')
    ap.add_argument('--check', action='store_true', help='disarmed mode-switch check (no arming)')
    ap.add_argument('--offboard', action='store_true',
                    help="PX4-direct stack: switch to Offboard + arm (instead of Altitude -> Position)")
    ap.add_argument('--switch_after_s', type=float, default=2.0)
    ap.add_argument('--land_after_s', type=float, default=0.0, help='land this long after Offboard starts (0 = Ctrl-C)')
    ap.add_argument('--ready_timeout_s', type=float, default=90.0)
    args, ros_args = ap.parse_known_args(argv)
    if not (args.arm or args.check):
        print(__doc__.split('Procedure (--arm)')[1].split('Ctrl-C at any')[0].rstrip())
        print('\nDRY RUN: nothing published. --check = disarmed mode test, --arm = fly (team approval); '
              '--offboard for PX4-direct stacks (maze_autonomy arm:=true).')
        return 0
    if rclpy is None:
        raise SystemExit('rclpy / px4_msgs not importable: source ROS 2 and the px4_msgs workspace first')
    from rclpy.signals import SignalHandlerOptions
    rclpy.init(args=ros_args, signal_handler_options=SignalHandlerOptions.NO)   # Ctrl-C must reach land_and_wait
    p = Pilot()
    log = p.get_logger()
    try:
        log.info('pilot: sticks centred, throttle low; waiting for PX4 (position + heading valid, disarmed)')
        if not p.spin_for(args.ready_timeout_s, until=p.ready):
            log.error('PX4 not ready (see sitl/run_native.sh status); nothing armed')
            return 1
        if args.offboard:
            return offboard_procedure(p, args)
        p.set_mode(ALTCTL)
        if not p.spin_for(5.0, until=lambda: p.nav() == VehicleStatus.NAVIGATION_STATE_ALTCTL):
            log.error(f'PX4 refused Altitude mode (now {p.nav_name()}): is SKYBETA=1 set (COM_RC_IN_MODE 0)?')
            return 1
        if args.check:
            p.spin_for(1.0)
            p.set_mode(POSCTL)
            ok = p.spin_for(5.0, until=lambda: p.nav() == VehicleStatus.NAVIGATION_STATE_POSCTL)
            log.info('check passed: PX4 takes RC input and Altitude -> Position on the ground (not armed)' if ok
                     else f'check FAILED: Position refused (now {p.nav_name()})')
            return 0 if ok else 1
        log.info('pilot: arming in Altitude')
        p.command(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, 1.0)
        if not p.spin_for(5.0, until=p.armed):
            log.error('arming refused (see the px4 log); nothing armed')
            return 1
        p.spin_for(args.switch_after_s)
        log.info('pilot: Position mode, throttle to centre; hands off')
        p.set_mode(POSCTL)
        p.throttle = 0.0
        if not p.spin_for(15.0, until=lambda: p.nav() == VehicleStatus.NAVIGATION_STATE_OFFBOARD or not p.armed()):
            log.warning('no Offboard within 15 s: is the team node streaming /skybeta/setpoint/twist and '
                        'flight_interface_sim running?')
        if not p.armed():
            log.warning('PX4 disarmed (not airborne 10 s after arming?)')
            return 1
        if p.nav() == VehicleStatus.NAVIGATION_STATE_OFFBOARD:
            log.info('Offboard active: the team node is flying' +
                     (f'; landing in {args.land_after_s:.0f} s' if args.land_after_s > 0 else '; Ctrl-C to land'))
            p.spin_for(args.land_after_s if args.land_after_s > 0 else 1e9, until=lambda: not p.armed())
        if p.armed():
            return land_and_wait(p)
        log.warning('disarmed during the flight')
        return 1
    except KeyboardInterrupt:
        if p.armed():
            return land_and_wait(p)
        return 0
    finally:
        p.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    sys.exit(main())
