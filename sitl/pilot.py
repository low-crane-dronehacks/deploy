#!/usr/bin/env python3
"""Scripted pilot for PX4 SITL: does what the safety pilot does on the skybeta drone, over MAVLink.

    pilot.py status                         watch mode / armed / position, never commands anything
    pilot.py fly                            DRY RUN: connect, check PX4 is ready, print the plan, send nothing
    pilot.py fly --arm [--autonomy-s 30]    the competition flow (needs the team's OK for this flight):
        1. stream centred sticks (MANUAL_CONTROL: SITL runs COM_RC_IN_MODE=1, joystick in place of the transmitter)
        2. Altitude, arm on the ground
        3. switch to Position -> skybeta's flight_interface requests Offboard (auto_offboard_on_arm) if the team node
           is publishing setpoints; we wait for it
        4. leave the team code flying for --autonomy-s, or until Offboard ends on its own (hold, time limit, ...)
        5. take over: throttle up for a moment (COM_RC_OVERRIDE bit 1, fleet file 09) -> Position
        6. throttle low until PX4 lands and disarms (COM_DISARM_LAND); MAV_CMD_NAV_LAND if it has not after 30 s
    Prints a JSON summary at the end; exit code 0 only if every step happened.

Connects to PX4 SITL's onboard MAVLink instance (PX4 sends to UDP 14540, px4-rc.mavlink), which nothing else in the
rig uses.
"""
import argparse
import json
import sys
import threading
import time

from pymavlink import mavutil

MAIN = {1: 'MANUAL', 2: 'ALTCTL', 3: 'POSCTL', 4: 'AUTO', 5: 'ACRO', 6: 'OFFBOARD', 7: 'STABILIZED', 8: 'RATTITUDE'}
ALTCTL, POSCTL, OFFBOARD = 2, 3, 6
CENTRE, LOW, TAKEOVER = 500, 150, 950      # MANUAL_CONTROL z: 0..1000 -> throttle -1..1 (500 = hold altitude)


class Pilot:
    def __init__(self, url):
        self.m = mavutil.mavlink_connection(url, source_system=255, source_component=190)
        self.lock = threading.Lock()
        self.mode, self.armed, self.z, self.landed = None, False, None, None
        self.throttle = CENTRE
        self.sticks = False
        self.texts = []
        self.alive = True

    def connect(self, timeout):
        hb = self.m.wait_heartbeat(timeout=timeout)
        if hb is None:
            raise SystemExit(f'no PX4 heartbeat in {timeout} s: is the rig up (run_rig.sh up)?')
        self._on(hb)
        threading.Thread(target=self._rx, daemon=True).start()
        threading.Thread(target=self._tx, daemon=True).start()

    def _on(self, msg):
        t = msg.get_type()
        if t == 'HEARTBEAT' and msg.get_srcComponent() == 1:
            self.mode = (msg.custom_mode >> 16) & 0xFF
            self.armed = bool(msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)
        elif t == 'LOCAL_POSITION_NED':
            self.z = msg.z
        elif t == 'EXTENDED_SYS_STATE':
            self.landed = msg.landed_state == mavutil.mavlink.MAV_LANDED_STATE_ON_GROUND
        elif t == 'STATUSTEXT':
            line = f'[px4] {msg.text}'
            self.texts.append(line)
            print(line, flush=True)

    def _rx(self):
        while self.alive:
            msg = self.m.recv_match(blocking=True, timeout=0.5)
            if msg is not None:
                self._on(msg)

    def _tx(self):
        n = 0
        while self.alive:
            if n % 25 == 0:
                self.m.mav.heartbeat_send(mavutil.mavlink.MAV_TYPE_GCS, mavutil.mavlink.MAV_AUTOPILOT_INVALID, 0, 0, 0)
            if self.sticks:
                self.m.mav.manual_control_send(self.m.target_system, 0, 0, int(self.throttle), 0, 0)
            n += 1
            time.sleep(0.04)

    def command(self, cmd, *p):
        p = list(p) + [0] * (7 - len(p))
        self.m.mav.command_long_send(self.m.target_system, self.m.target_component, cmd, 0, *p)

    def set_mode(self, main):
        self.command(mavutil.mavlink.MAV_CMD_DO_SET_MODE, mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED, main, 0)

    def wait(self, cond, timeout, retry=None, every=1.0):
        t0 = time.time()
        last = t0 - every                       # first retry immediately
        while time.time() - t0 < timeout:
            if cond():
                return time.time() - t0
            if retry and time.time() - last >= every:
                retry()
                last = time.time()
            time.sleep(0.05)
        return None

    def state(self):
        return {'mode': MAIN.get(self.mode, self.mode), 'armed': self.armed,
                'alt_m': None if self.z is None else round(-self.z, 2), 'landed': self.landed}


def status(p):
    while True:
        print(json.dumps(p.state()), flush=True)
        time.sleep(1.0)


def fly(p, a):
    out = {'steps': {}, 'ok': False}
    s = out['steps']
    p.sticks = True
    time.sleep(1.0)
    print(f'[pilot] PX4 {p.state()}')
    if p.armed:
        raise SystemExit('[pilot] vehicle is already armed: not touching it')
    if not a.arm:
        print('[pilot] DRY RUN (no --arm): would Altitude -> arm -> Position -> wait for Offboard -> '
              f'{a.autonomy_s:g} s of team control -> throttle takeover -> land. Nothing sent but centred sticks.')
        out['ok'] = True
        return out

    s['altctl_s'] = p.wait(lambda: p.mode == ALTCTL, 5, lambda: p.set_mode(ALTCTL))
    if s['altctl_s'] is None:
        return dict(out, why='PX4 did not enter Altitude')
    s['arm_s'] = p.wait(lambda: p.armed, a.ready_timeout,
                        lambda: p.command(mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 1), every=2.0)
    if s['arm_s'] is None:
        return dict(out, why='arming refused (see the [px4] lines: estimator not ready? VIO not fused?)')
    time.sleep(a.switch_after_s)
    t_switch = time.time()
    p.set_mode(POSCTL)
    s['offboard_s'] = p.wait(lambda: p.mode == OFFBOARD, a.offboard_timeout)
    if s['offboard_s'] is None:
        s['mode_after_switch'] = MAIN.get(p.mode, p.mode)
        p.throttle = LOW
        p.wait(lambda: not p.armed, 15)
        return dict(out, why='flight_interface did not take Offboard after the switch to Position: is the team launch '
                             'running and publishing setpoints (run_rig.sh team)?')
    print(f'[pilot] Offboard {time.time() - t_switch:.2f} s after the switch to Position; team code flying')
    alts = []
    t0 = time.time()
    while time.time() - t0 < a.autonomy_s and p.mode == OFFBOARD and p.armed:
        if p.z is not None:
            alts.append(-p.z)
        time.sleep(0.1)
    s['autonomy_s'] = round(time.time() - t0, 1)
    s['offboard_ended_by_itself'] = p.mode != OFFBOARD
    s['mode_at_takeover'] = MAIN.get(p.mode, p.mode)
    s['max_alt_m'] = round(max(alts), 2) if alts else None
    if p.armed and p.mode == OFFBOARD:
        p.throttle = TAKEOVER
        s['takeover_s'] = p.wait(lambda: p.mode != OFFBOARD, 3)
        p.throttle = CENTRE
        s['mode_after_takeover'] = MAIN.get(p.mode, p.mode)
        if s['takeover_s'] is None:
            print('[pilot] stick takeover did not leave Offboard; switching to Position by command')
            p.wait(lambda: p.mode != OFFBOARD, 5, lambda: p.set_mode(POSCTL))
    time.sleep(1.0)
    p.throttle = LOW
    s['disarm_s'] = p.wait(lambda: not p.armed, 30)
    if s['disarm_s'] is None:
        print('[pilot] not landed after 30 s of low throttle: MAV_CMD_NAV_LAND')
        p.command(mavutil.mavlink.MAV_CMD_NAV_LAND)
        s['disarm_s'] = p.wait(lambda: not p.armed, 60)
    p.throttle = CENTRE
    out['ok'] = s['disarm_s'] is not None and s.get('takeover_s', 0) is not None
    if s['disarm_s'] is None:
        out['why'] = 'still armed after the land command'
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('cmd', choices=['status', 'fly'])
    ap.add_argument('--url', default='udpin:0.0.0.0:14540')
    ap.add_argument('--arm', action='store_true', help='actually arm and fly (team approval for this flight)')
    ap.add_argument('--autonomy-s', type=float, default=30.0, help='seconds of team control before the takeover')
    ap.add_argument('--switch-after-s', type=float, default=1.0, help='armed in Altitude this long before Position')
    ap.add_argument('--ready-timeout', type=float, default=60.0, help='keep retrying the arm this long')
    ap.add_argument('--offboard-timeout', type=float, default=3.0)
    a = ap.parse_args(argv)
    p = Pilot(a.url)
    p.connect(30)
    try:
        if a.cmd == 'status':
            status(p)
        out = fly(p, a)
    finally:
        p.alive = False
    out['final'] = p.state()
    print(json.dumps(out, indent=2))
    return 0 if out['ok'] else 1


if __name__ == '__main__':
    sys.exit(main())
