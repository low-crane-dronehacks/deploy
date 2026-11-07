"""Pure tests for onboard/examples/climb_forward_hold.py (no ROS): python3 -m pytest -q tests"""
import importlib.util
import math
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location('cfh', os.path.join(ROOT, 'onboard', 'examples', 'climb_forward_hold.py'))
cfh = importlib.util.module_from_spec(_spec)
sys.modules['cfh'] = cfh          # dataclasses look their module up while the file executes
_spec.loader.exec_module(cfh)

DT = 0.05


class FakePx4:
    """Kinematic stand-in: Offboard velocity tracked with a first-order lag, position setpoints by a P loop,
    AUTO_LAND descends at 0.5 m/s and disarms on the ground (z = 0). Heading 30 deg."""

    def __init__(self, heading=math.radians(30)):
        self.pos, self.vel, self.heading = [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], heading
        self.armed = self.offboard = self.auto_land = False
        self.landed, self.pos_ok, self.t = True, True, 0.0
        self.mode_req = False

    def tel(self, pos_age=0.0):
        return cfh.Telemetry(t=self.t, pos=tuple(self.pos), heading=self.heading, pos_ok=self.pos_ok, pos_age=pos_age,
                             status_age=0.0, armed=self.armed, offboard=self.offboard, auto_land=self.auto_land,
                             landed=self.landed, landed_age=0.0)

    def apply(self, out):
        for c in out.commands:
            if c == 'offboard':
                self.mode_req = True
            elif c == 'arm' and self.mode_req:
                self.armed = self.offboard = True
            elif c == 'land' and self.armed:
                self.auto_land, self.offboard = True, False
        if self.auto_land:
            cmd = [0.0, 0.0, 0.5]
        elif self.armed and out.mode == 'velocity':
            cmd = list(out.velocity)
        elif self.armed and out.mode == 'position':
            cmd = [max(-1.0, min(1.0, 1.5 * (out.position[i] - self.pos[i]))) for i in range(3)]
        else:
            cmd = [0.0, 0.0, 0.0]
        for i in range(3):
            self.vel[i] += (cmd[i] - self.vel[i]) * DT / 0.3
            self.pos[i] += self.vel[i] * DT
        if self.pos[2] >= 0.0:                     # ground
            self.pos[2], self.vel[2] = 0.0, min(self.vel[2], 0.0)
            self.landed = True
            if self.auto_land:
                self.armed = self.auto_land = False
        elif self.pos[2] < -0.05:
            self.landed = False
        self.t += DT


def run(m, px4, until=60.0, hook=None):
    log = []
    while px4.t < until and m.phase not in ('DONE', 'ABORT'):
        if hook:
            hook(m, px4)
        out = m.step(px4.tel())
        log.append((m.phase, out, tuple(px4.pos)))
        px4.apply(out)
    return log


def test_dry_run_publishes_and_commands_nothing():
    m, px4 = cfh.ForwardHoldMission(cfh.Params(arm=False)), FakePx4()
    log = run(m, px4)
    assert m.phase == 'DONE' and 'dry run' in m.why
    assert all(out.mode is None and not out.commands for _, out, _ in log)
    assert not px4.armed


def test_full_flight_velocity_then_position_hold_where_it_ended():
    p = cfh.Params(arm=True, height=1.0, forward_m=1.5, hold_s=3.0)
    m, px4 = cfh.ForwardHoldMission(p), FakePx4()
    log = run(m, px4)
    assert m.phase == 'DONE', m.why
    phases = [ph for ph, _, _ in log]
    for ph in ('PRIME', 'ACTIVATE', 'CLIMB', 'FORWARD', 'HOLD', 'LAND'):
        assert ph in phases
    # Offboard + arm requested exactly once, after >= prime_s of heartbeats
    arm_ticks = [i for i, (_, out, _) in enumerate(log) if 'arm' in out.commands]
    assert len(arm_ticks) == 1 and arm_ticks[0] * DT >= p.prime_s - 1e-9
    # climb and forward are velocity commands, hold is a position setpoint
    assert all(out.mode == 'velocity' for ph, out, _ in log if ph in ('CLIMB', 'FORWARD') and out.mode)
    hold = [(out, pos) for ph, out, pos in log if ph == 'HOLD']
    assert hold and all(out.mode == 'position' for out, _ in hold)
    # the hold point is where FORWARD ended (not the planned point), and that is close to the plan
    hp = hold[0][0].position
    assert hp == m.hold_pos
    planned = (1.5 * math.cos(px4.heading), 1.5 * math.sin(px4.heading), -1.0)
    assert math.dist(hp, planned) < p.reach_tol + 1e-9
    # it actually holds: after settling, the vehicle stays within 10 cm of the hold point
    settled = [pos for out, pos in hold[int(1.5 / DT):]]
    assert settled and max(math.dist(pos, hp) for pos in settled) < 0.10
    # never faster than allowed, heading held at the takeoff heading
    for ph, out, _ in log:
        if out.mode == 'velocity':
            assert math.hypot(*out.velocity) <= max(p.climb_speed, p.forward_speed) + 1e-9
        if out.mode:
            assert out.yaw == pytest.approx(px4.heading)
    assert any('land' in out.commands for ph, out, _ in log if ph == 'LAND')


def test_climb_is_straight_up_and_drift_is_steered_out():
    m, px4 = cfh.ForwardHoldMission(cfh.Params(arm=True, forward_m=0.5)), FakePx4()

    def push(m, px4):                       # a gust during the climb
        if m.phase == 'CLIMB' and px4.pos[2] > -0.5:
            px4.vel[1] += 0.02
    log = run(m, px4, hook=push)
    climb_end = [pos for ph, _, pos in log if ph == 'CLIMB'][-1]
    assert math.hypot(climb_end[0], climb_end[1]) < 0.15
    assert m.phase == 'DONE'


def test_offboard_taken_away_yields_and_stops_publishing():
    m, px4 = cfh.ForwardHoldMission(cfh.Params(arm=True)), FakePx4()

    def pilot(m, px4):
        if m.phase == 'FORWARD':
            px4.offboard = False             # pilot switched to Position mode
    run(m, px4, hook=pilot)
    assert m.phase == 'ABORT'
    assert m.step(px4.tel()).mode is None


def test_position_invalid_lands_without_streaming_a_bad_setpoint():
    m, px4 = cfh.ForwardHoldMission(cfh.Params(arm=True)), FakePx4()

    def lose(m, px4):
        if m.phase == 'FORWARD':
            px4.pos_ok = False
    log = run(m, px4, hook=lose)
    land = [out for ph, out, _ in log if ph == 'LAND']
    assert land and all(out.mode is None for out in land)
    assert any('land' in out.commands for out in land)


def test_stale_telemetry_in_flight_aborts():
    m, px4 = cfh.ForwardHoldMission(cfh.Params(arm=True)), FakePx4()
    while m.phase != 'CLIMB' and px4.t < 20.0:
        px4.apply(m.step(px4.tel()))
    assert m.phase == 'CLIMB'
    out = m.step(px4.tel(pos_age=2.0))
    assert m.phase == 'ABORT' and out.mode is None


def test_someone_else_arming_while_priming_yields():
    m, px4 = cfh.ForwardHoldMission(cfh.Params(arm=True)), FakePx4()
    px4.apply(m.step(px4.tel()))
    assert m.phase == 'PRIME'
    px4.armed = True
    m.step(px4.tel())
    assert m.phase == 'ABORT'


def test_params_are_capped():
    with pytest.raises(ValueError):
        cfh.ForwardHoldMission(cfh.Params(forward_m=5.0))
    with pytest.raises(ValueError):
        cfh.ForwardHoldMission(cfh.Params(height=3.0))
    with pytest.raises(ValueError):
        cfh.ForwardHoldMission(cfh.Params(forward_speed=3.0))


def test_approach_speed_slows_near_target():
    assert cfh.approach_speed(5.0, 0.5, 0.5, 0.08) == 0.5
    assert cfh.approach_speed(0.04, 0.5, 0.5, 0.08) == pytest.approx(0.2)
    assert cfh.approach_speed(0.0, 0.5, 0.5, 0.08) == 0.08
