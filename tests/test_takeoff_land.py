"""Standalone hover's native PX4 messages and mission transitions, without ROS or flight."""
import math
import os
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'onboard', 'examples'))
import takeoff_land as H


class Command(SimpleNamespace):
    VEHICLE_CMD_DO_SET_MODE = 176
    VEHICLE_CMD_COMPONENT_ARM_DISARM = 400
    VEHICLE_CMD_NAV_LAND = 21


@pytest.fixture
def node(monkeypatch):
    # Only replace transport message constructors; exercise the actual mission methods.
    monkeypatch.setattr(H, 'OffboardControlMode', SimpleNamespace, raising=False)
    monkeypatch.setattr(H, 'TrajectorySetpoint', SimpleNamespace, raising=False)
    monkeypatch.setattr(H, 'VehicleCommand', Command, raising=False)
    monkeypatch.setattr(H, 'VehicleStatus', SimpleNamespace(
        ARMING_STATE_ARMED=2, NAVIGATION_STATE_OFFBOARD=14,
        NAVIGATION_STATE_AUTO_LAND=18), raising=False)
    n = object.__new__(H.TakeoffLand)
    n.now = 1.
    n.clock_seconds = lambda: n.now
    n.timestamp = lambda: int(n.now * 1e6)
    n.logs = []
    n.get_logger = lambda: SimpleNamespace(info=n.logs.append, warning=n.logs.append, error=n.logs.append)
    n.modes, n.setpoints, n.commands = [], [], []
    n.offboard_pub = SimpleNamespace(publish=n.modes.append)
    n.setpoint_pub = SimpleNamespace(publish=n.setpoints.append)
    n.command_pub = SimpleNamespace(publish=n.commands.append)
    n.phase, n.phase_start = 'WAIT', 1.
    n.last_tick_time = n.wait_started = None
    n.preflight_timeout_s = 45.
    n.last_wait_log = n.last_command = 0.
    n.target_system, n.command_target = 0, None
    n.travel, n.velocity_fault = H.VerticalTravel(), None
    n.yaw_aligned = False
    n.yaw_alignment_time, n.yaw_alignment_timestamp = None, 0
    n.ground_confirmation_started = None
    n.auto_land_observed = False
    n.landed, n.landed_time = None, 0.
    n.pos = None
    n.pos_time = n.status_time = 0.
    n.status = SimpleNamespace(system_id=17, component_id=1, arming_state=1, nav_state=1)
    feed(n)
    return n


def feed(n, dt=0., vz=0., **changes):
    n.now += dt
    msg = SimpleNamespace(timestamp=int(n.now * 1e6), timestamp_sample=int(n.now * 1e6),
                          vx=0., vy=0., vz=vz, heading=.7, v_xy_valid=True, v_z_valid=True,
                          heading_good_for_control=True, vxy_reset_counter=0, vz_reset_counter=0,
                          heading_reset_counter=0, xy_valid=False, z_valid=False,
                          x=math.nan, y=math.nan, z=math.nan)
    for k, value in changes.items():
        setattr(msg, k, value)
    n.on_position(msg)
    n.status_time = n.now


def test_system_id_uses_system_not_component():
    assert H.command_system_id(0, 17) == 17
    assert H.command_system_id(0, 1) == 1
    assert H.command_system_id(1, 1) == 1
    for configured, observed in [(1, 17), (0, 0), (0, 256)]:
        with pytest.raises(ValueError):
            H.command_system_id(configured, observed)


def test_commands_address_discovered_system_and_component_one(node):
    assert node.command(176, 1., 6.)
    msg = node.commands[-1]
    assert msg.target_system == msg.source_system == 17
    assert msg.target_component == 1
    node.status.system_id = 1
    assert node.command(176)
    assert node.commands[-1].target_system == 1
    node.target_system = 17
    assert not node.command(176)
    assert len(node.commands) == 2


@pytest.mark.parametrize('phase,want_z', [('PRIME', 0.), ('ACTIVATE', 0.), ('CLIMB', -.5),
                                         ('HOVER', 0.), ('LAND', 0.)])
def test_messages_enable_velocity_and_leave_position_and_yaw_unset(node, phase, want_z):
    node.phase = phase
    node.publish_setpoint()
    assert node.modes[-1].velocity and not node.modes[-1].position
    sp = node.setpoints[-1]
    assert sp.velocity == [0., 0., want_z]
    assert all(math.isnan(v) for v in sp.position + sp.acceleration)
    assert math.isnan(sp.yaw) and sp.yawspeed == 0.


def test_full_sequence_primes_arms_climbs_stops_and_lands_with_px4_feedback(node):
    node.tick()
    assert node.phase == 'PRIME' and not node.commands
    feed(node, 2.1)
    node.tick()
    assert node.phase == 'ACTIVATE'
    assert [m.command for m in node.commands] == [176, 400]
    assert all(m.target_system == 17 for m in node.commands)
    node.status.arming_state, node.status.nav_state = 2, 14
    feed(node, .1)
    node.tick()
    assert node.phase == 'CLIMB'
    for _ in range(25):
        feed(node, .1, vz=-.5)
        node.tick()
        if node.phase == 'HOVER':
            break
    assert node.phase == 'HOVER'
    assert node.travel.rise_m == pytest.approx(1., abs=.11)
    feed(node, .1)
    node.tick()
    assert node.setpoints[-1].velocity == [0., 0., 0.]
    for _ in range(22):
        feed(node, .1)
        node.tick()
    assert node.phase == 'LAND' and node.commands[-1].command == 21
    node.status.nav_state = 18
    feed(node, .1)
    node.tick()
    node.status.arming_state = 1
    node.on_landed(SimpleNamespace(landed=True))
    feed(node, .1)
    node.tick()
    assert node.phase == 'DONE'
    assert all(all(math.isnan(v) for v in sp.position) for sp in node.setpoints)


def test_velocity_not_command_time_measures_climb_and_duplicate_samples_do_not_count():
    travel = H.VerticalTravel()
    travel.observe(1., 0., (0, 0, 0))
    travel.observe(1.1, 0., (0, 0, 0))
    assert travel.rise_m == 0.
    travel.observe(1.2, -.5, (0, 0, 0))
    assert travel.rise_m == pytest.approx(.025)
    travel.observe(1.2, -.5, (0, 0, 0))
    assert travel.rise_m == pytest.approx(.025)


@pytest.mark.parametrize('sample,vz,resets', [(0., 0., (0, 0, 0)), (2., math.nan, (0, 0, 0)),
    (3., 0., (0, 0, 0)), (.9, 0., (0, 0, 0)), (1.1, 0., (1, 0, 0)),
    (1.1, 0., (0, 1, 0)), (1.1, 0., (0, 0, 1))])
def test_integrator_rejects_invalid_reset_and_discontinuous_feedback(sample, vz, resets):
    travel = H.VerticalTravel()
    travel.observe(1., 0., (0, 0, 0))
    with pytest.raises(ValueError):
        travel.observe(sample, vz, resets)


def test_velocity_fault_is_latched_and_requests_landing_without_upward_command(node):
    node.phase = 'CLIMB'
    node.status.arming_state, node.status.nav_state = 2, 14
    node.travel.observe(*H.vertical_sample(node.pos))
    feed(node, .1, v_z_valid=False)
    feed(node, .1)
    node.tick()
    assert node.phase == 'LAND' and node.commands[-1].command == 21
    assert node.setpoints[-1].velocity == [0., 0., 0.]


def test_activation_with_invalid_velocity_requests_landing_not_climb(node):
    node.phase = 'ACTIVATE'
    node.status.arming_state, node.status.nav_state = 2, 14
    feed(node, .1, vz=math.nan)
    node.tick()
    assert node.phase == 'LAND'
    assert node.setpoints[-1].velocity == [0., 0., 0.]


def test_stale_feedback_yields_before_any_new_message(node):
    node.phase = 'CLIMB'
    node.status.arming_state, node.status.nav_state = 2, 14
    node.now += 1.1
    node.tick()
    assert node.phase == 'ABORT' and not node.commands and not node.setpoints


def test_auto_discovery_does_not_switch_vehicles_after_priming(node):
    node.tick()
    assert node.command_target == 17
    node.status.system_id = 1
    feed(node, .1)
    before = len(node.setpoints)
    node.tick()
    assert node.phase == 'ABORT' and not node.commands and len(node.setpoints) == before


def test_pilot_takeover_yields_before_another_setpoint(node):
    node.phase = 'CLIMB'
    node.status.arming_state, node.status.nav_state = 2, 1
    node.tick()
    assert node.phase == 'ABORT' and not node.setpoints and not node.commands


def test_wrong_explicit_system_id_withholds_arming(node):
    node.target_system = 1
    node.tick()
    assert node.phase == 'WAIT' and not node.setpoints and not node.commands


def test_hover_primes_and_addresses_system_one_when_px4_reports_one(node):
    node.status.system_id = 1
    node.tick()
    feed(node, 2.1)
    node.tick()
    assert node.phase == 'ACTIVATE'
    assert [m.command for m in node.commands] == [176, 400]
    assert all(m.target_system == 1 for m in node.commands)
