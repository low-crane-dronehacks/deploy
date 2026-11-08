"""Offline policy input checks: PX4 frames, freshness, invalid data and resets."""
import math
from types import SimpleNamespace as Msg

import numpy as np
import pytest

from policynav import obs
from policynav.px4_feedback import Px4Feedback


def lpos(**changes):
    fields = dict(timestamp_sample=1_000_000, vx=0., vy=1., vz=-.2,
                  v_xy_valid=True, v_z_valid=True, heading_good_for_control=True,
                  dist_bottom=1.2, dist_bottom_valid=True,
                  vxy_reset_counter=0, vz_reset_counter=0, heading_reset_counter=0)
    return Msg(**(fields | changes))


def att(**changes):
    return Msg(**(dict(timestamp_sample=1_000_000,
                       q=[math.sqrt(.5), 0., 0., math.sqrt(.5)], quat_reset_counter=0) | changes))


def ready():
    f = Px4Feedback()
    f.on_lpos(lpos(), 10.)
    f.on_att(att(), 10.)
    return f


def test_px4_ned_velocity_and_frd_attitude_reach_policy_without_ros_conversion():
    f = ready()
    assert f.error(10.1, .5) is None
    pro = obs.proprio(f.vel, f.q, f.range_m, [0]*4, [1]*3)
    assert pro[:6] == pytest.approx([1., 0., -.2, .8, 0., 0.], abs=1e-6)
    v, yaw = obs.action_to_setpoint([1., 0., 0., .3], f.q, [1]*3, 1.)
    assert v == pytest.approx([0., 1., 0.], abs=1e-6)
    assert yaw == .3


@pytest.mark.parametrize('change', [dict(v_xy_valid=False), dict(v_z_valid=False),
    dict(heading_good_for_control=False), dict(vx=math.nan), dict(dist_bottom_valid=False),
    dict(dist_bottom=math.nan), dict(dist_bottom=0.)])
def test_invalid_input_immediately_blocks_policy_and_latches_during_flight(change):
    f = ready()
    f.on_lpos(lpos(timestamp_sample=1_100_000, **change), 10.1, active=True)
    assert f.error(10.1, .5)
    f.on_lpos(lpos(timestamp_sample=1_200_000), 10.2, active=True)
    assert f.error(10.2, .5)


def test_no_altitude_fallback_and_preflight_input_can_recover():
    f = ready()
    f.on_lpos(lpos(timestamp_sample=1_100_000, dist_bottom_valid=False), 10.1)
    assert f.range_m is None and f.error(10.1, .5)
    f.on_lpos(lpos(timestamp_sample=1_200_000), 10.2)
    assert f.error(10.2, .5) is None


@pytest.mark.parametrize('stream', ['lpos', 'att'])
def test_duplicate_samples_cannot_keep_input_fresh(stream):
    f = ready()
    getattr(f, 'on_' + stream)(lpos() if stream == 'lpos' else att(), 10.6)
    assert f.error(10.6, .5)


@pytest.mark.parametrize('change', [dict(vxy_reset_counter=1), dict(vz_reset_counter=1),
                                  dict(heading_reset_counter=1), dict(timestamp_sample=900_000)])
def test_reset_or_time_reversal_latches_fault(change):
    f = ready()
    f.on_lpos(lpos(**(dict(timestamp_sample=1_100_000) | change)), 10.1, active=True)
    assert f.error(10.1, .5)


@pytest.mark.parametrize('change', [dict(q=[0., 0., 0., 0.]), dict(q=[math.nan, 0., 0., 0.]),
                                  dict(quat_reset_counter=1)])
def test_bad_or_reset_attitude_blocks_policy(change):
    f = ready()
    f.on_att(att(timestamp_sample=1_100_000, **change), 10.1, active=True)
    assert f.error(10.1, .5)


def test_both_streams_required_and_independently_expire():
    f = Px4Feedback()
    f.on_lpos(lpos(), 10.)
    assert f.error(10., .5)
    f.on_att(att(), 10.)
    f.on_lpos(lpos(timestamp_sample=1_600_000), 10.6)
    assert f.error(10.6, .5)
