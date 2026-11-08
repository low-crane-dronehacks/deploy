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
    dict(heading_good_for_control=False), dict(vx=math.nan)])
def test_invalid_input_immediately_blocks_policy_and_latches_during_flight(change):
    f = ready()
    f.on_lpos(lpos(timestamp_sample=1_100_000, **change), 10.1, active=True)
    assert f.error(10.1, .5)
    f.on_lpos(lpos(timestamp_sample=1_200_000), 10.2, active=True)
    assert f.error(10.2, .5)


@pytest.mark.parametrize('change', [dict(dist_bottom_valid=False), dict(dist_bottom=math.nan), dict(dist_bottom=0.)])
def test_invalid_range_retains_last_valid_reading_during_flight(change):
    f = ready()
    f.on_lpos(lpos(timestamp_sample=1_100_000, **change), 10.1, active=True)
    assert f.range_m == 1.2 and f.range_t == 10.
    assert f.error(10.1, .5) is None
    f.on_lpos(lpos(timestamp_sample=1_200_000), 10.2)
    assert f.error(10.2, .5) is None


def test_range_before_first_valid_sample_is_rf_max():
    f = Px4Feedback(rf_max=4.)
    f.on_lpos(lpos(dist_bottom_valid=False), 10.)
    f.on_att(att(), 10.)
    assert f.range_m == 4. and f.range_t is None
    assert 'range' in f.error(10., .5)


def test_range_stays_cached_through_long_dropout_but_state_must_stay_fresh():
    f = ready()
    for i in range(1, 21):
        stamp = 1_000_000 + i * 100_000
        f.on_lpos(lpos(timestamp_sample=stamp, dist_bottom_valid=False), 10. + i*.1, active=True)
        f.on_att(att(timestamp_sample=stamp), 10. + i*.1, active=True)
    assert f.range_m == 1.2 and f.range_t == 10.
    assert f.error(12., .5) is None
    assert f.error(13., .5)


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


def flags(**changes):
    return Msg(**(dict(timestamp_sample=1_000_000, cs_tilt_align=True, cs_ev_yaw=True,
                       cs_ev_yaw_fault=False, fs_bad_hdg=False, reject_yaw=False,
                       cs_rng_hgt=True, cs_rng_fault=False, cs_rng_stuck=False, reject_hagl=False) | changes))


def test_local_vision_heading_allows_ground_start_with_real_range():
    f = Px4Feedback()
    f.on_lpos(lpos(heading_good_for_control=False, dist_bottom_valid=False), 10.)
    f.on_att(att(), 10.)
    f.on_estimator(flags(), 10.)
    f.on_lpos(lpos(timestamp_sample=1_100_000, heading_good_for_control=False,
                   dist_bottom_valid=False, dist_bottom=.09), 10.1)
    assert f.error(10.1, .5) is None
    assert f.range_m == .09


@pytest.mark.parametrize('change', [dict(cs_ev_yaw=False), dict(cs_tilt_align=False),
    dict(cs_ev_yaw_fault=True), dict(fs_bad_hdg=True), dict(reject_yaw=True)])
def test_vision_heading_failure_latches_in_flight(change):
    f = ready()
    f.on_estimator(flags(), 10.)
    f.on_lpos(lpos(timestamp_sample=1_100_000, heading_good_for_control=False), 10.1)
    f.on_estimator(flags(timestamp_sample=1_100_000, **change), 10.1, active=True)
    f.on_estimator(flags(timestamp_sample=1_200_000), 10.2, active=True)
    assert 'heading' in f.error(10.2, .5)


def test_vision_heading_requires_fresh_nonduplicate_estimator_flags():
    f = ready()
    f.on_estimator(flags(), 10.)
    f.on_lpos(lpos(timestamp_sample=3_400_000, heading_good_for_control=False), 12.4)
    f.on_att(att(timestamp_sample=3_400_000), 12.4)
    f.on_estimator(flags(), 12.6)
    assert 'estimator' in f.error(12.6, .5)


def test_global_heading_does_not_require_vision_flags():
    assert ready().error(10.1, .5) is None


@pytest.mark.parametrize('change', [dict(cs_rng_hgt=False), dict(cs_tilt_align=False),
    dict(cs_rng_fault=True), dict(cs_rng_stuck=True), dict(reject_hagl=True)])
def test_bad_range_fusion_retains_last_distance(change):
    f = ready()
    f.on_estimator(flags(**change), 10.)
    f.on_lpos(lpos(timestamp_sample=1_100_000, dist_bottom_valid=False, dist_bottom=2.), 10.1)
    assert f.range_m == 1.2 and f.range_t == 10.


def test_primary_range_height_is_accepted_without_terrain_validity():
    f = ready()
    f.on_estimator(flags(), 10.)
    f.on_lpos(lpos(timestamp_sample=1_100_000, dist_bottom_valid=False, dist_bottom=.09), 10.1)
    assert f.range_m == .09 and f.range_t == 10.1


def test_stale_range_fusion_flags_cannot_refresh_distance():
    f = ready()
    f.on_estimator(flags(), 10.)
    f.on_lpos(lpos(timestamp_sample=3_600_000, dist_bottom_valid=False, dist_bottom=2.), 12.6)
    assert f.range_m == 1.2 and f.range_t == 10.


def test_missing_initial_range_blocks_startup_without_claiming_four_metres():
    f = Px4Feedback()
    f.on_lpos(lpos(dist_bottom_valid=False), 10.)
    f.on_att(att(), 10.)
    assert 'range' in f.error(10.1, .5)


def test_duplicate_local_position_cannot_refresh_range():
    f = ready()
    f.on_estimator(flags(), 10.1)
    f.on_lpos(lpos(dist_bottom_valid=False, dist_bottom=2.), 10.1)
    assert f.range_m == 1.2 and f.range_t == 10.


def test_one_hz_flags_remain_usable_during_flight():
    f = ready()
    f.on_estimator(flags(), 10.)
    for i in range(1, 31):
        stamp, now = 1_000_000 + i * 100_000, 10. + i * .1
        if i % 10 == 0:
            f.on_estimator(flags(timestamp_sample=stamp), now, active=True)
        f.on_lpos(lpos(timestamp_sample=stamp, heading_good_for_control=False,
                       dist_bottom_valid=False, dist_bottom=.1), now, active=True)
        f.on_att(att(timestamp_sample=stamp), now, active=True)
        assert f.error(now, .5) is None
        assert f.range_m == .1
