"""PX4-feedback, heading-relative, velocity-only Skybeta mission (no ROS)."""
import math
import os
import sys
from types import SimpleNamespace
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'onboard', 'examples'))
import climb_forward_hold_skybeta as M


def step(m, t, velocity=(0., 0., 0.), heading=0., offboard=True, age=0., valid=True,
         resets=(0, 0, 0), sample_t=None):
    fb = M.Feedback(10. + t if sample_t is None else sample_t, velocity, heading, valid, resets)
    return m.step(M.State(t=t, feedback=fb, feedback_age=age, offboard=offboard, armed=offboard))


@pytest.mark.parametrize('heading,want', [
    (0., (0., .5, 0.)), (math.pi / 2, (.5, 0., 0.)),
    (math.pi, (0., -.5, 0.)), (-math.pi / 2, (-.5, 0., 0.)),
])
def test_body_forward_rotates_into_enu(heading, want):
    assert M.body_to_enu(.5, 0., 0., heading) == pytest.approx(want, abs=1e-12)


def test_right_and_down_signs():
    assert M.body_to_enu(0., .4, .2, 0.) == pytest.approx((.4, 0., -.2))
    assert M.body_to_enu(0., .4, .2, math.pi / 2) == pytest.approx((0., -.4, -.2), abs=1e-12)


def test_waits_with_zero_velocity_and_climbs_immediately_on_offboard():
    m = M.SkybetaForwardHold(M.Params(fly=True))
    assert step(m, 0., offboard=False).twist == (0., 0., 0.)
    assert m.phase == 'WAIT'
    assert step(m, .1).twist == (0., 0., .5)
    assert m.phase == 'CLIMB'


def test_full_sequence_uses_measured_velocity_and_finishes_with_zero_velocity():
    m = M.SkybetaForwardHold(M.Params(fly=True, hold_s=1.))
    velocity, seen, t = (0., 0., 0.), [], 0.
    for _ in range(300):
        out = step(m, t, velocity)
        seen.append(m.phase)
        assert math.sqrt(sum(v * v for v in out.twist)) <= math.sqrt(.5)
        velocity = (out.twist[1], out.twist[0], -out.twist[2])
        t += .1
        if m.phase == 'DONE':
            break
    assert list(dict.fromkeys(seen)) == ['CLIMB', 'FORWARD', 'HOLD', 'DONE']
    assert m.height_m == pytest.approx(1., abs=.1)
    assert m.forward_m == pytest.approx(1., abs=.1)
    assert out.twist == (0., 0., 0.)
    assert step(m, t).twist == (0., 0., 0.)


def reach_forward(m):
    step(m, 0., (0., 0., -.5))
    for i in range(1, 21):
        out = step(m, i * .1, (0., 0., -.5))
        if m.phase == 'FORWARD':
            return i * .1, out
    pytest.fail('did not reach forward phase')


def test_forward_tracks_current_heading_not_start_heading():
    m = M.SkybetaForwardHold(M.Params(fly=True))
    t, _ = reach_forward(m)
    out = step(m, t + .1, heading=math.pi / 2)
    assert out.twist[0] > 0.
    assert out.twist[1] == pytest.approx(0., abs=1e-12)


def test_commanded_motion_does_not_count_as_measured_travel():
    m = M.SkybetaForwardHold(M.Params(fly=True))
    for i in range(20):
        step(m, i * .1)
    assert m.phase == 'CLIMB' and m.height_m == 0.


def test_forward_progress_projects_measured_velocity_onto_current_heading():
    m = M.SkybetaForwardHold(M.Params(fly=True))
    t, _ = reach_forward(m)
    initial = m.forward_m
    step(m, t + .1, (0., .5, 0.), heading=0.)
    assert m.forward_m == initial
    step(m, t + .2, (0., .5, 0.), heading=math.pi / 2)
    step(m, t + .3, (0., .5, 0.), heading=math.pi / 2)
    assert m.forward_m == pytest.approx(initial + .075)


def test_duplicate_samples_do_not_integrate_again():
    m = M.SkybetaForwardHold(M.Params(fly=True))
    step(m, 0., (0., 0., -.5), sample_t=10.)
    step(m, .1, (0., 0., -.5), sample_t=10.1)
    height = m.height_m
    step(m, .2, (0., 0., -.5), sample_t=10.1)
    assert m.height_m == height


@pytest.mark.parametrize('change', [
    {'age': .6}, {'valid': False}, {'resets': (1, 0, 0)},
    {'resets': (0, 1, 0)}, {'resets': (0, 0, 1)},
    {'sample_t': 9.}, {'sample_t': 11.},
    {'velocity': (math.nan, 0., 0.)}, {'heading': math.nan},
])
def test_bad_feedback_aborts_to_zero_velocity_and_never_resumes(change):
    m = M.SkybetaForwardHold(M.Params(fly=True))
    step(m, 0.)
    assert step(m, .1, **change).twist == (0., 0., 0.)
    assert m.phase == 'ABORT'
    assert step(m, .2).twist == (0., 0., 0.)
    assert m.phase == 'ABORT'


def test_offboard_loss_stops_publishing_permanently():
    m = M.SkybetaForwardHold(M.Params(fly=True))
    step(m, 0.)
    assert step(m, .1, offboard=False).twist is None
    assert step(m, .2).twist is None
    assert m.phase == 'ABORT'


def test_offboard_without_feedback_requests_zero_velocity():
    m = M.SkybetaForwardHold(M.Params(fly=True))
    out = m.step(M.State(t=0., offboard=True, armed=True))
    assert m.phase == 'ABORT' and out.twist == (0., 0., 0.)


def test_climb_timeout_stops_with_zero_velocity():
    m = M.SkybetaForwardHold(M.Params(fly=True, climb_timeout_s=1.))
    for i in range(13):
        out = step(m, i * .1)
    assert m.phase == 'HOLD' and out.twist == (0., 0., 0.)


def test_feedback_parser_uses_px4_velocity_and_heading_not_position():
    msg = SimpleNamespace(timestamp_sample=1_000_000, vx=1., vy=2., vz=-.3, heading=.7,
                          v_xy_valid=True, v_z_valid=True, heading_good_for_control=True,
                          vxy_reset_counter=1, vz_reset_counter=2, heading_reset_counter=3,
                          x=math.nan, y=math.nan, z=math.nan)
    fb = M.feedback_from_px4(msg)
    assert fb.velocity_ned == (1., 2., -.3) and fb.heading_ned == .7 and fb.valid
    assert fb.sample_t == 1. and fb.resets == (1, 2, 3)
    for flag in ('v_xy_valid', 'v_z_valid', 'heading_good_for_control'):
        setattr(msg, flag, False)
        assert not M.feedback_from_px4(msg).valid
        setattr(msg, flag, True)


def test_validation_rejects_combined_speed_above_interface_limit():
    with pytest.raises(ValueError):
        M.Params(climb_speed=1.4, forward_speed=1.4).validate()
    with pytest.raises(ValueError):
        M.Params(height=2.5).validate()


def test_forward_timeout_and_zero_distance_use_zero_velocity():
    m = M.SkybetaForwardHold(M.Params(fly=True, forward_timeout_s=.2))
    t, _ = reach_forward(m)
    for i in range(1, 5):
        out = step(m, t + i * .1)
    assert m.phase == 'HOLD' and out.twist == (0., 0., 0.)
    m = M.SkybetaForwardHold(M.Params(fly=True, forward_m=0.))
    # At climb completion the forward stage should immediately stop.
    step(m, 0., (0., 0., -.5))
    for i in range(1, 21):
        out = step(m, i * .1, (0., 0., -.5))
        if m.phase == 'HOLD':
            break
    assert m.phase == 'HOLD' and out.twist == (0., 0., 0.)


def test_estimated_height_drift_is_corrected_with_vertical_velocity():
    m = M.SkybetaForwardHold(M.Params(fly=True))
    t, _ = reach_forward(m)
    # Continued measured climb while travelling forward must command descent.
    out = step(m, t + .1, (0., 0., -2.))
    out = step(m, t + .2, (0., 0., -2.))
    assert m.phase == 'FORWARD' and out.twist[2] < 0.


def test_duplicate_receipts_do_not_refresh_feedback_age():
    msg = SimpleNamespace(timestamp_sample=1_000_000, vx=0., vy=0., vz=0., heading=0.,
                          v_xy_valid=True, v_z_valid=True, heading_good_for_control=True,
                          vxy_reset_counter=0, vz_reset_counter=0, heading_reset_counter=0)
    node = SimpleNamespace(feedback=M.feedback_from_px4(msg), feedback_t=123., feedback_fault=False,
                           mission=M.SkybetaForwardHold(M.Params(fly=True)), p=M.Params())
    M.ClimbForwardHoldSkybeta._on_feedback(node, msg)
    assert node.feedback_t == 123.


def test_estimator_reset_between_timer_ticks_remains_latched():
    msg = SimpleNamespace(timestamp_sample=1_000_000, vx=0., vy=0., vz=0., heading=0.,
                          v_xy_valid=True, v_z_valid=True, heading_good_for_control=True,
                          vxy_reset_counter=0, vz_reset_counter=0, heading_reset_counter=0)
    mission = M.SkybetaForwardHold(M.Params(fly=True))
    step(mission, 0.)
    node = SimpleNamespace(feedback=M.feedback_from_px4(msg), feedback_t=123., feedback_fault=False,
                           mission=mission, p=M.Params())
    msg.timestamp_sample += 100_000
    msg.vxy_reset_counter = 1
    M.ClimbForwardHoldSkybeta._on_feedback(node, msg)
    msg.timestamp_sample += 100_000
    M.ClimbForwardHoldSkybeta._on_feedback(node, msg)
    assert node.feedback_fault


def test_feedback_fault_stop_ends_permanently_when_pilot_takes_over():
    m = M.SkybetaForwardHold(M.Params(fly=True))
    step(m, 0.)
    assert step(m, .1, valid=False).twist == (0., 0., 0.)
    assert step(m, .2, offboard=False).twist is None
    assert step(m, .3).twist is None


@pytest.mark.parametrize('heading', [0., math.pi / 2, math.pi, -math.pi / 2])
def test_hover_only_never_commands_horizontal_motion_or_position(heading):
    m = M.SkybetaForwardHold(M.Params(fly=True, forward_m=0., hold_s=.5))
    velocity, seen = (0., 0., 0.), []
    for i in range(150):
        out = step(m, i * .1, velocity, heading=heading)
        seen.append(m.phase)
        assert out.twist[:2] == (0., 0.)
        velocity = (0., 0., -out.twist[2])
        if m.phase == 'DONE':
            break
    assert list(dict.fromkeys(seen)) == ['CLIMB', 'HOLD', 'DONE']
    assert m.height_m == pytest.approx(1., abs=.1)
    assert m.forward_m == 0.
    assert out.twist == (0., 0., 0.)
    assert step(m, (i + 1) * .1, heading=heading).twist == (0., 0., 0.)
