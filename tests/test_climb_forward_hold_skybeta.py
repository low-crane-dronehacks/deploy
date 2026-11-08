"""Pure tests for the skybeta climb -> forward -> hold mission (no ROS)."""
import math
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'onboard', 'examples'))
import climb_forward_hold_skybeta as M                                        # noqa: E402


def _run(m, s, pos, yaw=0.0, offboard=True, age=0.0):
    return m.step(M.State(t=s, pos=pos, yaw=yaw, odom_age=age, offboard=offboard, armed=offboard))


def test_waits_with_zero_twist_and_never_requests_anything():
    m = M.SkybetaForwardHold(M.Params(fly=True))
    out = _run(m, 0.0, (0, 0, 0), offboard=False)
    assert m.phase == 'WAIT' and out.twist == (0.0, 0.0, 0.0) and out.pose is None and not out.hold


def test_full_sequence_climb_forward_hold_done():
    p = M.Params(fly=True, height=1.0, forward_m=1.0, hold_s=1.0)
    m = M.SkybetaForwardHold(p)
    yaw = math.radians(90)                         # facing north (ENU): forward = +y
    pos, t, dt = [0.0, 0.0, 0.0], 0.0, 0.1
    seen = []
    for _ in range(400):
        out = _run(m, t, tuple(pos), yaw)
        seen.append(m.phase)
        if out.twist:
            assert math.sqrt(sum(c * c for c in out.twist)) <= 1.4 + 1e-9
            for i in range(3):
                pos[i] += out.twist[i] * dt
        t += dt
        if m.phase == 'DONE':
            break
    assert [ph for i, ph in enumerate(seen) if i == 0 or ph != seen[i - 1]] == ['CLIMB', 'FORWARD', 'HOLD', 'DONE']
    assert abs(pos[2] - 1.0) < 0.1 and abs(pos[1] - 1.0) < 0.1 and abs(pos[0]) < 0.05
    assert out.pose is not None and out.twist is None       # DONE keeps publishing the hold pose


def test_climbs_immediately_when_offboard_starts():
    m = M.SkybetaForwardHold(M.Params(fly=True))
    _run(m, 0.0, (0, 0, 0), offboard=False)
    out = _run(m, 0.1, (0, 0, 0))
    assert m.phase == 'CLIMB' and out.twist[2] > 0                    # ENU up: takes off (PX4's 10 s rule)


def test_offboard_loss_aborts_and_stops_publishing():
    m = M.SkybetaForwardHold(M.Params(fly=True))
    _run(m, 0.0, (0, 0, 0))
    out = _run(m, 0.1, (0, 0, 0.2), offboard=False)
    assert m.phase == 'ABORT' and out.twist is None and out.pose is None
    out = _run(m, 0.2, (0, 0, 0.2), offboard=True)                   # never resumes on its own
    assert m.phase == 'ABORT' and out.twist is None


def test_stale_odometry_calls_hold():
    m = M.SkybetaForwardHold(M.Params(fly=True))
    _run(m, 0.0, (0, 0, 0))
    out = _run(m, 0.1, (0, 0, 0.3), age=1.0)
    assert m.phase == 'ABORT' and out.hold and out.twist is None


def test_offboard_without_odometry_holds():
    m = M.SkybetaForwardHold(M.Params(fly=True))
    out = m.step(M.State(t=0.0, pos=None, offboard=True))
    assert m.phase == 'ABORT' and out.hold


def test_climb_timeout_holds_in_place():
    m = M.SkybetaForwardHold(M.Params(fly=True, climb_timeout_s=1.0))
    _run(m, 0.0, (0, 0, 0))
    out = _run(m, 2.0, (0, 0, 0.4))                                  # stuck (e.g. clamped)
    assert m.phase == 'HOLD' and out.pose == (0, 0, 0.4)


def test_validation():
    with pytest.raises(ValueError):
        M.Params(forward_speed=1.6).validate()
    with pytest.raises(ValueError):
        M.Params(height=2.5).validate()
    assert abs(M.enu_yaw(0, 0, math.sin(0.5), math.cos(0.5)) - 1.0) < 1e-9
