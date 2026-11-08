"""Pure tests (numpy only, no ROS): python3 -m pytest -q tests"""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from policynav import mission as M, obs as O                                   # noqa: E402
from policynav.policy_np import NumpyPolicy                                    # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _vehicle():
    return M.Vehicle(pos=(0, 0, 0), vel=(0, 0, 0), q=(1, 0, 0, 0), pos_ok=True)


def test_never_arms_without_arm_true():
    m = M.Mission(M.MissionParams(arm=False))
    m.on_health(0.0, True)
    m.on_depth(0.0)
    stream, cmds = m.step(0.05, _vehicle())
    assert m.phase == 'wait' and not stream and cmds == []


def test_arms_flies_and_disarms_on_touchdown():
    m, v, t = M.Mission(M.MissionParams(arm=True)), _vehicle(), 0.0
    for _ in range(30):
        t += 0.1
        m.on_health(t, True)
        m.on_depth(t)
        _, cmds = m.step(t, v)
        v.armed |= 'arm' in cmds
        v.nav_state = 14 if 'offboard' in cmds else v.nav_state
    assert m.phase == 'fly'
    v.pos, v.landed = (0, 0, -1.0), False
    m.on_health(t + .1, True), m.on_depth(t + .1), m.step(t + .1, v)
    v.pos, v.landed = (0, 0, 0), True
    m.on_health(t + .2, True), m.on_depth(t + .2)
    assert m.step(t + .2, v) == (True, ['disarm']) and m.result == 'complete'


def test_forward_follows_heading():
    q = (math.cos(math.pi / 4), 0, 0, math.sin(math.pi / 4))                    # facing east
    vel, yaw = O.action_to_setpoint([1, 0, 0, 0], q, (5, 3, 2), 1.047)
    assert abs(vel[1] - 5) < 1e-9 and abs(vel[0]) < 1e-9 and yaw == 0.0


def test_invalid_depth_reads_near_and_shape():
    R = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], float)                   # optical -> body FLU
    rp = O.DepthReprojector((421.0, 421.0, 320.0, 200.0), (400, 640), R, np.zeros(3))
    d = np.full((400, 640), 2.0, np.float32)
    d[:, :320] = np.nan
    out = rp(d)
    assert out.shape == (80, 128)
    assert (out[:, :60] == 0).all() and np.allclose(out[:, 70:], 2.0 / 5.0, atol=0.01)


def test_policy_runs():
    pol = NumpyPolicy(os.path.join(ROOT, 'policies', 'comp_s2.npz'))
    a = pol(np.zeros((2, 80, 128), np.float32), np.zeros(pol.meta['proprio_dim'], np.float32))
    assert a.shape == (pol.meta['num_actions'],) and np.isfinite(a).all()


def test_edge_fill_right_band_takes_nearest_valid_inward():
    import numpy as np
    from policynav.obs import edge_fill
    out = np.tile(np.linspace(0.1, 0.9, 128, dtype=np.float32), (80, 1))
    valid = np.ones((80, 128), bool)
    valid[:, -14:] = False                         # drone 17's right-edge stereo band
    valid[:, :3] = False                           # and a thin left one
    out[~valid] = 0.0
    f, v = edge_fill(out, valid, 16)
    assert v.all()
    assert np.allclose(f[:, -14:], out[:, [-15]]) and np.allclose(f[:, :3], out[:, [3]])
    f2, v2 = edge_fill(out, valid, 0)
    assert (f2 == out).all() and (v2 == valid).all()
    valid[:, -20:] = False                          # band wider than the fill: inner columns stay invalid (near)
    out[~valid] = 0.0
    f3, v3 = edge_fill(out, valid, 16)
    assert not v3[:, -16:].any() and (f3[:, -16:] == 0).all()
