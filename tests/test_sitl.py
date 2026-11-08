"""Pure tests for the SITL rig's pieces (no Gazebo, no PX4, no docker): python3 -m pytest -q tests/test_sitl.py"""
import math
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
SITL = os.path.join(HERE, '..', 'sitl')
sys.path.insert(0, SITL)

import params as P  # noqa: E402

SKYBETA_PARAMS = os.environ.get('SKYBETA_PARAMS', os.path.join(HERE, '..', '..', 'skybeta', 'px4', 'params'))
need_skybeta = pytest.mark.skipif(not os.path.isdir(SKYBETA_PARAMS), reason='skybeta checkout not next to deploy')


# ── params.py ──
def write(tmp_path, name, text):
    p = tmp_path / name
    p.write_text(text)
    return str(p)


def test_read_params_types(tmp_path):
    f = write(tmp_path, 'a.params', '# c\n\n1\t1\tEKF2_EV_DELAY\t70.000000\t9\n1\t1\tEKF2_EV_CTRL\t9\t6\n1\t1\tX\t1.0\t6\n')
    assert P.read_params(f) == {'EKF2_EV_DELAY': '70.0', 'EKF2_EV_CTRL': '9', 'X': '1'}


def test_read_params_rejects_garbage(tmp_path):
    with pytest.raises(ValueError):
        P.read_params(write(tmp_path, 'a.params', 'EKF2_EV_CTRL 9\n'))


def test_overrides_parse_and_reject(tmp_path):
    assert P.read_overrides(write(tmp_path, 'o', 'A_B 1  # why\n\n# only a comment\n')) == {'A_B': '1'}
    with pytest.raises(ValueError):
        P.read_overrides(write(tmp_path, 'o2', 'A_B\n'))


@need_skybeta
def test_build_layers_and_filters(tmp_path):
    dump = write(tmp_path, 'd.params', '1\t1\tEKF2_EV_DELAY\t60.0\t9\n'      # allowed: overrides 03b's 70
                                        '1\t1\tMC_ROLLRATE_P\t0.2\t9\n'      # airframe gain: dropped
                                        '1\t1\tMPC_THR_HOVER\t0.3\t9\n'      # airframe: dropped
                                        '1\t1\tCA_ROTOR0_PX\t0.08\t9\n'      # geometry: dropped
                                        '1\t1\tCOM_RC_IN_MODE\t0\t6\n')      # replaced by the SITL override anyway
    ov = write(tmp_path, 'o', 'COM_RC_IN_MODE 1\n')
    text, merged = P.build(SKYBETA_PARAMS, dump, ov)
    assert merged['EKF2_EV_DELAY'] == '60.0'
    assert merged['EKF2_HGT_REF'] == '2' and merged['EKF2_EV_CTRL'] == '9' and merged['UXRCE_DDS_PTCFG'] == '2'
    assert merged['COM_RC_IN_MODE'] == '1' and merged['COM_RC_OVERRIDE'] == '3'
    for k in ('MC_ROLLRATE_P', 'MPC_THR_HOVER', 'CA_ROTOR0_PX', 'UXRCE_DDS_CFG', 'MAV_1_CONFIG', 'SER_TEL2_BAUD'):
        assert k not in merged
    lines = text.splitlines()
    src = next(i for i, ln in enumerate(lines) if ln.startswith('. ${R}etc/init.d-posix/airframes/4001_gz_x500'))
    assert all(not ln.startswith('param set-default') for ln in lines[:src])     # x500 first, ours after
    assert text.count('param set-default EKF2_EV_DELAY') == 1


@need_skybeta
def test_real_sitl_overrides_file_builds():
    _, merged = P.build(SKYBETA_PARAMS, None, os.path.join(SITL, 'px4_params.txt'))
    assert merged['COM_RC_IN_MODE'] == '1' and merged['NAV_DLL_ACT'] == '0'


# ── gz_vio.py (pure parts) ──
rclpy = pytest.importorskip('rclpy')
import gz_vio as V  # noqa: E402
import gz_depth as D  # noqa: E402


def test_quat_roundtrip():
    rng = np.random.default_rng(0)
    for _ in range(50):
        R = V.rotvec_to_mat(rng.normal(0, 1.5, 3))
        assert np.allclose(V.quat_to_mat(*V.mat_to_quat(R)), R, atol=1e-9)


def test_startup_aligned_frame():
    """The odom frame is base_link's startup pose: start at zero, forward motion along odom +x whatever the heading."""
    m = V.VioModel(sigma_p=0, sigma_a=0, drift_p=0, drift_yaw=0)
    R0 = V.rot_z(1.2)
    p, R = m.odom(0.0, np.array([3.0, -2.0, 0.3]), R0)
    assert np.allclose(p, 0) and np.allclose(R, np.eye(3))
    p, R = m.odom(1.0, np.array([3.0, -2.0, 0.3]) + R0 @ np.array([1.0, 0, 0.5]), V.rot_z(1.2 + 0.3))
    assert np.allclose(p, [1.0, 0, 0.5]) and np.allclose(R, V.rot_z(0.3))


def test_yaw_drift_rotates_about_gravity_only():
    m = V.VioModel(sigma_p=0, sigma_a=0, drift_p=0, drift_yaw=0.05, seed=3)
    m.odom(0.0, np.zeros(3), np.eye(3))
    p, R = m.odom(100.0, np.array([0, 0, 1.0]), np.eye(3))
    assert np.allclose(p, [0, 0, 1.0])                  # height untouched by heading drift
    assert abs(R[2, 2] - 1) < 1e-12 and abs(m.bias_yaw) > 0


def test_velocity_is_body_frame():
    R = V.rot_z(math.pi / 2)                            # nose along world +y
    hist = [(0.0, np.zeros(3), R), (0.1, np.array([0.0, 0.1, 0.0]), R)]
    v, w = V.velocity_body(hist)
    assert np.allclose(v, [1.0, 0, 0]) and np.allclose(w, 0)


# ── gz_depth.py (pure parts) ──
def test_intrinsics_match_drone_geometry():
    fx, fy, cx, cy = D.intrinsics(640, 400, 1.30376)
    assert abs(fx - 420) < 1.5 and (cx, cy) == (320, 200)
    assert abs(math.degrees(2 * math.atan(200 / fy)) - 51.0) < 0.2


def test_degrade_conventions():
    z = np.array([[0.2, 1.0, np.inf, 5.0]], np.float32)
    out = D.degrade(z, np.random.default_rng(0), min_range=0.33, noise_k=0.0, dropout=0.0)
    assert math.isnan(out[0, 0]) and out[0, 1] == 1.0 and math.isinf(out[0, 2]) and out[0, 3] == 5.0
    assert z[0, 0] == np.float32(0.2)                     # input untouched


def test_optical_rotation():
    """camera link (x forward) -> optical (z forward, x right, y down)."""
    x, y, z, w = D.quat_from_rpy(-math.pi / 2, 0, -math.pi / 2)
    R = V.quat_to_mat(w, x, y, z)
    assert np.allclose(R @ [0, 0, 1], [1, 0, 0], atol=1e-9)    # optical z = link forward
    assert np.allclose(R @ [1, 0, 0], [0, -1, 0], atol=1e-9)   # optical x = link right (-y in FLU)
    assert np.allclose(R @ [0, 1, 0], [0, 0, -1], atol=1e-9)   # optical y = link down
