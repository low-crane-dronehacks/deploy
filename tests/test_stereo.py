"""Stereo depth pieces (numpy; the matcher / rectification tests need OpenCV and are skipped without it)."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from policynav import stereo as S                                              # noqa: E402

FX, B = 421.0, 0.0757


def _P(fx, cx, cy, tx=0.0):
    return [fx, 0, cx, tx, 0, fx, cy, 0, 0, 0, 1, 0]


def test_disparity_to_depth_and_invalid():
    d = np.array([[0.0, 8.0, 16.0, 32.0]], np.float32)
    z = S.disparity_to_depth(d, FX, B, min_m=0.2, max_m=20.0)
    assert np.isnan(z[0, 0])
    assert np.allclose(z[0, 1:], FX * B / d[0, 1:])
    z = S.disparity_to_depth(d, FX, B, valid=np.array([[True, False, True, True]]))
    assert np.isnan(z[0, 1]) and np.isfinite(z[0, 2])
    assert np.isnan(S.disparity_to_depth(np.array([[400.0]]), FX, B, min_m=0.2)[0, 0])     # closer than min


def test_integer_disparity_rounds():
    z = S.disparity_to_depth(np.array([[7.6]]), FX, B, integer=True)
    assert np.isclose(z[0, 0], FX * B / 8.0)


def test_baseline_and_intrinsics_from_P():
    assert np.isclose(S.rect_baseline_m(_P(800.0, 512, 320, tx=-800.0 * B)), B)
    fx, fy, cx, cy = S.rectified_intrinsics(_P(800.0, 511.5, 319.5), 1024, 640, 640, 400)
    assert np.isclose(fx, 500.0) and np.isclose(fy, 500.0) and np.isclose(cx, 319.5) and np.isclose(cy, 199.5)


def test_gray_conversion():
    bgr = np.dstack([np.full((4, 4), 7, np.uint8)] * 3)
    assert S.to_gray8(bgr).shape == (4, 4) and S.to_gray8(bgr)[0, 0] == 7
    assert S.to_gray8(np.full((2, 2), 0x1234, np.uint16))[0, 0] == 0x12


def test_identity_rectification_is_a_resample():
    pytest.importorskip('cv2')
    import cv2
    K = [800.0, 0, 511.5, 0, 800.0, 319.5, 0, 0, 1]
    mx, my = S.rect_maps(K, [0, 0, 0, 0, 0], np.eye(3).ravel(), _P(800.0, 511.5, 319.5), 1024, 640, 640, 400)
    # output pixel (u, v) samples source pixel ((u + 0.5) * 1.6 - 0.5, ...)
    assert np.isclose(mx[200, 320], (320 + 0.5) * 1.6 - 0.5, atol=1e-3)
    assert np.isclose(my[200, 320], (200 + 0.5) * 1.6 - 0.5, atol=1e-3)
    assert cv2 is not None


def test_sgbm_recovers_a_plane_and_blanks_the_left_band():
    pytest.importorskip('cv2')
    rng = np.random.default_rng(0)
    H, W, d_true = 200, 320, 12
    tex = (rng.random((H, W + d_true)) * 255).astype(np.uint8)
    import cv2
    tex = cv2.GaussianBlur(tex, (3, 3), 0)
    left, right = tex[:, :W].copy(), tex[:, d_true:].copy()            # left pixel u matches right pixel u - d
    m, name = S.make_matcher(S.StereoParams(max_disparity=48), backend='sgbm')
    disp, valid = m(left, right)
    z = S.disparity_to_depth(disp, FX, B, valid=valid)
    core = z[20:-20, 80:-20]
    assert name == 'sgbm'
    assert np.nanmedian(core) == pytest.approx(FX * B / d_true, rel=0.03)
    assert np.isnan(z[:, :40]).mean() > 0.9                            # no-overlap band at the left edge


def test_rectify_from_extrinsics_puts_points_on_one_row():
    pytest.importorskip('cv2')
    # skybeta-17's OAK: raw per-eye K (different cx, cy, fx), right ~75 mm to +x, ~0.8 deg residual rotation
    K1 = np.array([[638.54, 0, 520.16], [0, 638.33, 303.46], [0, 0, 1]])
    K2 = np.array([[640.10, 0, 516.87], [0, 640.26, 320.29], [0, 0, 1]])
    ang = np.radians([0.04, 0.77, -0.42])
    import cv2
    R_lr, _ = cv2.Rodrigues(ang)
    t_lr = np.array([0.0753, -0.001, 0.0])
    m1, m2, fx, cx, cy, B_, R1 = S.stereo_rect_from_extrinsics(K1, np.zeros(5), K2, np.zeros(5), 1024, 640, R_lr,
                                                               t_lr, 640, 400)
    assert B_ == pytest.approx(0.0753, rel=0.02)
    rng = np.random.default_rng(1)
    for _ in range(20):
        X = np.array([rng.uniform(-0.8, 0.8), rng.uniform(-0.5, 0.5), rng.uniform(1.0, 4.0)])   # left optical
        xr = R1 @ X
        u, v = fx * xr[0] / xr[2] + cx, fx * xr[1] / xr[2] + cy
        if not (5 < u < 630 and 5 < v < 390):
            continue
        ur = u - fx * B_ / xr[2]
        Xr = R_lr.T @ (X - t_lr)                                         # the point in the right camera
        raw_l = K1 @ (X / X[2]); raw_r = K2 @ (Xr / Xr[2])
        iv, iu, iur = int(round(v)), int(round(u)), int(round(ur))
        if not (0 <= iur < 640):
            continue
        assert np.hypot(m1[0][iv, iu] - raw_l[0], m1[1][iv, iu] - raw_l[1]) < 2.0
        assert np.hypot(m2[0][iv, iur] - raw_r[0], m2[1][iv, iur] - raw_r[1]) < 2.0


def test_skybeta_camera_info_is_not_a_rectified_pair():
    P_l = _P(638.54, 520.16, 303.46, tx=-48.08)
    P_r = _P(640.10, 516.87, 320.29)
    assert not S.camera_info_is_rectified_pair(P_l, P_r, np.eye(3), np.eye(3))
    assert S.camera_info_is_rectified_pair(_P(500, 320, 200), _P(500, 320, 200, tx=-500 * B), np.eye(3), np.eye(3))
