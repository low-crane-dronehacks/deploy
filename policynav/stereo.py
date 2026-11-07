"""Stereo depth from the OAK's raw mono pair (skybeta publishes no depth). Pure: numpy + OpenCV (+ VPI on the Jetson).

  rectify   per-camera undistort+rectify maps from the two CameraInfo messages (K, D, R, P), resampled straight
            to the matching resolution (default 640x400, what our stereo model and FlyX use)
  match     VPI census + SGM on the CUDA backend with a confidence threshold (robust to glare / exposure
            differences: low-confidence matches become holes, which the policy reads as near). OpenCV SGBM is
            the fallback (no VPI, or for offline comparison on a laptop).
  depth     z = f * B / d in metres; NaN = invalid (no match, low confidence, or outside [min, max] range)

The policy node consumes the result exactly like the OAK depth topic (decode_depth: NaN -> invalid -> near).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class StereoParams:
    out_w: int = 640
    out_h: int = 400
    max_disparity: int = 96          # search range (px at out_w): min range = f*B/max_disparity (~0.33 m at 640)
    window: int = 5                  # matching window
    conf_threshold: int = 32767      # VPI confidence (U16, 0..65535): below -> invalid. ~50% is VPI's sample default
    uniqueness: float = 0.95         # VPI: best cost must beat the 2nd best by this ratio (1 = off)
    p1: int = 3                      # SGM smoothness penalties (VPI defaults)
    p2: int = 48
    integer_disparity: bool = False  # True: whole-pixel steps like FlyX's matcher / the OAK default
    min_depth_m: float = 0.2
    max_depth_m: float = 20.0


# ── calibration ────────────────────────────────────────────────────────────────────────────────────────────────

def _as(m, shape):
    return np.asarray(m, np.float64).reshape(shape)


def rect_baseline_m(P_right) -> float:
    """Baseline from the right camera's projection matrix (P[0,3] = -fx' * B)."""
    P = _as(P_right, (3, 4))
    return float(-P[0, 3] / P[0, 0])


def rectified_intrinsics(P_left, src_w, src_h, out_w, out_h):
    """(fx, fy, cx, cy) of the rectified LEFT image at the output resolution."""
    P = _as(P_left, (3, 4))
    sx, sy = out_w / src_w, out_h / src_h
    return P[0, 0] * sx, P[1, 1] * sy, (P[0, 2] + 0.5) * sx - 0.5, (P[1, 2] + 0.5) * sy - 0.5


def rect_maps(K, D, R, P, src_w, src_h, out_w, out_h, model='plumb_bob'):
    """cv2 remap tables: output (rectified, out_w x out_h) pixel -> raw source pixel."""
    import cv2
    K, R, P = _as(K, (3, 3)), _as(R, (3, 3)), _as(P, (3, 4)).copy()
    D = np.asarray(D, np.float64).ravel()
    sx, sy = out_w / src_w, out_h / src_h
    P[0, :] *= sx                                          # scale the new camera matrix to the output grid
    P[1, :] *= sy
    P[0, 2] += 0.5 * sx - 0.5
    P[1, 2] += 0.5 * sy - 0.5
    newK = P[:3, :3]
    if model in ('equidistant', 'fisheye'):
        return cv2.fisheye.initUndistortRectifyMap(K, D[:4], R, newK, (out_w, out_h), cv2.CV_32FC1)
    return cv2.initUndistortRectifyMap(K, D, R, newK, (out_w, out_h), cv2.CV_32FC1)


def camera_info_is_rectified_pair(P_left, P_right, R_left, R_right, tol_px: float = 1.0) -> bool:
    """True when camera_info already describes a rectified stereo pair (common P, right Tx != 0). The skybeta OAK
    driver publishes raw per-eye K with R = I, D = 0 and right Tx = 0, which is NOT one: rectify from extrinsics."""
    Pl, Pr = _as(P_left, (3, 4)), _as(P_right, (3, 4))
    return (abs(Pr[0, 3]) > 1e-6 and np.allclose(Pl[:3, :3], Pr[:3, :3], atol=tol_px))


def stereo_rect_from_extrinsics(K1, D1, K2, D2, src_w, src_h, R_lr, t_lr, out_w, out_h, alpha: float = 0.0):
    """Rectify from raw intrinsics + the RIGHT camera's pose in the LEFT optical frame (R_lr, t_lr: p_left =
    R_lr p_right + t_lr, e.g. from TF left_optical -> right_optical). -> (maps_left, maps_right, fx, cx, cy, baseline_m, R1)

    Output is out_w x out_h directly (the matching resolution). R1 rotates left-camera points into the rectified
    frame (x_rect = R1 x_left)."""
    import cv2
    K1, K2 = _as(K1, (3, 3)), _as(K2, (3, 3))
    D1 = np.asarray(D1, np.float64).ravel()[:8] if len(D1) else np.zeros(5)
    D2 = np.asarray(D2, np.float64).ravel()[:8] if len(D2) else np.zeros(5)
    R_lr, t_lr = _as(R_lr, (3, 3)), np.asarray(t_lr, np.float64).reshape(3)
    R = R_lr.T                                           # stereoRectify wants x_right = R x_left + T
    T = (-R_lr.T @ t_lr).reshape(3, 1)
    sx, sy = out_w / src_w, out_h / src_h
    S_ = np.array([[sx, 0, 0.5 * sx - 0.5], [0, sy, 0.5 * sy - 0.5], [0, 0, 1.0]])
    K1s, K2s = S_ @ K1, S_ @ K2                          # intrinsics re-expressed on the output grid
    R1, R2, P1, P2, _, _, _ = cv2.stereoRectify(K1s, D1, K2s, D2, (out_w, out_h), R, T,
                                                flags=cv2.CALIB_ZERO_DISPARITY, alpha=alpha)
    # remap: output pixel -> SOURCE pixel, so build with the source-resolution K and the output-grid P
    m1 = cv2.initUndistortRectifyMap(K1, D1, R1, P1[:3, :3], (out_w, out_h), cv2.CV_32FC1)
    m2 = cv2.initUndistortRectifyMap(K2, D2, R2, P2[:3, :3], (out_w, out_h), cv2.CV_32FC1)
    baseline = float(-P2[0, 3] / P2[0, 0])
    return m1, m2, float(P1[0, 0]), float(P1[0, 2]), float(P1[1, 2]), baseline, R1


def to_gray8(img: np.ndarray) -> np.ndarray:
    """mono8 / bgr8 / rgb8 / mono16 -> uint8 gray (the OAK mono eyes arrive as bgr8 with equal channels)."""
    if img.ndim == 3:
        img = img[..., 0] if img.shape[2] >= 1 else img
    if img.dtype == np.uint16:
        img = (img >> 8).astype(np.uint8)
    return np.ascontiguousarray(img.astype(np.uint8, copy=False))


# ── disparity -> depth ─────────────────────────────────────────────────────────────────────────────────────────

def disparity_to_depth(disp: np.ndarray, f_px: float, baseline_m: float, valid: np.ndarray | None = None,
                       min_m: float = 0.2, max_m: float = 20.0, integer: bool = False) -> np.ndarray:
    """(H,W) disparity in px -> depth in metres (float32), NaN where invalid."""
    d = np.asarray(disp, np.float32)
    if integer:
        d = np.floor(d + 0.5)
    ok = d > 0
    if valid is not None:
        ok &= valid
    z = np.full(d.shape, np.nan, np.float32)
    z[ok] = (f_px * baseline_m) / d[ok]
    z[(z < min_m) | (z > max_m)] = np.nan
    return z


# ── matchers ───────────────────────────────────────────────────────────────────────────────────────────────────

class VpiMatcher:
    """VPI stereo disparity (census + SGM), CUDA backend. Output disparity is S16 Q10.5 (divide by 32)."""

    def __init__(self, p: StereoParams):
        import vpi
        self.vpi, self.p = vpi, p

    def __call__(self, left8: np.ndarray, right8: np.ndarray):
        vpi, p = self.vpi, self.p
        with vpi.Backend.CUDA:
            L = vpi.asimage(left8).convert(vpi.Format.Y16_ER, scale=256)
            R = vpi.asimage(right8).convert(vpi.Format.Y16_ER, scale=256)
            conf = vpi.Image(L.size, vpi.Format.U16)
            try:
                disp = vpi.stereodisp(L, R, out_confmap=conf, backend=vpi.Backend.CUDA, window=p.window,
                                      maxdisp=p.max_disparity, confthreshold=p.conf_threshold,
                                      uniqueness=p.uniqueness, p1=p.p1, p2=p.p2)
            except TypeError:                                  # older bindings: only the core arguments
                disp = vpi.stereodisp(L, R, out_confmap=conf, backend=vpi.Backend.CUDA, window=p.window,
                                      maxdisp=p.max_disparity, confthreshold=p.conf_threshold)
        with disp.rlock_cpu() as d, conf.rlock_cpu() as c:
            d = np.asarray(d).astype(np.float32) / 32.0
            c = np.asarray(c)
        return d, (c >= p.conf_threshold) & (d > 0)


class SgbmMatcher:
    """OpenCV SGBM fallback (CPU). Left-right check + uniqueness + speckle filter turn doubtful matches into holes."""

    def __init__(self, p: StereoParams):
        import cv2
        nd = int(np.ceil(p.max_disparity / 16.0) * 16)
        self.sgbm = cv2.StereoSGBM_create(minDisparity=0, numDisparities=nd, blockSize=p.window,
                                          P1=8 * p.window * p.window, P2=32 * p.window * p.window,
                                          disp12MaxDiff=1, uniquenessRatio=10, speckleWindowSize=100,
                                          speckleRange=2, preFilterCap=31, mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)

    def __call__(self, left8: np.ndarray, right8: np.ndarray):
        d = self.sgbm.compute(left8, right8).astype(np.float32) / 16.0      # fixed point x16; invalid = (min-1)*16
        return d, d > 0


def make_matcher(p: StereoParams, backend: str = 'auto'):
    """backend: vpi | sgbm | auto (VPI if importable, else SGBM). -> (matcher, name)"""
    if backend in ('vpi', 'auto'):
        try:
            return VpiMatcher(p), 'vpi'
        except Exception:
            if backend == 'vpi':
                raise
    return SgbmMatcher(p), 'sgbm'
