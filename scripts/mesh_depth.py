#!/usr/bin/env python3
"""Synthesized OAK-D depth from a triangle mesh (NumPy only): exact ray casting + the dronehacks-sim stereo model.

Replaces the point-splat of the .ply scan / of surface samples in the depth spoofers:

  MeshDepthCamera   exact z-depth of the course mesh (every triangle, no splat cracks or pinholes) at a pinhole
                    camera, by rasterising each triangle's screen bounding box and testing the pixel rays against
                    it (signed triple products, valid for triangles crossing the camera plane too). Misses = +inf.
  StereoDepthModel  dronehacks-sim ``realism.py`` 'depth' effect (the model the RL policies were trained/evaluated
                    with), ported from torch: OAK-D S2 passive stereo at 640 px, baseline 7.5 cm; depth from noisy
                    (0.2 px), 1/8 px quantised disparity (error grows with z^2); nothing closer than f*B/95 px
                    (~0.33 m); spatially correlated textureless holes (4 %); stereo occlusion shadows beside
                    foreground edges and the no-match band at the left border. Invalid = NaN, miss stays +inf.

    from mesh_depth import MeshDepthCamera, StereoDepthModel
    cam = MeshDepthCamera(scene.triangles(), w=160, h=100, f=105.25, cx=79.5, cy=49.5)
    z = cam.render(R_wc, t_wc)                      # (h, w) metres, +inf = no hit
    z = StereoDepthModel(w=160, h=100, f_sensor=421.0, rng=rng).apply(z)

``python3 scripts/mesh_depth.py [scene.npz]`` renders one view and prints timing / statistics.
"""
from __future__ import annotations

import time

import numpy as np
from scipy.ndimage import zoom


class MeshDepthCamera:
    """Exact depth of a static triangle soup (world frame) from a pinhole camera.

    Camera convention: optical frame (x right, y down, z forward); ``R_wc`` maps camera -> world, ``t_wc`` is the
    camera centre in the world. Pixel (u, v) looks along ((u - cx) / f, (v - cy) / f, 1); the result is the
    camera-plane depth z (what a stereo camera reports)."""

    def __init__(self, triangles: np.ndarray, w: int, h: int, f: float, cx: float, cy: float,
                 near: float = 0.05, far: float = 12.0) -> None:
        tri = np.asarray(triangles, dtype=np.float64)
        area2 = np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
        self.tri = tri[area2 > 1e-12]
        self.w, self.h, self.f, self.cx, self.cy = int(w), int(h), float(f), float(cx), float(cy)
        self.near, self.far = float(near), float(far)
        self.last_pairs = 0

    def _bbox(self, V: np.ndarray):
        """Pixel bbox (u0, u1, v0, v1 inclusive) of each camera-frame triangle's part in front of the near plane."""
        z = V[..., 2]
        front = z > self.near
        # near-plane crossings on the 3 edges (a->b, b->c, c->a)
        A, B = V, np.roll(V, -1, axis=1)
        za, zb = A[..., 2], B[..., 2]
        cross = (za > self.near) != (zb > self.near)
        with np.errstate(divide="ignore", invalid="ignore"):
            s = (self.near - za) / (zb - za)
            P = A + s[..., None] * (B - A)
        pts = np.concatenate([V, P], axis=1)                          # (F, 6, 3)
        ok = np.concatenate([front, cross], axis=1)
        zz = np.where(ok, pts[..., 2], 1.0)
        u = np.where(ok, self.f * pts[..., 0] / zz + self.cx, np.nan)
        v = np.where(ok, self.f * pts[..., 1] / zz + self.cy, np.nan)
        with np.errstate(invalid="ignore"):
            u0 = np.floor(np.nanmin(u, 1)).astype(np.int64)
            u1 = np.ceil(np.nanmax(u, 1)).astype(np.int64)
            v0 = np.floor(np.nanmin(v, 1)).astype(np.int64)
            v1 = np.ceil(np.nanmax(v, 1)).astype(np.int64)
        return np.clip(u0, 0, self.w), np.clip(u1, -1, self.w - 1), np.clip(v0, 0, self.h), np.clip(v1, -1, self.h - 1)

    def render(self, R_wc: np.ndarray, t_wc: np.ndarray) -> np.ndarray:
        V = (self.tri - np.asarray(t_wc, dtype=np.float64)) @ np.asarray(R_wc, dtype=np.float64)   # world -> camera
        z = V[..., 2]
        keep = (z.max(1) > self.near) & (z.min(1) < self.far)
        # side culling (only where the whole triangle is in front, so x/z is meaningful)
        allf = z.min(1) > self.near
        with np.errstate(divide="ignore", invalid="ignore"):
            xu = self.f * V[..., 0] / z + self.cx
            yv = self.f * V[..., 1] / z + self.cy
        out_side = allf & ((xu.max(1) < -1) | (xu.min(1) > self.w) | (yv.max(1) < -1) | (yv.min(1) > self.h))
        V = V[keep & ~out_side]
        img = np.full(self.w * self.h, np.inf)
        if len(V) == 0:
            self.last_pairs = 0
            return img.reshape(self.h, self.w).astype(np.float32)
        u0, u1, v0, v1 = self._bbox(V)
        bw, bh = u1 - u0 + 1, v1 - v0 + 1
        good = (bw > 0) & (bh > 0)
        V, u0, v0, bw, bh = V[good], u0[good], v0[good], bw[good], bh[good]
        n = bw * bh
        N = int(n.sum())
        self.last_pairs = N
        if N == 0:
            return img.reshape(self.h, self.w).astype(np.float32)
        a, b, c = V[:, 0], V[:, 1], V[:, 2]
        cab, cbc, cca = np.cross(a, b), np.cross(b, c), np.cross(c, a)
        nrm = np.cross(b - a, c - a)
        na = np.einsum("ij,ij->i", nrm, a)
        ti = np.repeat(np.arange(len(V)), n)
        local = np.arange(N) - np.repeat(np.cumsum(n) - n, n)
        u = u0[ti] + local % bw[ti]
        v = v0[ti] + local // bw[ti]
        dx = (u - self.cx) / self.f
        dy = (v - self.cy) / self.f

        def dot(m):  # m (T,3) per triangle -> per pair d . m, with d = (dx, dy, 1)
            return dx * m[ti, 0] + dy * m[ti, 1] + m[ti, 2]

        s0, s1, s2 = dot(cab), dot(cbc), dot(cca)
        inside = ((s0 >= 0) & (s1 >= 0) & (s2 >= 0)) | ((s0 <= 0) & (s1 <= 0) & (s2 <= 0))
        nd = dot(nrm)
        with np.errstate(divide="ignore", invalid="ignore"):
            t = na[ti] / nd
        hit = inside & (np.abs(nd) > 1e-12) & (t > self.near) & (t < self.far)
        pix = (v * self.w + u)[hit]
        np.minimum.at(img, pix, t[hit])
        return img.reshape(self.h, self.w).astype(np.float32)


class StereoDepthModel:
    """dronehacks-sim ``RealismCfg`` depth effect (``realism.py`` ``_corrupt_depth_torch``), NumPy port.

    Works on a render at (h, w) whose horizontal field of view spans ``sensor_width_px`` sensor pixels with focal
    ``f_sensor`` (the OAK's 640 px / fx 421). Defaults are dronehacks-sim's: baseline 0.075 m, 95 px max
    disparity, sigma 0.2 px, 3 subpixel bits, 4 % holes of ~60 sensor px, occlusion on."""

    def __init__(self, w: int, h: int, f_sensor: float = 421.0, sensor_width_px: int = 640,
                 baseline_m: float = 0.075, max_disparity_px: float = 95.0, disparity_sigma_px: float = 0.2,
                 subpixel_bits: int = 3, hole_frac: float = 0.04, hole_scale_sensor_px: float = 60.0,
                 occlusion: bool = True, rng: np.random.Generator | None = None) -> None:
        self.w, self.h = int(w), int(h)
        self.fB = float(f_sensor) * float(baseline_m)
        self.px_scale = self.w / float(sensor_width_px)               # render px per sensor px
        self.max_disp, self.sigma = float(max_disparity_px), float(disparity_sigma_px)
        self.q = float(2 ** int(subpixel_bits))
        self.hole_frac = float(hole_frac)
        self.hole_px = max(1.0, hole_scale_sensor_px * self.px_scale)  # blob scale in render px
        self.occlusion = bool(occlusion)
        self.rng = rng if rng is not None else np.random.default_rng()
        self.z_min = self.fB / self.max_disp

    def apply(self, z: np.ndarray) -> np.ndarray:
        """z (h, w) metres, +inf = miss -> metres with NaN = invalid stereo pixel, +inf = miss."""
        z = np.asarray(z, dtype=np.float64)
        miss = ~np.isfinite(z)
        zc = np.where(miss, 1e3, np.maximum(z, 1e-3))
        d = self.fB / zc                                               # disparity in sensor px
        d = d + self.rng.normal(0.0, self.sigma, d.shape)
        d = np.round(d * self.q) / self.q
        invalid = (d > self.max_disp) | (d <= 0)
        out = self.fB / np.maximum(d, 1e-6)
        if self.hole_frac > 0:
            gh, gw = max(2, int(self.h / self.hole_px)), max(2, int(self.w / self.hole_px))
            field = zoom(self.rng.random((gh, gw)), (self.h / gh, self.w / gw), order=1, mode="nearest")[: self.h, : self.w]
            invalid |= field < self.hole_frac
        if self.occlusion:
            d_r = (self.fB / zc) * self.px_scale                       # disparity in render px (true depth)
            u = np.arange(self.w, dtype=np.float64)[None, :]
            xr = u - d_r                                               # column in the right image
            suffix = np.minimum.accumulate(xr[:, ::-1], axis=1)[:, ::-1]
            right_min = np.concatenate([suffix[:, 1:], np.full((self.h, 1), np.inf)], axis=1)
            invalid |= ((right_min <= xr - 0.5) | (xr < 0)) & ~miss
        out = np.where(invalid & ~miss, np.nan, out)
        out[miss] = np.inf
        return out.astype(np.float32)


def _demo(argv=None) -> None:
    import argparse
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import real_world_scene as rws

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scene", nargs="?", default=str(rws.DEFAULT_SCENE))
    ap.add_argument("--pos", type=float, nargs=3, default=[-1.0, 0.5, -0.8], help="camera centre NED")
    ap.add_argument("--yaw-deg", type=float, default=-90.0)
    a = ap.parse_args(argv)
    s = rws.RealWorldScene(rws.resolve_scene(a.scene))
    cam = MeshDepthCamera(s.triangles(rws.DEPTH_GROUPS), 160, 100, 421.0 / 4, 79.5, 49.5)
    y = np.radians(a.yaw_deg)
    R_nb = np.array([[np.cos(y), -np.sin(y), 0], [np.sin(y), np.cos(y), 0], [0, 0, 1.0]])
    R_wc = R_nb @ np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=float)
    model = StereoDepthModel(160, 100, rng=np.random.default_rng(0))
    t0 = time.perf_counter()
    for _ in range(10):
        z = cam.render(R_wc, np.array(a.pos))
    t1 = time.perf_counter()
    zn = model.apply(z)
    t2 = time.perf_counter()
    fin = np.isfinite(z)
    print(f"{len(cam.tri)} triangles, {cam.last_pairs} pixel tests, render {1e3 * (t1 - t0) / 10:.1f} ms, "
          f"stereo {1e3 * (t2 - t1):.1f} ms")
    print(f"hits {fin.mean():.1%} (z {np.nanmin(np.where(fin, z, np.nan)):.2f}..{np.nanmax(np.where(fin, z, np.nan)):.2f} m), "
          f"after stereo: invalid {np.isnan(zn).mean():.1%}, miss {np.isinf(zn).mean():.1%}")


if __name__ == "__main__":
    _demo()
