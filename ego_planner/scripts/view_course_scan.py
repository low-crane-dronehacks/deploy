#!/usr/bin/env python3
"""Quick look: a surveyed point cloud (.ply, e.g. a Scaniverse scan) next to the course prior.

    python3 ego_planner/scripts/view_course_scan.py assets/dronehacks-course-scan-2026-11-06.ply
    python3 ego_planner/scripts/view_course_scan.py scan.ply --save /tmp/scan_vs_prior.png
    python3 ego_planner/scripts/view_course_scan.py scan.ply --yaw 92 --tx 0.3 --ty -1.1   # manual alignment
    python3 ego_planner/scripts/view_course_scan.py scan.ply --3d

The scan has an arbitrary frame, so it is aligned to the prior (assets/arena_prior.bin, NED) in 2D:
  1. z-up scan -> z-down (x, y, z) -> (y, x, -z), with height measured from the scan's floor;
  2. for every yaw (1 deg steps; also mirrored, unless --mirror says otherwise), the wall band
     (--band, default 0.3-1.5 m) is rasterised at 5 cm and FFT-correlated with the prior's walls,
     which gives the best translation for that yaw; per handedness the best-covering (yaw, translation)
     is kept, and the one whose wall band lies closest to the prior surface (median) wins.
Resolution of the fit is one cell (5 cm) and 1 deg. It is a visual aid, not the online registration.

Panels: the scan in NED (RGB), the prior (walls, gates, pad, launch mark, net), and the overlay coloured by
each scan point's distance to the nearest prior surface, so as-built differences stand out (red > --diff_m).
Needs numpy + matplotlib only.
"""
from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np

DEPLOY_ROOT = Path(__file__).resolve().parents[2]
PLY_TYPES = {'char': 'i1', 'uchar': 'u1', 'int8': 'i1', 'uint8': 'u1', 'short': '<i2', 'ushort': '<u2',
             'int16': '<i2', 'uint16': '<u2', 'int': '<i4', 'uint': '<u4', 'int32': '<i4', 'uint32': '<u4',
             'float': '<f4', 'float32': '<f4', 'double': '<f8', 'float64': '<f8'}


def load_ply(path: Path) -> tuple[np.ndarray, np.ndarray | None]:
    """Vertices (N,3) float64 and colours (N,3) in [0,1] (or None) of a binary little-endian / ASCII PLY."""
    with open(path, 'rb') as f:
        header = []
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f'{path}: no end_header')
            line = line.decode('ascii', 'replace').strip()
            header.append(line)
            if line == 'end_header':
                break
        offset = f.tell()
    fmt = next(l.split()[1] for l in header if l.startswith('format'))
    count, props, in_vertex, before = 0, [], False, 0
    for l in header:
        t = l.split()
        if t[0] == 'element':
            in_vertex = t[1] == 'vertex'
            if in_vertex:
                count = int(t[2])
            elif count == 0:
                before += 1
        elif t[0] == 'property' and in_vertex:
            if t[1] == 'list':
                raise ValueError('list properties on vertices are not supported')
            props.append((t[2], PLY_TYPES[t[1]]))
    if before:
        raise ValueError('elements before "vertex" are not supported')
    if fmt == 'binary_little_endian':
        data = np.fromfile(path, dtype=np.dtype(props), count=count, offset=offset)
    elif fmt == 'ascii':
        raw = np.loadtxt(path, skiprows=len(header), max_rows=count)
        data = np.zeros(count, dtype=np.dtype(props))
        for k, (name, _) in enumerate(props):
            data[name] = raw[:, k]
    else:
        raise ValueError(f'PLY format {fmt} not supported')
    xyz = np.stack([data['x'], data['y'], data['z']], 1).astype(np.float64)
    names = data.dtype.names
    rgb = None
    if all(c in names for c in ('red', 'green', 'blue')):
        rgb = np.stack([data['red'], data['green'], data['blue']], 1).astype(np.float64)
        rgb /= 255.0 if rgb.max() > 1.0 else 1.0
    ok = np.isfinite(xyz).all(1)
    return xyz[ok], (rgb[ok] if rgb is not None else None)


class Prior:
    """assets/arena_prior.bin (clearance grid, NED, node-centred at lo + res*i) + arena_prior.json."""

    def __init__(self, bin_path: Path, json_path: Path):
        b = bin_path.read_bytes()
        magic, _, nx, ny, nz, res, lx, ly, lz = struct.unpack('<4sIiiiffff', b[:36])
        if magic != b'EGOP':
            raise ValueError(f'{bin_path}: bad magic {magic!r}')
        self.meta = json.loads(json_path.read_text())
        scale = self.meta['grid'].get('distance_scale', 200.0)
        self.dist = np.frombuffer(b[36:], np.uint8).reshape(nx, ny, nz).astype(np.float32) / scale
        self.occ = self.dist <= self.meta['grid'].get('occupied_threshold_m', 0.035)
        self.res = float(res)
        self.lo = np.array([lx, ly, lz], dtype=np.float64)
        self.shape = np.array([nx, ny, nz])

    def axis(self, k: int) -> np.ndarray:
        return self.lo[k] + self.res * np.arange(self.shape[k])

    def footprint(self, h_lo: float, h_hi: float) -> np.ndarray:
        """2D (nx, ny) occupancy of the heights [h_lo, h_hi] (height = -z)."""
        h = -self.axis(2)
        return self.occ[:, :, (h >= h_lo) & (h <= h_hi)].any(2)

    def distance(self, p: np.ndarray) -> np.ndarray:
        """Clearance (m, nearest node; saturates at 1.275 m) of NED points; NaN outside the grid."""
        idx = np.rint((p - self.lo) / self.res).astype(int)
        ok = ((idx >= 0) & (idx < self.shape)).all(1)
        out = np.full(len(p), np.nan, dtype=np.float32)
        i = idx[ok]
        out[ok] = self.dist[i[:, 0], i[:, 1], i[:, 2]]
        return out


def floor_height(z_up: np.ndarray) -> float:
    """Floor = densest 2 cm slab in the lowest metre of the scan."""
    lo = np.percentile(z_up, 0.5)
    hist, edges = np.histogram(z_up, bins=np.arange(lo - 0.2, lo + 1.0, 0.02))
    k = int(np.argmax(hist))
    return 0.5 * (edges[k] + edges[k + 1])


def to_ned(xyz: np.ndarray, up: str) -> np.ndarray:
    """Scan (right-handed, `up` axis vertical) -> right-handed z-down frame, z = -height above the floor."""
    if up == 'y':          # (x, y, z) y-up -> z-up (x, -z, y)
        xyz = np.stack([xyz[:, 0], -xyz[:, 2], xyz[:, 1]], 1)
    out = np.stack([xyz[:, 1], xyz[:, 0], -xyz[:, 2]], 1)
    out[:, 2] += floor_height(xyz[:, 2])
    return out


def rasterise(p2: np.ndarray, res: float) -> tuple[np.ndarray, np.ndarray]:
    origin = p2.min(0)
    ij = np.rint((p2 - origin) / res).astype(int)
    img = np.zeros(ij.max(0) + 1, dtype=np.float32)
    np.add.at(img, (ij[:, 0], ij[:, 1]), 1.0)
    return img, origin


def dilate(img: np.ndarray) -> np.ndarray:
    out = img.copy()
    for a in (0, 1):
        for s in (1, -1):
            out = np.maximum(out, np.roll(img, s, axis=a))
    return out


def rot2(yaw_deg: float) -> np.ndarray:
    c, s = np.cos(np.radians(yaw_deg)), np.sin(np.radians(yaw_deg))
    return np.array([[c, -s], [s, c]])


def align(p_ned: np.ndarray, prior: Prior, band: tuple[float, float], mirror: str, step_deg: float = 1.0):
    """Best 2D (mirror, yaw, t) so that mirror -> rotate -> translate puts the scan wall band on the prior walls."""
    h = -p_ned[:, 2]
    wall = p_ned[(h >= band[0]) & (h <= band[1]), :2]
    if len(wall) < 100:
        raise SystemExit(f'only {len(wall)} scan points in the {band} m height band; check --up / --band')
    if len(wall) > 200_000:
        wall = wall[np.random.default_rng(0).choice(len(wall), 200_000, replace=False)]
    P = prior.footprint(*band).astype(np.float32)
    best = {}
    for m in {'auto': (False, True), 'no': (False,), 'yes': (True,)}[mirror]:
        w = wall * np.array([1.0, -1.0]) if m else wall
        for yaw in np.arange(0.0, 360.0, step_deg):
            S, origin = rasterise(w @ rot2(yaw).T, prior.res)
            S = dilate((S >= 2).astype(np.float32))
            shp = np.array(P.shape) + np.array(S.shape)
            corr = np.fft.irfft2(np.fft.rfft2(P, shp) * np.conj(np.fft.rfft2(S, shp)), shp)
            k = np.unravel_index(int(np.argmax(corr)), corr.shape)
            score = float(corr[k]) / max(P.sum(), 1.0)
            k = np.where(np.array(k) > np.array(P.shape) - 1, np.array(k) - shp, np.array(k))
            t = prior.lo[:2] + k * prior.res - origin
            if m not in best or score > best[m][0]:
                best[m] = (score, float(yaw), t)
    return best


def apply(p_ned: np.ndarray, mirror: bool, yaw: float, t: np.ndarray) -> np.ndarray:
    q = p_ned.copy()
    if mirror:
        q[:, 1] = -q[:, 1]
    q[:, :2] = q[:, :2] @ rot2(yaw).T + t
    return q


def draw_course(ax, prior: Prior, band, walls=True):
    """Top-down, east right / north up (x_ned on the vertical axis)."""
    m = prior.meta
    if walls:
        fp = prior.footprint(*band)
        ax.imshow(np.where(fp, 1.0, np.nan), origin='lower', cmap='Greys', vmin=0, vmax=1.3,
                  extent=[prior.lo[1] - prior.res / 2, prior.lo[1] + prior.res * (prior.shape[1] - 0.5),
                          prior.lo[0] - prior.res / 2, prior.lo[0] + prior.res * (prior.shape[0] - 0.5)],
                  interpolation='nearest', zorder=1)
    a = m['arena']['arena_half_m']
    ax.plot([-a, a, a, -a, -a], [-a, -a, a, a, -a], color='tab:blue', lw=1, ls='--', zorder=3, label='net')
    for g in m['gates']:
        (an, ae), (bn, be) = g['A_ned'], g['B_ned']
        ax.plot([ae, be], [an, bn], color='tab:orange', lw=2.5, zorder=4)
        ax.annotate(f"{g['index']} {g['name']}", (g['center_ned'][1], g['center_ned'][0]), fontsize=7,
                    color='tab:orange', zorder=5, xytext=(3, 3), textcoords='offset points')
    lm = m['landmarks']
    pn, pe = lm['pad_center_ned'][:2]
    ax.add_patch(__import__('matplotlib.patches', fromlist=['Circle']).Circle(
        (pe, pn), lm['pad_radius_m'], fill=False, color='tab:green', lw=1.5, zorder=4))
    ax.plot(pe, pn, '+', color='tab:green', zorder=4)
    ln, le = lm['launch_mark_ned'][:2]
    ax.plot(le, ln, 'x', color='tab:purple', ms=8, mew=2, zorder=4, label='launch')
    ax.set_aspect('equal')
    ax.set_xlabel('east  y_ned [m]')
    ax.set_ylabel('north  x_ned [m]')


def thin(n: int, k: int, seed: int = 1) -> np.ndarray:
    return np.arange(n) if n <= k else np.random.default_rng(seed).choice(n, k, replace=False)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('ply', type=Path)
    ap.add_argument('--prior', type=Path, default=DEPLOY_ROOT / 'assets' / 'arena_prior.bin')
    ap.add_argument('--up', choices=('z', 'y'), default='z', help='vertical axis of the scan (default z)')
    ap.add_argument('--band', type=float, nargs=2, default=(0.3, 1.5), metavar=('LO', 'HI'),
                    help='height band [m] used for the fit and the overlay (default 0.3 1.5)')
    ap.add_argument('--mirror', choices=('auto', 'no', 'yes'), default='auto')
    ap.add_argument('--yaw', type=float, help='manual yaw [deg] (skips the search; give --tx/--ty too)')
    ap.add_argument('--tx', type=float, default=0.0, help='manual north offset [m]')
    ap.add_argument('--ty', type=float, default=0.0, help='manual east offset [m]')
    ap.add_argument('--diff_m', type=float, default=0.15, help='overlay: red beyond this distance to the prior')
    ap.add_argument('--points', type=int, default=250_000, help='max points drawn per panel')
    ap.add_argument('--save', type=Path, help='write a PNG instead of opening a window')
    ap.add_argument('--3d', dest='three_d', action='store_true', help='also open a 3D view (scan + prior voxels)')
    args = ap.parse_args(argv)

    if args.save:
        import matplotlib
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    prior = Prior(args.prior, args.prior.with_suffix('.json'))
    xyz, rgb = load_ply(args.ply)
    p = to_ned(xyz, args.up)
    print(f'{args.ply.name}: {len(p)} points, extent {np.ptp(p, 0).round(2)} m (N, E, down)')

    if args.yaw is not None:
        mirror, yaw, t = args.mirror == 'yes', args.yaw, np.array([args.tx, args.ty])
        print(f'manual alignment: mirror={mirror} yaw={yaw:.1f} deg t=({t[0]:.2f}, {t[1]:.2f}) m')
    else:
        # Wall coverage alone barely separates a scan from its mirror image (corridors look alike both ways),
        # so each handedness' best fit is judged by the median distance of its wall band to the prior surface.
        cands = []
        for m, (score, yaw_m, t_m) in align(p, prior, tuple(args.band), args.mirror).items():
            q = apply(p, m, yaw_m, t_m)
            h = -q[:, 2]
            sel = (h >= args.band[0]) & (h <= args.band[1])
            med = float(np.nanmedian(prior.distance(q[sel])))
            cands.append((med, m, yaw_m, t_m, score))
            print(f'  mirror={m}: yaw {yaw_m:.1f} deg t=({t_m[0]:.2f}, {t_m[1]:.2f}) m, prior walls covered '
                  f'{100 * score:.0f}%, median distance to prior {med:.3f} m')
        med, mirror, yaw, t, score = min(cands, key=lambda c: c[0])
        print(f'auto alignment: mirror={mirror} yaw={yaw:.1f} deg t=({t[0]:.2f}, {t[1]:.2f}) m')
        if mirror:
            print('  WARNING: the best fit is mirrored; a right-handed scan should not need that. Check by eye.')
        print(f'  reproduce / tweak: --mirror {"yes" if mirror else "no"} --yaw {yaw:.1f} '
              f'--tx {t[0]:.2f} --ty {t[1]:.2f}')
    q = apply(p, mirror, yaw, t)

    h = -q[:, 2]
    a = prior.meta['arena']['arena_half_m']
    inside = (np.abs(q[:, 0]) < a - 0.05) & (np.abs(q[:, 1]) < a - 0.05)
    in_band = inside & (h >= args.band[0]) & (h <= args.band[1])
    d = prior.distance(q[in_band])
    far = d > args.diff_m
    print(f'scan points in the arena, {args.band[0]}-{args.band[1]} m: {in_band.sum()}, distance to prior surface '
          f'median {np.nanmedian(d):.3f} m, p90 {np.nanpercentile(d, 90):.3f} m, '
          f'{100 * np.mean(far):.0f}% beyond {args.diff_m} m (as-built differences or clutter)')

    fig, axs = plt.subplots(1, 3, figsize=(19, 6.8), constrained_layout=True)
    fig.suptitle(f'{args.ply.name} vs {args.prior.name}  (mirror={mirror}, yaw={yaw:.1f} deg, '
                 f't=({t[0]:.2f}, {t[1]:.2f}) m)')
    lim = (-a - 1.0, a + 1.0)

    ax = axs[0]
    sel = np.where(inside | (np.abs(q[:, :2]).max(1) < a + 1.0))[0]
    sel = sel[h[sel] < prior.meta['arena'].get('net_top_m', 3.0)]          # drop the ceiling
    sel = sel[np.argsort(h[sel])][thin(len(sel), args.points)]
    sel = sel[np.argsort(h[sel])]                                            # draw low first: walls on top
    ax.scatter(q[sel, 1], q[sel, 0], s=0.3, c=rgb[sel] if rgb is not None else h[sel], lw=0, rasterized=True)
    ax.set_title('scan, top-down (RGB, below the net top)')
    ax.set_aspect('equal')
    ax.set_xlabel('east  y_ned [m]')
    ax.set_ylabel('north  x_ned [m]')

    ax = axs[1]
    draw_course(ax, prior, args.band)
    ax.set_title(f'course prior: walls {args.band[0]}-{args.band[1]} m, gates, pad, launch')
    ax.legend(loc='upper right', fontsize=8)

    ax = axs[2]
    draw_course(ax, prior, args.band)
    idx = np.where(in_band)[0]
    k = thin(len(idx), args.points)
    sc = ax.scatter(q[idx[k], 1], q[idx[k], 0], s=0.6, c=np.minimum(d[k], 2 * args.diff_m), cmap='RdYlGn_r',
                    vmin=0, vmax=2 * args.diff_m, lw=0, zorder=2, rasterized=True)
    fig.colorbar(sc, ax=ax, shrink=0.8, label='distance to prior surface [m]')
    ax.set_title(f'overlay: scan {args.band[0]}-{args.band[1]} m on the prior')

    for ax in axs:
        ax.set_xlim(*lim)
        ax.set_ylim(*lim)
        ax.grid(alpha=0.3)

    if args.three_d:
        fig3 = plt.figure(figsize=(10, 8))
        ax3 = fig3.add_subplot(projection='3d')
        s3 = np.where(inside & (h < prior.meta['arena'].get('net_top_m', 3.0)))[0]
        s3 = s3[thin(len(s3), 60_000)]
        ax3.scatter(q[s3, 1], q[s3, 0], h[s3], s=0.5, c=rgb[s3] if rgb is not None else h[s3], lw=0)
        surf = prior.occ & (prior.dist >= 0.0)
        i, j, kk = np.nonzero(surf)
        m3 = thin(len(i), 15_000, 2)
        ax3.scatter(prior.axis(1)[j[m3]], prior.axis(0)[i[m3]], -prior.axis(2)[kk[m3]], s=1, c='tab:red',
                    alpha=0.25, lw=0, label='prior walls')
        ax3.set_xlabel('east [m]')
        ax3.set_ylabel('north [m]')
        ax3.set_zlabel('height [m]')
        ax3.set_box_aspect((2 * a, 2 * a, 3.0))
        ax3.legend()

    if args.save:
        fig.savefig(args.save, dpi=130)
        print(f'wrote {args.save}')
    else:
        plt.show()
    return 0


if __name__ == '__main__':
    sys.exit(main())
