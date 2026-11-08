#!/usr/bin/env python3
"""Make the Gazebo course textures (sitl/world/textures/*.jpg, net.png), used by sitl/texture_world.py.

    python3 sitl/make_textures.py --wall WALL_PHOTO.png --window WINDOW_PHOTO.png [--out sitl/world/textures]

walls.jpg   seamless mottling in the wall photo's colours (orange); walls + tunnel
window.jpg  seamless mottling from the window photo's luminance, on light blue
carpet.jpg  grey carpet: periodic noise (fine pile + mottling), tiles seamlessly
net.png     faint black netting, RGBA: strands opaque, holes alpha 0 (Gazebo alpha-tests albedo maps, so the
            cameras see through the holes)
boulder.jpg grey stone noise for the boulders / pillars
hall.jpg    painted cinder block: the hall walls and ceiling around the arena, seen through the nets
Why: the untextured course gave cuVSLAM almost nothing to track (2026-11-08: lost tracking on the takeoff in SITL).
Deterministic (fixed seeds).
"""
from __future__ import annotations

import argparse
import os

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))


def periodic_noise(n: int, beta: float, rng: np.random.Generator, lo_cut: int = 1) -> np.ndarray:
    """n x n noise with a 1/f^beta spectrum, periodic (tiles seamlessly), scaled to zero mean / unit std."""
    f = np.fft.fftfreq(n) * n
    r = np.hypot(*np.meshgrid(f, f, indexing='ij'))
    amp = np.where(r >= lo_cut, 1.0 / np.maximum(r, 1.0) ** beta, 0.0)
    x = np.real(np.fft.ifft2(amp * np.exp(2j * np.pi * rng.random((n, n)))))
    return (x - x.mean()) / (x.std() + 1e-12)


def palette_noise(photo: np.ndarray, n: int, rng: np.random.Generator, contrast: float = 1.0) -> np.ndarray:
    """n x n seamless texture: multi-scale periodic noise mapped through the photo's colours by luminance rank."""
    field = (periodic_noise(n, 1.4, rng, lo_cut=2) + 0.55 * periodic_noise(n, 0.6, rng)
             + 0.25 * periodic_noise(n, 0.0, rng))
    cols = photo.reshape(-1, 3)
    cols = cols[np.argsort(cols @ np.array([0.299, 0.587, 0.114]))]
    if contrast != 1.0:
        m = cols.mean(0)
        cols = np.clip((cols - m) * contrast + m, 0, 1)
    rank = np.argsort(np.argsort(field.ravel())) / (field.size - 1)
    return cols[np.round(rank * (len(cols) - 1)).astype(int)].reshape(n, n, 3)


def load_rgb(path: str) -> np.ndarray:
    return np.asarray(Image.open(path).convert('RGB'), dtype=np.float64) / 255.0


def save(a: np.ndarray, path: str) -> None:
    """RGB -> .jpg (noise does not compress as PNG), RGBA -> .png (alpha)."""
    rgba = a.shape[2] == 4
    path = os.path.splitext(path)[0] + ('.png' if rgba else '.jpg')
    im = Image.fromarray(np.clip(a * 255.0 + 0.5, 0, 255).astype(np.uint8), 'RGBA' if rgba else 'RGB')
    im.save(path, optimize=True) if rgba else im.save(path, quality=92)
    print(f'wrote {path} {a.shape[1]}x{a.shape[0]}')


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--wall', required=True, help='wall photo (orange)')
    ap.add_argument('--window', required=True, help='window photo')
    ap.add_argument('--out', default=os.path.join(HERE, 'world', 'textures'))
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)

    # walls + tunnel: seamless mottling in the wall photo's own colours (each noise level takes the photo colour of
    # the same luminance rank). Mirror-tiling the photo itself gave a kaleidoscope: symmetric repeats mislead matching.
    wall = palette_noise(load_rgb(a.wall), 1024, np.random.default_rng(4), contrast=1.2)
    # the cameras are monochrome, where orange is a dark, flat grey (mono std 0.10, rendered darker still: SITL lost
    # tracking facing the corridor-end wall at ~0.5 m). Re-scale luminance to mean 0.52 / std 0.19 and add a fine grain
    # (detail at close range), keeping the hue: RGB scaled by new / old luminance.
    lum = wall @ np.array([0.299, 0.587, 0.114])
    grain = 0.07 * periodic_noise(1024, 0.0, np.random.default_rng(7))
    new_lum = np.clip(0.52 + (lum - lum.mean()) * (0.19 / lum.std()) + grain, 0.05, 0.98)
    wall = np.clip(wall * (new_lum / np.maximum(lum, 1e-3))[..., None], 0, 1)
    save(wall, os.path.join(a.out, 'walls.png'))

    # window: the same, from the window photo's luminance, on a light blue
    win = load_rgb(a.window)
    lum = palette_noise(win, 512, np.random.default_rng(5)) @ np.array([0.299, 0.587, 0.114])
    lum = (lum - lum.mean()) / (lum.std() + 1e-9)
    save(np.clip(np.array([0.66, 0.78, 0.92]) + 0.09 * lum[..., None], 0, 1), os.path.join(a.out, 'window.png'))

    # carpet: grey, fine pile (white-ish noise) + low-frequency mottling + faint fibre streaks
    rng = np.random.default_rng(1)
    n = 512
    pile = periodic_noise(n, 0.3, rng)
    mottle = periodic_noise(n, 1.6, rng)
    streak = np.repeat(periodic_noise(n, 0.8, rng)[:1], n, 0)
    g = 0.47 + 0.055 * pile + 0.05 * mottle + 0.015 * streak
    save(np.stack([g * 0.98, g, g * 1.03], -1), os.path.join(a.out, 'carpet.png'))

    # net: square mesh, thin dark strands (opaque), holes transparent; slightly wobbly strands, faint knots
    m = 128
    yy, xx = np.meshgrid(np.arange(m), np.arange(m), indexing='ij')
    wob = 1.5 * periodic_noise(m, 2.0, np.random.default_rng(2))
    d = np.minimum(np.abs(((xx + wob) % m) - m / 2), np.abs(((yy - wob) % m) - m / 2))
    strand = d < 2.5
    rgba = np.zeros((m, m, 4))
    rgba[..., :3] = 0.08
    rgba[..., 3] = strand.astype(float)
    save(rgba, os.path.join(a.out, 'net.png'))

    # hall walls / ceiling around the arena (seen through the nets): painted cinder block, 0.4 x 0.2 m blocks,
    # 8 x 4 blocks per tile (1.6 m), staggered rows, mortar lines, paint mottling
    k = 1024
    yy, xx = np.meshgrid(np.arange(k), np.arange(k), indexing='ij')
    bw, bh = k // 4, k // 8                                   # block 256 x 128 px = 0.4 x 0.2 m
    row = yy // bh
    xs = (xx + (row % 2) * (bw // 2)) % k
    mortar = ((xs % bw) < 6) | ((yy % bh) < 6)
    rng = np.random.default_rng(6)
    tone = rng.normal(0, 0.035, (k // bh, k // bw * 2))[row, (xs // bw)]   # per-block shade
    hall = 0.70 + tone + 0.035 * periodic_noise(k, 1.0, rng) + 0.02 * periodic_noise(k, 0.1, rng)
    hall = np.where(mortar, hall - 0.22, hall)
    save(np.stack([hall, hall * 0.97, hall * 0.90], -1), os.path.join(a.out, 'hall.jpg'))

    # boulders / pillars: grey stone
    rng = np.random.default_rng(3)
    s = 0.42 + 0.09 * periodic_noise(256, 1.2, rng) + 0.04 * periodic_noise(256, 0.2, rng)
    save(np.stack([s, s * 0.98, s * 0.95], -1), os.path.join(a.out, 'boulder.png'))


if __name__ == '__main__':
    main()
