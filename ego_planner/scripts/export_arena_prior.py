#!/usr/bin/env python3
"""Export the DroneHacks 2026 Arena Prior (3D distance/occupancy grid + gate/landmark metadata) for C++ EGO-Planner-V2.

Writes:
  assets/arena_prior.json   Gate geometry, landmarks, sectors, and nominal route anchor waypoints (NED, metres)
  assets/arena_prior.bin    Compact 3D grid (123x123x62 at 5 cm resolution over [-3.05,3.05]x[-3.05,3.05]x[-3.05,0.0] NED)
                            storing uint8(clamp(round(exact_clearance_m * 200), 0, 255)) -> 5 mm precision, 916 KiB.
"""
from __future__ import annotations

import json
import os
import struct
import sys

import numpy as np

SIM_DIR = os.environ.get('DRONEHACKS_SIM', '/home/ejkaba/prj/dronehacks-sim')
sys.path.insert(0, SIM_DIR)

from dronehacks import rules  # noqa: E402
from dronehacks.mesh import CourseMesh  # noqa: E402
from dronehacks.render import MeshScene  # noqa: E402

DEPLOY_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main() -> None:
    assets_dir = os.path.join(SIM_DIR, 'assets')
    course = CourseMesh.load(os.path.join(assets_dir, 'course.npz'))
    scene = MeshScene(course.select(('wall', 'tunnel', 'boulder')), 1, device='cpu')

    res = 0.05
    lo = np.array([-3.05, -3.05, -3.05], dtype=np.float64)
    hi = np.array([3.05, 3.05, 0.00], dtype=np.float64)
    shape = tuple(int(round((hi[k] - lo[k]) / res)) + 1 for k in range(3))  # (123, 123, 62)
    axes = [lo[k] + res * np.arange(shape[k], dtype=np.float32) for k in range(3)]
    pts = np.stack(np.meshgrid(*axes, indexing='ij'), axis=-1).reshape(-1, 3)

    print(f'Computing exact triangle-mesh clearance over {shape} = {len(pts)} voxels...')
    dist = scene.distance_at(pts).reshape(shape)

    # Quantize to uint8 in units of 0.005 m (5 mm), saturating at 255 (1.275 m)
    q_dist = np.clip(np.rint(dist * 200.0), 0, 255).astype(np.uint8)

    out_dir = os.path.join(DEPLOY_ROOT, 'assets')
    os.makedirs(out_dir, exist_ok=True)
    bin_path = os.path.join(out_dir, 'arena_prior.bin')
    with open(bin_path, 'wb') as f:
        # Header: magic "EGOP" (4B), version=1 (uint32), nx, ny, nz (3x int32), res (float32), lo (3x float32)
        f.write(struct.pack('<4sIiiiffff', b'EGOP', 1, shape[0], shape[1], shape[2],
                            float(res), float(lo[0]), float(lo[1]), float(lo[2])))
        f.write(q_dist.tobytes(order='C'))

    gates_meta = []
    # Optimal gate crossing heights and span fractions
    gate_alt_bounds = {
        'corridor_end': (0.45, 1.35, 0.80),
        'gate_a':       (0.45, 1.35, 0.80),
        'tunnel_exit':  (0.55, 1.15, 0.80),
        'gate_b':       (0.45, 1.35, 0.80),
        'window':       (0.62, 0.92, 0.78),
    }
    for idx, g in enumerate(rules.GATES):
        name = g['name']
        alt_min, alt_max, alt_nom = gate_alt_bounds[name]
        A, B, t, n, L = g['A'], g['B'], g['t'], g['n'], float(g['L'])
        us = np.linspace(0.22, L - 0.22, 41)
        sample_pts = np.array([[*(A + u * t), -alt_nom] for u in us], dtype=np.float32)
        clr = scene.distance_at(sample_pts)
        # Also penalize proximity to outer net at +-3.0 m
        net_clr = np.minimum(3.0 - np.abs(sample_pts[:, 0]), 3.0 - np.abs(sample_pts[:, 1]))
        eff_clr = np.minimum(clr, net_clr)
        best_u = float(us[int(np.argmax(eff_clr))])
        center_xy = (A + best_u * t).tolist()
        gates_meta.append({
            'index': idx,
            'name': name,
            'checkpoint': int(g['cp']),
            'is_window': bool(idx == rules.WINDOW_GATE),
            'A_ned': [round(float(A[0]), 4), round(float(A[1]), 4)],
            'B_ned': [round(float(B[0]), 4), round(float(B[1]), 4)],
            'tangent_ned': [round(float(t[0]), 6), round(float(t[1]), 6)],
            'normal_ned': [round(float(n[0]), 6), round(float(n[1]), 6)],
            'length_m': round(L, 4),
            'center_ned': [round(float(center_xy[0]), 4), round(float(center_xy[1]), 4), round(-alt_nom, 4)],
            'alt_min_m': alt_min,
            'alt_max_m': alt_max,
            'alt_nominal_m': alt_nom,
            'nominal_clearance_m': round(float(eff_clr.max()), 4),
        })

    prior_json = {
        'course_name': 'DroneHacks 2026 Course v3',
        'frame': 'NED',
        'grid': {
            'binary_file': 'arena_prior.bin',
            'resolution_m': res,
            'lo_ned': lo.tolist(),
            'hi_ned': hi.tolist(),
            'shape': list(shape),
            'distance_scale': 200.0,
            'occupied_threshold_m': 0.035,
        },
        'arena': {
            'arena_half_m': 3.0,
            'net_top_m': 3.0,
            'wall_top_m': float(rules.WALL_TOP_M),
            'collision_radius_m': 0.20,
            'cp_margin_m': 0.15,
            'touchdown_alt_m': 0.05,
            'rest_alt_m': 0.08,
            'lift_alt_m': 0.30,
            'land_max_speed_ms': 1.0,
        },
        'landmarks': {
            'launch_mark_ned': [round(float(rules.LAUNCH_MARK[0]), 4), round(float(rules.LAUNCH_MARK[1]), 4), 0.0],
            'pad_center_ned': [round(float(rules.PAD_CENTER[0]), 4), round(float(rules.PAD_CENTER[1]), 4), 0.0],
            'pad_radius_m': round(float(rules.PAD_RADIUS_M), 4),
            'return_radius_m': round(float(rules.RETURN_RADIUS_M), 4),
            'return_hold_s': 3.0,
        },
        'gates': gates_meta,
    }
    json_path = os.path.join(out_dir, 'arena_prior.json')
    with open(json_path, 'w') as f:
        json.dump(prior_json, f, indent=2)
    print(f'Wrote {bin_path} ({os.path.getsize(bin_path) / 1024:.1f} KiB) and {json_path}')


if __name__ == '__main__':
    main()
