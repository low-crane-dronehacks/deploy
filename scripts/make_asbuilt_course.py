#!/usr/bin/env python3
"""Build the as-built course (the "orange" map the RL policy trains on) as a real-world scene npz + its checkpoints.

dronehacks-sim does not ship this map as a file: its training recipes start from the scan-registered course and
apply ``--course-mods`` (``dronehacks/perturb.py`` ``apply_mods``) per episode, and score it with as-built
checkpoints (``--cp0-wall``, ``--gate-extend``, ``--window-gate``). With severity 0 / ``rng=None`` every ranged
modification takes its midpoint: the nominal as-built course. This script runs exactly that once (dronehacks-sim's
own ``perturb.build_variants``) and saves the result, so the demos need no dronehacks-sim checkout at run time.

Recipe: ``recipes/ftw0_s330.sh`` (window wall re-laid on the scan, randomized window height):
  gapwalls 0-0.30 m (0.15)  |  window sill 0.30-0.60 (0.45 m), top 1.20-1.55 (1.375 m)  |  wall moves / rotations /
  re-placements / spans measured from the Nov 6 LiDAR scan  |  launch mark shifted 0.38 m north  |  checkpoint 0 =
  NW corner -> wall3 tip, every checkpoint line extended to the walls, window line re-laid on the scan.

Inputs (default):  assets/main_direction/course_w0scan2.npz   (team source of truth for wall 0 / the window wall)
Output (default):  assets/main_direction/course_asbuilt.npz  + course_asbuilt.json (recipe, window, launch mark and
                   the as-built checkpoint gates in assets/arena_prior.json's format)

Run with the system NumPy 1.x (dronehacks-sim's scan helpers + SciPy):

    PYTHONNOUSERSITE=1 python3 scripts/make_asbuilt_course.py --dronesim ../dronesim
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

DEPLOY_ROOT = Path(__file__).resolve().parent.parent
MAIN_DIR = DEPLOY_ROOT / "assets" / "main_direction"
ARENA_PRIOR = DEPLOY_ROOT / "assets" / "arena_prior.json"

#: dronehacks-sim recipes/ftw0_s330.sh (the deployed as-built course): --course-mods, --window-gate, --cp0-wall,
#: --anchor-wall, --gate-extend, --perturb-scale
FTW0_S330_MODS = (
    "gapwalls:0.0:0.30;winh:wall0:0.30:0.60:1.20:1.55;move:wall1:-0.13:0.0:0;move:wall4:-0.08:-0.46:0;"
    "move:tunnel:0.10:-0.07:0;seg:wall2:0.75:0.98:-1.2:1.6;shift:wall1:0.30:0.46:north:mark;rot:wall3:40:50:south;"
    "place:wall5:north:0.05:0.3:-2.40:-2.10:wall3;span:wall6:wall1:-0.2:0.2:nw:-0.3:0.0"
)
FTW0_S330_WINDOW_GATE = (-2.902, -0.507, -1.811, -0.304)
WINDOW_SILL_M, WINDOW_TOP_M = 0.45, 1.375          # winh midpoints
ALT_MARGIN_M = 0.31                                 # arena_prior.json's window band: 0.30-1.22 m opening -> 0.62-0.92


def _jsonable(x):
    if isinstance(x, np.ndarray):
        return np.round(x, 4).tolist()
    if isinstance(x, dict):
        return {k: _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    return x


def _surface_points(mesh, step: float = 0.01) -> np.ndarray:
    T = mesh.vertices[mesh.faces].astype(np.float64)
    area = 0.5 * np.linalg.norm(np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0]), axis=1)
    rng = np.random.default_rng(0)
    n = np.maximum(1, rng.poisson(area / step ** 2))
    ti = np.repeat(np.arange(len(T)), n)
    u = rng.random((len(ti), 2))
    f = u.sum(1) > 1
    u[f] = 1 - u[f]
    return T[ti, 0] + u[:, :1] * (T[ti, 1] - T[ti, 0]) + u[:, 1:] * (T[ti, 2] - T[ti, 0])


def _clearance_factory(mesh):
    """dronehacks-sim uses its Warp BVH (MeshScene.distance_at); a 1 cm surface-sample KD-tree is within ~1 cm."""
    tree = cKDTree(_surface_points(mesh))
    return lambda p: tree.query(np.asarray(p, np.float64).reshape(-1, 3))[0]


def _arena_gate(g: dict, idx: int, cad: dict) -> dict:
    """dronehacks-sim gate (A, B, n, cp) -> an assets/arena_prior.json gate entry (altitudes from the CAD entry)."""
    A, B, n = (np.asarray(g[k], np.float64) for k in ("A", "B", "n"))
    t = (B - A) / np.linalg.norm(B - A)
    win = g["name"] == "window"
    if win:
        lo, hi = WINDOW_SILL_M + ALT_MARGIN_M, WINDOW_TOP_M - ALT_MARGIN_M
        alt = dict(alt_min_m=round(lo, 3), alt_max_m=round(hi, 3), alt_nominal_m=round(0.5 * (lo + hi), 3))
    else:
        alt = {k: cad[k] for k in ("alt_min_m", "alt_max_m", "alt_nominal_m")}
    mid = 0.5 * (A + B)
    return {
        "index": idx, "name": g["name"], "checkpoint": int(g["cp"]), "is_window": win,
        "A_ned": np.round(A, 4).tolist(), "B_ned": np.round(B, 4).tolist(),
        "tangent_ned": np.round(t, 6).tolist(), "normal_ned": np.round(n, 6).tolist(),
        "length_m": round(float(np.linalg.norm(B - A)), 4),
        "center_ned": [round(float(mid[0]), 4), round(float(mid[1]), 4), -alt["alt_nominal_m"]],
        **alt,
        "nominal_clearance_m": cad.get("nominal_clearance_m"),
    }


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dronesim", type=Path, default=DEPLOY_ROOT.parent / "dronesim", help="dronehacks-sim checkout")
    ap.add_argument("--base", type=Path, default=MAIN_DIR / "course_w0scan2.npz")
    ap.add_argument("--mods", default=FTW0_S330_MODS)
    ap.add_argument("--out", type=Path, default=MAIN_DIR / "course_asbuilt.npz")
    a = ap.parse_args(argv)

    sys.path.insert(0, str(a.dronesim.resolve()))
    from dronehacks import perturb, rules              # noqa: E402  (dronehacks-sim)
    from dronehacks.mesh import CourseMesh             # noqa: E402

    base = CourseMesh.load(str(a.base))
    bodies = perturb.rigid_bodies(base)
    info: dict = {}
    asb = perturb.apply_mods(base, bodies, a.mods, None, info)   # rng=None: every range at its midpoint
    asb.save(str(a.out))

    # as-built checkpoints, exactly as the env builds them for this recipe (env._gates + build_variants severity 0)
    gates = [dict(g) for g in rules.GATES]
    n0, e0, n1, e1 = FTW0_S330_WINDOW_GATE
    A, B = np.array([n0, e0]), np.array([n1, e1])
    t = (B - A) / np.linalg.norm(B - A)
    nrm = np.array([-t[1], t[0]])
    nrm = -nrm if nrm[1] > 0 else nrm
    gates[rules.WINDOW_GATE].update(A=A, B=B, t=t, L=float(np.linalg.norm(B - A)), n=nrm)
    sim_dir = a.dronesim.resolve() / "assets"
    mark = np.asarray(json.loads((sim_dir / "scene.json").read_text())["landmarks"]["launch_mark_ned"][:2], np.float64)
    goal = np.asarray(np.load(sim_dir / "route.npz")["goal"], np.float64)
    lay = perturb.build_variants(
        base, gates, rules.PAD_CENTER, mark, goal, severities=[0.0], seed=0, clearance_factory=_clearance_factory,
        launch_xy=mark, mods=a.mods, scale=2.5, cp0_wall="wall3", anchor_wall="wall1", gate_extend=True,
    ).layouts[0]
    cad = {g["name"]: g for g in json.loads(ARENA_PRIOR.read_text())["gates"]}
    arena_gates = [_arena_gate(g, i, cad[g["name"]]) for i, g in enumerate(lay.gates)]

    win = next(b for b in bodies if b.name == "wall0")
    wz = np.unique(np.round(-asb.vertices[np.unique(asb.faces[win.faces]), 2], 4))
    meta = {
        "what": "nominal as-built course (dronehacks-sim perturb.apply_mods / build_variants, severity 0 -> range "
                "midpoints): the map the RL policy trains on",
        "base": a.base.name,
        "recipe": "dronehacks-sim recipes/ftw0_s330.sh",
        "course_mods": a.mods,
        "window_gate": list(FTW0_S330_WINDOW_GATE),
        "window_wall_heights_m": wz.tolist(),
        "window_sill_m": WINDOW_SILL_M,
        "window_top_m": WINDOW_TOP_M,
        "wall_gap_under_m": 0.15,
        "launch_mark_ned": np.round(lay.mark, 4).tolist(),
        "pad_center_ned": np.round(np.asarray(lay.pad_center), 4).tolist(),
        "gates": arena_gates,
        "info": _jsonable({k: v for k, v in info.items()}),
    }
    a.out.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"wrote {a.out} ({len(asb.faces)} faces); window wall heights {wz.tolist()} m; launch mark "
          f"{np.round(lay.mark, 3).tolist()}")
    for g in arena_gates:
        print(f"  {g['name']:13s} cp{g['checkpoint']} A {g['A_ned']} B {g['B_ned']} n {g['normal_ned']} "
              f"alt {g['alt_min_m']}-{g['alt_max_m']} m")


if __name__ == "__main__":
    main()
