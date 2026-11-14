#!/usr/bin/env python3
"""The real-world course model: the simplified as-built scene from dronehacks-sim, NumPy only.

The demos used to rebuild the real course from the noisy .ply scan (SE(2) ICP of CAD pieces, splatted scan points).
They now use dronehacks-sim's simplified as-built geometry instead, copied (not linked) into
``assets/real_world/``:

  main_direction/course_asbuilt.npz  DEFAULT (iSAM2 / full demos): the as-built course the RL policy trains on
                        (scripts/make_asbuilt_course.py: course_w0scan2 + dronehacks-sim's ftw0_s330 course mods)
  course_w0scan2.npz    the MPPI demo's world: course_w0scan.npz with wall 0 ~4 cm east and a 0.84 m wide
                        window at 0.305-1.219 m (see assets/real_world/README.md)
  course_w0scan.npz     the Course v3 CAD with wall 0 + the window re-laid where the scan of the built course
                        puts them (dronehacks-sim ``assets/course_w0scan.npz``; its "as-built" eval condition)
  course_w0scan_hi.npz  the same with the window sill / top raised to 0.50 / 1.50 m (as-built height unsure)

File format (dronehacks-sim ``CourseMesh``): ``vertices`` (V,3) f32 NED metres (origin floor centre, z down),
``faces`` (F,3) i32, ``face_group`` (F,) i8 into ``groups`` = (floor, net, paint, wall, tunnel, boulder),
``face_object`` (F,) i32 source OBJ object.

The pre-baked CAD prior (``assets/main_direction/dronehacks-course-v3-obj.zip`` / ``assets/cad_prior_esdf.bin``) is deliberately NOT
this file: the demos show iSAM2 / MPPI generalizing from that prior to this different real world.

    python3 scripts/real_world_scene.py                 # summary of the default scene
    python3 scripts/real_world_scene.py --points 0.02   # + surface-sampled point count at 2 cm
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

DEPLOY_ROOT = Path(__file__).resolve().parent.parent
REAL_WORLD_DIR = DEPLOY_ROOT / "assets" / "real_world"
MAIN_DIR = DEPLOY_ROOT / "assets" / "main_direction"
#: the as-built course the RL policy trains on (scripts/make_asbuilt_course.py): what the iSAM2 / full demos fly in
ASBUILT_SCENE = MAIN_DIR / "course_asbuilt.npz"
#: course_w0scan2: the MPPI demo's world
W0SCAN2_SCENE = REAL_WORLD_DIR / "course_w0scan2.npz"
DEFAULT_SCENE = ASBUILT_SCENE

GROUPS = ("floor", "net", "paint", "wall", "tunnel", "boulder")
#: physical groups a depth camera sees (net: stereo sees through it; paint: 1-3 mm floor markings)
DEPTH_GROUPS = ("floor", "wall", "tunnel", "boulder")
#: obstacles (what the drone can hit)
OBSTACLE_GROUPS = ("wall", "tunnel", "boulder")

#: opaque display colours (RGB u8) per group; also the colours of the sampled "scan" points
GROUP_RGB = {
    "floor": (156, 152, 146),
    "net": (40, 40, 40),
    "paint": (236, 236, 228),
    "wall": (198, 160, 112),     # plywood
    "tunnel": (84, 112, 150),
    "boulder": (128, 122, 114),
}


class RealWorldScene:
    """Triangle soup in NED with one group per face."""

    def __init__(self, path: Path | str = DEFAULT_SCENE) -> None:
        self.path = Path(path)
        d = np.load(self.path, allow_pickle=False)
        self.vertices = d["vertices"].astype(np.float64)
        self.faces = d["faces"].astype(np.int64)
        self.face_group = d["face_group"].astype(np.int64)
        self.face_object = d["face_object"].astype(np.int64)
        names = [str(g) for g in d["groups"]] if "groups" in d.files else list(GROUPS)
        if tuple(names) != GROUPS:
            raise ValueError(f"{self.path}: unexpected groups {names}, want {GROUPS}")

    # ------------------------------------------------------------------ selection
    def face_mask(self, groups=DEPTH_GROUPS) -> np.ndarray:
        return np.isin(self.face_group, [GROUPS.index(g) for g in groups])

    def triangles(self, groups=DEPTH_GROUPS) -> np.ndarray:
        """(F,3,3) NED triangles of ``groups``."""
        return self.vertices[self.faces[self.face_mask(groups)]]

    def submesh(self, groups) -> tuple[np.ndarray, np.ndarray]:
        """(verts NED, faces) of ``groups`` with unused vertices dropped."""
        f = self.faces[self.face_mask(groups)]
        used, inv = np.unique(f, return_inverse=True)
        return self.vertices[used], inv.reshape(f.shape)

    # ------------------------------------------------------------------ bodies
    def bodies(self, groups=("floor", "paint", "wall", "tunnel", "boulder"), weld: float = 1e-3) -> list[dict]:
        """Rigid pieces: per group, OBJ objects that share a vertex position are one body.

        Returns dicts ``name`` (e.g. ``wall_0``), ``group``, ``verts`` (V,3 NED), ``faces`` (F,3), ``rgb``.
        Bodies are numbered in order of their centroid (north then east) so names are stable."""
        out: list[dict] = []
        for g in groups:
            fidx = np.nonzero(self.face_group == GROUPS.index(g))[0]
            if len(fidx) == 0:
                continue
            objs = np.unique(self.face_object[fidx])
            parent = {int(o): int(o) for o in objs}

            def find(a: int) -> int:
                while parent[a] != a:
                    parent[a] = parent[parent[a]]
                    a = parent[a]
                return a

            seen: dict[tuple, int] = {}
            for o in objs:
                vids = np.unique(self.faces[fidx[self.face_object[fidx] == o]])
                keys = np.round(self.vertices[vids] / weld).astype(np.int64)
                for k in map(tuple, keys):
                    if k in seen:
                        ra, rb = find(int(o)), find(seen[k])
                        if ra != rb:
                            parent[ra] = rb
                    else:
                        seen[k] = int(o)
            comps: dict[int, list[int]] = {}
            for o in objs:
                comps.setdefault(find(int(o)), []).append(int(o))
            pieces = []
            for members in comps.values():
                f = self.faces[fidx[np.isin(self.face_object[fidx], members)]]
                used, inv = np.unique(f, return_inverse=True)
                v = self.vertices[used]
                pieces.append((tuple(np.round(v.mean(0)[:2], 3)), v, inv.reshape(f.shape)))
            pieces.sort(key=lambda p: (-p[0][0], p[0][1]))
            for i, (_, v, f) in enumerate(pieces):
                out.append({"name": f"{g}_{i}", "group": g, "verts": v, "faces": f, "rgb": GROUP_RGB[g]})
        return out

    # ------------------------------------------------------------------ sampling
    def sample_points(self, spacing: float = 0.015, groups=DEPTH_GROUPS, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
        """Uniform surface samples (~1 point per ``spacing``^2) -> (xyz NED (N,3) f64, rgb (N,3) f64 in [0,1]).

        A drop-in for the aligned .ply scan in the depth spoofers: same frame, same (points, colours) shape."""
        rng = np.random.default_rng(seed)
        mask = self.face_mask(groups)
        T = self.vertices[self.faces[mask]]
        grp = self.face_group[mask]
        area = 0.5 * np.linalg.norm(np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0]), axis=1)
        n = rng.poisson(area / spacing ** 2)
        n = np.where((n == 0) & (area > 0.25 * spacing ** 2), 1, n)
        tri = np.repeat(np.arange(len(T)), n)
        u = rng.random((len(tri), 2))
        flip = u.sum(1) > 1.0
        u[flip] = 1.0 - u[flip]
        A, B, C = T[tri, 0], T[tri, 1], T[tri, 2]
        xyz = A + u[:, :1] * (B - A) + u[:, 1:] * (C - A)
        lut = np.array([GROUP_RGB[g] for g in GROUPS], np.float64) / 255.0
        rgb = lut[grp[tri]]
        rgb = np.clip(rgb * (0.92 + 0.16 * rng.random((len(tri), 1))), 0.0, 1.0)   # slight texture for stereo
        return xyz, rgb

    def summary(self) -> str:
        lo, hi = self.vertices.min(0), self.vertices.max(0)
        counts = {g: int((self.face_group == i).sum()) for i, g in enumerate(GROUPS)}
        return f"{self.path.name}: {len(self.faces)} faces {counts}, NED bounds {lo.round(2)} .. {hi.round(2)}"


def scene_meta(path: str | Path | None) -> dict:
    """The scene's ``<stem>.json`` sidecar (e.g. course_asbuilt.json from make_asbuilt_course.py), or {}."""
    if path is None:
        return {}
    j = Path(path).with_suffix(".json")
    if not j.exists():
        return {}
    import json
    return json.loads(j.read_text())


def launch_mark_ned(path: str | Path | None, cad_mark_ne) -> np.ndarray:
    """The launch mark (N, E) in this scene: the CAD mark plus the as-built ``mark_shift`` (dronehacks-sim moves the
    launch mark with ``shift:...:mark``; the paint stays, the env launches from the shifted mark)."""
    meta = scene_meta(path)
    if "launch_mark_ned" in meta:
        return np.asarray(meta["launch_mark_ned"], dtype=float)[:2]
    shift = np.asarray(meta.get("info", {}).get("mark_shift", [0.0, 0.0]), dtype=float)
    return np.asarray(cad_mark_ne, dtype=float)[:2] + shift[:2]


def scene_arena_meta(path: str | Path | None, arena_meta: dict) -> dict:
    """``assets/arena_prior.json`` metadata for this scene: the as-built scene's sidecar replaces the checkpoint
    gates (dronehacks-sim's as-built scoring: checkpoint 0 from the NW corner to a wall tip, lines extended to the
    walls, the window re-laid) and the launch mark; everything else (arena, net, pad) is the surveyed prior."""
    meta = scene_meta(path)
    if not meta.get("gates"):
        return arena_meta
    out = dict(arena_meta)
    out["gates"] = meta["gates"]
    lm = dict(arena_meta.get("landmarks", {}))
    if "launch_mark_ned" in meta and "launch_mark_ned" in lm:
        lm["launch_mark_ned"] = [*meta["launch_mark_ned"][:2], *lm["launch_mark_ned"][2:]]
    out["landmarks"] = lm
    return out


def resolve_scene(path: str | Path | None) -> Path | None:
    """'' / None -> None (use the .ply instead); relative paths resolve against the deploy root."""
    if path is None or str(path) == "":
        return None
    p = Path(path).expanduser()
    if not p.is_absolute() and not p.exists():
        p = DEPLOY_ROOT / p
    if not p.exists():
        raise FileNotFoundError(f"real-world scene not found: {p}")
    return p.resolve()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scene", nargs="?", default=str(DEFAULT_SCENE))
    ap.add_argument("--points", type=float, default=0.0, help="also sample the surface at this spacing [m]")
    a = ap.parse_args()
    s = RealWorldScene(resolve_scene(a.scene))
    print(s.summary())
    for b in s.bodies():
        print(f"  {b['name']:10s} {len(b['faces']):6d} faces  centre NED {b['verts'].mean(0).round(2)}")
    if a.points > 0:
        xyz, _ = s.sample_points(a.points)
        print(f"sampled {len(xyz)} points at {a.points} m")


if __name__ == "__main__":
    main()
