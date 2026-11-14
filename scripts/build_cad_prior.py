#!/usr/bin/env python3
"""Build CAD-derived 3D ESDF priors directly from assets/main_direction/dronehacks-course-v3-obj.zip / .obj + .mtl.

Imports each named CAD object (`g <piece>`) individually with its MTL diffuse colour (matching
`course-sc.png`, including default [0.80, 0.80, 0.80] grey when a group omits `usemtl` as `Barrier 2`
does in Onshape's export), converts ENU -> NED, and writes:

  1. assets/cad_prior_esdf.bin   ("CADP" v1) Closest-surface-point + analytical surface-normal 3D ESDF
                                 for iSAM2 point-to-CAD-plane prior registration (129x129x69 at 5 cm
                                 over [-3.20, 3.20]x[-3.20, 3.20]x[-3.20, 0.20] NED; 10 bytes/cell:
                                 int16 dx,dy,dz in 0.1 mm + int8 nx,ny,nz,flags).
  2. assets/arena_prior.bin      ("EGOP" v1) 123x123x62 @ 5 cm NED obstacle clearance grid derived from
                                 the CAD structural obstacles (walls, barriers, tunnel, cradles, boulders).
"""
from __future__ import annotations

import argparse
import json
import re
import struct
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

DEPLOY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MTL_KD = (0.80, 0.80, 0.80)


@dataclass
class CadObject:
    name: str
    slug: str
    category: str          # 'obstacle' | 'floor' | 'paint' | 'net'
    kd_rgb: tuple[float, float, float]
    verts_enu: np.ndarray  # (V, 3) float32
    norms_enu: np.ndarray  # (V, 3) float32
    faces: np.ndarray      # (F, 3) uint32


def classify_piece(name: str) -> str:
    low = name.lower()
    if low.startswith("net "):
        return "net"
    if low.startswith("floor"):
        return "floor"
    if any(k in low for k in ("landing pad", "guideline", "launch mark", "approach ring")):
        return "paint"
    return "obstacle"


def slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def parse_mtl(mtl_text: str) -> dict[str, tuple[float, float, float]]:
    mtls: dict[str, tuple[float, float, float]] = {}
    cur: str | None = None
    for raw in mtl_text.splitlines():
        s = raw.strip()
        if s.startswith("newmtl "):
            cur = s.split(None, 1)[1].strip()
        elif s.startswith("Kd ") and cur is not None:
            r, g, b = (float(x) for x in s.split()[1:4])
            mtls[cur] = (r, g, b)
    return mtls


def read_obj_mtl_texts(source: Path) -> tuple[str, str]:
    if source.suffix.lower() == ".zip":
        with zipfile.ZipFile(source) as zf:
            names = zf.namelist()
            obj_name = next(n for n in names if n.lower().endswith(".obj"))
            mtl_name = next((n for n in names if n.lower().endswith(".mtl")), None)
            obj_txt = zf.read(obj_name).decode("utf-8", "replace")
            mtl_txt = zf.read(mtl_name).decode("utf-8", "replace") if mtl_name else ""
            return obj_txt, mtl_txt
    obj_txt = source.read_text(encoding="utf-8", errors="replace")
    mtl_path = source.with_suffix(".mtl")
    mtl_txt = mtl_path.read_text(encoding="utf-8", errors="replace") if mtl_path.exists() else ""
    return obj_txt, mtl_txt


def load_cad_objects(source: Path) -> list[CadObject]:
    """Import every `g <name>` CAD piece individually with per-corner normals and MTL colour."""
    obj_txt, mtl_txt = read_obj_mtl_texts(source)
    mtls = parse_mtl(mtl_txt)

    global_v: list[tuple[float, float, float]] = []
    global_vn: list[tuple[float, float, float]] = []

    raw_groups: list[tuple[str, str | None, list[list[tuple[int, int]]]]] = []
    cur_name = "piece_0"
    cur_mtl: str | None = None
    cur_faces: list[list[tuple[int, int]]] = []

    def flush() -> None:
        nonlocal cur_faces
        if cur_faces:
            raw_groups.append((cur_name, cur_mtl, cur_faces))
            cur_faces = []

    for line in obj_txt.splitlines():
        if line.startswith("v "):
            p = line.split()
            global_v.append((float(p[1]), float(p[2]), float(p[3])))
        elif line.startswith("vn "):
            p = line.split()
            global_vn.append((float(p[1]), float(p[2]), float(p[3])))
        elif line.startswith("g "):
            flush()
            cur_name = line[2:].strip() or "unnamed"
            cur_mtl = None  # Reset per object so pieces without usemtl (e.g. Barrier 2) get default grey
        elif line.startswith("usemtl "):
            cur_mtl = line[7:].strip()
        elif line.startswith("f "):
            toks = line.split()[1:]
            corners: list[tuple[int, int]] = []
            for tok in toks:
                parts = tok.split("/")
                vi = int(parts[0])
                vi = vi - 1 if vi > 0 else len(global_v) + vi
                ni = -1
                if len(parts) >= 3 and parts[2]:
                    ni_raw = int(parts[2])
                    ni = ni_raw - 1 if ni_raw > 0 else len(global_vn) + ni_raw
                corners.append((vi, ni))
            for k in range(1, len(corners) - 1):
                cur_faces.append([corners[0], corners[k], corners[k + 1]])
    flush()

    gv = np.asarray(global_v, dtype=np.float32)
    gvn = np.asarray(global_vn, dtype=np.float32) if global_vn else np.zeros((0, 3), dtype=np.float32)

    objects: list[CadObject] = []
    seen_slugs: dict[str, int] = {}
    for name, mtl_key, tris in raw_groups:
        base_slug = slugify(name)
        cnt = seen_slugs.get(base_slug, 0)
        seen_slugs[base_slug] = cnt + 1
        slug = base_slug if cnt == 0 else f"{base_slug}_{cnt}"

        vert_map: dict[tuple[int, int], int] = {}
        local_v: list[np.ndarray] = []
        local_n: list[np.ndarray] = []
        local_f: list[tuple[int, int, int]] = []

        for tri in tris:
            p0, p1, p2 = gv[tri[0][0]], gv[tri[1][0]], gv[tri[2][0]]
            fn = np.cross(p1 - p0, p2 - p0)
            fn_norm = float(np.linalg.norm(fn))
            fn = fn / fn_norm if fn_norm > 1e-12 else np.array([0.0, 0.0, 1.0], dtype=np.float32)
            tri_idx = []
            for vi, ni in tri:
                key = (vi, ni)
                idx = vert_map.get(key)
                if idx is None:
                    idx = len(local_v)
                    vert_map[key] = idx
                    local_v.append(gv[vi])
                    local_n.append(gvn[ni] if 0 <= ni < len(gvn) else fn)
                tri_idx.append(idx)
            local_f.append((tri_idx[0], tri_idx[1], tri_idx[2]))

        kd = mtls.get(mtl_key, DEFAULT_MTL_KD) if mtl_key is not None else DEFAULT_MTL_KD
        objects.append(
            CadObject(
                name=name,
                slug=slug,
                category=classify_piece(name),
                kd_rgb=kd,
                verts_enu=np.asarray(local_v, dtype=np.float32),
                norms_enu=np.asarray(local_n, dtype=np.float32),
                faces=np.asarray(local_f, dtype=np.uint32),
            )
        )
    return objects


def triangles_ned(objects: list[CadObject]) -> tuple[np.ndarray, np.ndarray]:
    """Return (tris_ned (T,3,3), outward_normals_ned (T,3)) in float64."""
    tri_list = []
    nrm_list = []
    for obj in objects:
        # ENU (E, N, U) -> NED (N, E, -U). Reflection flips triangle winding: swap (1, 2).
        v_ned = np.column_stack([obj.verts_enu[:, 1], obj.verts_enu[:, 0], -obj.verts_enu[:, 2]]).astype(np.float64)
        f = obj.faces[:, [0, 2, 1]]
        t = v_ned[f]  # (F, 3, 3)
        e1 = t[:, 1] - t[:, 0]
        e2 = t[:, 2] - t[:, 0]
        fn = np.cross(e1, e2)
        ln = np.linalg.norm(fn, axis=1, keepdims=True)
        valid = ln[:, 0] > 1e-12
        t = t[valid]
        fn = fn[valid] / ln[valid]
        tri_list.append(t)
        nrm_list.append(fn)
    return np.concatenate(tri_list, axis=0), np.concatenate(nrm_list, axis=0)


def sample_triangles_dense(tris: np.ndarray, step: float = 0.008, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Sample surface points at ~`step` m spacing across all triangles; return (points (M,3), tri_idx (M,))."""
    rng = np.random.default_rng(seed)
    a, b, c = tris[:, 0], tris[:, 1], tris[:, 2]
    areas = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    # Include all 3 vertices + centroid of every triangle, plus area-proportional interior samples
    pts_parts = [a, b, c, (a + b + c) / 3.0]
    idx_parts = [np.arange(len(tris), dtype=np.int32)] * 4

    counts = np.floor(areas / (step * step)).astype(np.int32)
    big = np.where(counts > 0)[0]
    if len(big):
        rep_tri = np.repeat(big, counts[big])
        r1 = np.sqrt(rng.random(len(rep_tri)))
        r2 = rng.random(len(rep_tri))
        w0 = 1.0 - r1
        w1 = r1 * (1.0 - r2)
        w2 = r1 * r2
        samp = w0[:, None] * a[rep_tri] + w1[:, None] * b[rep_tri] + w2[:, None] * c[rep_tri]
        pts_parts.append(samp)
        idx_parts.append(rep_tri.astype(np.int32))

    all_pts = np.concatenate(pts_parts, axis=0)
    all_idx = np.concatenate(idx_parts, axis=0)
    # Deduplicate at 2 mm voxels to keep KD-tree compact and ultra-fast
    keys = np.floor(all_pts / 0.002).astype(np.int64)
    _, uniq = np.unique(keys, axis=0, return_index=True)
    return all_pts[uniq], all_idx[uniq]


def closest_point_on_triangles_batch(
    p: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray
) -> np.ndarray:
    """Vectorized exact closest point on triangle (a,b,c) for each row of p,a,b,c: shape (N,3)."""
    ab = b - a
    ac = c - a
    ap = p - a
    d1 = np.einsum("ij,ij->i", ab, ap)
    d2 = np.einsum("ij,ij->i", ac, ap)

    bp = p - b
    d3 = np.einsum("ij,ij->i", ab, bp)
    d4 = np.einsum("ij,ij->i", ac, bp)

    cp = p - c
    d5 = np.einsum("ij,ij->i", ab, cp)
    d6 = np.einsum("ij,ij->i", ac, cp)

    vc = d1 * d4 - d3 * d2
    vb = d5 * d2 - d1 * d6
    va = d3 * d6 - d5 * d4

    denom = 1.0 / np.maximum(va + vb + vc, 1e-30)
    v_face = vb * denom
    w_face = vc * denom
    out = a + v_face[:, None] * ab + w_face[:, None] * ac

    # Vertex regions
    m_a = (d1 <= 0.0) & (d2 <= 0.0)
    m_b = (d3 >= 0.0) & (d4 <= d3)
    m_c = (d6 >= 0.0) & (d5 <= d6)
    out[m_a] = a[m_a]
    out[m_b] = b[m_b]
    out[m_c] = c[m_c]

    # Edge regions
    m_ab = (vc <= 0.0) & (d1 >= 0.0) & (d3 <= 0.0) & ~(m_a | m_b)
    if np.any(m_ab):
        t = d1[m_ab] / np.maximum(d1[m_ab] - d3[m_ab], 1e-30)
        out[m_ab] = a[m_ab] + t[:, None] * ab[m_ab]

    m_ac = (vb <= 0.0) & (d2 >= 0.0) & (d6 <= 0.0) & ~(m_a | m_c)
    if np.any(m_ac):
        t = d2[m_ac] / np.maximum(d2[m_ac] - d6[m_ac], 1e-30)
        out[m_ac] = a[m_ac] + t[:, None] * ac[m_ac]

    m_bc = (va <= 0.0) & ((d4 - d3) >= 0.0) & ((d5 - d6) >= 0.0) & ~(m_b | m_c)
    if np.any(m_bc):
        t = (d4[m_bc] - d3[m_bc]) / np.maximum((d4[m_bc] - d3[m_bc]) + (d5[m_bc] - d6[m_bc]), 1e-30)
        out[m_bc] = b[m_bc] + t[:, None] * (c[m_bc] - b[m_bc])

    return out


def compute_exact_closest_field(
    grid_pts: np.ndarray, tris: np.ndarray, tri_normals: np.ndarray, k_cand: int = 6
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (closest_q (N,3), closest_nrm (N,3), dist (N,)) for all grid points."""
    samp_pts, samp_tri = sample_triangles_dense(tris, step=0.008)
    print(f"  surface KD-tree: {len(samp_pts)} dense samples across {len(tris)} triangles")
    tree = cKDTree(samp_pts)
    _, nn_idx = tree.query(grid_pts, k=k_cand, workers=-1)
    cand_tris = samp_tri[nn_idx]  # (N, K)

    best_d2 = np.full(len(grid_pts), np.inf, dtype=np.float64)
    best_q = np.zeros_like(grid_pts)
    best_n = np.zeros_like(grid_pts)

    chunk = 200_000
    for s in range(0, len(grid_pts), chunk):
        e = min(len(grid_pts), s + chunk)
        p_sub = grid_pts[s:e]
        for k in range(k_cand):
            ti = cand_tris[s:e, k]
            q = closest_point_on_triangles_batch(p_sub, tris[ti, 0], tris[ti, 1], tris[ti, 2])
            d2 = np.einsum("ij,ij->i", p_sub - q, p_sub - q)
            better = d2 < best_d2[s:e]
            if np.any(better):
                best_d2[s:e][better] = d2[better]
                best_q[s:e][better] = q[better]
                best_n[s:e][better] = tri_normals[ti[better]]

    return best_q, best_n, np.sqrt(best_d2)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--cad",
        type=Path,
        default=DEPLOY_ROOT / "assets" / "main_direction" / "dronehacks-course-v3-obj.zip",
        help="Path to assets/main_direction/dronehacks-course-v3-obj.zip or dronehacks-course-v3.obj",
    )
    ap.add_argument("--out-dir", type=Path, default=DEPLOY_ROOT / "assets")
    ap.add_argument("--esdf-res", type=float, default=0.05, help="CAD ESDF voxel size [m] (default 0.05)")
    args = ap.parse_args(argv)

    objects = load_cad_objects(args.cad)
    by_cat: dict[str, list[CadObject]] = {}
    for o in objects:
        by_cat.setdefault(o.category, []).append(o)
    print(f"Imported {len(objects)} individual CAD objects from {args.cad.name}:")
    for cat, items in by_cat.items():
        print(f"  {cat:10s}: {len(items):2d} pieces ({sum(len(x.faces) for x in items)} tris)")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Build iSAM2 3D Closest-Point + Normal CAD ESDF (obstacles + floor + side net panels)
    isam_objs = [
        o for o in objects
        if o.category in ("obstacle", "floor") or (o.category == "net" and "roof" not in o.name.lower())
    ]
    tris_isam, nrms_isam = triangles_ned(isam_objs)
    res = float(args.esdf_res)
    lo = np.array([-3.20, -3.20, -3.20], dtype=np.float64)
    hi = np.array([3.20, 3.20, 0.20], dtype=np.float64)
    shape = tuple(int(round((hi[k] - lo[k]) / res)) + 1 for k in range(3))
    axes = [lo[k] + res * np.arange(shape[k], dtype=np.float64) for k in range(3)]
    grid_pts = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)

    print(f"Computing iSAM2 CAD ESDF over {shape} ({len(grid_pts)} cells @ {100*res:.1f} cm)...")
    best_q, best_n, _ = compute_exact_closest_field(grid_pts, tris_isam, nrms_isam)

    # Encode offset (q - cell_center) in 0.1 mm (1e-4 m) as int16, normal as int8 * 127
    dq_01mm = np.clip(np.rint((best_q - grid_pts) * 10000.0), -32767, 32767).astype("<i2")
    n_i8 = np.clip(np.rint(best_n * 127.0), -127, 127).astype("<i1")
    flags = np.zeros((len(grid_pts), 1), dtype="<i1")

    rec = np.empty(
        len(grid_pts),
        dtype=[("dx", "<i2"), ("dy", "<i2"), ("dz", "<i2"), ("nx", "i1"), ("ny", "i1"), ("nz", "i1"), ("flags", "i1")],
    )
    rec["dx"], rec["dy"], rec["dz"] = dq_01mm[:, 0], dq_01mm[:, 1], dq_01mm[:, 2]
    rec["nx"], rec["ny"], rec["nz"] = n_i8[:, 0], n_i8[:, 1], n_i8[:, 2]
    rec["flags"] = flags[:, 0]

    esdf_path = args.out_dir / "cad_prior_esdf.bin"
    with open(esdf_path, "wb") as f:
        # Header (36 bytes, matching EGOP layout): magic "CADP", version=1, nx, ny, nz, res, lo_x, lo_y, lo_z
        f.write(
            struct.pack(
                "<4sIiiiffff",
                b"CADP",
                1,
                shape[0],
                shape[1],
                shape[2],
                float(res),
                float(lo[0]),
                float(lo[1]),
                float(lo[2]),
            )
        )
        f.write(rec.tobytes(order="C"))
    print(f"Wrote {esdf_path} ({esdf_path.stat().st_size / (1024 * 1024):.2f} MiB)")

    # 2. Rebuild assets/arena_prior.bin ("EGOP" obstacle clearance) from CAD obstacle pieces
    obs_objs = [o for o in objects if o.category == "obstacle"]
    tris_obs, nrms_obs = triangles_ned(obs_objs)
    res_a = 0.05
    lo_a = np.array([-3.05, -3.05, -3.05], dtype=np.float64)
    hi_a = np.array([3.05, 3.05, 0.00], dtype=np.float64)
    shape_a = tuple(int(round((hi_a[k] - lo_a[k]) / res_a)) + 1 for k in range(3))
    axes_a = [lo_a[k] + res_a * np.arange(shape_a[k], dtype=np.float64) for k in range(3)]
    pts_a = np.stack(np.meshgrid(*axes_a, indexing="ij"), axis=-1).reshape(-1, 3)

    print(f"Computing arena_prior.bin obstacle clearance over {shape_a} ({len(pts_a)} cells)...")
    _, _, dist_obs = compute_exact_closest_field(pts_a, tris_obs, nrms_obs)
    q_dist = np.clip(np.rint(dist_obs * 200.0), 0, 255).astype(np.uint8)

    arena_bin = args.out_dir / "arena_prior.bin"
    with open(arena_bin, "wb") as f:
        f.write(
            struct.pack(
                "<4sIiiiffff",
                b"EGOP",
                1,
                shape_a[0],
                shape_a[1],
                shape_a[2],
                float(res_a),
                float(lo_a[0]),
                float(lo_a[1]),
                float(lo_a[2]),
            )
        )
        f.write(q_dist.tobytes(order="C"))
    print(f"Wrote {arena_bin} ({arena_bin.stat().st_size / 1024:.1f} KiB)")

    arena_json = args.out_dir / "arena_prior.json"
    if arena_json.exists():
        meta = json.loads(arena_json.read_text())
        meta["cad_source"] = "dronehacks-course-v3-obj.zip (dronehacks-course-v3.obj + .mtl)"
        meta["isam2_prior"] = {
            "binary_file": "cad_prior_esdf.bin",
            "format": "CADP_v1_closest_point_normal_esdf",
            "resolution_m": res,
            "lo_ned": lo.tolist(),
            "hi_ned": hi.tolist(),
            "shape": list(shape),
            "pieces_included": [o.name for o in isam_objs],
        }
        arena_json.write_text(json.dumps(meta, indent=2) + "\n")
        print(f"Updated {arena_json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
