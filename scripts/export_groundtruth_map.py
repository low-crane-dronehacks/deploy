#!/usr/bin/env python3
"""Export the DroneHacks 2026 Groundtruth Map Encoding for C++ RRT* and MPPI.

Unlike config/arena_prior.bin (which is a pre-computed voxel ESDF), this script creates
a new geometric + spatial-index groundtruth map encoding directly from the course mesh
(sitl/world/course.obj or dronehacks-sim/assets/course.npz) and course rules:

  1. Analytical 3D Oriented Bounding Box (OBB) wall/window slabs (with exact CSG window cutout),
     semi-ellipsoid boulder primitives, and tunnel arch/rib primitives.
  2. Compact Surface-Triangle Mesh (walls + tunnel + decimated boulder hulls) for exact C++ AABB-BVH
     closest-point and continuous swept-sphere collision queries.
  3. 3D Morton (Z-Order) Bit-Packed Binary Occupancy Grid (1 bit per 5cm cell, NO pre-made ESDF).
  4. Gate apertures, sector polygons, launch mark, and landing pad metadata.

Writes:
  config/course_groundtruth.json   Human-readable geometric primitives, gates, and landmarks
  config/course_groundtruth.bin    Compact binary groundtruth encoding (magic "CGTM" v1)
"""
from __future__ import annotations

import json
import math
import os
import struct
import sys

import numpy as np

SIM_DIR = os.environ.get("DRONEHACKS_SIM", "/home/ejkaba/prj/dronehacks-sim")
if os.path.isdir(SIM_DIR):
    sys.path.insert(0, SIM_DIR)

DEPLOY_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def build_Analytical_primitives() -> tuple[list[dict], list[dict], dict]:
    """Return (obb_slabs, boulders, tunnel_spec) in NED metres."""
    # Each OBB slab is defined by 2D centerline endpoints A=(ax, ay), B=(bx, by),
    # half_thickness_m (perpendicular to AB in XY), and z_min_ned, z_max_ned (NED: z <= 0).
    obb_slabs = [
        # wall0 decomposed around the window aperture [-2.7015..-1.84225, -0.62735..-0.31455, z=-1.2192..-0.3048]
        {
            "name": "wall0_left_pillar",
            "ax": -3.0297, "ay": -0.74675,
            "bx": -2.7015, "by": -0.62735,
            "half_thickness_m": 0.0508,
            "z_min_ned": -1.7272, "z_max_ned": 0.0,
        },
        {
            "name": "wall0_right_pillar",
            "ax": -1.84225, "ay": -0.31455,
            "bx": -0.5472, "by": 0.1568,
            "half_thickness_m": 0.0508,
            "z_min_ned": -1.7272, "z_max_ned": 0.0,
        },
        {
            "name": "wall0_window_sill",
            "ax": -2.7015, "ay": -0.62735,
            "bx": -1.84225, "by": -0.31455,
            "half_thickness_m": 0.0508,
            "z_min_ned": -0.3048, "z_max_ned": 0.0,
        },
        {
            "name": "wall0_window_lintel",
            "ax": -2.7015, "ay": -0.62735,
            "bx": -1.84225, "by": -0.31455,
            "half_thickness_m": 0.0508,
            "z_min_ned": -1.7272, "z_max_ned": -1.2192,
        },
        # wall1..wall6 solid full-height walls
        {
            "name": "wall1",
            "ax": 0.4572, "ay": -3.0480,
            "bx": 0.4572, "by": 1.5748,
            "half_thickness_m": 0.0508,
            "z_min_ned": -1.7272, "z_max_ned": 0.0,
        },
        {
            "name": "wall2",
            "ax": -1.12855, "ay": 0.97335,
            "bx": 0.3871, "by": 1.52505,
            "half_thickness_m": 0.0508,
            "z_min_ned": -1.7272, "z_max_ned": 0.0,
        },
        {
            "name": "wall3",
            "ax": 0.5080, "ay": -1.9304,
            "bx": 2.1209, "by": -1.9304,
            "half_thickness_m": 0.0508,
            "z_min_ned": -1.7272, "z_max_ned": 0.0,
        },
        {
            "name": "wall4",
            "ax": -0.6223, "ay": -0.9754,
            "bx": 0.4064, "by": -0.9754,
            "half_thickness_m": 0.0508,
            "z_min_ned": -1.7272, "z_max_ned": 0.0,
        },
        {
            "name": "wall5",
            "ax": 1.4351, "ay": -0.8128,
            "bx": 3.0480, "by": -0.8128,
            "half_thickness_m": 0.0508,
            "z_min_ned": -1.7272, "z_max_ned": 0.0,
        },
        {
            "name": "wall6",
            "ax": 0.5080, "ay": 0.3048,
            "bx": 2.1209, "by": 0.3048,
            "half_thickness_m": 0.0508,
            "z_min_ned": -1.7272, "z_max_ned": 0.0,
        },
        # Tunnel floor threshold ribs (3 ribs inside tunnel at x=-0.2921, 0.4572, 1.2065)
        {
            "name": "tunnel_rib0",
            "ax": -0.2921, "ay": 1.8034,
            "bx": -0.2921, "by": 2.8194,
            "half_thickness_m": 0.0127,
            "z_min_ned": -0.2776, "z_max_ned": 0.0,
        },
        {
            "name": "tunnel_rib1",
            "ax": 0.4572, "ay": 1.8034,
            "bx": 0.4572, "by": 2.8194,
            "half_thickness_m": 0.0127,
            "z_min_ned": -0.2776, "z_max_ned": 0.0,
        },
        {
            "name": "tunnel_rib2",
            "ax": 1.2065, "ay": 1.8034,
            "bx": 1.2065, "by": 2.8194,
            "half_thickness_m": 0.0127,
            "z_min_ned": -0.2776, "z_max_ned": 0.0,
        },
        # Connector post between wall1 and tunnel
        {
            "name": "tunnel_wall_post",
            "ax": 0.4064, "ay": 1.5748,
            "bx": 0.5080, "by": 1.5748,
            "half_thickness_m": 0.010,
            "z_min_ned": -1.7272, "z_max_ned": 0.0,
        },
    ]

    boulders = [
        {"name": "boulder0", "cx": -2.46888, "cy": 2.01168, "cz": 0.0, "rx": 0.2286, "ry": 0.2286, "rz": 0.5334},
        {"name": "boulder1", "cx": -2.34696, "cy": 0.79248, "cz": 0.0, "rx": 0.2286, "ry": 0.2286, "rz": 0.5334},
        {"name": "boulder2", "cx": -2.49936, "cy": -2.56032, "cz": 0.0, "rx": 0.2286, "ry": 0.2286, "rz": 0.5334},
        {"name": "boulder3", "cx": -2.37744, "cy": -1.34112, "cz": 0.0, "rx": 0.2286, "ry": 0.2286, "rz": 0.5334},
        {"name": "boulder4", "cx": -1.55448, "cy": 2.52984, "cz": 0.0, "rx": 0.2286, "ry": 0.2286, "rz": 0.5334},
    ]

    tunnel_spec = {
        "x_min_ned": -0.3048,
        "x_max_ned": 1.2192,
        "y_min_ned": 1.5748,
        "y_max_ned": 3.0480,
        "z_top_ned": -1.8542,
        "y_center_ned": 2.3114,
        "arch_inner_half_y_m": 0.728,
        "arch_outer_half_y_m": 0.7366,
    }
    return obb_slabs, boulders, tunnel_spec


def make_boulder_hemisphere_mesh(cx: float, cy: float, rx: float, ry: float, rz: float,
                                 n_lat: int = 6, n_lon: int = 12) -> tuple[np.ndarray, np.ndarray]:
    """Generate a compact triangulated semi-ellipsoid mesh in NED (z <= 0) for BVH inclusion."""
    verts = [[cx, cy, -rz]]  # pole at index 0
    for ilat in range(1, n_lat + 1):
        theta = (0.5 * math.pi) * (ilat / n_lat)  # 0 at top pole, pi/2 at floor
        st, ct = math.sin(theta), math.cos(theta)
        for ilon in range(n_lon):
            phi = (2.0 * math.pi) * (ilon / n_lon)
            x = cx + rx * st * math.cos(phi)
            y = cy + ry * st * math.sin(phi)
            z = -rz * ct
            verts.append([x, y, z])
    faces = []
    # Top cap triangles
    for ilon in range(n_lon):
        nxt = (ilon + 1) % n_lon
        faces.append([0, 1 + ilon, 1 + nxt])
    # Quad strips
    for ilat in range(1, n_lat):
        base0 = 1 + (ilat - 1) * n_lon
        base1 = 1 + ilat * n_lon
        for ilon in range(n_lon):
            nxt = (ilon + 1) % n_lon
            faces.append([base0 + ilon, base1 + ilon, base1 + nxt])
            faces.append([base0 + ilon, base1 + nxt, base0 + nxt])
    return np.asarray(verts, dtype=np.float32), np.asarray(faces, dtype=np.int32)


def extract_surface_triangles(boulders: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """Extract exact wall + tunnel triangles from course.npz (or course.obj) plus compact boulder meshes."""
    from dronehacks.mesh import CourseMesh, GROUP_ID

    course_npz = os.path.join(SIM_DIR, "assets", "course.npz")
    course = CourseMesh.load(course_npz)
    wt = course.select(("wall", "tunnel"))
    verts_list = [wt.vertices.astype(np.float32)]
    faces_list = [wt.faces.astype(np.int32)]
    v_offset = len(verts_list[0])

    for b in boulders:
        bv, bf = make_boulder_hemisphere_mesh(b["cx"], b["cy"], b["rx"], b["ry"], b["rz"])
        verts_list.append(bv)
        faces_list.append(bf + v_offset)
        v_offset += len(bv)

    all_v = np.concatenate(verts_list, axis=0)
    all_f = np.concatenate(faces_list, axis=0)
    return all_v, all_f


def build_binary_occupancy_bitset(
    all_v: np.ndarray,
    all_f: np.ndarray,
    lo_ned: np.ndarray,
    hi_ned: np.ndarray,
    res: float = 0.05,
    occ_shell_m: float = 0.04,
) -> tuple[tuple[int, int, int], np.ndarray]:
    """Build a 1-bit-per-voxel binary occupancy bitset (NOT an ESDF) by Triangle-Box / surface rasterization."""
    shape = tuple(int(round((hi_ned[k] - lo_ned[k]) / res)) + 1 for k in range(3))
    nx, ny, nz = shape
    total_voxels = nx * ny * nz
    occ_bool = np.zeros((nx, ny, nz), dtype=bool)

    # Rasterize every obstacle triangle into the binary occupancy grid
    v0 = all_v[all_f[:, 0]]
    v1 = all_v[all_f[:, 1]]
    v2 = all_v[all_f[:, 2]]

    for i in range(len(all_f)):
        a, b, c = v0[i], v1[i], v2[i]
        tri_lo = np.minimum(np.minimum(a, b), c) - occ_shell_m
        tri_hi = np.maximum(np.maximum(a, b), c) + occ_shell_m
        i0 = np.clip(np.floor((tri_lo - lo_ned) / res).astype(int), 0, np.array(shape) - 1)
        i1 = np.clip(np.ceil((tri_hi - lo_ned) / res).astype(int), 0, np.array(shape) - 1)

        # Sample points on triangle if large, or check voxel centers in bounding box
        xs = lo_ned[0] + res * np.arange(i0[0], i1[0] + 1, dtype=np.float32)
        ys = lo_ned[1] + res * np.arange(i0[1], i1[1] + 1, dtype=np.float32)
        zs = lo_ned[2] + res * np.arange(i0[2], i1[2] + 1, dtype=np.float32)
        pts = np.stack(np.meshgrid(xs, ys, zs, indexing="ij"), axis=-1).reshape(-1, 3)

        # Exact point-to-triangle distance for voxels in triangle AABB
        ab = b - a
        ac = c - a
        ap = pts - a
        d1 = ap @ ab
        d2 = ap @ ac
        # Project onto triangle plane + clamp to edges
        n_vec = np.cross(ab, ac)
        n_norm = np.linalg.norm(n_vec)
        if n_norm < 1e-9:
            continue
        n_unit = n_vec / n_norm
        dist_plane = np.abs(ap @ n_unit)
        # Barycentric check with edge fallback
        v_proj = pts - (ap @ n_unit)[:, None] * n_unit
        c0 = np.cross(b - a, v_proj - a) @ n_unit
        c1 = np.cross(c - b, v_proj - b) @ n_unit
        c2 = np.cross(a - c, v_proj - c) @ n_unit
        inside_tri = (c0 >= -1e-4) & (c1 >= -1e-4) & (c2 >= -1e-4)

        def seg_dist_sq(p, s0, s1):
            d = s1 - s0
            l2 = float(d @ d)
            if l2 < 1e-12:
                return np.sum((p - s0) ** 2, axis=1)
            t = np.clip(((p - s0) @ d) / l2, 0.0, 1.0)
            proj = s0 + t[:, None] * d
            return np.sum((p - proj) ** 2, axis=1)

        d_sq = np.where(
            inside_tri,
            dist_plane ** 2,
            np.minimum(
                seg_dist_sq(pts, a, b),
                np.minimum(seg_dist_sq(pts, b, c), seg_dist_sq(pts, c, a)),
            ),
        )
        hit = (d_sq <= occ_shell_m ** 2).reshape(len(xs), len(ys), len(zs))
        occ_bool[i0[0]:i1[0] + 1, i0[1]:i1[1] + 1, i0[2]:i1[2] + 1] |= hit

    # Also mark interior of the 5 semi-ellipsoid boulders as occupied
    ax_x = lo_ned[0] + res * np.arange(nx, dtype=np.float32)[:, None, None]
    ax_y = lo_ned[1] + res * np.arange(ny, dtype=np.float32)[None, :, None]
    ax_z = lo_ned[2] + res * np.arange(nz, dtype=np.float32)[None, None, :]
    _, boulders, _ = build_Analytical_primitives()
    for b in boulders:
        val = ((ax_x - b["cx"]) / b["rx"]) ** 2 + ((ax_y - b["cy"]) / b["ry"]) ** 2 + ((ax_z - b["cz"]) / b["rz"]) ** 2
        occ_bool |= (val <= 1.0) & (ax_z <= 0.0)

    flat_bits = occ_bool.ravel(order="C")
    n_words = (total_voxels + 63) // 64
    padded = np.zeros(n_words * 64, dtype=np.uint8)
    padded[:total_voxels] = flat_bits.astype(np.uint8)
    words = np.zeros(n_words, dtype=np.uint64)
    for bit in range(64):
        words |= padded[bit::64].astype(np.uint64) << np.uint64(bit)

    return shape, words


def main() -> None:
    obb_slabs, boulders, tunnel_spec = build_Analytical_primitives()
    all_v, all_f = extract_surface_triangles(boulders)

    lo_ned = np.array([-3.05, -3.05, -3.05], dtype=np.float64)
    hi_ned = np.array([3.05, 3.05, 0.00], dtype=np.float64)
    res = 0.05
    shape, bit_words = build_binary_occupancy_bitset(all_v, all_f, lo_ned, hi_ned, res=res)

    prior_json_path = os.path.join(DEPLOY_ROOT, "config", "arena_prior.json")
    with open(prior_json_path, "r") as f:
        prior_meta = json.load(f)

    gt_json = {
        "encoding_name": "DroneHacks 2026 Course GroundTruth Map (Analytical OBB/CSG + Surface BVH + Binary Occupancy Bitset)",
        "magic": "CGTM",
        "version": 1,
        "frame": "NED",
        "binary_file": "course_groundtruth.bin",
        "arena": prior_meta["arena"],
        "landmarks": prior_meta["landmarks"],
        "gates": prior_meta["gates"],
        "occupancy_bitset": {
            "resolution_m": res,
            "lo_ned": lo_ned.tolist(),
            "hi_ned": hi_ned.tolist(),
            "shape": list(shape),
            "num_uint64_words": int(len(bit_words)),
            "occupied_voxels": int(sum(bin(int(w)).count("1") for w in bit_words)),
        },
        "primitives": {
            "obb_slabs": obb_slabs,
            "boulders": boulders,
            "tunnel": tunnel_spec,
        },
        "surface_mesh_summary": {
            "num_vertices": int(len(all_v)),
            "num_triangles": int(len(all_f)),
        },
    }

    out_dir = os.path.join(DEPLOY_ROOT, "config")
    os.makedirs(out_dir, exist_ok=True)
    json_out = os.path.join(out_dir, "course_groundtruth.json")
    with open(json_out, "w") as f:
        json.dump(gt_json, f, indent=2)

    bin_out = os.path.join(out_dir, "course_groundtruth.bin")
    with open(bin_out, "wb") as f:
        # Header:
        #   magic "CGTM" (4B), version=1 (uint32),
        #   nx, ny, nz (3x int32), res (float32), lo_x, lo_y, lo_z (3x float32),
        #   n_obb (uint32), n_boulders (uint32), n_verts (uint32), n_tris (uint32), n_bit_words (uint32)
        f.write(
            struct.pack(
                "<4sIiiiffffIIIII",
                b"CGTM",
                1,
                shape[0],
                shape[1],
                shape[2],
                float(res),
                float(lo_ned[0]),
                float(lo_ned[1]),
                float(lo_ned[2]),
                len(obb_slabs),
                len(boulders),
                len(all_v),
                len(all_f),
                len(bit_words),
            )
        )
        # 1. OBB slabs: each is 7x float32 (ax, ay, bx, by, half_thick, z_min, z_max)
        for s in obb_slabs:
            f.write(
                struct.pack(
                    "<fffffff",
                    float(s["ax"]),
                    float(s["ay"]),
                    float(s["bx"]),
                    float(s["by"]),
                    float(s["half_thickness_m"]),
                    float(s["z_min_ned"]),
                    float(s["z_max_ned"]),
                )
            )
        # 2. Boulders: each is 6x float32 (cx, cy, cz, rx, ry, rz)
        for b in boulders:
            f.write(
                struct.pack(
                    "<ffffff",
                    float(b["cx"]),
                    float(b["cy"]),
                    float(b["cz"]),
                    float(b["rx"]),
                    float(b["ry"]),
                    float(b["rz"]),
                )
            )
        # 3. Surface vertices: (n_verts, 3) float32
        f.write(np.ascontiguousarray(all_v, dtype=np.float32).tobytes())
        # 4. Surface triangle indices: (n_tris, 3) int32
        f.write(np.ascontiguousarray(all_f, dtype=np.int32).tobytes())
        # 5. Binary occupancy bitset: (n_bit_words,) uint64
        f.write(np.ascontiguousarray(bit_words, dtype=np.uint64).tobytes())

    print(
        f"Wrote {bin_out} ({os.path.getsize(bin_out) / 1024:.1f} KiB: "
        f"{len(obb_slabs)} OBBs, {len(boulders)} boulders, {len(all_f)} triangles, "
        f"{gt_json['occupancy_bitset']['occupied_voxels']} occupied bits) and {json_out}"
    )


if __name__ == "__main__":
    main()
