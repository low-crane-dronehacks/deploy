"""Pytest suite for C++ GroundTruthMap, Informed RRT*, PathTransformer, STVL Costmap, and 2048-rollout MPPI."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np

from policynav.maze_planning_py import MazePlanningPipeline

ROOT = Path(__file__).resolve().parents[1]


def test_groundtruth_map_encoding_and_clearance():
    gt_bin = ROOT / "config" / "course_groundtruth.bin"
    gt_json = ROOT / "config" / "course_groundtruth.json"
    assert gt_bin.exists(), "course_groundtruth.bin must exist"
    assert gt_json.exists(), "course_groundtruth.json must exist"

    with open(gt_json) as f:
        meta = json.load(f)
    assert meta["magic"] == "CGTM"
    assert meta["version"] == 1
    assert len(meta["primitives"]["obb_slabs"]) == 14
    assert len(meta["primitives"]["boulders"]) == 5
    assert meta["surface_mesh_summary"]["num_triangles"] >= 2000

    pipe = MazePlanningPipeline()
    d_launch, grad_launch = pipe.query_groundtruth([0.9144, -2.5146, -0.78])
    assert d_launch > 0.38, f"Expected launch clearance > 0.38m, got {d_launch:.3f}m"
    assert np.all(np.isfinite(grad_launch))

    d_win, _ = pipe.query_groundtruth([-2.2830, -0.4750, -0.78])
    assert d_win > 0.22, f"Expected window aperture clearance > 0.22m, got {d_win:.3f}m"


def test_rrt_star_point_to_point_and_course_planning():
    pipe = MazePlanningPipeline()
    wps = pipe.plan_point_to_point([0.9144, -2.5146, -0.78], [1.60, -1.35, -0.78])
    assert wps.shape[0] >= 2 and wps.shape[1] == 3
    for pt in wps:
        d, _ = pipe.query_groundtruth(pt)
        assert d > 0.16, f"RRT* waypoint {pt} too close to obstacle (d={d:.3f}m)"


def test_mppi_2048_rollouts_and_pillar_evasion():
    pipe = MazePlanningPipeline(mppi_samples=2048, mppi_horizon_steps=16)
    pipe.plan_point_to_point([0.9144, -2.5146, -0.78], [2.55, -2.5146, -0.78])

    # Place an unmodeled vertical pole dead-center in the launch corridor (Milestone 3 & Section 6 FMEA)
    pipe.inject_pillar(1.75, -2.5146, radius_m=0.10, height_m=1.65)
    assert pipe.query_costmap([1.75, -2.5146, -0.78]) < 0.12

    res = pipe.step(
        pos_odom=[1.25, -2.5146, -0.78],
        vel_odom=[1.2, 0.0, 0.0],
        quat_wxyz=[1.0, 0.0, 0.0, 0.0],
    )
    assert np.all(np.isfinite(res["vel_ned"]))
    assert res["vel_ned"][0] > 0.3, "Expected forward velocity along corridor"
    assert abs(res["vel_ned"][1]) > 0.12, "Expected lateral evasive velocity around dead-center pillar"
    assert res["rl_lookaheads_body"].shape == (4, 3)


def test_path_transformer_map_to_odom_shift():
    pipe = MazePlanningPipeline()
    pipe.reset((0.9144, -2.5146, -0.78))
    # Simulate 10 cm iSAM2 correction shift on map -> odom
    pipe.set_map_to_odom((0.10, -0.05, 0.0), (1.0, 0.0, 0.0, 0.0))
    res = pipe.step(
        pos_odom=[0.8144, -2.4646, -0.78],
        vel_odom=[0.5, 0.0, 0.0],
        quat_wxyz=[1.0, 0.0, 0.0, 0.0],
    )
    assert np.all(np.isfinite(res["body_action"]))
    assert np.all(np.isfinite(res["rl_lookaheads_body"]))
