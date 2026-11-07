"""Unit and integration tests for C++ EGO-Planner-V2 via Python ctypes bindings."""
from __future__ import annotations

import os
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ego_planner" / "python"))

from ego_planner_py import EgoPlannerV2  # noqa: E402


def test_esdf_arena_prior():
    planner = EgoPlannerV2(arm_authorized=True)
    # Launch pose at [0.9144, -2.5146, -0.78] should be well clear of obstacles
    d_launch, grad = planner.query_esdf([0.9144, -2.5146, -0.78])
    assert d_launch > 0.38, f"Expected launch clearance > 0.38m, got {d_launch:.3f}m"
    assert np.all(np.isfinite(grad))

    # Window aperture center at [-2.2830, -0.4750, -0.78] should be open
    d_win, _ = planner.query_esdf([-2.2830, -0.4750, -0.78])
    assert d_win > 0.25, f"Expected window center clearance > 0.25m, got {d_win:.3f}m"


def test_mission_state_machine_and_safety():
    # Verify arm_authorized=False never arms
    p_disarmed = EgoPlannerV2(arm_authorized=False)
    p_disarmed.reset(ready_for_sim_flight=False)
    for t in [0.0, 0.5, 1.5, 2.5]:
        out = p_disarmed.step_raw(
            t_s=t,
            pos_ned=(0.9144, -2.5146, -0.08),
            vel_ned=(0.0, 0.0, 0.0),
            quat_wxyz=(1.0, 0.0, 0.0, 0.0),
            armed=False,
            offboard=False,
            landed=True,
            pose_valid=True,
            healthy=True,
        )
        assert out["phase"] == "WAIT"
        assert not out["publish_arm_cmd"]

    # Verify arm_authorized=True transitions WAIT -> ARM -> FLY and commands climb
    p_armed = EgoPlannerV2(arm_authorized=True)
    p_armed.reset(ready_for_sim_flight=True)
    action, info = p_armed.act_sim_single(
        pos_ned=np.array([0.9144, -2.5146, -0.08], dtype=np.float32),
        vel_ned=np.zeros(3, dtype=np.float32),
        quat_wxyz=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
    )
    assert info["phase"] == "FLY"
    assert info["active_leg"] == "run_course"
    assert action.shape == (4,)
    assert action[2] < 0.0, "Expected upward velocity command (negative z in NED) on takeoff"


def test_depth_raycast_updates_esdf():
    planner = EgoPlannerV2(arm_authorized=True)
    planner.reset(ready_for_sim_flight=True)
    # Inject a synthetic wall at 1.4m ahead of launch pose
    depth = np.full((80, 128), 1.4 / 5.0, dtype=np.float32)
    planner.act_sim_single(
        pos_ned=np.array([0.9144, -2.5146, -0.78], dtype=np.float32),
        vel_ned=np.zeros(3, dtype=np.float32),
        quat_wxyz=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        depth_norm=depth,
    )
    d_wall, _ = planner.query_esdf([0.9144 + 1.4, -2.5146, -0.78])
    assert d_wall < 0.15, f"Expected raycasted obstacle to lower ESDF to <0.15m, got {d_wall:.3f}m"


def test_arena_priors_and_slam_config():
    import json
    prior_json_path = ROOT / "config" / "arena_prior.json"
    assert prior_json_path.exists(), "arena_prior.json must exist"
    with open(prior_json_path) as f:
        prior = json.load(f)
    assert "gates" in prior and len(prior["gates"]) >= 4, "Must contain all competition gates"
    assert "landmarks" in prior and "launch_mark_ned" in prior["landmarks"], "Must declare launch mark anchor"
    assert "pad_center_ned" in prior["landmarks"], "Must declare touchdown pad anchor"

