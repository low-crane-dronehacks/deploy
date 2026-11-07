#!/usr/bin/env python3
"""Closed-loop evaluation of C++ Informed RRT* + PathTransformer + STVL Costmap + 2048-rollout MPPI
in dronehacks-sim (CourseEnv), including Milestone 3 unmodeled vertical pole evasion tests.
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

DEPLOY_ROOT = Path(__file__).resolve().parents[1]
SIM_ROOT = Path(os.environ.get("DRONEHACKS_SIM", "/home/ejkaba/prj/dronehacks-sim"))
sys.path.insert(0, str(DEPLOY_ROOT))
sys.path.insert(0, str(SIM_ROOT))

from policynav.maze_planning_py import MazePlanningPipeline  # noqa: E402
from dronehacks.bank import BANK_LAUNCH  # noqa: E402
from dronehacks.env import CourseEnv, CourseEnvCfg  # noqa: E402


def rpy_to_quat_wxyz(roll: float, pitch: float, yaw: float) -> tuple[float, float, float, float]:
    cr, sr = math.cos(0.5 * roll), math.sin(0.5 * roll)
    cp, sp = math.cos(0.5 * pitch), math.sin(0.5 * pitch)
    cy, sy = math.cos(0.5 * yaw), math.sin(0.5 * yaw)
    return (
        cy * cr * cp + sy * sr * sp,
        cy * sr * cp - sy * cr * sp,
        cy * cr * sp + sy * sr * cp,
        sy * cr * cp - cy * sr * sp,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate C++ RRT* + 2048-rollout MPPI in dronehacks-sim")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--mppi-samples", type=int, default=2048)
    ap.add_argument("--inject-pillar", action="store_true", help="Place unmodeled vertical pole in launch chute")
    args = ap.parse_args()

    cfg = CourseEnvCfg(
        num_envs=1,
        device=args.device,
        seed=1000,
        assets_dir=str(SIM_ROOT / "assets"),
    )
    env = CourseEnv(cfg)
    launch_cells = np.nonzero(env._bank_kind.cpu().numpy() == BANK_LAUNCH)[0]
    env.set_eval_queue(torch.tensor([[int(launch_cells[0])]], device=env.device))
    env.get_observations()

    pipe = MazePlanningPipeline(mppi_samples=args.mppi_samples, cruise_speed_m_s=2.65)
    pos0 = env.plant.pos_ned[0].detach().cpu().numpy()
    pipe.reset((float(pos0[0]), float(pos0[1]), -0.78))
    if args.inject_pillar:
        pipe.inject_pillar(1.75, -2.5146, radius_m=0.10, height_m=1.65)

    solve_times = []
    min_clearances = []
    max_cp = 0

    for step_idx in range(args.steps):
        pos_ned = env.plant.pos_ned[0].detach().cpu().numpy()
        vel_ned = env.plant.vel_ned[0].detach().cpu().numpy()
        rpy_ned = env.plant.rpy_ned()[0].detach().cpu().numpy()
        quat_wxyz = rpy_to_quat_wxyz(float(rpy_ned[0]), float(rpy_ned[1]), float(rpy_ned[2]))

        # Takeoff climb until altitude >= 0.55 m, then hand over to RRT* + MPPI
        if -pos_ned[2] < 0.55 and step_idx < 12:
            action = np.array([0.0, 0.0, -0.65, 0.0], dtype=np.float32)
        else:
            t0 = time.perf_counter()
            out = pipe.step(pos_odom=pos_ned, vel_odom=vel_ned, quat_wxyz=quat_wxyz, dt_s=0.10)
            solve_times.append((time.perf_counter() - t0) * 1e3)
            action = out["body_action"]

        d_c = pipe.query_costmap(pos_ned)
        min_clearances.append(d_c)
        max_cp = max(max_cp, int(env._next_cp[0].item()))

        act_tensor = torch.from_numpy(action).to(env.device).unsqueeze(0)
        _, _, dones, _ = env.step(act_tensor)
        if int(dones[0]):
            break

    print(
        f"Completed {step_idx + 1} steps | max_cp={max_cp} | "
        f"mean_mppi_ms={np.mean(solve_times):.2f}ms (p99={np.percentile(solve_times, 99):.2f}ms) | "
        f"min_clearance={np.min(min_clearances):.3f}m"
    )


if __name__ == "__main__":
    main()
