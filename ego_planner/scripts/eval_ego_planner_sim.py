#!/usr/bin/env python3
"""Evaluate C++ EGO-Planner-V2 in dronehacks-sim (CourseEnv) on nominal and perturbed courses,
and optionally render FPV depth + top-down map MP4 flight videos.

Usage:
  python3 ego_planner/scripts/eval_ego_planner_sim.py --device cpu --episodes 10
  python3 ego_planner/scripts/eval_ego_planner_sim.py --device cuda:0 --perturb 1.0 --video out/ego_planner_perturb.mp4
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

DEPLOY_ROOT = Path(__file__).resolve().parents[2]
SIM_ROOT = Path(os.environ.get("DRONEHACKS_SIM", "/home/ejkaba/prj/dronehacks-sim"))
sys.path.insert(0, str(DEPLOY_ROOT / "ego_planner" / "python"))
sys.path.insert(0, str(SIM_ROOT))

from ego_planner_py import EgoPlannerV2  # noqa: E402
from dronehacks import rules  # noqa: E402
from dronehacks.bank import BANK_LAUNCH  # noqa: E402
from dronehacks.env import CourseEnv, CourseEnvCfg  # noqa: E402
from dronehacks.mesh import CourseMesh  # noqa: E402
from dronehacks.video import FPS, PHASES, write_fpv  # noqa: E402


def rpy_to_quat_wxyz(roll: float, pitch: float, yaw: float) -> tuple[float, float, float, float]:
    cr, sr = math.cos(0.5 * roll), math.sin(0.5 * roll)
    cp, sp = math.cos(0.5 * pitch), math.sin(0.5 * pitch)
    cy, sy = math.cos(0.5 * yaw), math.sin(0.5 * yaw)
    qw = cy * cr * cp + sy * sr * sp
    qx = cy * sr * cp - sy * cr * sp
    qy = cy * cr * sp + sy * sr * cp
    qz = sy * cr * cp - cy * sr * sp
    return (qw, qx, qy, qz)


def run_episode(
    env: CourseEnv,
    planner: EgoPlannerV2,
    cell: int,
    record_trace: bool = False,
) -> dict:
    env.set_eval_queue(torch.tensor([[cell]], device=env.device))
    planner.reset(ready_for_sim_flight=True)

    if record_trace:
        env.plant.trace = []
        first = torch.cat([env.plant.pos_ned, env.plant.rpy_ned()], dim=-1).clone()
    else:
        first = None

    obs = env.get_observations()
    outcome = "timeout"
    points = 0.0
    phases = []
    step_times_ms = []
    max_cp_reached = 0

    for step_idx in range(env.max_episode_length):
        ph = int(env._phase[0]) if env._rp else 0
        pos_ned = env.plant.pos_ned[0].detach().cpu().numpy()
        vel_ned = env.plant.vel_ned[0].detach().cpu().numpy()
        rpy_ned = env.plant.rpy_ned()[0].detach().cpu().numpy()
        quat_wxyz = rpy_to_quat_wxyz(float(rpy_ned[0]), float(rpy_ned[1]), float(rpy_ned[2]))
        depth_newest = obs["policy"][0, 0].detach().cpu().numpy()

        t0 = time.perf_counter()
        action_np, info = planner.act_sim_single(
            pos_ned=pos_ned,
            vel_ned=vel_ned,
            quat_wxyz=np.array(quat_wxyz, dtype=np.float32),
            depth_norm=depth_newest,
            dt_s=0.10,
            raycast_stride=4,
        )
        step_times_ms.append((time.perf_counter() - t0) * 1e3)
        max_cp_reached = max(max_cp_reached, int(env._next_cp[0].item()))

        act_tensor = torch.from_numpy(action_np).to(env.device).unsqueeze(0)
        obs, _, dones, _ = env.step(act_tensor)
        phases.append(ph)

        if int(dones[0]):
            outcome = env.OUTCOMES[int(env.finished[-1][3, 0])]
            points = float(env.finished_points[-1][0])
            if outcome == "success":
                max_cp_reached = 5
            break

    duration_s = (step_idx + 1) * 0.10
    res = {
        "cell": cell,
        "outcome": outcome,
        "points": points,
        "duration_s": duration_s,
        "checkpoints": max_cp_reached,
        "mean_planner_ms": float(np.mean(step_times_ms)) if step_times_ms else 0.0,
        "p99_planner_ms": float(np.percentile(step_times_ms, 99)) if step_times_ms else 0.0,
    }
    if record_trace and first is not None and env.plant.trace is not None:
        res["trace"] = torch.cat([first] + env.plant.trace, dim=0).cpu().numpy()
        res["phases"] = phases
        env.plant.trace = None
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate C++ EGO-Planner-V2 in dronehacks-sim")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--perturb", type=float, default=0.0)
    ap.add_argument("--speed", type=float, default=3.35)
    ap.add_argument("--video", default="", help="Optional output MP4 path")
    ap.add_argument("--json-out", default="", help="Optional summary JSON output path")
    args = ap.parse_args()

    cfg = CourseEnvCfg(
        num_envs=1,
        device=args.device,
        seed=args.seed,
        assets_dir=str(SIM_ROOT / "assets"),
        perturb=args.perturb,
        perturb_variants=8 if args.perturb > 0 else 1,
        perturb_seed=args.seed,
    )
    env = CourseEnv(cfg)
    planner = EgoPlannerV2(
        arm_authorized=True,
        enable_return_leg=False,
        prefer_window=True,
        desired_speed_ms=args.speed,
    )

    launch_cells = np.nonzero(env._bank_kind.cpu().numpy() == BANK_LAUNCH)[0]
    cell = int(launch_cells[0])

    results = []
    first_trace_ep = None
    for ep in range(args.episodes):
        rec = bool(args.video and ep == 0)
        ep_res = run_episode(env, planner, cell, record_trace=rec)
        if rec:
            first_trace_ep = ep_res
        results.append(
            {k: v for k, v in ep_res.items() if k not in ("trace", "phases")}
        )
        print(
            f"[Episode {ep + 1}/{args.episodes}] outcome={ep_res['outcome']} "
            f"points={ep_res['points']:.2f}/60.0 time={ep_res['duration_s']:.1f}s "
            f"cp={ep_res['checkpoints']}/5 planner_ms={ep_res['mean_planner_ms']:.2f}ms"
        )

    succ_rate = float(np.mean([r["outcome"] == "success" for r in results]))
    mean_pts = float(np.mean([r["points"] for r in results]))
    mean_time = float(np.mean([r["duration_s"] for r in results]))
    mean_ms = float(np.mean([r["mean_planner_ms"] for r in results]))
    summary = {
        "perturb": args.perturb,
        "episodes": args.episodes,
        "success_rate": succ_rate,
        "mean_points": round(mean_pts, 2),
        "mean_duration_s": round(mean_time, 2),
        "mean_planner_ms": round(mean_ms, 3),
        "results": results,
    }
    print(
        f"=== Summary (perturb={args.perturb}): success={succ_rate*100:.1f}%, "
        f"mean_points={mean_pts:.2f}/60, mean_time={mean_time:.2f}s, "
        f"mean_planner_ms={mean_ms:.2f}ms ==="
    )

    if args.json_out:
        os.makedirs(os.path.dirname(os.path.abspath(args.json_out)), exist_ok=True)
        with open(args.json_out, "w") as f:
            json.dump(summary, f, indent=2)

    if args.video and first_trace_ep is not None and "trace" in first_trace_ep:
        tr = first_trace_ep["trace"]
        phases = first_trace_ep["phases"]
        outcome = first_trace_ep["outcome"]
        points = first_trace_ep["points"]
        ts = np.arange(len(tr)) * env.plant.dt
        t = np.arange(0.0, ts[-1], 1.0 / FPS)
        pos = np.stack([np.interp(t, ts, tr[:, k]) for k in range(3)], axis=1).astype(np.float32)
        rpy = np.stack([np.interp(t, ts, np.unwrap(tr[:, 3 + k])) for k in range(3)], axis=1).astype(np.float32)
        speed = np.linalg.norm(np.gradient(pos, 1.0 / FPS, axis=0), axis=1)
        course = CourseMesh.load(os.path.join(env.cfg.assets_dir, "course.npz"))
        mark, pad = env._mark[0].cpu().numpy(), env._pad_c[0].cpu().numpy()
        circles = [
            (pad[0], pad[1], rules.PAD_RADIUS_M, (80, 220, 120)),
            (mark[0], mark[1], rules.RETURN_RADIUS_M, (255, 210, 60)),
        ]
        step_of = np.minimum((t / 0.1).astype(int), len(phases) - 1)
        labels = [
            f"EGO-Planner-V2 ({PHASES[phases[k]]})"
            + (f" -> {outcome} ({points:.1f} pts)" if k == len(phases) - 1 else "")
            for k in step_of
        ]
        write_fpv(args.video, course, t, pos, rpy, speed, args.device, circles=circles, labels=labels)
        print(f"Wrote flight video to {args.video}")


if __name__ == "__main__":
    main()
