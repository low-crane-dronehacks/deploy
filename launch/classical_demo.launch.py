"""Fused Classical Autonomy (A* + MPPI) + iSAM2 CAD-Prior Generalization Demo.

Joins both halves of the C++ ROS 2 stack against the real physical course scan (.ply) with
zero synthetic/arbitrary obstacles:
  - sim_loopback_node (C++ QuadPlant): executes /control/cmd_vel closed-loop -> /truth/odometry
  - classical_scan_bridge.py: renders live 640x400 OAK depth from the .ply scan + VIO-drifted
    PX4 /fmu/out/* telemetry, and bridges iSAM2's smoothed odometry onto /odometry/local
  - isam2_depth_node (C++ GTSAM iSAM2 + PriorEsdfMap): corrects VIO drift against CAD prior +
    depth ICP, carves stale CAD walls, populates shifted real-world walls, and publishes
    /isam2_depth/obstacles for MPPI
  - astar_planner_node (C++): plans initial global trajectory on the CAD prior, and replans
    around blocked corridors when MPPI detects real scan walls blocking the CAD route
  - path_transformer_node (C++): extracts timed local reference horizon in odom_ned
  - mppi_controller_node (C++): 1024-rollout QuadPlant MPPI planning on the iSAM2 local grid,
    requesting /plan/replan_blocked_corridor when the natural CAD trajectory is blocked
  - control_mux_node (C++): watchdogs & mode mux (strictly arm:=false)

Usage:
  python3 ../rerun/rerun_classical_demo.py --launch
  PYTHONNOUSERSITE=1 ros2 launch launch/classical_demo.launch.py
"""
from __future__ import annotations

import json
import os
import shlex
from pathlib import Path

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    EmitEvent,
    ExecuteProcess,
    LogInfo,
    OpaqueFunction,
    RegisterEventHandler,
    TimerAction,
)
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.substitutions import LaunchConfiguration

DEPLOY = Path(__file__).resolve().parents[1]
MAP_FRAME = "map_ned"
ODOM_FRAME = "odom_ned"
BASE_FRAME = "base_link_frd"
TRUTH_BASE_FRAME = "truth_base_link_frd"
LAUNCH_MARK_NED = (0.9144, -2.5146, -0.78)


def _course_goals(prior_json: Path, prefer_window: bool):
    with open(prior_json) as f:
        meta = json.load(f)
    by_cp: dict[int, list[dict]] = {}
    for g in sorted(meta["gates"], key=lambda g: g["index"]):
        by_cp.setdefault(g["checkpoint"], []).append(g)
    flat, names = [], []
    for cp in sorted(by_cp):
        options = by_cp[cp]
        pick = next((g for g in options if g["is_window"] == prefer_window), options[0])
        c = pick["center_ned"]
        flat += [c[0], c[1], c[2], *pick["A_ned"], *pick["B_ned"], *pick["normal_ned"]]
        names.append(pick["name"])
    pad = meta["landmarks"]["pad_center_ned"]
    flat += [pad[0], pad[1], -0.80, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    names.append("pad")
    return flat, names, meta["arena"]


def _node_cmd(pkg_name: str, exe_name: str) -> list[str]:
    candidates = [
        DEPLOY / "install" / pkg_name / "lib" / pkg_name / exe_name,
        DEPLOY / "install" / "isam2_depth" / "lib" / "isam2_depth" / exe_name,
    ]
    for c in candidates:
        if c.exists():
            return [str(c)]
    return ["ros2", "run", pkg_name, exe_name]


def _setup(context):
    arm = LaunchConfiguration("arm").perform(context)
    if arm.lower() in ("1", "true", "yes"):
        raise RuntimeError("classical_demo is a simulation/scan demo; arm:=true is strictly forbidden")

    ply = LaunchConfiguration("ply").perform(context)
    if not ply:
        ply = str(DEPLOY / "dronehacks-course-scan-2026-11-06.ply")
    cad_prior = LaunchConfiguration("cad_prior").perform(context)
    gt_map_path = LaunchConfiguration("groundtruth_map_path").perform(context)
    if not os.path.isabs(gt_map_path):
        gt_map_path = str(DEPLOY / gt_map_path)

    cruise_speed = LaunchConfiguration("cruise_speed_m_s").perform(context)
    mppi_samples = LaunchConfiguration("mppi_samples").perform(context)
    prefer_window = LaunchConfiguration("prefer_window").perform(context).lower() in ("1", "true", "yes")
    start_delay = float(LaunchConfiguration("start_delay_s").perform(context))
    rviz = LaunchConfiguration("rviz").perform(context).lower() in ("1", "true", "yes")
    auto_shutdown = LaunchConfiguration("auto_shutdown").perform(context).lower() in ("1", "true", "yes")
    pad_hold_s = LaunchConfiguration("pad_hold_s").perform(context)
    max_duration_s = LaunchConfiguration("max_duration_s").perform(context)
    bridge_extra = shlex.split(LaunchConfiguration("bridge_args").perform(context))

    frames = ["-p", f"map_frame:={MAP_FRAME}", "-p", f"odom_frame:={ODOM_FRAME}"]
    flat, names, arena = _course_goals(DEPLOY / "config" / "arena_prior.json", prefer_window)

    actions = [
        LogInfo(
            msg=f"classical_demo: speed={cruise_speed} mppi_samples={mppi_samples} "
                f"prefer_window={prefer_window} ply={ply} arm=false"
        ),
        # 1. Closed-loop C++ QuadPlant truth simulator
        ExecuteProcess(
            cmd=_node_cmd("pkg_maze_planning", "sim_loopback_node")
            + [
                "--ros-args",
                "-p", "use_sim_time:=false",
                "-p", f"map_frame:={MAP_FRAME}",
                "-p", f"odom_frame:={MAP_FRAME}",
                "-p", f"base_frame:={TRUTH_BASE_FRAME}",
                "-p", "odom_topic:=/truth/odometry",
                "-p", "publish_map_odom_tf:=false",
                "-p", f"start_x:={LAUNCH_MARK_NED[0]}",
                "-p", f"start_y:={LAUNCH_MARK_NED[1]}",
                "-p", f"start_z:={LAUNCH_MARK_NED[2]}",
                "-p", "start_yaw:=0.0",
            ],
            cwd=str(DEPLOY),
            output="screen",
        ),
        # 2. C++ iSAM2 Depth Factor Graph + Online CAD Prior ESDF Adaptation
        ExecuteProcess(
            cmd=_node_cmd("isam2_depth", "isam2_depth_node")
            + [
                "--ros-args",
                "-p", "use_sim_time:=false",
                "-p", f"world_frame:={ODOM_FRAME}",
                "-p", "body_frame:=isam2_body_frd",
                "-p", "base_frame:=base_link",
                "-p", "depth_topic:=/camera/depth/image_raw",
                "-p", "depth_info_topic:=/camera/depth/camera_info",
                "-p", f"cad_prior_path:={cad_prior}",
                "-p", "grid_resolution:=0.08",
                "-p", "max_odom_gap:=0.30",
                "-p", "kf_max_interval:=0.35",
                "-p", "prior_pixel_stride:=4",
                "-p", "prior_l_free:=-0.45",
            ],
            cwd=str(DEPLOY),
            output="screen",
        ),
    ]

    # 3. Scan Depth Renderer + Drifted PX4 Telemetry + iSAM2->LocalOdom Bridge
    bridge = ExecuteProcess(
        cmd=[
            "/usr/bin/python3",
            str(DEPLOY / "scripts" / "classical_scan_bridge.py"),
            ply,
            "--pad_hold_s", str(pad_hold_s),
            "--max_duration_s", str(max_duration_s),
            *bridge_extra,
        ],
        cwd=str(DEPLOY),
        output="screen",
        additional_env={"PYTHONNOUSERSITE": "1"},
    )
    actions.append(bridge)
    if auto_shutdown and not rviz:
        actions.append(RegisterEventHandler(OnProcessExit(target_action=bridge, on_exit=[EmitEvent(event=Shutdown())])))

    # 4. C++ A* Global Planner (initializes on CAD prior; replans on blocked corridor using iSAM2 adapted map)
    astar = ExecuteProcess(
        cmd=_node_cmd("pkg_maze_planning", "astar_planner_node")
        + [
            "--ros-args",
            "-p", "use_sim_time:=false",
            "-p", f"groundtruth_map_path:={gt_map_path}",
            "-p", "isam2_priors_topic:=/isam2_depth/priors",
            "-p", "isam2_new_obstacles_topic:=/isam2_depth/new_updated_obstacles",
            "-p", "isam2_voxel_resolution_m:=0.08",
            "-p", "block_radius_m:=0.38",
            "-p", "replan_min_interval_s:=1.0",
            "-p", f"cruise_speed_m_s:={cruise_speed}",
            "-p", f"arena_half_m:={arena['arena_half_m']}",
            "-p", f"cp_margin_m:={arena['cp_margin_m']}",
            "-p", "clearance_m:=0.26",
            "-p", "clearance_fallback_ratio:=0.78",
            "-p", "snap_radius_m:=1.50",
            "-p", "goals:=[" + ",".join(f"{v:.6f}" for v in flat) + "]",
            "-p", "goal_names:=[" + ",".join(f"'{n}'" for n in names) + "]",
        ]
        + frames,
        cwd=str(DEPLOY),
        output="screen",
    )
    actions.append(TimerAction(period=start_delay, actions=[astar]) if start_delay > 0 else astar)

    # 5. C++ Path Transformer Node
    actions.append(
        ExecuteProcess(
            cmd=_node_cmd("pkg_maze_planning", "path_transformer_node")
            + ["--ros-args", "-p", "use_sim_time:=false", "-p", f"cruise_speed_m_s:={cruise_speed}"]
            + frames,
            cwd=str(DEPLOY),
            output="screen",
        )
    )

    # 6. C++ MPPI Controller Node (plans strictly on iSAM2 local grid; zero synthetic obstacles)
    actions.append(
        ExecuteProcess(
            cmd=_node_cmd("pkg_maze_planning", "mppi_controller_node")
            + [
                "--ros-args",
                "-p", "use_sim_time:=false",
                "-p", f"groundtruth_map_path:={gt_map_path}",
                "-p", f"mppi_num_samples:={mppi_samples}",
                "-p", f"cruise_speed_m_s:={cruise_speed}",
                "-p", "inject_demo_pillar:=false",
                "-p", "obstacle_cloud_topic:=''",
                "-p", "isam2_obstacles_topic:=/isam2_depth/obstacles",
                "-p", "isam2_grid_resolution:=0.08",
                "-p", "isam2_grid_timeout_s:=12.0",
                "-p", "trust_isam2_prior_grid:=true",
                "-p", "mppi_safe_distance_m:=0.38",
            ]
            + frames,
            cwd=str(DEPLOY),
            output="screen",
        )
    )

    # 7. C++ Control Mode Multiplexer Node (arm:=false)
    actions.append(
        ExecuteProcess(
            cmd=_node_cmd("pkg_maze_control", "control_mux_node")
            + ["--ros-args", "-p", "use_sim_time:=false", "-p", "mode:=classical", "-p", "arm:=false"],
            cwd=str(DEPLOY),
            output="screen",
        )
    )

    if rviz:
        actions.append(
            ExecuteProcess(
                cmd=["rviz2", "-d", str(DEPLOY / "config" / "maze_rviz.rviz"), "--ros-args", "-p", "use_sim_time:=false"],
                cwd=str(DEPLOY),
                output="screen",
            )
        )

    return actions


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument("ply", default_value=str(DEPLOY / "dronehacks-course-scan-2026-11-06.ply")),
            DeclareLaunchArgument("cad_prior", default_value=str(DEPLOY / "config" / "cad_prior_esdf.bin")),
            DeclareLaunchArgument("groundtruth_map_path", default_value="config/course_groundtruth.bin"),
            DeclareLaunchArgument("arm", default_value="false"),
            DeclareLaunchArgument("mppi_samples", default_value="1024"),
            DeclareLaunchArgument("cruise_speed_m_s", default_value="0.95"),
            DeclareLaunchArgument("prefer_window", default_value="true"),
            DeclareLaunchArgument("start_delay_s", default_value="2.5"),
            DeclareLaunchArgument("pad_hold_s", default_value="2.5"),
            DeclareLaunchArgument("max_duration_s", default_value="95.0"),
            DeclareLaunchArgument("auto_shutdown", default_value="true"),
            DeclareLaunchArgument("rviz", default_value="false"),
            DeclareLaunchArgument("bridge_args", default_value=""),
            OpaqueFunction(function=_setup),
        ]
    )
