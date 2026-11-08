"""Launch the Dual-Track Autonomous Quadrotor Maze Navigation System (system_architecture_plan.md).

Decoupled ROS 2 nodes (Section 2.2, 2.3, WP-3, WP-4); each talks to the others only through topics / TF:
  - odometry source: onboard/adapters/px4_local_odom_ned.py (PX4 estimate -> /odometry/local, NED) on
    sitl / flyx / drone, or sim_loopback_node for the standalone RViz demo
  - map -> odom alignment: static launch-mark offset (map_odom_source:=static), or published by an
    estimator such as iSAM2 / the merged stack (map_odom_source:=external)
  - isam2_depth_node (isam2:=true; needs GTSAM + px4_msgs): ~/obstacles -> MPPI's local distance grid
  - global planner (planner:=astar | rrt_star): astar_planner_node (grid A* through the arena_prior.json
    checkpoints) or rrt_star_planner_node -> /plan/global_rrt (latched, timed, on each new plan)
  - path_transformer_node: /plan/global_rrt + TF map->odom -> /plan/local_reference (20 Hz),
    /plan/rl_lookaheads (50 Hz)
  - mppi_controller_node: /plan/local_reference + obstacle clouds -> /control/trajectory_setpoint (50 Hz),
    /costmap/local_voxels, /viz/mppi_candidates (MarkerArray)
  - control_mux_node: classical MPPI vs RL policy -> /fmu/in/* (only with arm:=true)

Frames (all NED, like PX4): map_ned (arena / prior map) -> odom_ned (PX4 local origin) -> base_link_frd.
They are deliberately not the ROS ENU frames (odom, base_link) that onboard/adapters/px4_odom.py publishes.

Usage:
  ros2 launch launch/maze_autonomy.launch.py sim_loopback:=true inject_demo_pillar:=true rviz:=true arm:=false
  ros2 launch launch/maze_autonomy.launch.py hardware:=sitl arm:=false
  ros2 launch launch/maze_autonomy.launch.py hardware:=sitl isam2:=true arm:=false
Arming (arm:=true) needs team approval for that specific flight (AGENTS.md).
"""
import json
import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, LogInfo, OpaqueFunction
from launch.substitutions import LaunchConfiguration

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MAP_FRAME = "map_ned"
ODOM_FRAME = "odom_ned"
BASE_FRAME = "base_link_frd"
ISAM2_WORLD_FRAME = "isam2_world_ned"   # isam2_depth_node's world: PX4's local NED frame (= odom_ned)

PROFILES = {
    "sitl": dict(
        use_sim_time=True,
        depth_topic="/camera/depth/image_raw",
        depth_info_topic="/camera/depth/camera_info",
    ),
    "flyx": dict(
        use_sim_time=True,
        depth_topic="/camera/depth/image_raw",
        depth_info_topic="/camera/depth/camera_info",
    ),
    "drone": dict(
        use_sim_time=False,
        depth_topic="/oak/stereo/image_raw",
        depth_info_topic="/oak/stereo/camera_info",
    ),
}

# Launch mark in the map (arena) frame, NED. PX4's local origin is where the vehicle sits at boot, so with a
# launch from the launch mark, map_ned -> odom_ned is this translation (and the heading offset, which is 0
# when PX4 has a true-north heading as in SITL).
LAUNCH_MARK_NED = (0.9144, -2.5146, 0.0)


def _course_goals(prior_json, prefer_window):
    """arena_prior.json checkpoints in order, one gate per checkpoint (the window or Gate B for the shared
    one, +5 for the window), then the pad: astar_planner_node's flat `goals` (9 doubles each) + names."""
    with open(prior_json) as f:
        meta = json.load(f)
    by_cp = {}
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
    flat += [pad[0], pad[1], -0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    names.append("pad")
    return flat, names, meta["arena"]


def _static_tf(parent, child, xyz=(0.0, 0.0, 0.0), yaw=0.0, sim_time="false"):
    x, y, z = xyz
    return ExecuteProcess(
        cmd=["ros2", "run", "tf2_ros", "static_transform_publisher",
             "--x", str(x), "--y", str(y), "--z", str(z), "--yaw", str(yaw),
             "--frame-id", parent, "--child-frame-id", child,
             "--ros-args", "-p", f"use_sim_time:={sim_time}"],
        output="screen",
    )


def _setup(context):
    hw = LaunchConfiguration("hardware").perform(context)
    if hw not in PROFILES:
        raise RuntimeError(f"hardware must be one of {sorted(PROFILES)}, got {hw!r}")
    p = PROFILES[hw]
    arm = LaunchConfiguration("arm").perform(context)
    mode = LaunchConfiguration("mode").perform(context)
    mppi_samples = LaunchConfiguration("mppi_samples").perform(context)
    cruise_speed = LaunchConfiguration("cruise_speed_m_s").perform(context)
    sim_loopback = LaunchConfiguration("sim_loopback").perform(context).lower() == "true"
    inject_demo_pillar = LaunchConfiguration("inject_demo_pillar").perform(context)
    rviz = LaunchConfiguration("rviz").perform(context)
    isam2 = LaunchConfiguration("isam2").perform(context).lower() == "true"
    map_odom_source = LaunchConfiguration("map_odom_source").perform(context)
    map_odom_yaw = LaunchConfiguration("map_odom_yaw").perform(context)
    gt_map_path = LaunchConfiguration("groundtruth_map_path").perform(context)
    if not os.path.isabs(gt_map_path):
        gt_map_path = os.path.join(ROOT, gt_map_path)
    if map_odom_source not in ("static", "external"):
        raise RuntimeError(f"map_odom_source must be static | external, got {map_odom_source!r}")
    if sim_loopback and arm.lower() == "true":
        raise RuntimeError("sim_loopback:=true is a standalone demo; it cannot be combined with arm:=true")

    actions = [
        LogInfo(
            msg=f"maze_autonomy: hardware={hw} mode={mode} arm={arm} mppi_samples={mppi_samples} "
                f"speed={cruise_speed} sim_loopback={sim_loopback} isam2={isam2} map_odom={map_odom_source}"
        )
    ]

    if hw == "drone" and not sim_loopback:
        dev = LaunchConfiguration("fc_device").perform(context)
        actions += [
            ExecuteProcess(
                cmd=["MicroXRCEAgent", "serial", "--dev", dev, "-b", "921600"], output="screen"
            ),
            ExecuteProcess(
                cmd=[
                    "ros2",
                    "launch",
                    "depthai_ros_driver",
                    "camera.launch.py",
                    f'params_file:={os.path.join(ROOT, "config", "oak.yaml")}',
                ],
                output="screen",
            ),
            ExecuteProcess(
                cmd=["python3", os.path.join(ROOT, "onboard", "services.py"), "start"],
                output="screen",
            ),
        ]

    sim_time_str = "false" if sim_loopback else str(p["use_sim_time"]).lower()
    frames = ["-p", f"map_frame:={MAP_FRAME}", "-p", f"odom_frame:={ODOM_FRAME}"]

    def _node_cmd(pkg_name: str, exe_name: str) -> list[str]:
        candidates = [
            os.path.join(ROOT, "install", pkg_name, "lib", pkg_name, exe_name),
            os.path.join(ROOT, "install", "isam2_depth", "lib", "isam2_depth", exe_name),
        ]
        for c in candidates:
            if os.path.exists(c):
                return [c]
        return ["ros2", "run", pkg_name, exe_name]

    # 0. Odometry + TF. Exactly one source of /odometry/local and of each TF edge.
    if sim_loopback:
        # Publishes /odometry/local, map_ned -> odom_ned (identity) and odom_ned -> base_link_frd.
        actions.append(ExecuteProcess(
            cmd=_node_cmd("pkg_maze_planning", "sim_loopback_node")
            + ["--ros-args", "-p", "use_sim_time:=false", "-p", f"base_frame:={BASE_FRAME}"] + frames,
            cwd=ROOT, output="screen"))
    else:
        actions.append(ExecuteProcess(
            cmd=["python3", os.path.join(ROOT, "onboard", "adapters", "px4_local_odom_ned.py"),
                 "--ros-args", "-p", f"use_sim_time:={sim_time_str}",
                 "-p", f"odom_frame:={ODOM_FRAME}", "-p", f"base_frame:={BASE_FRAME}"],
            cwd=ROOT, output="screen"))
        if map_odom_source == "static":
            actions.append(_static_tf(MAP_FRAME, ODOM_FRAME, LAUNCH_MARK_NED, float(map_odom_yaw), sim_time_str))
        actions.append(_static_tf(ODOM_FRAME, ISAM2_WORLD_FRAME, sim_time=sim_time_str))

    # 0b. Optional iSAM2 depth graph (planning-only estimator; never fed back to PX4).
    if isam2 and not sim_loopback:
        actions.append(ExecuteProcess(
            cmd=_node_cmd("isam2_depth", "isam2_depth_node")
            + ["--ros-args", "-p", f"use_sim_time:={sim_time_str}",
               "-p", f"depth_topic:={p['depth_topic']}", "-p", f"depth_info_topic:={p['depth_info_topic']}",
               "-p", f"world_frame:={ISAM2_WORLD_FRAME}"],
            cwd=ROOT, output="screen"))

    # 1. Global planner (/plan/global_rrt, latched + /viz/groundtruth_map_cloud): A* (default) or RRT*.
    planner = LaunchConfiguration("planner").perform(context)
    if planner not in ("astar", "rrt_star"):
        raise RuntimeError(f"planner must be astar | rrt_star, got {planner!r}")
    if planner == "astar":
        prefer_window = LaunchConfiguration("prefer_window").perform(context).lower() == "true"
        flat, names, arena = _course_goals(os.path.join(ROOT, "config", "arena_prior.json"), prefer_window)
        actions.append(ExecuteProcess(
            cmd=_node_cmd("pkg_maze_planning", "astar_planner_node")
            + ["--ros-args", "-p", f"use_sim_time:={sim_time_str}",
               "-p", f"groundtruth_map_path:={gt_map_path}",
               "-p", f"cruise_speed_m_s:={cruise_speed}",
               "-p", f"arena_half_m:={arena['arena_half_m']}",
               "-p", f"cp_margin_m:={arena['cp_margin_m']}",
               "-p", "goals:=[" + ",".join(f"{v:.6f}" for v in flat) + "]",
               "-p", "goal_names:=[" + ",".join(f"'{n}'" for n in names) + "]"]
            + frames,
            cwd=ROOT, output="screen"))
    if planner == "rrt_star":
        actions.append(
            ExecuteProcess(
                cmd=_node_cmd("pkg_maze_planning", "rrt_star_planner_node")
                + [
                    "--ros-args",
                    "-p",
                    f"use_sim_time:={sim_time_str}",
                    "-p",
                    f"groundtruth_map_path:={gt_map_path}",
                    "-p",
                    f"cruise_speed_m_s:={cruise_speed}",
                ]
                + frames,
                cwd=ROOT,
                output="screen",
            )
        )

    # 2. Path Transformer Node (map -> odom -> /plan/local_reference @ 20 Hz & /plan/rl_lookaheads @ 50 Hz)
    actions.append(
        ExecuteProcess(
            cmd=_node_cmd("pkg_maze_planning", "path_transformer_node")
            + [
                "--ros-args",
                "-p",
                f"use_sim_time:={sim_time_str}",
                "-p",
                f"cruise_speed_m_s:={cruise_speed}",
            ]
            + frames,
            cwd=ROOT,
            output="screen",
        )
    )

    # 3. MPPI Controller & STVL VoxelCostmap Node
    actions.append(
        ExecuteProcess(
            cmd=_node_cmd("pkg_maze_planning", "mppi_controller_node")
            + [
                "--ros-args",
                "-p",
                f"use_sim_time:={sim_time_str}",
                "-p",
                f"groundtruth_map_path:={gt_map_path}",
                "-p",
                f"mppi_num_samples:={mppi_samples}",
                "-p",
                f"cruise_speed_m_s:={cruise_speed}",
                "-p",
                f"inject_demo_pillar:={inject_demo_pillar.lower()}",
                "-p",
                f"isam2_obstacles_topic:={'/isam2_depth_node/obstacles' if isam2 else ''}",
            ]
            + frames,
            cwd=ROOT,
            output="screen",
        )
    )

    # 4. Control Mode Multiplexer Node (Classical MPPI vs. RL Policy + Watchdogs)
    actions.append(
        ExecuteProcess(
            cmd=_node_cmd("pkg_maze_control", "control_mux_node")
            + [
                "--ros-args",
                "-p",
                f"use_sim_time:={sim_time_str}",
                "-p",
                f"mode:={mode}",
                "-p",
                f"arm:={arm}",
            ],
            cwd=ROOT,
            output="screen",
        )
    )

    # 5. Optional RViz2 realtime 3D visualizer
    if rviz.lower() == "true":
        rviz_cfg = os.path.join(ROOT, "config", "maze_rviz.rviz")
        actions.append(
            ExecuteProcess(
                cmd=["rviz2", "-d", rviz_cfg],
                cwd=ROOT,
                output="screen",
            )
        )

    return actions


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument("hardware", default_value="sitl", description="drone | sitl | flyx"),
            DeclareLaunchArgument("mode", default_value="classical", description="classical | rl_policy"),
            DeclareLaunchArgument("arm", default_value="false", description="true = arm and publish to PX4"),
            DeclareLaunchArgument("mppi_samples", default_value="2048", description="Number of MPPI rollouts"),
            DeclareLaunchArgument("cruise_speed_m_s", default_value="2.60", description="Nominal cruise speed m/s"),
            DeclareLaunchArgument("sim_loopback", default_value="false", description="true = standalone closed-loop RViz demo without Gazebo"),
            DeclareLaunchArgument("inject_demo_pillar", default_value="false", description="true = inject unmodeled vertical pole at [1.75, -2.51] for MPPI evasion demo"),
            DeclareLaunchArgument("rviz", default_value="false", description="true = launch rviz2 with config/maze_rviz.rviz"),
            DeclareLaunchArgument("planner", default_value="astar", description="astar | rrt_star (global planner)"),
            DeclareLaunchArgument("prefer_window", default_value="true",
                                  description="true = window for the shared checkpoint (+5), false = Gate B"),
            DeclareLaunchArgument("isam2", default_value="false", description="true = run isam2_depth_node and feed its obstacles to MPPI"),
            DeclareLaunchArgument("map_odom_source", default_value="static",
                                  description="static = launch-mark offset TF map_ned -> odom_ned; "
                                              "external = an estimator (iSAM2 / merged stack) publishes it"),
            DeclareLaunchArgument("map_odom_yaw", default_value="0.0",
                                  description="heading of PX4's local frame in the map frame (rad), static source only"),
            DeclareLaunchArgument("groundtruth_map_path", default_value="config/course_groundtruth.bin"),
            DeclareLaunchArgument("fc_device", default_value="/dev/ttyTHS1"),
            OpaqueFunction(function=_setup),
        ]
    )
