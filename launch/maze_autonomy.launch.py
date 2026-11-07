"""Launch the Dual-Track Autonomous Quadrotor Maze Navigation System (system_architecture_plan.md).

Launches:
  - isam2_depth_node (when GTSAM is built): TF map -> odom + local DistanceGrid
  - maze_planning_node: GroundTruthMap + Informed RRT* (/plan/global_rrt) + PathTransformer
    (/plan/local_reference, /plan/rl_lookaheads) + STVL Costmap (/costmap/local_voxels) +
    2048-rollout MPPI Controller (/control/trajectory_setpoint)
  - control_mux_node: Switches Classical MPPI vs. RL Policy (/fmu/in/trajectory_setpoint or
    /fmu/in/vehicle_rates_setpoint) with safety fallback interlocks

Usage:
  ros2 launch launch/maze_autonomy.launch.py hardware:=sitl arm:=false
  ros2 launch launch/maze_autonomy.launch.py hardware:=sitl mode:=classical arm:=true
  ros2 launch launch/maze_autonomy.launch.py hardware:=drone mode:=classical arm:=true
"""
import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, LogInfo, OpaqueFunction
from launch.substitutions import LaunchConfiguration

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

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


def _setup(context):
    hw = LaunchConfiguration("hardware").perform(context)
    if hw not in PROFILES:
        raise RuntimeError(f"hardware must be one of {sorted(PROFILES)}, got {hw!r}")
    p = PROFILES[hw]
    arm = LaunchConfiguration("arm").perform(context)
    mode = LaunchConfiguration("mode").perform(context)
    mppi_samples = LaunchConfiguration("mppi_samples").perform(context)
    cruise_speed = LaunchConfiguration("cruise_speed_m_s").perform(context)
    gt_map_path = LaunchConfiguration("groundtruth_map_path").perform(context)
    if not os.path.isabs(gt_map_path):
        gt_map_path = os.path.join(ROOT, gt_map_path)

    actions = [
        LogInfo(
            msg=f"maze_autonomy: hardware={hw} mode={mode} arm={arm} mppi_samples={mppi_samples} speed={cruise_speed} gt_map={gt_map_path}"
        )
    ]

    if hw == "drone":
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

    sim_time_str = str(p["use_sim_time"]).lower()

    # 1. Maze Planning Node (RRT* + PathTransformer + VoxelCostmap + 2048-rollout MPPI)
    plan_bin = os.path.join(ROOT, "install", "isam2_depth", "lib", "isam2_depth", "maze_planning_node")
    plan_cmd = [plan_bin] if os.path.exists(plan_bin) else ["ros2", "run", "isam2_depth", "maze_planning_node"]
    actions.append(
        ExecuteProcess(
            cmd=plan_cmd
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
            ],
            cwd=ROOT,
            output="screen",
        )
    )

    # 2. Control Mode Multiplexer Node (Classical MPPI vs. RL Policy + Watchdogs)
    mux_bin = os.path.join(ROOT, "install", "isam2_depth", "lib", "isam2_depth", "control_mux_node")
    mux_cmd = [mux_bin] if os.path.exists(mux_bin) else ["ros2", "run", "isam2_depth", "control_mux_node"]
    actions.append(
        ExecuteProcess(
            cmd=mux_cmd
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

    return actions


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument("hardware", default_value="sitl", description="drone | sitl | flyx"),
            DeclareLaunchArgument("mode", default_value="classical", description="classical | rl_policy"),
            DeclareLaunchArgument("arm", default_value="false", description="true = arm and publish to PX4"),
            DeclareLaunchArgument("mppi_samples", default_value="2048", description="Number of MPPI rollouts"),
            DeclareLaunchArgument("cruise_speed_m_s", default_value="2.60", description="Nominal cruise speed m/s"),
            DeclareLaunchArgument("groundtruth_map_path", default_value="config/course_groundtruth.bin"),
            DeclareLaunchArgument("fc_device", default_value="/dev/ttyTHS1"),
            OpaqueFunction(function=_setup),
        ]
    )
