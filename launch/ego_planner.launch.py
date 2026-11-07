"""Launch C++ ROS 2 EGO-Planner-V2 across all platforms (sitl, flyx, drone).

    ros2 launch launch/ego_planner.launch.py hardware:=sitl arm:=true
    ros2 launch launch/ego_planner.launch.py hardware:=flyx arm:=true
    ros2 launch launch/ego_planner.launch.py hardware:=drone arm:=true
"""
import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, LogInfo, OpaqueFunction
from launch.substitutions import LaunchConfiguration

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PROFILES = {
    "sitl": dict(
        use_sim_time=True,
        range_source="altitude",
        depth_topic="/camera/depth/image_raw",
        depth_info_topic="/camera/depth/camera_info",
    ),
    "flyx": dict(
        use_sim_time=True,
        range_source="sim",
        depth_topic="/camera/depth/image_raw",
        depth_info_topic="/camera/depth/camera_info",
    ),
    "drone": dict(
        use_sim_time=False,
        range_source="distance_sensor",
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
    enable_return = LaunchConfiguration("enable_return_leg").perform(context)
    desired_speed = LaunchConfiguration("desired_speed_ms").perform(context)
    planner = LaunchConfiguration("planner").perform(context)
    controller = LaunchConfiguration("controller").perform(context)
    prior_bin = LaunchConfiguration("prior_bin_path").perform(context)
    if not os.path.isabs(prior_bin):
        prior_bin = os.path.join(ROOT, prior_bin)

    actions = [
        LogInfo(
            msg=f"ego_planner: hardware={hw} arm={arm} planner={planner} controller={controller} speed={desired_speed} return={enable_return} prior={prior_bin}"
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

    # Prefer compiled C++ ROS 2 node binary if installed, else run via ros2 run
    node_bin = os.path.join(ROOT, "install", "ego_planner", "lib", "ego_planner", "ego_planner_node")
    cmd_prefix = [node_bin] if os.path.exists(node_bin) else ["ros2", "run", "ego_planner", "ego_planner_node"]

    args = cmd_prefix + [
        "--ros-args",
        "-p",
        f'use_sim_time:={str(p["use_sim_time"]).lower()}',
        "-p",
        f'range_source:={p["range_source"]}',
        "-p",
        f'depth_topic:={p["depth_topic"]}',
        "-p",
        f'depth_info_topic:={p["depth_info_topic"]}',
        "-p",
        f"prior_bin_path:={prior_bin}",
        "-p",
        f"arm:={arm}",
        "-p",
        f"planner:={planner}",
        "-p",
        f"controller:={controller}",
        "-p",
        f"desired_speed_ms:={desired_speed}",
        "-p",
        f"enable_return_leg:={enable_return}",
    ]
    actions.append(ExecuteProcess(cmd=args, cwd=ROOT, output="screen"))
    return actions


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument("hardware", default_value="sitl", description="drone | sitl | flyx"),
            DeclareLaunchArgument("arm", default_value="false", description="true = arm and fly"),
            DeclareLaunchArgument("planner", default_value="bspline", description="bspline | rrt_star"),
            DeclareLaunchArgument("controller", default_value="ego", description="ego | mppi"),
            DeclareLaunchArgument("desired_speed_ms", default_value="2.25", description="Cruise speed in m/s"),
            DeclareLaunchArgument(
                "enable_return_leg", default_value="false", description="true = fly return leg after pad"
            ),
            DeclareLaunchArgument("prior_bin_path", default_value="config/arena_prior.bin"),
            DeclareLaunchArgument("fc_device", default_value="/dev/ttyTHS1"),
            OpaqueFunction(function=_setup),
        ]
    )
