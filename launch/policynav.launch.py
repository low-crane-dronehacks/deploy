"""One launch for every platform: the same policy node + mission logic, only the hardware layer differs.

    ros2 launch launch/policynav.launch.py hardware:=sitl                 # our PX4 SITL rig (sitl/), dry run
    ros2 launch launch/policynav.launch.py hardware:=flyx                 # the organizers' cloud simulator
    ros2 launch launch/policynav.launch.py hardware:=drone arm:=true      # the aircraft (UNTESTED: see README)
    ... policy:=policies/<name>.npz

arm defaults to false: the node runs the policy on live data and reports what it would command, publishes nothing.
"""
import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, LogInfo, OpaqueFunction
from launch.substitutions import LaunchConfiguration

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PROFILES = {
    # sitl: sitl/run_rig.sh up provides PX4 SITL + uXRCE (UDP) + depth relay + truth-based external vision
    'sitl': dict(use_sim_time=True, range_source='altitude', depth_topic='/camera/depth/image_raw',
                 depth_info_topic='/camera/depth/camera_info'),
    # flyx: the organizers' simulator provides PX4, cuVSLAM (onboard/) and the OAK depth topics already
    'flyx': dict(use_sim_time=True, range_source='sim', depth_topic='/camera/depth/image_raw',
                 depth_info_topic='/camera/depth/camera_info'),
    # drone: OAK-D S2 via depthai_ros_driver, VL53L1X via PX4 distance_sensor, uXRCE over the Jetson UART
    'drone': dict(use_sim_time=False, range_source='distance_sensor', depth_topic='/oak/stereo/image_raw',
                  depth_info_topic='/oak/stereo/camera_info'),
}


def _setup(context):
    hw = LaunchConfiguration('hardware').perform(context)
    if hw not in PROFILES:
        raise RuntimeError(f'hardware must be one of {sorted(PROFILES)}, got {hw!r}')
    p = PROFILES[hw]
    arm = LaunchConfiguration('arm').perform(context)
    policy = LaunchConfiguration('policy').perform(context)
    actions = [LogInfo(msg=f'policynav: hardware={hw} arm={arm} policy={policy}')]
    if hw == 'drone':
        dev = LaunchConfiguration('fc_device').perform(context)
        actions += [
            ExecuteProcess(cmd=['MicroXRCEAgent', 'serial', '--dev', dev, '-b', '921600'], output='screen'),
            # OAK-D S2 depth (depthai_ros_driver); set the stereo preset / resolution in config/oak.yaml
            ExecuteProcess(cmd=['ros2', 'launch', 'depthai_ros_driver', 'camera.launch.py',
                                f'params_file:={os.path.join(ROOT, "config", "oak.yaml")}'], output='screen'),
            # onboard VIO (cuVSLAM -> external vision -> PX4) and /estimator/healthy
            ExecuteProcess(cmd=['python3', os.path.join(ROOT, 'onboard', 'services.py'), 'start'], output='screen'),
        ]
    args = ['python3', os.path.join(ROOT, 'main_policy.py'), '--ros-args',
            '-p', f'use_sim_time:={str(p["use_sim_time"]).lower()}', '-p', f'range_source:={p["range_source"]}',
            '-p', f'depth_topic:={p["depth_topic"]}', '-p', f'depth_info_topic:={p["depth_info_topic"]}',
            '-p', f'policy_file:={policy}', '-p', f'arm:={arm}']
    actions.append(ExecuteProcess(cmd=args, cwd=ROOT, output='screen'))
    return actions


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('hardware', default_value='sitl', description='drone | sitl | flyx'),
        DeclareLaunchArgument('arm', default_value='false', description='true = may arm and fly (operator approval)'),
        DeclareLaunchArgument('policy', default_value='policies/comp_s2.npz'),
        DeclareLaunchArgument('fc_device', default_value='/dev/ttyTHS1', description='drone: Pixhawk UART'),
        OpaqueFunction(function=_setup),
    ])
