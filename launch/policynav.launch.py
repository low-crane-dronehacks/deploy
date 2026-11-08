"""One launch for every platform: the same policy node + mission logic, only the hardware layer differs.

    ros2 launch launch/policynav.launch.py hardware:=sitl                 # our PX4 SITL rig (sitl/), dry run
    ros2 launch launch/policynav.launch.py hardware:=flyx                 # the organizers' cloud simulator
    ros2 launch launch/policynav.launch.py hardware:=drone arm:=true      # the aircraft (UNTESTED: see README)
    ros2 launch launch/policynav.launch.py hardware:=handheld [state:=px4|odom|none] [record:=true]
                                                                          # camera + policy, carried; never commands PX4
    ... policy:=policies/<name>.npz

arm defaults to false: the node runs the policy on live data and reports what it would command, publishes nothing.
"""
import os
import time

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, LogInfo, OpaqueFunction
from launch.substitutions import LaunchConfiguration

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PROFILES = {
    # sitl: the PX4 1.17 / Jazzy rig this profile was written for is gone; sitl/ now runs PX4 v1.16.2 with skybeta.launch.py
    'sitl': dict(use_sim_time=True, range_source='altitude', depth_topic='/camera/depth/image_raw',
                 depth_info_topic='/camera/depth/camera_info'),
    # flyx: the organizers' simulator provides PX4, cuVSLAM (onboard/) and the OAK depth topics already
    'flyx': dict(use_sim_time=True, range_source='sim', depth_topic='/camera/depth/image_raw',
                 depth_info_topic='/camera/depth/camera_info'),
    # drone: OAK-D S2 via depthai_ros_driver, VL53L1X via PX4 distance_sensor, uXRCE over the Jetson UART
    'drone': dict(use_sim_time=False, range_source='distance_sensor', depth_topic='/oak/stereo/image_raw',
                  depth_info_topic='/oak/stereo/camera_info'),
    # handheld: OAK + the policy, carried through the course (props OFF). Never commands PX4 (docs/HANDHELD.md)
    'handheld': dict(use_sim_time=False, range_source='auto', depth_topic='/oak/stereo/image_raw',
                     depth_info_topic='/oak/stereo/camera_info'),
}

# what record:=true captures (raw mono pair only with record_stereo:=true: ~15 MB/s)
RECORD = ['/oak/stereo/image_raw', '/oak/stereo/camera_info', '/oak/imu/data', '/tf', '/tf_static',
          '/policynav/handheld/obs', '/policynav/handheld/action', '/policynav/handheld/cmd', '/policynav/status',
          '/fmu/out/vehicle_local_position_v1', '/fmu/out/vehicle_attitude', '/fmu/out/distance_sensor',
          '/visual_slam/tracking/odometry', '/visual_slam/status']
RECORD_STEREO = ['/oak/left/image_raw', '/oak/left/camera_info', '/oak/right/image_raw', '/oak/right/camera_info']


def _handheld(context, p, policy):
    arg = lambda n: LaunchConfiguration(n).perform(context)      # noqa: E731
    state, dev = arg('state'), arg('fc_device')
    x, y, z, yaw, pitch, roll = (arg('cam_pose').split() + ['0'] * 6)[:6]
    actions = [LogInfo(msg=f'policynav HANDHELD: state={state}, policy={policy}. Nothing is sent to PX4.'),
               ExecuteProcess(cmd=['ros2', 'launch', 'depthai_ros_driver', 'camera.launch.py',
                                   f'params_file:={os.path.join(ROOT, "config", "oak.yaml")}'], output='screen'),
               # camera mount: base_link -> the driver's base frame (measure it; config/oak.yaml notes)
               ExecuteProcess(cmd=['ros2', 'run', 'tf2_ros', 'static_transform_publisher', '--x', x, '--y', y,
                                   '--z', z, '--yaw', yaw, '--pitch', pitch, '--roll', roll,
                                   '--frame-id', 'base_link', '--child-frame-id', arg('cam_frame')], output='screen')]
    if state == 'px4':      # Pixhawk on USB (props off): its estimate + rangefinder; we only READ /fmu/out
        actions.append(ExecuteProcess(cmd=['MicroXRCEAgent', 'serial', '--dev', dev, '-b', '921600'], output='screen'))
    if state == 'odom' and arg('vio') == 'true':
        actions.append(ExecuteProcess(cmd=['python3', os.path.join(ROOT, 'onboard', 'services.py'), 'start'],
                                      output='screen'))
    if arg('record') == 'true':
        out = os.path.join(ROOT, 'logs', 'handheld', 'bag_' + time.strftime('%Y%m%d_%H%M%S'))
        topics = RECORD + (RECORD_STEREO if arg('record_stereo') == 'true' else [])
        actions.append(ExecuteProcess(cmd=['ros2', 'bag', 'record', '-o', out] + topics, output='screen'))
    actions.append(ExecuteProcess(
        cmd=['python3', os.path.join(ROOT, 'main_handheld.py'), '--ros-args', '-p', f'policy_file:={policy}',
             '-p', f'state_source:={state}', '-p', f'range_source:={arg("range")}',
             '-p', f'memory:={arg("memory")}', '-p', f'depth_topic:={p["depth_topic"]}',
             '-p', f'depth_info_topic:={p["depth_info_topic"]}'], cwd=ROOT, output='screen'))
    return actions


def _setup(context):
    hw = LaunchConfiguration('hardware').perform(context)
    if hw not in PROFILES:
        raise RuntimeError(f'hardware must be one of {sorted(PROFILES)}, got {hw!r}')
    p = PROFILES[hw]
    arm = LaunchConfiguration('arm').perform(context)
    policy = LaunchConfiguration('policy').perform(context)
    if hw == 'handheld':
        return _handheld(context, p, policy)
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
        DeclareLaunchArgument('hardware', default_value='sitl', description='drone | sitl | flyx | handheld'),
        DeclareLaunchArgument('arm', default_value='false', description='true = may arm and fly (operator approval)'),
        DeclareLaunchArgument('policy', default_value='policies/cur4_clean50_800.npz'),
        DeclareLaunchArgument('fc_device', default_value='/dev/ttyTHS1',
                              description='drone: Pixhawk UART (handheld over USB: /dev/ttyACM0)'),
        # handheld only
        DeclareLaunchArgument('state', default_value='px4', description='handheld state: px4 | odom | none'),
        DeclareLaunchArgument('vio', default_value='false', description='handheld state:=odom: also start onboard VIO'),
        DeclareLaunchArgument('range', default_value='auto', description='auto | distance_sensor | altitude | fixed'),
        DeclareLaunchArgument('memory', default_value='closed', description='closed (as in flight) | fresh'),
        DeclareLaunchArgument('record', default_value='false', description='handheld: ros2 bag to logs/handheld/'),
        DeclareLaunchArgument('record_stereo', default_value='false', description='also the raw mono pair'),
        DeclareLaunchArgument('cam_frame', default_value='oak-d-base-frame', description='driver base frame'),
        DeclareLaunchArgument('cam_pose', default_value='0.10 0 0 0 0 0',
                              description='base_link -> cam_frame: "x y z yaw pitch roll" (m, rad); MEASURE it'),
        OpaqueFunction(function=_setup),
    ])
