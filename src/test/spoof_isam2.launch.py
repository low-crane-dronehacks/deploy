"""isam2_depth_node fed by spoof_px4_depth.py (fake PX4 + depth rendered from a course scan), shown in RViz.

    source ~/isam2_ws/install/setup.bash          # px4_msgs + isam2_depth (docs/ISAM2.md)
    ros2 launch src/test/spoof_isam2.launch.py ply:=$PWD/dronehacks-course-scan-2026-11-06.ply
    ros2 launch src/test/spoof_isam2.launch.py ply:=... rviz:=false spoof_args:='--no_drift --csv /tmp/run.csv'

No PX4, no vehicle, nothing is armed: the spoofer only publishes fake /fmu/out/* telemetry.
Use a ROS_DOMAIN_ID of your own on a shared network, so the fake PX4 topics stay on this machine.
"""
import shlex
from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

HERE = Path(__file__).resolve().parent


def setup(context):
    ply = LaunchConfiguration('ply').perform(context)
    if not ply:
        raise RuntimeError('ply:=<path to the course scan .ply> is required')
    extra = shlex.split(LaunchConfiguration('spoof_args').perform(context))
    actions = [
        Node(package='isam2_depth', executable='isam2_depth_node', name='isam2_depth', output='screen',
             parameters=[{'use_sim_time': True}]),
        ExecuteProcess(cmd=['/usr/bin/python3', str(HERE / 'spoof_px4_depth.py'), ply, *extra], output='screen'),
    ]
    if LaunchConfiguration('rviz').perform(context).lower() in ('1', 'true', 'yes'):
        actions.append(Node(package='rviz2', executable='rviz2', arguments=['-d', str(HERE / 'spoof_isam2.rviz')],
                            parameters=[{'use_sim_time': True}], output='log'))
    return actions


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('ply', default_value='', description='course scan (.ply)'),
        DeclareLaunchArgument('rviz', default_value='true'),
        DeclareLaunchArgument('spoof_args', default_value='', description='extra spoof_px4_depth.py arguments'),
        OpaqueFunction(function=setup),
    ])
