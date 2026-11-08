"""isam2_depth_node fed by spoof_px4_depth.py (fake PX4 + depth rendered from a course scan), shown in RViz.

    source ~/isam2_ws/install/setup.bash          # px4_msgs + isam2_depth (docs/ISAM2.md)
    ros2 launch src/test/spoof_isam2.launch.py ply:=$PWD/dronehacks-course-scan-2026-11-06.ply
    ros2 launch src/test/spoof_isam2.launch.py ply:=... rviz:=false spoof_args:='--no_drift --csv /tmp/run.csv'
    ros2 launch src/test/spoof_isam2.launch.py ply:=... map_save:=/tmp/isam2_map    # map at Ctrl-C
    ros2 launch src/test/spoof_isam2.launch.py ply:=... stereo:=true    # + rendered stereo/IMU -> cuVSLAM
                                                                        # (docs/CUVSLAM_LOCAL.md for the install)

No PX4, no vehicle, nothing is armed: the spoofer only publishes fake /fmu/out/* telemetry.
Use a ROS_DOMAIN_ID of your own on a shared network, so the fake PX4 topics stay on this machine.
"""
import os
import shlex
from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

HERE = Path(__file__).resolve().parent
DEPLOY = HERE.parents[1]


def setup(context):
    ply = LaunchConfiguration('ply').perform(context)
    if not ply:
        raise RuntimeError('ply:=<path to the course scan .ply> is required')
    extra = shlex.split(LaunchConfiguration('spoof_args').perform(context))
    stereo = LaunchConfiguration('stereo').perform(context).lower() in ('1', 'true', 'yes')
    if stereo and '--stereo' not in extra:
        extra.append('--stereo')
    actions = [
        Node(package='isam2_depth', executable='isam2_depth_node', name='isam2_depth', output='screen',
             parameters=[{'use_sim_time': True,
                          'map_save_path': LaunchConfiguration('map_save').perform(context)}]),
        ExecuteProcess(cmd=['/usr/bin/python3', str(HERE / 'spoof_px4_depth.py'), ply, *extra], output='screen'),
    ]
    if stereo:
        # cuVSLAM lives in its own venv (docs/CUVSLAM_LOCAL.md); its CUDA math libraries come from NVIDIA's wheels
        venv = Path(os.path.expanduser(LaunchConfiguration('cuvslam_venv').perform(context)))
        py = venv / 'bin' / 'python'
        if not py.exists():
            raise RuntimeError(f'stereo:=true needs the cuVSLAM venv at {venv} (docs/CUVSLAM_LOCAL.md)')
        nv = venv / 'lib' / 'python3.12' / 'site-packages' / 'nvidia'
        ld = ':'.join(str(nv / d / 'lib') for d in ('cublas', 'cusolver', 'cusparse', 'nvjitlink'))
        imu = LaunchConfiguration('vslam_imu').perform(context).lower() in ('1', 'true', 'yes')
        imu_hz = next((extra[k + 1] for k, a in enumerate(extra[:-1]) if a == '--imu_hz'), '200.0')
        actions.append(ExecuteProcess(
            cmd=[str(py), '-m', 'onboard.slam.cuvslam_lib_node', '--ros-args',
                 '--params-file', str(DEPLOY / 'onboard' / 'config' / 'vio.yaml'),
                 '-p', f'imu:={str(imu).lower()}', '-p', f'imu_frequency:={float(imu_hz)}'],
            cwd=str(DEPLOY), output='screen',
            additional_env={'LD_LIBRARY_PATH': ld + ':' + os.environ.get('LD_LIBRARY_PATH', '')}))
    if LaunchConfiguration('rviz').perform(context).lower() in ('1', 'true', 'yes'):
        actions.append(Node(package='rviz2', executable='rviz2', arguments=['-d', str(HERE / 'spoof_isam2.rviz')],
                            parameters=[{'use_sim_time': True}], output='log'))
    return actions


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('ply', default_value='', description='course scan (.ply)'),
        DeclareLaunchArgument('rviz', default_value='true'),
        DeclareLaunchArgument('spoof_args', default_value='', description='extra spoof_px4_depth.py arguments'),
        DeclareLaunchArgument('stereo', default_value='false',
                              description='true = render a stereo pair + IMU from the scan and run cuVSLAM on it'),
        DeclareLaunchArgument('vslam_imu', default_value='true', description='cuVSLAM Inertial (true) or stereo-only'),
        DeclareLaunchArgument('cuvslam_venv', default_value='~/cuvslam-venv'),
        DeclareLaunchArgument('map_save', default_value='',
                              description='write the corrected map to <map_save>.ply (+ _keyframes.csv) at shutdown'),
        OpaqueFunction(function=setup),
    ])
