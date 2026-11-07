"""The real drone (organizers' skybeta stack): flight interface + our stereo depth + the policy.

Set in /etc/skybeta/drone.env:  COMPETITOR_PKG=policynav  COMPETITOR_LAUNCH=skybeta.launch.py
(after cloning this repo into ~/team_ws/src/ and building with skybeta's container_scripts/team_build.sh), or:

    ros2 launch policynav skybeta.launch.py                          # fly: pilot arms, then Position
    ros2 launch policynav skybeta.launch.py mode:=handheld           # props OFF: camera + policy, no setpoints
    ros2 launch policynav skybeta.launch.py policy:=policies/<name>.npz

flight_interface is kept exactly as skybeta's competitor_example starts it (it owns the Offboard heartbeat and
the safety envelope). mode:=handheld does not start it and publishes no setpoints at all.
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import LaunchConfigurationEquals, LaunchConfigurationNotEquals
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node

ARGUMENTS = [
    # skybeta's flight interface, with its own defaults (competitor_example/launch/competitor.launch.py)
    ('max_altitude_m', '3.0', 'ceiling above the odometry origin, metres'),
    ('min_altitude_m', '0.3', 'floor above the odometry origin, metres (pose setpoints only)'),
    ('max_speed_mps', '1.5', 'cap on the norm of a twist setpoint, m/s'),
    ('setpoint_rate_hz', '10.0', 'OffboardControlMode heartbeat rate -- PX4 needs > 2 Hz'),
    ('allow_mode_request', 'false', 'SIM ONLY: enable /skybeta/request_offboard'),
    ('auto_offboard_on_arm', 'true', 'enter Offboard once when the pilot arms, then switches to Position'),
    ('odometry_topic', '/visual_slam/tracking/odometry', 'ENU pose source'),
    # ours
    ('mode', 'fly', 'fly | handheld (props off: no flight interface, no setpoints)'),
    ('policy', 'policies/cur4_clean50_800.npz', 'exported policy (.npz), relative to the package share'),
    ('stereo_backend', 'vpi', 'vpi (C++ vpi_stereo, CUDA census+SGM) | sgbm (OpenCV, CPU, ~29 ms at 640x400 on 6 cores)'),
    ('max_disparity', '96', 'stereo search range at 640 px: min range = f*B/max (~0.33 m at 96, 0.40 m at 80)'),
    ('depth_rate_hz', '13.5', 'stereo rate, averaged by camera stamps (training models 13.5-15 Hz frames)'),
    ('speed_cap', '1.45', 'our own cap on |v| (m/s), under the interface clamp'),
    ('yaw_rate_cap', '0.6', 'rad/s; the interface does not limit yaw, and stereo VO loses track when spun fast'),
    ('max_flight_s', '120.0', 'then /skybeta/hold'),
]


def generate_launch_description():
    lc = LaunchConfiguration
    declared = [DeclareLaunchArgument(n, default_value=d, description=h) for n, d, h in ARGUMENTS]
    fly = LaunchConfigurationEquals('mode', 'fly')
    flight_interface = Node(
        package='skybeta_flight_interface', executable='flight_interface', name='flight_interface',
        output='screen', emulate_tty=True, condition=fly,
        parameters=[{k: lc(k) for k in ('max_altitude_m', 'min_altitude_m', 'max_speed_mps', 'setpoint_rate_hz',
                                        'allow_mode_request', 'auto_offboard_on_arm', 'odometry_topic')}],
    )
    depth = Node(
        package='policynav', executable='stereo_depth', name='policynav_depth', output='screen', emulate_tty=True,
        condition=LaunchConfigurationNotEquals('stereo_backend', 'vpi'),    # vpi: vpi_stereo rectifies + matches in C++
        parameters=[{'backend': PythonExpression(["'external' if '", lc('stereo_backend'), "' == 'vpi' else '",
                                                  lc('stereo_backend'), "'"]),
                     'max_disparity': lc('max_disparity'), 'rate_hz': lc('depth_rate_hz')}],
    )
    vpi = Node(
        package='policynav', executable='vpi_stereo', name='policynav_vpi_stereo', output='screen', emulate_tty=True,
        condition=LaunchConfigurationEquals('stereo_backend', 'vpi'),
        parameters=[{'max_disparity': lc('max_disparity'), 'rate_hz': lc('depth_rate_hz')}],
    )
    policy = Node(
        package='policynav', executable='skybeta_policy', name='policynav_skybeta', output='screen',
        emulate_tty=True, condition=fly,
        parameters=[{'policy_file': lc('policy'), 'speed_cap': lc('speed_cap'), 'yaw_rate_cap': lc('yaw_rate_cap'),
                     'max_flight_s': lc('max_flight_s')}],
    )
    handheld = Node(
        package='policynav', executable='handheld', name='policynav_handheld', output='screen', emulate_tty=True,
        condition=LaunchConfigurationEquals('mode', 'handheld'),
        parameters=[{'policy_file': lc('policy'), 'state_source': 'odom', 'range_source': 'auto',
                     'depth_topic': '/policynav/depth/image', 'depth_info_topic': '/policynav/depth/camera_info',
                     'log_dir': '/root/team_ws/logs/handheld'}],
    )
    return LaunchDescription(declared + [flight_interface, depth, vpi, policy, handheld])
