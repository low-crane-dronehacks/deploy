"""The real drone (organizers' skybeta stack): flight interface + our stereo depth + the policy.

Set in /etc/skybeta/drone.env:  COMPETITOR_PKG=policynav  COMPETITOR_LAUNCH=skybeta.launch.py
(after cloning this repo into ~/team_ws/src/ and building with skybeta's container_scripts/team_build.sh), or:

    ros2 launch policynav skybeta.launch.py                          # fly: pilot arms, then Position
    ros2 launch policynav skybeta.launch.py mode:=handheld           # props OFF: camera + policy, no setpoints
    ros2 launch policynav skybeta.launch.py policy:=policies/<name>.npz
    ros2 launch policynav skybeta.launch.py dds_profile:=stack            # our nodes on skybeta's DDS profile as-is

flight_interface is kept exactly as skybeta's competitor_example starts it (it owns the Offboard heartbeat and
the safety envelope). mode:=handheld does not start it and publishes no setpoints at all.
"""
import os
import xml.etree.ElementTree as ET

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo, OpaqueFunction
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
    ('dds_profile', 'team', 'team (config/fastdds_team.xml: skybeta\'s localhost profile + 8 MiB SHM segment, our '
                            'nodes only) | stack (inherit skybeta\'s profile unchanged)'),
]


# NumPy's OpenBLAS starts one thread per core and they busy-wait between calls. The policy's matmuls are tiny
# (~2 ms), so 6 threads spun ~350% CPU at 10 Hz and starved cuVSLAM; one thread costs ~10% of a core.
ONE_BLAS_THREAD = {'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1'}
PROFILE_VARS = ('FASTRTPS_DEFAULT_PROFILES_FILE', 'FASTDDS_DEFAULT_PROFILES_FILE')


def dds_env(context):
    """Env for OUR nodes' DDS participants (never flight_interface's): the team profile, or {} to inherit skybeta's.

    The team profile is only used when skybeta runs in its default local scope (its profile env is set), and only if
    the installed file parses: Fast DDS silently falls back to every interface on a bad file, which is what skybeta's
    run.sh refuses to allow, so we fall back to skybeta's own profile instead.
    """
    choice = LaunchConfiguration('dds_profile').perform(context)
    if choice == 'stack':
        return {}, 'DDS: our nodes inherit skybeta\'s profile'
    if choice != 'team':
        raise ValueError(f"dds_profile must be team | stack, got {choice!r}")
    if not os.environ.get(PROFILE_VARS[0]):
        return {}, ('DDS: skybeta profile env not set (SKYBETA_DDS_SCOPE=network?): NOT applying the localhost-only '
                    'team profile; our nodes inherit the environment')
    path = os.path.join(get_package_share_directory('policynav'), 'config', 'fastdds_team.xml')
    try:
        ET.parse(path)
    except (OSError, ET.ParseError) as e:
        return {}, f'DDS: team profile unusable ({e}); our nodes inherit skybeta\'s profile'
    return {v: path for v in PROFILE_VARS}, f'DDS: our nodes use {path} (skybeta localhost + 8 MiB SHM segment)'


def generate_launch_description():
    declared = [DeclareLaunchArgument(n, default_value=d, description=h) for n, d, h in ARGUMENTS]
    return LaunchDescription(declared + [OpaqueFunction(function=nodes)])


def nodes(context):
    lc = LaunchConfiguration
    env, dds_note = dds_env(context)
    ours = dict(ONE_BLAS_THREAD, **env)
    fly = LaunchConfigurationEquals('mode', 'fly')
    flight_interface = Node(
        package='skybeta_flight_interface', executable='flight_interface', name='flight_interface',
        output='screen', emulate_tty=True, condition=fly,
        parameters=[{k: lc(k) for k in ('max_altitude_m', 'min_altitude_m', 'max_speed_mps', 'setpoint_rate_hz',
                                        'allow_mode_request', 'auto_offboard_on_arm', 'odometry_topic')}],
    )
    depth = Node(
        package='policynav', executable='stereo_depth', name='policynav_depth', output='screen', emulate_tty=True, additional_env=ours,
        condition=LaunchConfigurationNotEquals('stereo_backend', 'vpi'),    # vpi: vpi_stereo rectifies + matches in C++
        parameters=[{'backend': PythonExpression(["'external' if '", lc('stereo_backend'), "' == 'vpi' else '",
                                                  lc('stereo_backend'), "'"]),
                     'max_disparity': lc('max_disparity'), 'rate_hz': lc('depth_rate_hz')}],
    )
    vpi = Node(
        package='policynav', executable='vpi_stereo', name='policynav_vpi_stereo', output='screen', emulate_tty=True, additional_env=env,
        condition=LaunchConfigurationEquals('stereo_backend', 'vpi'),
        parameters=[{'max_disparity': lc('max_disparity'), 'rate_hz': lc('depth_rate_hz')}],
    )
    policy = Node(
        package='policynav', executable='skybeta_policy', name='policynav_skybeta', output='screen', additional_env=ours,
        emulate_tty=True, condition=fly,
        parameters=[{'policy_file': lc('policy'), 'speed_cap': lc('speed_cap'), 'yaw_rate_cap': lc('yaw_rate_cap'),
                     'max_flight_s': lc('max_flight_s')}],
    )
    handheld = Node(
        package='policynav', executable='handheld', name='policynav_handheld', output='screen', emulate_tty=True, additional_env=ours,
        condition=LaunchConfigurationEquals('mode', 'handheld'),
        parameters=[{'policy_file': lc('policy'), 'state_source': 'odom', 'range_source': 'auto',
                     'depth_topic': '/policynav/depth/image', 'depth_info_topic': '/policynav/depth/camera_info',
                     'log_dir': '/root/team_ws/logs/handheld'}],
    )
    return [LogInfo(msg=dds_note), flight_interface, depth, vpi, policy, handheld]
