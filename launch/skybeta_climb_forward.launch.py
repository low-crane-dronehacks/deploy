"""Real drone: skybeta's flight_interface (exactly as competitor_example starts it) + climb_forward_hold_skybeta.

    ros2 launch policynav skybeta_climb_forward.launch.py                       # dry run: prints the plan only
    ros2 launch policynav skybeta_climb_forward.launch.py fly:=true height:=1.0 forward_m:=1.0 target_system:=17
Flies only after the PILOT arms (Altitude) and switches to Position. The pilot lands.
fly:=true is the competitor-window use (COMPETITOR_LAUNCH=skybeta_climb_forward.launch.py): it starts flight_interface.
Mission feedback is PX4 fused velocity + heading, commands are heading-relative velocity only.
Target system defaults to drone 17; override it for other aircraft.
The dry run starts only our node (it publishes nothing), so it can run beside the stack's own flight_interface.
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

ARGS = [('max_altitude_m', '3.0'), ('min_altitude_m', '0.3'), ('max_speed_mps', '1.5'), ('setpoint_rate_hz', '10.0'),
        ('allow_mode_request', 'false'), ('auto_offboard_on_arm', 'true'), ('target_system', '17'),
        ('px4_feedback_topic', '/fmu/out/vehicle_local_position'),
        ('odometry_topic', '/visual_slam/tracking/odometry'),
        ('fly', 'false'), ('height', '1.0'), ('forward_m', '1.0'), ('climb_speed', '0.5'), ('forward_speed', '0.5'),
        ('hold_s', '5.0')]


def generate_launch_description():
    lc = LaunchConfiguration
    return LaunchDescription([DeclareLaunchArgument(n, default_value=d) for n, d in ARGS] + [
        # only when flying: a dry run next to the running stack must not add a second Offboard heartbeat source
        Node(package='skybeta_flight_interface', executable='flight_interface', name='flight_interface',
             output='screen', emulate_tty=True, condition=IfCondition(lc('fly')),
             parameters=[{k: lc(k) for k in ('max_altitude_m', 'min_altitude_m', 'max_speed_mps', 'setpoint_rate_hz',
                                             'allow_mode_request', 'auto_offboard_on_arm', 'odometry_topic', 'target_system')}]),
        Node(package='policynav', executable='climb_forward_hold_skybeta.py', name='climb_forward_hold_skybeta',
             output='screen', emulate_tty=True,
             parameters=[{k: lc(k) for k in ('fly', 'height', 'forward_m', 'climb_speed', 'forward_speed', 'hold_s',
                                             'px4_feedback_topic')}]),
    ])
