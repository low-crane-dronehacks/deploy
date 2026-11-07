#!/usr/bin/env python3

import math
import os
from pathlib import Path
import sys

# This is ordinary ROS/PX4 code: no runtime, cloud API or mission wrapper.
# Use the topic versions exported by the installed px4_msgs generation.
def px4_topic(name, message_type):
    version = int(getattr(message_type, "MESSAGE_VERSION", 0))
    return "/fmu/out/" + name + (f"_v{version}" if version else "")

import rclpy
from rclpy.node import Node
from rclpy.qos import (
    QoSProfile, ReliabilityPolicy,
    DurabilityPolicy, HistoryPolicy
)

from px4_msgs.msg import (
    OffboardControlMode,
    TrajectorySetpoint,
    VehicleCommand,
    VehicleLocalPosition,
    VehicleStatus,
    VehicleLandDetected,
    EstimatorStatusFlags
)

HEIGHT = 1.0       # Takeoff height in meters
HOVER_TIME = 2.0   # Seconds to hover before landing


class TakeoffLand(Node):
    def __init__(self):
        super().__init__("takeoff_land")

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        self.offboard_pub = self.create_publisher(
            OffboardControlMode, "/fmu/in/offboard_control_mode", qos
        )
        self.setpoint_pub = self.create_publisher(
            TrajectorySetpoint, "/fmu/in/trajectory_setpoint", qos
        )
        self.command_pub = self.create_publisher(
            VehicleCommand, "/fmu/in/vehicle_command", qos
        )

        self.create_subscription(
            VehicleLocalPosition, px4_topic("vehicle_local_position", VehicleLocalPosition),
            self.on_position, qos
        )
        self.create_subscription(
            VehicleStatus, px4_topic("vehicle_status", VehicleStatus),
            self.on_status, qos
        )
        self.create_subscription(
            VehicleLandDetected, px4_topic("vehicle_land_detected", VehicleLandDetected),
            self.on_landed, qos
        )
        self.create_subscription(
            EstimatorStatusFlags, px4_topic("estimator_status_flags", EstimatorStatusFlags),
            self.on_estimator_flags, qos
        )

        self.pos = None
        self.status = None
        self.pos_time = 0.0
        self.status_time = 0.0
        self.landed = None
        self.landed_time = 0.0
        self.yaw_aligned = False
        self.yaw_alignment_time = None
        self.yaw_alignment_timestamp = 0
        self.last_wait_log = 0.0
        self.ground_confirmation_started = None
        self.auto_land_observed = False

        self.phase = "WAIT"
        self.phase_start = self.clock_seconds()
        self.last_tick_time = None
        self.wait_started = None
        self.preflight_timeout_s = float(self.declare_parameter("preflight_timeout_s", 45.0).value)
        if not math.isfinite(self.preflight_timeout_s) or self.preflight_timeout_s <= 0:
            raise ValueError("preflight_timeout_s must be finite and positive")
        self.last_command = 0.0

        self.x = self.y = self.z = self.yaw = 0.0
        self.target_z = 0.0

        # Control loop at 10 Hz
        self.create_timer(0.1, self.tick)

    def on_position(self, msg):
        self.pos = msg
        self.pos_time = self.clock_seconds()

    def on_status(self, msg):
        self.status = msg
        self.status_time = self.clock_seconds()

    def on_landed(self, msg):
        self.landed = bool(msg.landed)
        self.landed_time = self.clock_seconds()

    def on_estimator_flags(self, msg):
        self.yaw_aligned = bool(msg.cs_yaw_align)
        self.yaw_alignment_time = self.clock_seconds()
        self.yaw_alignment_timestamp = int(msg.timestamp)

    def position_ready(self, now):
        if self.pos is None:
            return False
        # PX4's compass final-heading flag waits for an in-flight alignment.
        # Its native yaw alignment is usable on the ground, as in refnav/node.py.
        # Require fresh reception AND compatible source time; replayed/future
        # flags must not make an unknown heading eligible for an arm request.
        flags_age_us = int(self.pos.timestamp) - self.yaw_alignment_timestamp
        aligned = (self.yaw_aligned and self.yaw_alignment_time is not None
                   and 0 <= now - self.yaw_alignment_time <= 3.0
                   and self.yaw_alignment_timestamp > 0
                   and 0 <= flags_age_us <= 3_000_000)
        return bool(self.pos.xy_valid and self.pos.z_valid
                    and (self.pos.heading_good_for_control or aligned)
                    and all(math.isfinite(value) for value in
                            (self.pos.x, self.pos.y, self.pos.z, self.pos.heading)))

    def clock_seconds(self):
        # Standard ROS time: hardware uses system time; deployment selects
        # use_sim_time for simulation. No simulator or lab API is imported.
        return self.get_clock().now().nanoseconds * 1e-9

    def timestamp(self):
        return self.get_clock().now().nanoseconds // 1000

    def command(self, cmd, p1=0.0, p2=0.0):
        msg = VehicleCommand()
        msg.timestamp = self.timestamp()
        msg.command = cmd
        msg.param1 = float(p1)
        msg.param2 = float(p2)
        msg.target_system = 1
        msg.target_component = 1
        msg.source_system = 1
        msg.source_component = 1
        msg.from_external = True
        self.command_pub.publish(msg)

    def publish_setpoint(self):
        # Tell PX4 we're controlling position
        mode = OffboardControlMode()
        mode.timestamp = self.timestamp()
        mode.position = True
        self.offboard_pub.publish(mode)

        # PX4 uses NED coordinates: negative Z is UP
        sp = TrajectorySetpoint()
        sp.timestamp = self.timestamp()
        sp.position = [self.x, self.y, self.target_z]
        sp.velocity = [math.nan] * 3
        sp.acceleration = [math.nan] * 3
        sp.yaw = self.yaw
        self.setpoint_pub.publish(sp)

    def begin_landing(self):
        self.get_logger().info("Landing")
        self.phase = "LAND"
        self.last_command = 0.0
        self.ground_confirmation_started = None
        self.auto_land_observed = False

    def tick(self):
        now = self.clock_seconds()

        if self.phase in ("DONE", "ABORT"):
            return

        if self.last_tick_time is not None and now < self.last_tick_time:
            self.get_logger().error("ROS clock moved backwards; yielding control")
            self.phase = "ABORT"
            return
        self.last_tick_time = now
        if self.wait_started is None:
            self.wait_started = now
        if self.phase in ("WAIT", "PRIME") and now - self.wait_started >= self.preflight_timeout_s:
            self.get_logger().error("Preflight timeout: no fresh valid PX4 position/heading; no arm request sent")
            self.phase = "ABORT"
            return

        # Wait for position and status telemetry
        if self.pos is None or self.status is None:
            return

        armed = (
            self.status.arming_state ==
            VehicleStatus.ARMING_STATE_ARMED
        )
        offboard = (
            self.status.nav_state ==
            VehicleStatus.NAVIGATION_STATE_OFFBOARD
        )

        # Applies during landing too: stale status must never authorize retries.
        if now - self.pos_time > 1.0 or now - self.status_time > 1.0:
            if self.phase in ("WAIT", "PRIME"):
                # No mode/arm request has been sent. Cancel priming and require
                # a whole fresh interval after recovery; emit nothing meanwhile.
                self.phase = "WAIT"
                return
            self.get_logger().error("Telemetry lost; stopping Offboard stream")
            self.phase = "ABORT"
            return

        if self.phase == "PRIME" and armed:
            self.get_logger().warning("Vehicle armed before our request; yielding control")
            self.phase = "ABORT"
            return

        if self.phase == "LAND":
            if armed and self.ground_confirmation_started is not None:
                self.get_logger().error("Vehicle re-armed during ground confirmation; yielding control")
                self.phase = "ABORT"
                return
            if armed and self.status.nav_state == VehicleStatus.NAVIGATION_STATE_AUTO_LAND:
                self.auto_land_observed = True
            if not armed:
                if self.landed is True and now - self.landed_time <= 1.0:
                    self.get_logger().info("Landed and disarmed")
                    self.phase = "DONE"
                else:
                    if (self.status.nav_state != VehicleStatus.NAVIGATION_STATE_AUTO_LAND
                            and not self.auto_land_observed):
                        self.get_logger().error("Unexpected disarm without confirmed ground; inspect vehicle")
                        self.phase = "ABORT"
                        return
                    # Status and land detection arrive independently. Keep the
                    # one-second freshness requirement; wait for a new ground
                    # report without publishing anything to a disarmed vehicle.
                    if self.ground_confirmation_started is None:
                        self.ground_confirmation_started = now
                    if now - self.ground_confirmation_started >= 5.0:
                        self.get_logger().error("Disarmed without fresh ground confirmation; inspect vehicle")
                        self.phase = "ABORT"
                return

            # Let PX4 handle the automatic descent
            if self.status.nav_state == VehicleStatus.NAVIGATION_STATE_AUTO_LAND:
                return

            if self.auto_land_observed:
                self.get_logger().warning("Flight mode changed after automatic landing began; yielding control")
                self.phase = "ABORT"
                return

            if offboard:
                self.publish_setpoint()
                if now - self.last_command > 1.0:
                    self.command(VehicleCommand.VEHICLE_CMD_NAV_LAND)
                    self.last_command = now
            else:
                self.get_logger().warning(
                    "Flight mode changed; yielding control to PX4/pilot"
                )
                self.phase = "ABORT"
            return

        # Yield before publishing another setpoint after loss of our authority.
        if self.phase in ("CLIMB", "HOVER") and (not armed or not offboard):
            self.phase = "ABORT"
            return

        if self.phase == "WAIT":
            if armed or not self.position_ready(now):
                if now - self.last_wait_log >= 5.0:
                    self.get_logger().info("Waiting for valid position and fresh PX4 heading/yaw alignment")
                    self.last_wait_log = now
                return

            # Record the current position as our starting point
            self.x = float(self.pos.x)
            self.y = float(self.pos.y)
            self.z = float(self.pos.z)
            self.yaw = float(self.pos.heading)
            self.target_z = self.z - HEIGHT

            self.get_logger().info("Position valid; priming Offboard")
            self.phase = "PRIME"
            self.phase_start = now

        if self.phase == "PRIME" and not self.position_ready(now):
            self.get_logger().warning("Position/heading lost while priming; withholding arm request")
            self.phase = "WAIT"
            return

        # Continuously send Offboard heartbeat and position target
        self.publish_setpoint()

        if self.phase == "PRIME":
            # PX4 needs over 1 second of Offboard heartbeats
            if now - self.phase_start >= 2.0:
                self.command(
                    VehicleCommand.VEHICLE_CMD_DO_SET_MODE, 1.0, 6.0
                )
                self.command(
                    VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, 1.0
                )
                self.get_logger().info("Offboard and arm requested")
                self.phase = "ACTIVATE"
                self.phase_start = now

        elif self.phase == "ACTIVATE":
            if armed and offboard:
                self.get_logger().info("Armed; climbing")
                self.phase = "CLIMB"
                self.phase_start = now
            elif now - self.phase_start > 8.0:
                self.get_logger().error("Arming/Offboard timeout")
                if armed:
                    self.begin_landing()
                else:
                    self.phase = "ABORT"

        elif self.phase == "CLIMB":
            if not armed or not offboard:
                self.get_logger().error("Offboard control lost")
                self.phase = "ABORT"
            elif abs(self.pos.z - self.target_z) < 0.15:
                self.get_logger().info("Reached target height")
                self.phase = "HOVER"
                self.phase_start = now
            elif now - self.phase_start > 25.0:
                self.get_logger().warning("Climb timeout")
                self.begin_landing()

        elif self.phase == "HOVER":
            if not armed or not offboard:
                self.phase = "ABORT"
            elif now - self.phase_start >= HOVER_TIME:
                self.begin_landing()


def ros_arguments():
    # Ordinary ROS deployment data, used on both the Jetson and the simulator.
    # No file means standard ROS system time. An explicitly configured missing
    # file is an installation error, not permission to silently select a clock.
    configured = os.environ.get("DRONE_PARAMS_FILE")
    parameters = Path(configured or "/etc/flyx/mission.yaml")
    args = list(sys.argv[1:])
    if parameters.is_file():
        return ["--ros-args", "--params-file", str(parameters), *args]
    if configured:
        raise FileNotFoundError("Configured ROS parameter file is missing: " + str(parameters))
    return args


def main():
    rclpy.init(args=ros_arguments())
    node = TakeoffLand()
    try:
        while rclpy.ok() and node.phase not in ("DONE", "ABORT"):
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        node.get_logger().warning(
            "Node interrupted; verify vehicle state in QGroundControl"
        )
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
    return 0 if node.phase == "DONE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
