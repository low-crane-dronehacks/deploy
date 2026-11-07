// control_mux_node: Switches between Classical MPPI (/control/trajectory_setpoint -> /fmu/in/trajectory_setpoint)
// and RL Policy (/control/rates_setpoint -> /fmu/in/vehicle_rates_setpoint) with safety fallback interlocks.
// Implements Section 2.2, 2.3, WP-4 Task 2-4, WP-5 Task 3, and Section 6 FMEA of system_architecture_plan.md.
#include <chrono>
#include <memory>
#include <string>

#include <geometry_msgs/msg/twist_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#ifdef HAVE_PX4_MSGS
#include <px4_msgs/msg/offboard_control_mode.hpp>
#include <px4_msgs/msg/trajectory_setpoint.hpp>
#include <px4_msgs/msg/vehicle_rates_setpoint.hpp>
#endif
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/string.hpp>

#include "maze_planning/control_mux.hpp"

namespace maze_planning {

class ControlMuxNode : public rclcpp::Node {
 public:
  ControlMuxNode() : Node("control_mux_node") {
    const std::string initial_mode = declare_parameter<std::string>("mode", "classical");
    arm_authorized_ = declare_parameter<bool>("arm", false);

    if (initial_mode == "rl_policy") {
      mux_.setRequestedPathway(ControlPathway::kRLPolicy);
    } else {
      mux_.setRequestedPathway(ControlPathway::kClassicalMPPI);
    }

#ifdef HAVE_PX4_MSGS
    rclcpp::QoS px4_qos(1);
    px4_qos.best_effort();
    px4_qos.transient_local();

    pub_ocm_ = create_publisher<px4_msgs::msg::OffboardControlMode>(
        "/fmu/in/offboard_control_mode", px4_qos);
    pub_fmu_traj_ = create_publisher<px4_msgs::msg::TrajectorySetpoint>(
        "/fmu/in/trajectory_setpoint", px4_qos);
    pub_fmu_rates_ = create_publisher<px4_msgs::msg::VehicleRatesSetpoint>(
        "/fmu/in/vehicle_rates_setpoint", px4_qos);

    sub_traj_ = create_subscription<px4_msgs::msg::TrajectorySetpoint>(
        "/control/trajectory_setpoint", px4_qos,
        [this](const px4_msgs::msg::TrajectorySetpoint::SharedPtr m) { onTrajectorySetpoint(*m); });
    sub_rates_ = create_subscription<px4_msgs::msg::VehicleRatesSetpoint>(
        "/control/rates_setpoint", px4_qos,
        [this](const px4_msgs::msg::VehicleRatesSetpoint::SharedPtr m) { onRatesSetpoint(*m); });
#endif
    pub_status_ = create_publisher<std_msgs::msg::String>("/control/mux_status", 10);
    sub_cmd_vel_ = create_subscription<geometry_msgs::msg::TwistStamped>(
        "/control/cmd_vel", 10,
        [this](const geometry_msgs::msg::TwistStamped::SharedPtr m) { onCmdVel(*m); });
    sub_odom_ = create_subscription<nav_msgs::msg::Odometry>(
        "/odometry/local", rclcpp::SensorDataQoS(),
        [this](const nav_msgs::msg::Odometry::SharedPtr m) { onOdometry(*m); });
    sub_mode_ = create_subscription<std_msgs::msg::String>(
        "/control/mux_mode", 10,
        [this](const std_msgs::msg::String::SharedPtr m) {
          if (m->data == "rl_policy") {
            mux_.setRequestedPathway(ControlPathway::kRLPolicy);
          } else {
            mux_.setRequestedPathway(ControlPathway::kClassicalMPPI);
          }
        });

    timer_ = create_wall_timer(std::chrono::milliseconds(10), [this]() { onMuxTick(); });
    RCLCPP_INFO(get_logger(), "control_mux_node started (mode=%s, arm=%s)",
                initial_mode.c_str(), arm_authorized_ ? "true" : "false");
  }

 private:
  void onCmdVel(const geometry_msgs::msg::TwistStamped& m) {
    TrajectorySetpointCmd cmd{};
    cmd.timestamp_s = get_clock()->now().seconds();
    cmd.pos_ned = state_odom_.pos_odom;
    cmd.vel_ned = Eigen::Vector3d(m.twist.linear.x, m.twist.linear.y, m.twist.linear.z);
    cmd.yawspeed_rad_s = m.twist.angular.z;
    mux_.onClassicalCommand(cmd);
  }

#ifdef HAVE_PX4_MSGS
  void onTrajectorySetpoint(const px4_msgs::msg::TrajectorySetpoint& m) {
    TrajectorySetpointCmd cmd{};
    cmd.timestamp_s = get_clock()->now().seconds();
    cmd.pos_ned = Eigen::Vector3d(m.position[0], m.position[1], m.position[2]);
    cmd.vel_ned = Eigen::Vector3d(m.velocity[0], m.velocity[1], m.velocity[2]);
    cmd.acc_ned = Eigen::Vector3d(m.acceleration[0], m.acceleration[1], m.acceleration[2]);
    cmd.yaw_rad = m.yaw;
    cmd.yawspeed_rad_s = m.yawspeed;
    mux_.onClassicalCommand(cmd);
  }

  void onRatesSetpoint(const px4_msgs::msg::VehicleRatesSetpoint& m) {
    RatesSetpointCmd cmd{};
    cmd.timestamp_s = get_clock()->now().seconds();
    cmd.omega_body_rad_s = Eigen::Vector3d(m.roll, m.pitch, m.yaw);
    cmd.thrust_norm = static_cast<double>(-m.thrust_body[2]);
    cmd.inference_latency_ms = 2.0;
    mux_.onRLRatesCommand(cmd);
  }
#endif

  void onOdometry(const nav_msgs::msg::Odometry& m) {
    state_odom_.timestamp_s = rclcpp::Time(m.header.stamp).seconds();
    state_odom_.pos_odom = Eigen::Vector3d(
        m.pose.pose.position.x, m.pose.pose.position.y, m.pose.pose.position.z);
    state_odom_.q_odom_base = Eigen::Quaterniond(
        m.pose.pose.orientation.w, m.pose.pose.orientation.x,
        m.pose.pose.orientation.y, m.pose.pose.orientation.z).normalized();
    state_odom_.pos_variance =
        std::max({m.pose.covariance[0], m.pose.covariance[7], m.pose.covariance[14]});
    state_odom_.tracking_valid = state_odom_.pos_odom.allFinite();
  }

  void onMuxTick() {
    const double now_s = get_clock()->now().seconds();
    const MuxDecision dec = mux_.evaluate(now_s, state_odom_, 1.0);

    std_msgs::msg::String st;
    st.data = dec.status_reason;
    pub_status_->publish(st);

    if (!arm_authorized_) return; // Never publish to PX4 unless arm:=true

#ifdef HAVE_PX4_MSGS
    const uint64_t us = static_cast<uint64_t>(get_clock()->now().nanoseconds() / 1000);
    px4_msgs::msg::OffboardControlMode ocm{};
    ocm.timestamp = us;

    if (dec.publish_rates_setpoint && dec.active_pathway == ControlPathway::kRLPolicy) {
      ocm.position = false;
      ocm.velocity = false;
      ocm.acceleration = false;
      ocm.attitude = false;
      ocm.body_rate = true;
      pub_ocm_->publish(ocm);

      px4_msgs::msg::VehicleRatesSetpoint r{};
      r.timestamp = us;
      r.roll = static_cast<float>(dec.rates_cmd.omega_body_rad_s.x());
      r.pitch = static_cast<float>(dec.rates_cmd.omega_body_rad_s.y());
      r.yaw = static_cast<float>(dec.rates_cmd.omega_body_rad_s.z());
      r.thrust_body = {0.0f, 0.0f, static_cast<float>(-dec.rates_cmd.thrust_norm)};
      pub_fmu_rates_->publish(r);
    } else if (dec.publish_trajectory_setpoint) {
      ocm.position = true;
      ocm.velocity = true;
      ocm.acceleration = true;
      ocm.attitude = false;
      ocm.body_rate = false;
      pub_ocm_->publish(ocm);

      px4_msgs::msg::TrajectorySetpoint sp{};
      sp.timestamp = us;
      sp.position = {
          static_cast<float>(dec.trajectory_cmd.pos_ned.x()),
          static_cast<float>(dec.trajectory_cmd.pos_ned.y()),
          static_cast<float>(dec.trajectory_cmd.pos_ned.z()),
      };
      sp.velocity = {
          static_cast<float>(dec.trajectory_cmd.vel_ned.x()),
          static_cast<float>(dec.trajectory_cmd.vel_ned.y()),
          static_cast<float>(dec.trajectory_cmd.vel_ned.z()),
      };
      sp.acceleration = {
          static_cast<float>(dec.trajectory_cmd.acc_ned.x()),
          static_cast<float>(dec.trajectory_cmd.acc_ned.y()),
          static_cast<float>(dec.trajectory_cmd.acc_ned.z()),
      };
      sp.yaw = static_cast<float>(dec.trajectory_cmd.yaw_rad);
      sp.yawspeed = static_cast<float>(dec.trajectory_cmd.yawspeed_rad_s);
      pub_fmu_traj_->publish(sp);
    }
#endif
  }

  ControlMux mux_;
  VehicleKinematics state_odom_{};
  bool arm_authorized_{false};

#ifdef HAVE_PX4_MSGS
  rclcpp::Publisher<px4_msgs::msg::OffboardControlMode>::SharedPtr pub_ocm_;
  rclcpp::Publisher<px4_msgs::msg::TrajectorySetpoint>::SharedPtr pub_fmu_traj_;
  rclcpp::Publisher<px4_msgs::msg::VehicleRatesSetpoint>::SharedPtr pub_fmu_rates_;
  rclcpp::Subscription<px4_msgs::msg::TrajectorySetpoint>::SharedPtr sub_traj_;
  rclcpp::Subscription<px4_msgs::msg::VehicleRatesSetpoint>::SharedPtr sub_rates_;
#endif
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr pub_status_;
  rclcpp::Subscription<geometry_msgs::msg::TwistStamped>::SharedPtr sub_cmd_vel_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr sub_odom_;
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr sub_mode_;

  rclcpp::TimerBase::SharedPtr timer_;
};

}  // namespace maze_planning

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<maze_planning::ControlMuxNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
