#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <std_msgs/msg/bool.hpp>
#include <std_msgs/msg/string.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>

#include <px4_msgs/msg/distance_sensor.hpp>
#include <px4_msgs/msg/offboard_control_mode.hpp>
#include <px4_msgs/msg/trajectory_setpoint.hpp>
#include <px4_msgs/msg/vehicle_attitude.hpp>
#include <px4_msgs/msg/vehicle_command.hpp>
#include <px4_msgs/msg/vehicle_land_detected.hpp>
#include <px4_msgs/msg/vehicle_local_position.hpp>
#include <px4_msgs/msg/vehicle_status.hpp>

#include "ego_planner/mission_manager.hpp"

namespace ego_planner {

namespace {

constexpr uint16_t kMavCmdSetMode = 176;
constexpr uint16_t kMavCmdArm = 400;
constexpr uint16_t kMavCmdLand = 21;
constexpr uint8_t kNavStateOffboard = 14;
constexpr uint8_t kArmingStateArmed = 2;

constexpr int kOutWidth = 128;
constexpr int kOutHeight = 80;
constexpr double kHfovDeg = 72.0;
constexpr double kVfovDeg = 50.0;
constexpr double kFarClipM = 5.0;

bool parseBoolParam(const rclcpp::ParameterValue& pv, bool fallback) {
  if (pv.get_type() == rclcpp::ParameterType::PARAMETER_BOOL) {
    return pv.get<bool>();
  }
  if (pv.get_type() == rclcpp::ParameterType::PARAMETER_STRING) {
    std::string s = pv.get<std::string>();
    for (char& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return (s == "1" || s == "true" || s == "yes");
  }
  return fallback;
}

}  // namespace

class EgoPlannerNode : public rclcpp::Node {
 public:
  EgoPlannerNode()
      : Node("ego_planner_node"),
        sys_clock_(RCL_SYSTEM_TIME),
        tf_buffer_(this->get_clock()),
        tf_listener_(tf_buffer_) {
    MissionConfig cfg{};

    auto arm_pv = this->declare_parameter("arm", rclcpp::ParameterValue(false));
    cfg.arm_authorized = parseBoolParam(arm_pv, false);

    auto ret_pv = this->declare_parameter("enable_return_leg", rclcpp::ParameterValue(false));
    cfg.enable_return_leg = parseBoolParam(ret_pv, false);

    auto win_pv = this->declare_parameter("prefer_window", rclcpp::ParameterValue(true));
    cfg.prefer_window = parseBoolParam(win_pv, true);

    cfg.prior_bin_path = this->declare_parameter<std::string>("prior_bin_path", "config/arena_prior.bin");
    cfg.control_dt_s = this->declare_parameter<double>("control_dt", 0.10);
    cfg.stream_lead_s = this->declare_parameter<double>("stream_lead", 1.0);
    cfg.arm_wait_s = this->declare_parameter<double>("arm_wait", 15.0);
    cfg.depth_stale_s = this->declare_parameter<double>("depth_stale", 0.50);
    cfg.depth_lost_land_s = this->declare_parameter<double>("depth_lost_land", 3.0);
    cfg.health_bad_land_s = this->declare_parameter<double>("health_bad_land", 1.0);
    cfg.max_flight_s = this->declare_parameter<double>("max_flight_s", 180.0);
    cfg.geofence_m = this->declare_parameter<double>("geofence_m", 8.0);
    cfg.airborne_alt_m = this->declare_parameter<double>("airborne_alt", 0.30);
    cfg.cruise_alt_m = this->declare_parameter<double>("cruise_alt_m", 0.78);
    cfg.desired_speed_ms = this->declare_parameter<double>("desired_speed_ms", 2.25);
    cfg.replan_period_s = this->declare_parameter<double>("replan_period_s", 0.20);
    cfg.planner_type = this->declare_parameter<std::string>("planner", "bspline");
    cfg.controller_type = this->declare_parameter<std::string>("controller", "ego");

    depth_topic_ = this->declare_parameter<std::string>("depth_topic", "/camera/depth/image_raw");
    info_topic_ = this->declare_parameter<std::string>("depth_info_topic", "/camera/depth/camera_info");
    health_topic_ = this->declare_parameter<std::string>("health_topic", "/estimator/healthy");
    status_topic_ = this->declare_parameter<std::string>("status_topic", "/ego_planner/status");
    base_frame_ = this->declare_parameter<std::string>("base_frame", "base_link");
    range_source_ = this->declare_parameter<std::string>("range_source", "distance_sensor");

    manager_ = std::make_unique<MissionManager>(cfg);
    depth_norm_.assign(static_cast<std::size_t>(kOutWidth * kOutHeight), 1.0f);

    rclcpp::QoS px4_qos(1);
    px4_qos.best_effort();
    px4_qos.transient_local();

    pub_ocm_ = this->create_publisher<px4_msgs::msg::OffboardControlMode>(
        "/fmu/in/offboard_control_mode", px4_qos);
    pub_sp_ = this->create_publisher<px4_msgs::msg::TrajectorySetpoint>(
        "/fmu/in/trajectory_setpoint", px4_qos);
    pub_cmd_ = this->create_publisher<px4_msgs::msg::VehicleCommand>(
        "/fmu/in/vehicle_command", px4_qos);
    pub_status_ = this->create_publisher<std_msgs::msg::String>(status_topic_, 10);
    pub_policynav_status_ = this->create_publisher<std_msgs::msg::String>("/policynav/status", 10);

    sub_lpos_ = this->create_subscription<px4_msgs::msg::VehicleLocalPosition>(
        "/fmu/out/vehicle_local_position_v1", px4_qos,
        [this](const px4_msgs::msg::VehicleLocalPosition::SharedPtr m) { onLocalPos(*m); });
    sub_att_ = this->create_subscription<px4_msgs::msg::VehicleAttitude>(
        "/fmu/out/vehicle_attitude", px4_qos,
        [this](const px4_msgs::msg::VehicleAttitude::SharedPtr m) { onAttitude(*m); });
    sub_status_ = this->create_subscription<px4_msgs::msg::VehicleStatus>(
        "/fmu/out/vehicle_status_v1", px4_qos,
        [this](const px4_msgs::msg::VehicleStatus::SharedPtr m) { onStatus(*m); });
    sub_land_ = this->create_subscription<px4_msgs::msg::VehicleLandDetected>(
        "/fmu/out/vehicle_land_detected", px4_qos,
        [this](const px4_msgs::msg::VehicleLandDetected::SharedPtr m) { onLand(*m); });
    sub_health_ = this->create_subscription<std_msgs::msg::Bool>(
        health_topic_, 10,
        [this](const std_msgs::msg::Bool::SharedPtr m) { onHealth(*m); });
    sub_info_ = this->create_subscription<sensor_msgs::msg::CameraInfo>(
        info_topic_, rclcpp::SensorDataQoS(),
        [this](const sensor_msgs::msg::CameraInfo::SharedPtr m) { onCameraInfo(*m); });
    sub_depth_ = this->create_subscription<sensor_msgs::msg::Image>(
        depth_topic_, rclcpp::SensorDataQoS(),
        [this](const sensor_msgs::msg::Image::SharedPtr m) { onDepth(*m); });

    if (range_source_ == "distance_sensor") {
      sub_range_ = this->create_subscription<px4_msgs::msg::DistanceSensor>(
          "/fmu/out/distance_sensor", px4_qos,
          [this](const px4_msgs::msg::DistanceSensor::SharedPtr m) { onRange(*m); });
    }

    const auto dt_ctrl = std::chrono::duration<double>(cfg.control_dt_s);
    timer_ctrl_ = this->create_timer(
        std::chrono::duration_cast<std::chrono::nanoseconds>(dt_ctrl),
        [this]() { onControlTimer(); });
    timer_stream_ = this->create_timer(
        std::chrono::milliseconds(50),
        [this]() { onStreamTimer(); });
    timer_status_ = this->create_timer(
        std::chrono::milliseconds(500),
        [this]() { onStatusTimer(); });

    RCLCPP_INFO(this->get_logger(),
                "ego_planner_node initialized (prior=%s, arm=%s, planner=%s, controller=%s, speed=%.2f m/s, alt=%.2f m)",
                cfg.prior_bin_path.c_str(), cfg.arm_authorized ? "true" : "false",
                cfg.planner_type.c_str(), cfg.controller_type.c_str(),
                cfg.desired_speed_ms, cfg.cruise_alt_m);
  }

 private:
  double nowSec() const { return static_cast<double>(this->get_clock()->now().nanoseconds()) * 1e-9; }

  void onLocalPos(const px4_msgs::msg::VehicleLocalPosition& m) {
    state_.pos_ned = Vec3{static_cast<double>(m.x), static_cast<double>(m.y), static_cast<double>(m.z)};
    state_.vel_ned = Vec3{static_cast<double>(m.vx), static_cast<double>(m.vy), static_cast<double>(m.vz)};
    state_.pos_valid = m.xy_valid && m.z_valid && m.v_xy_valid && m.v_z_valid &&
                       std::isfinite(m.x) && std::isfinite(m.y) && std::isfinite(m.z) &&
                       std::isfinite(m.vx) && std::isfinite(m.vy) && std::isfinite(m.vz);
    if (range_source_ != "distance_sensor" || !has_range_sensor_) {
      state_.range_m = std::clamp(-state_.pos_ned.z + 0.08, 0.04, 4.0);
    }
  }

  void onAttitude(const px4_msgs::msg::VehicleAttitude& m) {
    state_.q_wxyz = Quat{static_cast<double>(m.q[0]), static_cast<double>(m.q[1]),
                         static_cast<double>(m.q[2]), static_cast<double>(m.q[3])};
    state_.att_valid = true;
  }

  void onStatus(const px4_msgs::msg::VehicleStatus& m) {
    state_.armed = (m.arming_state == kArmingStateArmed);
    state_.nav_state = static_cast<int>(m.nav_state);
  }

  void onLand(const px4_msgs::msg::VehicleLandDetected& m) { state_.landed = m.landed; }

  void onHealth(const std_msgs::msg::Bool& m) { manager_->onEstimatorHealth(nowSec(), m.data); }

  void onRange(const px4_msgs::msg::DistanceSensor& m) {
    if (m.signal_quality != 0 && m.current_distance >= m.min_distance &&
        m.current_distance <= m.max_distance) {
      state_.range_m = static_cast<double>(m.current_distance);
      has_range_sensor_ = true;
    }
  }

  void onCameraInfo(const sensor_msgs::msg::CameraInfo& m) {
    if (m.k[0] > 0.0 && !has_cam_info_) {
      fx_ = m.k[0];
      fy_ = m.k[4];
      cx_ = m.k[2];
      cy_ = m.k[5];
      src_w_ = static_cast<int>(m.width);
      src_h_ = static_cast<int>(m.height);
      has_cam_info_ = true;
    }
  }

  void onDepth(const sensor_msgs::msg::Image& m) {
    const double t = nowSec();
    if (!has_cam_info_) {
      manager_->onDepthTimestamp(t);
      return;
    }

    if (!has_tf_) {
      try {
        const auto tf = tf_buffer_.lookupTransform(base_frame_, m.header.frame_id, tf2::TimePointZero);
        const auto& q = tf.transform.rotation;
        const auto& tr = tf.transform.translation;
        R_body_cam_ = Quat{q.w, q.x, q.y, q.z}.toRotationMatrix();
        t_body_cam_ = Vec3{tr.x, tr.y, tr.z};
        has_tf_ = true;
      } catch (const std::exception& e) {
        RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
                             "Waiting for TF %s <- %s: %s",
                             base_frame_.c_str(), m.header.frame_id.c_str(), e.what());
        return;
      }
    }

    // Always update depth timestamp for freshness check; rate-limit full 3D raycast to ~10 Hz
    manager_->onDepthTimestamp(t);
    if ((t - last_depth_raycast_t_) < 0.085) {
      return;
    }
    last_depth_raycast_t_ = t;

    // Reproject incoming depth into normalized 128x80 buffer (z_depth / 5.0 m).
    // Use 2.0f as "unhit" sentinel so unhit pixels become 0.0f (invalid return, never clears walls).
    std::fill(depth_norm_.begin(), depth_norm_.end(), 2.0f);
    const double out_fx = (0.5 * kOutWidth) / std::tan(0.5 * kHfovDeg * M_PI / 180.0);
    const double out_fy = (0.5 * kOutHeight) / std::tan(0.5 * kVfovDeg * M_PI / 180.0);
    const double out_cx = 0.5 * (kOutWidth - 1);
    const double out_cy = 0.5 * (kOutHeight - 1);

    const bool is_16u = (m.encoding == "16UC1" || m.encoding == "mono16");
    const bool is_32f = (m.encoding == "32FC1");
    if (!is_16u && !is_32f) {
      return;
    }

    const int stride = 2;
    for (int v = 0; v < static_cast<int>(m.height); v += stride) {
      for (int u = 0; u < static_cast<int>(m.width); u += stride) {
        float d_m = 0.0f;
        if (is_16u) {
          const uint16_t raw = *reinterpret_cast<const uint16_t*>(&m.data[v * m.step + u * 2]);
          d_m = static_cast<float>(raw) * 1e-3f;
        } else {
          d_m = *reinterpret_cast<const float*>(&m.data[v * m.step + u * 4]);
        }
        if (std::isinf(d_m) && d_m > 0.0f) d_m = static_cast<float>(kFarClipM);
        if (!std::isfinite(d_m) || d_m <= 0.05f) continue;

        const Vec3 p_cam{
            ((static_cast<double>(u) - cx_) / fx_) * d_m,
            ((static_cast<double>(v) - cy_) / fy_) * d_m,
            static_cast<double>(d_m),
        };
        const Vec3 p_body = R_body_cam_ * p_cam + t_body_cam_;
        // Body FLU (x fwd, y left, z up): depth along +x_body
        if (p_body.x <= 0.08) continue;
        const int u_out = static_cast<int>(std::round(out_cx - out_fx * (p_body.y / p_body.x)));
        const int v_out = static_cast<int>(std::round(out_cy - out_fy * (p_body.x > 0 ? p_body.z / p_body.x : 0.0)));
        if (u_out >= 0 && u_out < kOutWidth && v_out >= 0 && v_out < kOutHeight) {
          const float norm_val =
              static_cast<float>(std::clamp(p_body.x / kFarClipM, 0.0, 1.0));
          float& dst = depth_norm_[static_cast<std::size_t>(v_out * kOutWidth + u_out)];
          if (norm_val < dst) dst = norm_val;
        }
      }
    }
    for (float& val : depth_norm_) {
      if (val > 1.0f) val = 0.0f;
    }

    manager_->onDepthFrame(t, depth_norm_.data(), kOutWidth, kOutHeight,
                           kHfovDeg, kVfovDeg, kFarClipM, state_.pos_ned, state_.q_wxyz, 4);
  }

  void onControlTimer() {
    const double t = nowSec();
    if (t <= 0.0) return;

    const auto t0 = std::chrono::steady_clock::now();
    last_out_ = manager_->step(t, state_);
    const auto t1 = std::chrono::steady_clock::now();
    last_step_ms_ = std::chrono::duration<double, std::milli>(t1 - t0).count();

    for (const auto& ev : manager_->consumeEvents()) {
      RCLCPP_INFO(this->get_logger(), "Event: %s", ev.c_str());
    }
    manager_->clearEvents();

    if (last_out_.mission_phase == "FLY") {
      RCLCPP_INFO_THROTTLE(
          this->get_logger(), *this->get_clock(), 1000,
          "FLY cp=%d prim=%s pos=(%.2f,%.2f,%.2f) vel_sp=(%.2f,%.2f,%.2f) step=%.2fms",
          last_out_.next_checkpoint, last_out_.active_primitive.c_str(),
          state_.pos_ned.x, state_.pos_ned.y, state_.pos_ned.z,
          last_out_.vel_ned.x, last_out_.vel_ned.y, last_out_.vel_ned.z,
          last_step_ms_);
    }

    if (last_out_.command_flags & CommandFlag::kCmdOffboard) {
      publishVehicleCommand(kMavCmdSetMode, 1.0f, 6.0f);
    }
    if (last_out_.command_flags & CommandFlag::kCmdArm) {
      publishVehicleCommand(kMavCmdArm, 1.0f, 0.0f);
    }
    if (last_out_.command_flags & CommandFlag::kCmdLand) {
      publishVehicleCommand(kMavCmdLand, 0.0f, 0.0f);
    }
    if (last_out_.command_flags & CommandFlag::kCmdDisarm) {
      publishVehicleCommand(kMavCmdArm, 0.0f, 0.0f);
    }

    if (last_out_.mission_phase == "DONE") {
      RCLCPP_INFO(this->get_logger(), "Mission complete: result=%s why=%s",
                  last_out_.result.c_str(), last_out_.why.c_str());
      rclcpp::shutdown();
    }
  }

  void onStreamTimer() {
    if (!last_out_.stream_velocity) return;
    const uint64_t us = static_cast<uint64_t>(sys_clock_.now().nanoseconds() / 1000);

    px4_msgs::msg::OffboardControlMode ocm{};
    ocm.timestamp = us;
    ocm.position = false;
    ocm.velocity = true;
    ocm.acceleration = false;
    ocm.attitude = false;
    ocm.body_rate = false;
    pub_ocm_->publish(ocm);

    px4_msgs::msg::TrajectorySetpoint sp{};
    sp.timestamp = us;
    const float nan_f = std::numeric_limits<float>::quiet_NaN();
    sp.position = {nan_f, nan_f, nan_f};
    sp.velocity = {
        static_cast<float>(last_out_.vel_ned.x),
        static_cast<float>(last_out_.vel_ned.y),
        static_cast<float>(last_out_.vel_ned.z),
    };
    sp.acceleration = {nan_f, nan_f, nan_f};
    sp.yaw = nan_f;
    sp.yawspeed = static_cast<float>(last_out_.yawspeed_rad_s);
    pub_sp_->publish(sp);
  }

  void onStatusTimer() {
    std::ostringstream oss;
    oss.setf(std::ios::fixed);
    oss.precision(3);
    oss << "{\"phase\":\"" << last_out_.mission_phase << "\","
        << "\"why\":\"" << last_out_.why << "\","
        << "\"result\":\"" << last_out_.result << "\","
        << "\"leg\":\"" << last_out_.active_leg << "\","
        << "\"primitive\":\"" << last_out_.active_primitive << "\","
        << "\"next_cp\":" << last_out_.next_checkpoint << ","
        << "\"vel_sp\":[" << last_out_.vel_ned.x << "," << last_out_.vel_ned.y << ","
        << last_out_.vel_ned.z << "],"
        << "\"yawspeed\":" << last_out_.yawspeed_rad_s << ","
        << "\"planner_ms\":" << last_step_ms_ << "}";

    std_msgs::msg::String msg;
    msg.data = oss.str();
    pub_status_->publish(msg);
    pub_policynav_status_->publish(msg);
  }

  void publishVehicleCommand(uint16_t cmd, float p1, float p2) {
    px4_msgs::msg::VehicleCommand c{};
    c.timestamp = static_cast<uint64_t>(sys_clock_.now().nanoseconds() / 1000);
    c.command = cmd;
    c.param1 = p1;
    c.param2 = p2;
    c.target_system = 1;
    c.target_component = 1;
    c.source_system = 1;
    c.source_component = 1;
    c.from_external = true;
    pub_cmd_->publish(c);
  }

  rclcpp::Clock sys_clock_;
  tf2_ros::Buffer tf_buffer_;
  tf2_ros::TransformListener tf_listener_;

  std::unique_ptr<MissionManager> manager_;
  VehicleState state_{};
  ControlOutput last_out_{};
  double last_step_ms_{0.0};
  double last_depth_raycast_t_{-1e9};

  std::string depth_topic_;
  std::string info_topic_;
  std::string health_topic_;
  std::string status_topic_;
  std::string base_frame_;
  std::string range_source_;

  bool has_cam_info_{false};
  bool has_tf_{false};
  bool has_range_sensor_{false};
  double fx_{1.0}, fy_{1.0}, cx_{0.0}, cy_{0.0};
  int src_w_{0}, src_h_{0};
  Mat3 R_body_cam_{Mat3::identity()};
  Vec3 t_body_cam_{0.0, 0.0, 0.0};
  std::vector<float> depth_norm_;

  rclcpp::Publisher<px4_msgs::msg::OffboardControlMode>::SharedPtr pub_ocm_;
  rclcpp::Publisher<px4_msgs::msg::TrajectorySetpoint>::SharedPtr pub_sp_;
  rclcpp::Publisher<px4_msgs::msg::VehicleCommand>::SharedPtr pub_cmd_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr pub_status_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr pub_policynav_status_;

  rclcpp::Subscription<px4_msgs::msg::VehicleLocalPosition>::SharedPtr sub_lpos_;
  rclcpp::Subscription<px4_msgs::msg::VehicleAttitude>::SharedPtr sub_att_;
  rclcpp::Subscription<px4_msgs::msg::VehicleStatus>::SharedPtr sub_status_;
  rclcpp::Subscription<px4_msgs::msg::VehicleLandDetected>::SharedPtr sub_land_;
  rclcpp::Subscription<px4_msgs::msg::DistanceSensor>::SharedPtr sub_range_;
  rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr sub_health_;
  rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr sub_info_;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr sub_depth_;

  rclcpp::TimerBase::SharedPtr timer_ctrl_;
  rclcpp::TimerBase::SharedPtr timer_stream_;
  rclcpp::TimerBase::SharedPtr timer_status_;
};

}  // namespace ego_planner

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<ego_planner::EgoPlannerNode>();
  rclcpp::spin(node);
  if (rclcpp::ok()) {
    rclcpp::shutdown();
  }
  return 0;
}
