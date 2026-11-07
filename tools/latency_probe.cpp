// latency_probe: read-only. For each topic: messages, rate by header stamp, age on arrival (now - header.stamp),
// bytes per message and MB/s. Prints one table after `seconds` and exits.
//
//   ros2 run policynav latency_probe --ros-args -p seconds:=20 -p label:=ours
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "geometry_msgs/msg/twist_stamped.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "sensor_msgs/msg/imu.hpp"

struct Stat {
  std::vector<double> age_ms;
  std::vector<double> stamp_s;
  double bytes = 0;
};

class Probe : public rclcpp::Node {
 public:
  Probe() : Node("policynav_latency_probe") {
    seconds_ = declare_parameter<double>("seconds", 20.0);
    label_ = declare_parameter<std::string>("label", "probe");
    const auto img = declare_parameter<std::vector<std::string>>(
        "image_topics", {"/oak/left/image_raw", "/oak/right/image_raw", "/oak/left/image_mono", "/oak/right/image_mono",
                         "/policynav/depth/image"});
    const auto odom = declare_parameter<std::vector<std::string>>("odom_topics", {"/visual_slam/tracking/odometry"});
    const auto imu = declare_parameter<std::vector<std::string>>("imu_topics", {"/oak/imu/data"});
    const auto tw = declare_parameter<std::vector<std::string>>("twist_topics", {"/policynav/handheld/cmd"});
    auto qos = rclcpp::SensorDataQoS();
    for (const auto& t : img)
      subs_.push_back(create_subscription<sensor_msgs::msg::Image>(t, qos, [this, t](sensor_msgs::msg::Image::ConstSharedPtr m) {
        record(t, m->header.stamp, m->data.size());
      }));
    for (const auto& t : odom)
      subs_.push_back(create_subscription<nav_msgs::msg::Odometry>(t, qos, [this, t](nav_msgs::msg::Odometry::ConstSharedPtr m) {
        record(t, m->header.stamp, sizeof(*m));
      }));
    for (const auto& t : imu)
      subs_.push_back(create_subscription<sensor_msgs::msg::Imu>(t, qos, [this, t](sensor_msgs::msg::Imu::ConstSharedPtr m) {
        record(t, m->header.stamp, sizeof(*m));
      }));
    for (const auto& t : tw)
      subs_.push_back(create_subscription<geometry_msgs::msg::TwistStamped>(
          t, qos, [this, t](geometry_msgs::msg::TwistStamped::ConstSharedPtr m) { record(t, m->header.stamp, sizeof(*m)); }));
    t0_ = std::chrono::steady_clock::now();
    timer_ = create_wall_timer(std::chrono::duration<double>(seconds_), [this]() { report(); rclcpp::shutdown(); });
  }

 private:
  void record(const std::string& t, const builtin_interfaces::msg::Time& st, size_t bytes) {
    const double s = st.sec + 1e-9 * st.nanosec;
    auto& x = stats_[t];
    x.age_ms.push_back(1e3 * (now().seconds() - s));
    x.stamp_s.push_back(s);
    x.bytes += bytes;
  }
  static double pct(std::vector<double> v, double q) {
    if (v.empty()) return 0;
    std::sort(v.begin(), v.end());
    return v[std::min(v.size() - 1, static_cast<size_t>(q * (v.size() - 1) + 0.5))];
  }
  void report() {
    const double el = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0_).count();
    std::printf("[%s] %.0f s window\n%-34s %6s %8s %8s %8s %8s %9s %8s\n", label_.c_str(), el, "topic", "n", "Hz(stamp)",
                "age p50", "age p95", "age max", "KB/msg", "MB/s");
    for (auto& kv : stats_) {
      auto& x = kv.second;
      const size_t n = x.age_ms.size();
      double hz = 0;
      if (n > 1) {
        std::vector<double> s = x.stamp_s;
        std::sort(s.begin(), s.end());
        hz = (n - 1) / std::max(1e-9, s.back() - s.front());
      }
      std::printf("%-34s %6zu %8.2f %8.1f %8.1f %8.1f %9.1f %8.2f\n", kv.first.c_str(), n, hz, pct(x.age_ms, 0.5),
                  pct(x.age_ms, 0.95), pct(x.age_ms, 1.0), n ? x.bytes / n / 1024.0 : 0.0, x.bytes / el / 1e6);
    }
    std::fflush(stdout);
  }
  double seconds_;
  std::string label_;
  std::map<std::string, Stat> stats_;
  std::vector<rclcpp::SubscriptionBase::SharedPtr> subs_;
  rclcpp::TimerBase::SharedPtr timer_;
  std::chrono::steady_clock::time_point t0_;
};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  auto n = std::make_shared<Probe>();
  rclcpp::spin(n);
  if (rclcpp::ok()) rclcpp::shutdown();
  return 0;
}
