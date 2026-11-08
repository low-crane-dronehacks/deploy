// path_transformer_node: Standalone ROS 2 Path Transformer node (WP-3 Task 2).
// Subscribes to /plan/global_rrt (map frame), /odometry/local (odom frame), and TF map -> odom;
// publishes /plan/local_reference (20 Hz, nav_msgs/Path in odom frame) and
// /plan/rl_lookaheads (50 Hz, std_msgs/Float32MultiArray: 4 lookaheads at d=[0.5, 1.0, 2.0, 3.5]m in body frame B).
//
// /plan/global_rrt is latched (transient_local) and only sent on a new plan. Its pose stamps carry the timing
// (ros_conversions.hpp); /plan/local_reference keeps that timing (pose stamps relative to its header stamp).
// It consumes any planner that follows that contract, not only rrt_star_planner_node.
#include <array>
#include <chrono>
#include <cmath>
#include <memory>
#include <string>

#include <geometry_msgs/msg/pose_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/float32_multi_array.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>

#include "maze_planning/path_transformer.hpp"
#include "maze_planning/ros_conversions.hpp"

namespace maze_planning {

class PathTransformerNode : public rclcpp::Node {
 public:
  PathTransformerNode()
      : Node("path_transformer_node"),
        tf_buffer_(get_clock()),
        tf_listener_(tf_buffer_) {
    map_frame_ = declare_parameter<std::string>("map_frame", "map_ned");
    odom_frame_ = declare_parameter<std::string>("odom_frame", "odom_ned");
    cruise_speed_m_s_ = declare_parameter<double>("cruise_speed_m_s", 1.50);

    pub_local_ref_ = create_publisher<nav_msgs::msg::Path>("/plan/local_reference", 10);
    pub_rl_lookaheads_ =
        create_publisher<std_msgs::msg::Float32MultiArray>("/plan/rl_lookaheads", 10);

    sub_global_ = create_subscription<nav_msgs::msg::Path>(
        "/plan/global_rrt", rclcpp::QoS(1).reliable().transient_local(),
        [this](const nav_msgs::msg::Path::SharedPtr m) { onGlobalPath(*m); });
    sub_odom_ = create_subscription<nav_msgs::msg::Odometry>(
        "/odometry/local", rclcpp::SensorDataQoS(),
        [this](const nav_msgs::msg::Odometry::SharedPtr m) { onOdometry(*m); });

    timer_local_ref_ = create_wall_timer(
        std::chrono::milliseconds(50), [this]() { onPublishLocalRef(); });
    timer_rl_ = create_wall_timer(
        std::chrono::milliseconds(20), [this]() { onPublishRLLookaheads(); });
  }

 private:
  void updateTF() {
    SE3Transform T_map_odom{};
    if (!ros_conv::lookupSE3(tf_buffer_, map_frame_, odom_frame_, T_map_odom)) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000,
                           "no TF %s <- %s; treating them as identical", map_frame_.c_str(), odom_frame_.c_str());
    }
    transformer_.setMapToOdomTransform(T_map_odom);
  }

  void onGlobalPath(const nav_msgs::msg::Path& m) {
    PlannedPath p = ros_conv::fromRosPath(m, map_frame_, cruise_speed_m_s_);
    if (p.frame_id != map_frame_) {
      RCLCPP_ERROR(get_logger(), "global path is in frame '%s', expected '%s'; ignoring it", p.frame_id.c_str(),
                   map_frame_.c_str());
      return;
    }
    if (p.points.size() < 2) return;
    RCLCPP_INFO(get_logger(), "new global path: %zu poses, %.2f m, %.2f s", p.points.size(), p.total_length_m,
                p.total_duration_s);
    transformer_.setGlobalPathMap(p);
  }

  void onOdometry(const nav_msgs::msg::Odometry& m) {
    state_odom_ = ros_conv::fromOdometry(m);
    has_odom_ = true;
  }

  void onPublishLocalRef() {
    if (!has_odom_ || !transformer_.hasGlobalPath()) return;
    updateTF();
    const PlannedPath local_odom = transformer_.extractLocalReferenceOdom(state_odom_);
    pub_local_ref_->publish(ros_conv::toRosPath(local_odom, odom_frame_, get_clock()->now()));
  }

  void onPublishRLLookaheads() {
    if (!has_odom_ || !transformer_.hasGlobalPath()) return;
    updateTF();
    std::array<float, 12> flat{};
    transformer_.computeRLLookaheadsBody(state_odom_, &flat);
    std_msgs::msg::Float32MultiArray msg;
    msg.data.assign(flat.begin(), flat.end());
    pub_rl_lookaheads_->publish(msg);
  }

  tf2_ros::Buffer tf_buffer_;
  tf2_ros::TransformListener tf_listener_;
  std::string map_frame_{"map"};
  std::string odom_frame_{"odom"};
  double cruise_speed_m_s_{1.50};

  PathTransformer transformer_;
  VehicleKinematics state_odom_{};
  bool has_odom_{false};

  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr pub_local_ref_;
  rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr pub_rl_lookaheads_;
  rclcpp::Subscription<nav_msgs::msg::Path>::SharedPtr sub_global_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr sub_odom_;
  rclcpp::TimerBase::SharedPtr timer_local_ref_;
  rclcpp::TimerBase::SharedPtr timer_rl_;
};

}  // namespace maze_planning

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<maze_planning::PathTransformerNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
