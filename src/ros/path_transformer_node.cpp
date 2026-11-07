// path_transformer_node: Standalone ROS 2 Path Transformer node (WP-3 Task 2).
// Subscribes to /plan/global_rrt (map frame), /odometry/local (odom frame), and TF map -> odom;
// publishes /plan/local_reference (20 Hz, nav_msgs/Path in odom frame) and
// /plan/rl_lookaheads (50 Hz, std_msgs/Float32MultiArray: 4 lookaheads at d=[0.5, 1.0, 2.0, 3.5]m in body frame B).
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

namespace maze_planning {

class PathTransformerNode : public rclcpp::Node {
 public:
  PathTransformerNode()
      : Node("path_transformer_node"),
        tf_buffer_(get_clock()),
        tf_listener_(tf_buffer_) {
    map_frame_ = declare_parameter<std::string>("map_frame", "map");
    odom_frame_ = declare_parameter<std::string>("odom_frame", "odom");
    cruise_speed_m_s_ = declare_parameter<double>("cruise_speed_m_s", 2.60);

    pub_local_ref_ = create_publisher<nav_msgs::msg::Path>("/plan/local_reference", 10);
    pub_rl_lookaheads_ =
        create_publisher<std_msgs::msg::Float32MultiArray>("/plan/rl_lookaheads", 10);

    sub_global_ = create_subscription<nav_msgs::msg::Path>(
        "/plan/global_rrt", 10,
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
    try {
      const auto tf = tf_buffer_.lookupTransform(map_frame_, odom_frame_, tf2::TimePointZero);
      SE3Transform T_map_odom{};
      T_map_odom.t = Eigen::Vector3d(
          tf.transform.translation.x, tf.transform.translation.y, tf.transform.translation.z);
      T_map_odom.q = Eigen::Quaterniond(
          tf.transform.rotation.w, tf.transform.rotation.x,
          tf.transform.rotation.y, tf.transform.rotation.z).normalized();
      transformer_.setMapToOdomTransform(T_map_odom);
    } catch (const std::exception&) {
    }
  }

  void onGlobalPath(const nav_msgs::msg::Path& m) {
    PlannedPath p;
    p.frame_id = m.header.frame_id.empty() ? map_frame_ : m.header.frame_id;
    p.points.reserve(m.poses.size());
    double s = 0.0;
    for (std::size_t i = 0; i < m.poses.size(); ++i) {
      const auto& pos = m.poses[i].pose.position;
      const auto& q = m.poses[i].pose.orientation;
      PathPoint pt{};
      pt.pos = Eigen::Vector3d(pos.x, pos.y, pos.z);
      pt.yaw_rad = std::atan2(2.0 * (q.w * q.z + q.x * q.y),
                              1.0 - 2.0 * (q.y * q.y + q.z * q.z));
      if (i > 0) {
        const Eigen::Vector3d d = pt.pos - p.points.back().pos;
        const double ds = d.norm();
        s += ds;
        if (ds > 1e-5) {
          p.points.back().vel = (d / ds) * cruise_speed_m_s_;
        }
      }
      pt.s_m = s;
      pt.t_s = s / std::max(0.2, cruise_speed_m_s_);
      p.points.push_back(pt);
    }
    p.total_length_m = s;
    p.total_duration_s = s / std::max(0.2, cruise_speed_m_s_);
    transformer_.setGlobalPathMap(p);
  }

  void onOdometry(const nav_msgs::msg::Odometry& m) {
    state_odom_.timestamp_s = rclcpp::Time(m.header.stamp).seconds();
    state_odom_.pos_odom = Eigen::Vector3d(
        m.pose.pose.position.x, m.pose.pose.position.y, m.pose.pose.position.z);
    state_odom_.q_odom_base = Eigen::Quaterniond(
        m.pose.pose.orientation.w, m.pose.pose.orientation.x,
        m.pose.pose.orientation.y, m.pose.pose.orientation.z).normalized();
    const Eigen::Vector3d v_body(
        m.twist.twist.linear.x, m.twist.twist.linear.y, m.twist.twist.linear.z);
    state_odom_.vel_odom = state_odom_.q_odom_base * v_body;
    has_odom_ = true;
  }

  void onPublishLocalRef() {
    if (!has_odom_ || !transformer_.hasGlobalPath()) return;
    updateTF();
    const PlannedPath local_odom = transformer_.extractLocalReferenceOdom(state_odom_);
    nav_msgs::msg::Path msg;
    msg.header.stamp = get_clock()->now();
    msg.header.frame_id = odom_frame_;
    msg.poses.reserve(local_odom.points.size());
    for (const auto& pt : local_odom.points) {
      geometry_msgs::msg::PoseStamped ps;
      ps.header = msg.header;
      ps.pose.position.x = pt.pos.x();
      ps.pose.position.y = pt.pos.y();
      ps.pose.position.z = pt.pos.z();
      ps.pose.orientation.w = std::cos(0.5 * pt.yaw_rad);
      ps.pose.orientation.z = std::sin(0.5 * pt.yaw_rad);
      msg.poses.push_back(ps);
    }
    pub_local_ref_->publish(msg);
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
  double cruise_speed_m_s_{2.60};

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
