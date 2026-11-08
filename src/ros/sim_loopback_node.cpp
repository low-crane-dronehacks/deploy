// sim_loopback_node: kinematic stand-in for the vehicle, for the standalone RViz demo (no Gazebo, no PX4).
//
// in   /control/cmd_vel    geometry_msgs/TwistStamped (NED velocity + yaw rate) from mppi_controller_node
// out  /odometry/local     nav_msgs/Odometry (odom frame; twist in the body frame per REP-105)
//      /viz/drone_pose     geometry_msgs/PoseStamped (odom frame)
//      TF map -> odom      identity (the demo starts odom at the launch mark in map coordinates)
//      TF odom -> base     vehicle pose
//
// This was inside mppi_controller_node; it is its own node so the controller never simulates or owns TF.
// Never run it next to a real or SITL vehicle: it publishes /odometry/local and TF.
#include <chrono>
#include <cmath>
#include <memory>
#include <string>
#include <vector>

#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <geometry_msgs/msg/twist_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <rclcpp/rclcpp.hpp>
#include <tf2_ros/transform_broadcaster.h>

#include <Eigen/Core>
#include <Eigen/Geometry>

class SimLoopbackNode : public rclcpp::Node {
 public:
  SimLoopbackNode() : Node("sim_loopback_node"), tf_broadcaster_(this) {
    map_frame_ = declare_parameter<std::string>("map_frame", "map_ned");
    odom_frame_ = declare_parameter<std::string>("odom_frame", "odom_ned");
    base_frame_ = declare_parameter<std::string>("base_frame", "base_link_frd");
    pos_ = Eigen::Vector3d(declare_parameter<double>("start_x", 0.9144),
                           declare_parameter<double>("start_y", -2.5146),
                           declare_parameter<double>("start_z", -0.78));
    yaw_ = declare_parameter<double>("start_yaw", 0.0);
    vel_response_ = declare_parameter<double>("velocity_response", 0.20);   // per 20 ms step
    cmd_timeout_s_ = declare_parameter<double>("cmd_timeout_s", 0.3);

    pub_odom_ = create_publisher<nav_msgs::msg::Odometry>("/odometry/local", rclcpp::SensorDataQoS());
    pub_pose_ = create_publisher<geometry_msgs::msg::PoseStamped>("/viz/drone_pose", 10);
    sub_cmd_ = create_subscription<geometry_msgs::msg::TwistStamped>(
        "/control/cmd_vel", 10, [this](const geometry_msgs::msg::TwistStamped::SharedPtr m) {
          cmd_vel_ = Eigen::Vector3d(m->twist.linear.x, m->twist.linear.y, m->twist.linear.z);
          cmd_yawspeed_ = m->twist.angular.z;
          last_cmd_s_ = now().seconds();
        });
    timer_ = create_wall_timer(std::chrono::milliseconds(20), [this]() { step(0.02); });
    RCLCPP_WARN(get_logger(), "sim_loopback_node: publishing simulated /odometry/local and TF (demo only)");
  }

 private:
  void step(double dt) {
    const bool fresh = (now().seconds() - last_cmd_s_) < cmd_timeout_s_;
    const Eigen::Vector3d v_cmd = fresh ? cmd_vel_ : Eigen::Vector3d::Zero();
    vel_ = (1.0 - vel_response_) * vel_ + vel_response_ * v_cmd;
    pos_ += vel_ * dt;
    yaw_ = std::remainder(yaw_ + (fresh ? cmd_yawspeed_ : 0.0) * dt, 2.0 * M_PI);
    const Eigen::Quaterniond q(Eigen::AngleAxisd(yaw_, Eigen::Vector3d::UnitZ()));
    const rclcpp::Time stamp = now();

    nav_msgs::msg::Odometry o;
    o.header.stamp = stamp;
    o.header.frame_id = odom_frame_;
    o.child_frame_id = base_frame_;
    o.pose.pose.position.x = pos_.x();
    o.pose.pose.position.y = pos_.y();
    o.pose.pose.position.z = pos_.z();
    o.pose.pose.orientation.w = q.w();
    o.pose.pose.orientation.x = q.x();
    o.pose.pose.orientation.y = q.y();
    o.pose.pose.orientation.z = q.z();
    const Eigen::Vector3d v_body = q.conjugate() * vel_;
    o.twist.twist.linear.x = v_body.x();
    o.twist.twist.linear.y = v_body.y();
    o.twist.twist.linear.z = v_body.z();
    o.twist.twist.angular.z = fresh ? cmd_yawspeed_ : 0.0;
    pub_odom_->publish(o);

    geometry_msgs::msg::PoseStamped ps;
    ps.header = o.header;
    ps.pose = o.pose.pose;
    pub_pose_->publish(ps);

    geometry_msgs::msg::TransformStamped map_odom;
    map_odom.header.stamp = stamp;
    map_odom.header.frame_id = map_frame_;
    map_odom.child_frame_id = odom_frame_;
    map_odom.transform.rotation.w = 1.0;
    geometry_msgs::msg::TransformStamped odom_base;
    odom_base.header.stamp = stamp;
    odom_base.header.frame_id = odom_frame_;
    odom_base.child_frame_id = base_frame_;
    odom_base.transform.translation.x = pos_.x();
    odom_base.transform.translation.y = pos_.y();
    odom_base.transform.translation.z = pos_.z();
    odom_base.transform.rotation = o.pose.pose.orientation;
    tf_broadcaster_.sendTransform(std::vector<geometry_msgs::msg::TransformStamped>{map_odom, odom_base});
  }

  tf2_ros::TransformBroadcaster tf_broadcaster_;
  std::string map_frame_, odom_frame_, base_frame_;
  Eigen::Vector3d pos_{Eigen::Vector3d::Zero()};
  Eigen::Vector3d vel_{Eigen::Vector3d::Zero()};
  double yaw_{0.0};
  double vel_response_{0.2};
  double cmd_timeout_s_{0.3};
  Eigen::Vector3d cmd_vel_{Eigen::Vector3d::Zero()};
  double cmd_yawspeed_{0.0};
  double last_cmd_s_{-1e9};

  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pub_odom_;
  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr pub_pose_;
  rclcpp::Subscription<geometry_msgs::msg::TwistStamped>::SharedPtr sub_cmd_;
  rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<SimLoopbackNode>());
  rclcpp::shutdown();
  return 0;
}
