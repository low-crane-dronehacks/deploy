// sim_loopback_node: stand-in for the vehicle, for the standalone RViz demo (no Gazebo, no PX4). It steps the same
// plant MPPI rolls out (QuadPlant, the C++ port of dronehacks-sim quad.py: Lee velocity controller, deadcat rotors,
// motor lag, drag), so the demo shows real attitude, lag and yaw-rate limits instead of an ideal velocity source.
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

#include "maze_planning/quad_plant.hpp"

class SimLoopbackNode : public rclcpp::Node {
 public:
  SimLoopbackNode() : Node("sim_loopback_node"), tf_broadcaster_(this) {
    map_frame_ = declare_parameter<std::string>("map_frame", "map_ned");
    odom_frame_ = declare_parameter<std::string>("odom_frame", "odom_ned");
    base_frame_ = declare_parameter<std::string>("base_frame", "base_link_frd");
    const std::string odom_topic = declare_parameter<std::string>("odom_topic", "/odometry/local");
    publish_map_odom_tf_ = declare_parameter<bool>("publish_map_odom_tf", true);
    const Eigen::Vector3d start(declare_parameter<double>("start_x", 0.9144),
                                declare_parameter<double>("start_y", -2.5146),
                                declare_parameter<double>("start_z", -0.78));
    const double start_yaw = declare_parameter<double>("start_yaw", 0.0);
    state_ = plant_.fromNed(start, Eigen::Vector3d::Zero(),
                            Eigen::Quaterniond(Eigen::AngleAxisd(start_yaw, Eigen::Vector3d::UnitZ())));
    cmd_timeout_s_ = declare_parameter<double>("cmd_timeout_s", 0.3);

    pub_odom_ = create_publisher<nav_msgs::msg::Odometry>(odom_topic, rclcpp::SensorDataQoS());
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
    plant_.stepWorld(state_, v_cmd, fresh ? cmd_yawspeed_ : 0.0, dt);
    const Eigen::Vector3d pos = state_.posNed();
    const Eigen::Vector3d vel = state_.velNed();
    const Eigen::Quaterniond q(state_.rotNed());   // body FRD -> NED
    const rclcpp::Time stamp = now();

    nav_msgs::msg::Odometry o;
    o.header.stamp = stamp;
    o.header.frame_id = odom_frame_;
    o.child_frame_id = base_frame_;
    o.pose.pose.position.x = pos.x();
    o.pose.pose.position.y = pos.y();
    o.pose.pose.position.z = pos.z();
    o.pose.pose.orientation.w = q.w();
    o.pose.pose.orientation.x = q.x();
    o.pose.pose.orientation.y = q.y();
    o.pose.pose.orientation.z = q.z();
    const Eigen::Vector3d v_body = q.conjugate() * vel;
    o.twist.twist.linear.x = v_body.x();
    o.twist.twist.linear.y = v_body.y();
    o.twist.twist.linear.z = v_body.z();
    const Eigen::Vector3d w_frd(state_.omega_b.x(), -state_.omega_b.y(), -state_.omega_b.z());   // FLU -> FRD
    o.twist.twist.angular.x = w_frd.x();
    o.twist.twist.angular.y = w_frd.y();
    o.twist.twist.angular.z = w_frd.z();
    pub_odom_->publish(o);

    geometry_msgs::msg::PoseStamped ps;
    ps.header = o.header;
    ps.pose = o.pose.pose;
    pub_pose_->publish(ps);

    std::vector<geometry_msgs::msg::TransformStamped> tfs;
    if (publish_map_odom_tf_ && map_frame_ != odom_frame_) {
      geometry_msgs::msg::TransformStamped map_odom;
      map_odom.header.stamp = stamp;
      map_odom.header.frame_id = map_frame_;
      map_odom.child_frame_id = odom_frame_;
      map_odom.transform.rotation.w = 1.0;
      tfs.push_back(map_odom);
    }
    geometry_msgs::msg::TransformStamped odom_base;
    odom_base.header.stamp = stamp;
    odom_base.header.frame_id = odom_frame_;
    odom_base.child_frame_id = base_frame_;
    odom_base.transform.translation.x = pos.x();
    odom_base.transform.translation.y = pos.y();
    odom_base.transform.translation.z = pos.z();
    odom_base.transform.rotation = o.pose.pose.orientation;
    tfs.push_back(odom_base);
    tf_broadcaster_.sendTransform(tfs);
  }

  tf2_ros::TransformBroadcaster tf_broadcaster_;
  std::string map_frame_, odom_frame_, base_frame_;
  bool publish_map_odom_tf_{true};
  maze_planning::QuadPlant plant_;
  maze_planning::QuadState state_;
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
