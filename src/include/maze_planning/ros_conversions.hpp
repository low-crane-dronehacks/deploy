#pragma once
// ROS <-> maze_planning conversions shared by every node (header-only; ROS nodes only, never the core lib).
//
// Path timing convention (the global / local reference contract, docs/AUTONOMY_ARCHITECTURE.md):
//   poses[i].header.stamp = path.header.stamp + t_i   (time from the start of the path)
// A planner that knows its timing (a trajectory optimizer) writes it there and the consumers keep it. A plain
// geometric path (all pose stamps equal, or zero) is timed at a constant fallback speed instead.

#include <algorithm>
#include <cmath>
#include <string>

#include <geometry_msgs/msg/transform_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <rclcpp/rclcpp.hpp>
#include <tf2_ros/buffer.h>

#include "maze_planning/types.hpp"

namespace maze_planning {
namespace ros_conv {

inline double yawFromQuat(double w, double x, double y, double z) {
  return std::atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z));
}

inline nav_msgs::msg::Path toRosPath(const PlannedPath& p, const std::string& frame_id,
                                     const rclcpp::Time& t0) {
  nav_msgs::msg::Path msg;
  msg.header.stamp = t0;
  msg.header.frame_id = frame_id;
  msg.poses.reserve(p.points.size());
  for (const auto& pt : p.points) {
    geometry_msgs::msg::PoseStamped ps;
    ps.header.frame_id = frame_id;
    ps.header.stamp = t0 + rclcpp::Duration::from_seconds(std::max(0.0, pt.t_s));
    ps.pose.position.x = pt.pos.x();
    ps.pose.position.y = pt.pos.y();
    ps.pose.position.z = pt.pos.z();
    ps.pose.orientation.w = std::cos(0.5 * pt.yaw_rad);
    ps.pose.orientation.z = std::sin(0.5 * pt.yaw_rad);
    msg.poses.push_back(ps);
  }
  return msg;
}

// Parse a nav_msgs/Path. Uses per-pose stamps for t_s / velocity when they increase along the path,
// otherwise times it at fallback_speed_m_s.
inline PlannedPath fromRosPath(const nav_msgs::msg::Path& m, const std::string& default_frame,
                               double fallback_speed_m_s) {
  PlannedPath p;
  p.frame_id = m.header.frame_id.empty() ? default_frame : m.header.frame_id;
  p.stamp_s = rclcpp::Time(m.header.stamp).seconds();
  const std::size_t n = m.poses.size();
  p.points.reserve(n);

  bool timed = n >= 2;
  if (timed) {
    const rclcpp::Time t_first(m.poses.front().header.stamp);
    rclcpp::Time t_prev = t_first;
    for (std::size_t i = 1; i < n && timed; ++i) {
      const rclcpp::Time t_i(m.poses[i].header.stamp);
      if (t_i < t_prev) timed = false;   // stamps must not go backwards
      t_prev = t_i;
    }
    timed = timed && (t_prev - t_first).seconds() > 1e-3;
  }
  const double speed = std::max(0.2, fallback_speed_m_s);
  const rclcpp::Time t_base = n ? rclcpp::Time(m.poses.front().header.stamp) : rclcpp::Time(m.header.stamp);

  double s = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    const auto& pos = m.poses[i].pose.position;
    const auto& q = m.poses[i].pose.orientation;
    PathPoint pt{};
    pt.pos = Eigen::Vector3d(pos.x, pos.y, pos.z);
    pt.yaw_rad = yawFromQuat(q.w, q.x, q.y, q.z);
    if (i > 0) s += (pt.pos - p.points.back().pos).norm();
    pt.s_m = s;
    pt.t_s = timed ? (rclcpp::Time(m.poses[i].header.stamp) - t_base).seconds() : s / speed;
    p.points.push_back(pt);
  }
  // Velocity by finite differences over the chosen timing (forward, last copies the previous one).
  for (std::size_t i = 0; i + 1 < n; ++i) {
    const double dt = p.points[i + 1].t_s - p.points[i].t_s;
    if (dt > 1e-6) p.points[i].vel = (p.points[i + 1].pos - p.points[i].pos) / dt;
  }
  if (n >= 2) p.points.back().vel = p.points[n - 2].vel;
  p.total_length_m = s;
  p.total_duration_s = n ? p.points.back().t_s : 0.0;
  return p;
}

// nav_msgs/Odometry (pose in header.frame_id, twist in child_frame_id per REP-105) -> VehicleKinematics.
inline VehicleKinematics fromOdometry(const nav_msgs::msg::Odometry& m) {
  VehicleKinematics k{};
  k.timestamp_s = rclcpp::Time(m.header.stamp).seconds();
  k.pos_odom = Eigen::Vector3d(m.pose.pose.position.x, m.pose.pose.position.y, m.pose.pose.position.z);
  k.q_odom_base = Eigen::Quaterniond(m.pose.pose.orientation.w, m.pose.pose.orientation.x,
                                     m.pose.pose.orientation.y, m.pose.pose.orientation.z).normalized();
  const Eigen::Vector3d v_body(m.twist.twist.linear.x, m.twist.twist.linear.y, m.twist.twist.linear.z);
  k.vel_odom = k.q_odom_base * v_body;
  k.omega_body = Eigen::Vector3d(m.twist.twist.angular.x, m.twist.twist.angular.y, m.twist.twist.angular.z);
  k.pos_variance = std::max({m.pose.covariance[0], m.pose.covariance[7], m.pose.covariance[14]});
  k.tracking_valid = k.pos_odom.allFinite() && k.vel_odom.allFinite();
  return k;
}

inline SE3Transform fromTransformMsg(const geometry_msgs::msg::Transform& t) {
  SE3Transform out;
  out.t = Eigen::Vector3d(t.translation.x, t.translation.y, t.translation.z);
  out.q = Eigen::Quaterniond(t.rotation.w, t.rotation.x, t.rotation.y, t.rotation.z).normalized();
  return out;
}

// T_target_source from TF (latest). Returns false (and leaves `out` alone) when TF has no such transform.
inline bool lookupSE3(const tf2_ros::Buffer& buf, const std::string& target, const std::string& source,
                      SE3Transform& out) {
  try {
    out = fromTransformMsg(buf.lookupTransform(target, source, tf2::TimePointZero).transform);
    return true;
  } catch (const std::exception&) {
    return false;
  }
}

}  // namespace ros_conv
}  // namespace maze_planning
