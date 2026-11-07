#pragma once

#include <array>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>

namespace maze_planning {

inline double wrapPi(double a) {
  while (a > M_PI) a -= 2.0 * M_PI;
  while (a < -M_PI) a += 2.0 * M_PI;
  return a;
}

/**
 * @brief 3D Rigid Transform (e.g. T_map_odom or T_odom_base).
 */
struct SE3Transform {
  Eigen::Quaterniond q{Eigen::Quaterniond::Identity()};
  Eigen::Vector3d t{Eigen::Vector3d::Zero()};

  Eigen::Vector3d apply(const Eigen::Vector3d& p) const { return q * p + t; }
  Eigen::Vector3d rotate(const Eigen::Vector3d& v) const { return q * v; }
  Eigen::Vector3d unrotate(const Eigen::Vector3d& v) const { return q.conjugate() * v; }

  SE3Transform inverse() const {
    SE3Transform inv;
    inv.q = q.conjugate();
    inv.t = -(inv.q * t);
    return inv;
  }

  SE3Transform operator*(const SE3Transform& rhs) const {
    SE3Transform out;
    out.q = (q * rhs.q).normalized();
    out.t = q * rhs.t + t;
    return out;
  }

  double yaw() const {
    const Eigen::Matrix3d R = q.toRotationMatrix();
    return std::atan2(R(1, 0), R(0, 0));
  }
};

/**
 * @brief High-rate local vehicle kinematics in the continuous odom frame (and body frame B).
 */
struct VehicleKinematics {
  double timestamp_s{0.0};
  Eigen::Vector3d pos_odom{Eigen::Vector3d::Zero()};
  Eigen::Vector3d vel_odom{Eigen::Vector3d::Zero()};
  Eigen::Quaterniond q_odom_base{Eigen::Quaterniond::Identity()};
  Eigen::Vector3d omega_body{Eigen::Vector3d::Zero()};
  double pos_variance{0.01};
  bool tracking_valid{true};

  double yaw() const {
    const Eigen::Matrix3d R = q_odom_base.toRotationMatrix();
    return std::atan2(R(1, 0), R(0, 0));
  }

  Eigen::Vector3d velBody() const { return q_odom_base.conjugate() * vel_odom; }
  Eigen::Vector3d projectedGravityBody() const {
    return q_odom_base.conjugate() * Eigen::Vector3d(0.0, 0.0, -1.0);
  }
};

/**
 * @brief Gate / Window aperture specification in the maze map frame.
 */
struct GateAperture {
  int id{0};
  std::string name;
  Eigen::Vector3d center_map{Eigen::Vector3d::Zero()};
  Eigen::Vector3d normal_map{Eigen::Vector3d(1.0, 0.0, 0.0)};
  Eigen::Vector2d post_a_xy{Eigen::Vector2d::Zero()};
  Eigen::Vector2d post_b_xy{Eigen::Vector2d::Zero()};
  double width_m{1.0};
  double z_min_map{-1.50};
  double z_max_map{-0.20};
  bool is_window{false};
};

/**
 * @brief Discrete / spline-evaluated point along a planned reference path.
 */
struct PathPoint {
  Eigen::Vector3d pos{Eigen::Vector3d::Zero()};
  Eigen::Vector3d vel{Eigen::Vector3d::Zero()};
  Eigen::Vector3d acc{Eigen::Vector3d::Zero()};
  double yaw_rad{0.0};
  double yawspeed_rad_s{0.0};
  double s_m{0.0};
  double t_s{0.0};
};

/**
 * @brief Global or local reference path with arc-length parameterization.
 */
struct PlannedPath {
  std::string frame_id{"map"};
  double stamp_s{0.0};
  std::vector<PathPoint> points;
  double total_length_m{0.0};
  double total_duration_s{0.0};

  bool empty() const { return points.empty(); }
  PathPoint sampleByTime(double t_s) const;
  PathPoint sampleByArcLength(double s_m) const;
  int findClosestIndex(const Eigen::Vector3d& pos, int min_idx = 0) const;
};

/**
 * @brief Classical MPPI output command (/control/trajectory_setpoint -> PX4 TrajectorySetpoint).
 */
struct TrajectorySetpointCmd {
  double timestamp_s{0.0};
  Eigen::Vector3d pos_ned{Eigen::Vector3d::Zero()};
  Eigen::Vector3d vel_ned{Eigen::Vector3d::Zero()};
  Eigen::Vector3d acc_ned{Eigen::Vector3d::Zero()};
  double yaw_rad{0.0};
  double yawspeed_rad_s{0.0};
  std::array<double, 4> body_action{0.0, 0.0, 0.0, 0.0}; // [fwd, right, down, yaw_rate] normalized

  // FMEA Pillar Dead-Center / Stall detection flags
  bool obstacle_hold{false};
  bool request_global_replan{false};
  Eigen::Vector3d blocked_obstacle_pos{Eigen::Vector3d::Zero()};
  double min_rollout_cost{0.0};
  double solve_time_ms{0.0};
};

/**
 * @brief RL Policy output command (/control/rates_setpoint -> PX4 VehicleRatesSetpoint).
 */
struct RatesSetpointCmd {
  double timestamp_s{0.0};
  Eigen::Vector3d omega_body_rad_s{Eigen::Vector3d::Zero()}; // [wx, wy, wz] rad/s
  double thrust_norm{0.5};                                   // normalized collective thrust [0.1, 0.9]
  double inference_latency_ms{0.0};
};

}  // namespace maze_planning
