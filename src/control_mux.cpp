#include "maze_planning/control_mux.hpp"

#include <cmath>

namespace maze_planning {

ControlMux::ControlMux(const ControlMuxConfig& cfg) : cfg_(cfg) {}

void ControlMux::onClassicalCommand(const TrajectorySetpointCmd& cmd) {
  last_classical_cmd_ = cmd;
  has_classical_ = true;
}

void ControlMux::onRLRatesCommand(const RatesSetpointCmd& cmd) {
  last_rl_cmd_ = cmd;
  has_rl_ = true;
}

bool ControlMux::isRLCommandValid(double now_s, std::string& out_reason) const {
  if (!has_rl_) {
    out_reason = "rl_no_command";
    return false;
  }
  if ((now_s - last_rl_cmd_.timestamp_s) > cfg_.max_rl_command_age_s) {
    out_reason = "rl_command_stale";
    return false;
  }
  if (last_rl_cmd_.inference_latency_ms > cfg_.max_rl_latency_ms) {
    out_reason = "rl_latency_exceeded_20ms";
    return false;
  }
  if (!last_rl_cmd_.omega_body_rad_s.allFinite() || !std::isfinite(last_rl_cmd_.thrust_norm)) {
    out_reason = "rl_nan_output";
    return false;
  }
  if (std::abs(last_rl_cmd_.omega_body_rad_s.x()) > cfg_.max_roll_pitch_rate_rad_s ||
      std::abs(last_rl_cmd_.omega_body_rad_s.y()) > cfg_.max_roll_pitch_rate_rad_s ||
      std::abs(last_rl_cmd_.omega_body_rad_s.z()) > cfg_.max_yaw_rate_rad_s) {
    out_reason = "rl_rates_envelope_violation";
    return false;
  }
  if (last_rl_cmd_.thrust_norm < cfg_.min_thrust_norm ||
      last_rl_cmd_.thrust_norm > cfg_.max_thrust_norm) {
    out_reason = "rl_thrust_envelope_violation";
    return false;
  }
  out_reason = "rl_policy_ok";
  return true;
}

MuxDecision ControlMux::evaluate(double now_s,
                                 const VehicleKinematics& state_odom,
                                 double current_clearance_m) {
  MuxDecision out{};

  // 1. Check VIO tracking health & covariance watchdog (WP-4 Task 4 & FMEA VIO Tracking Loss)
  if (!state_odom.tracking_valid || state_odom.pos_variance > cfg_.max_pos_variance) {
    out.active_pathway = ControlPathway::kSafetyHold;
    out.publish_trajectory_setpoint = true;
    out.publish_rates_setpoint = false;
    out.fallback_tripped = true;
    out.status_reason = "vio_covariance_hold";
    out.trajectory_cmd.timestamp_s = now_s;
    out.trajectory_cmd.pos_ned = state_odom.pos_odom;
    out.trajectory_cmd.vel_ned = Eigen::Vector3d::Zero();
    out.trajectory_cmd.acc_ned = Eigen::Vector3d::Zero();
    out.trajectory_cmd.yaw_rad = state_odom.yaw();
    return out;
  }

  // 2. Check imminent collision watchdog (< 0.3s time-to-impact along velocity vector, WP-4 Task 4)
  const double speed = state_odom.vel_odom.norm();
  if (speed > 0.35 && current_clearance_m < 0.21 + speed * cfg_.collision_lookahead_s &&
      current_clearance_m < 0.22) {
    out.active_pathway = ControlPathway::kSafetyHold;
    out.publish_trajectory_setpoint = true;
    out.publish_rates_setpoint = false;
    out.fallback_tripped = true;
    out.status_reason = "imminent_collision_hold";
    out.trajectory_cmd.timestamp_s = now_s;
    out.trajectory_cmd.pos_ned = state_odom.pos_odom;
    out.trajectory_cmd.vel_ned = Eigen::Vector3d::Zero();
    out.trajectory_cmd.acc_ned = Eigen::Vector3d::Zero();
    out.trajectory_cmd.yaw_rad = state_odom.yaw();
    return out;
  }

  // 3. Evaluate requested pathway
  if (requested_pathway_ == ControlPathway::kRLPolicy) {
    std::string rl_reason;
    if (isRLCommandValid(now_s, rl_reason)) {
      out.active_pathway = ControlPathway::kRLPolicy;
      out.publish_trajectory_setpoint = false;
      out.publish_rates_setpoint = true;
      out.rates_cmd = last_rl_cmd_;
      out.status_reason = rl_reason;
      return out;
    }
    // WP-5 Task 3 & FMEA: Trip watchdog and switch authority back to Classical MPPI / Hold
    out.fallback_tripped = true;
    out.status_reason = "rl_fallback_to_mppi:" + rl_reason;
  }

  out.active_pathway = ControlPathway::kClassicalMPPI;
  out.publish_trajectory_setpoint = true;
  out.publish_rates_setpoint = false;
  if (has_classical_) {
    out.trajectory_cmd = last_classical_cmd_;
    if (!out.fallback_tripped) {
      out.status_reason = last_classical_cmd_.obstacle_hold ? "mppi_obstacle_hold" : "classical_mppi";
    }
  } else {
    out.trajectory_cmd.timestamp_s = now_s;
    out.trajectory_cmd.pos_ned = state_odom.pos_odom;
    out.trajectory_cmd.yaw_rad = state_odom.yaw();
  }
  return out;
}

}  // namespace maze_planning
