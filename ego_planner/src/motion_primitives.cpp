#include "ego_planner/motion_primitives.hpp"

#include <algorithm>
#include <cmath>

namespace ego_planner {

// ============================================================================
// HoldPrimitive
// ============================================================================
HoldPrimitive::HoldPrimitive(const Vec3& hold_pos_ned, double hold_yaw_rad, double duration_s)
    : hold_pos_ned_(hold_pos_ned), hold_yaw_rad_(hold_yaw_rad), duration_s_(duration_s) {}

StateSetpoint HoldPrimitive::evaluate(double /*t_rel_s*/, const VehicleState& current_state) const {
  StateSetpoint sp;
  sp.pos_ned = hold_pos_ned_;
  Vec3 err = hold_pos_ned_ - current_state.pos_ned;
  sp.vel_ned = {
      clamp(1.2 * err.x, -1.0, 1.0),
      clamp(1.2 * err.y, -1.0, 1.0),
      clamp(1.2 * err.z, -0.8, 0.8),
  };
  sp.yaw_rad = hold_yaw_rad_;
  double cur_yaw = current_state.q_wxyz.toEulerZYX().z;
  sp.yawspeed_rad_s = clamp(1.5 * wrapPi(hold_yaw_rad_ - cur_yaw), -1.0, 1.0);
  return sp;
}

bool HoldPrimitive::isCompleted(double t_rel_s, const VehicleState& /*current_state*/) const {
  return t_rel_s >= duration_s_;
}

// ============================================================================
// TakeoffPrimitive
// ============================================================================
TakeoffPrimitive::TakeoffPrimitive(const Vec3& start_pos_ned, double target_alt_m,
                                   double start_yaw_rad, double target_yaw_rad,
                                   double duration_s)
    : start_pos_ned_(start_pos_ned),
      target_alt_m_(target_alt_m),
      start_yaw_rad_(start_yaw_rad),
      target_yaw_rad_(target_yaw_rad),
      duration_s_(std::max(0.25, duration_s)) {}

StateSetpoint TakeoffPrimitive::evaluate(double t_rel_s, const VehicleState& current_state) const {
  StateSetpoint sp;
  const double tau = clamp(t_rel_s / duration_s_, 0.0, 1.0);
  // Minimum-jerk profile: s(tau) = 10 tau^3 - 15 tau^4 + 6 tau^5
  const double s = tau * tau * tau * (10.0 + tau * (-15.0 + 6.0 * tau));
  const double ds = (t_rel_s < duration_s_)
                        ? (30.0 * tau * tau * (1.0 - 2.0 * tau + tau * tau)) / duration_s_
                        : 0.0;

  const double z0 = start_pos_ned_.z;
  const double z1 = -target_alt_m_;
  sp.pos_ned = {start_pos_ned_.x, start_pos_ned_.y, z0 + (z1 - z0) * s};

  Vec3 pos_err = sp.pos_ned - current_state.pos_ned;
  // Brisk liftoff below 0.35m to clear ground clamp, then smooth PD settle to target_alt_m_
  double vz_ff = (z1 - z0) * ds;
  double vz_cmd = vz_ff + 2.4 * pos_err.z - 0.45 * current_state.vel_ned.z;
  if (-current_state.pos_ned.z < 0.35) {
    vz_cmd = std::min(vz_cmd, -1.25);  // negative z is UP in NED
  }
  sp.vel_ned = {
      clamp(2.0 * pos_err.x - 0.3 * current_state.vel_ned.x, -1.2, 1.2),
      clamp(2.0 * pos_err.y - 0.3 * current_state.vel_ned.y, -1.2, 1.2),
      clamp(vz_cmd, -1.8, 1.0),
  };

  double dyaw = wrapPi(target_yaw_rad_ - start_yaw_rad_);
  sp.yaw_rad = wrapPi(start_yaw_rad_ + s * dyaw);
  double cur_yaw = current_state.q_wxyz.toEulerZYX().z;
  sp.yawspeed_rad_s = clamp(2.2 * wrapPi(target_yaw_rad_ - cur_yaw), -1.047, 1.047);
  return sp;
}

bool TakeoffPrimitive::isCompleted(double t_rel_s, const VehicleState& current_state) const {
  double alt = -current_state.pos_ned.z;
  double yaw_err = std::fabs(wrapPi(target_yaw_rad_ - current_state.q_wxyz.toEulerZYX().z));
  if (alt >= target_alt_m_ - 0.12 && yaw_err < 0.45 && t_rel_s >= 0.50) return true;
  if (alt >= 0.40 && t_rel_s >= duration_s_ + 0.55) return true;
  return false;
}

// ============================================================================
// UniformBSplinePrimitive
// ============================================================================
UniformBSplinePrimitive::UniformBSplinePrimitive(const std::vector<Vec3>& control_points,
                                                 double knot_dt, double yaw_lookahead_s)
    : yaw_lookahead_s_(yaw_lookahead_s) {
  setControlPoints(control_points, knot_dt);
}

void UniformBSplinePrimitive::setControlPoints(const std::vector<Vec3>& control_points, double knot_dt) {
  ctrl_pts_ = control_points;
  knot_dt_ = std::max(0.02, knot_dt);
  int segs = std::max(0, static_cast<int>(ctrl_pts_.size()) - 3);
  total_duration_s_ = segs * knot_dt_;
}

void UniformBSplinePrimitive::evaluatePVA(double t_rel_s, Vec3& pos, Vec3& vel, Vec3& acc) const {
  const int n = static_cast<int>(ctrl_pts_.size());
  if (n == 0) {
    pos = vel = acc = {0.0, 0.0, 0.0};
    return;
  }
  if (n < 4) {
    pos = ctrl_pts_.back();
    vel = acc = {0.0, 0.0, 0.0};
    return;
  }

  const int num_segs = n - 3;
  const double tc = clamp(t_rel_s, 0.0, total_duration_s_);
  double s = tc / knot_dt_;
  int seg = std::clamp(static_cast<int>(std::floor(s)), 0, num_segs - 1);
  double u = clamp(s - seg, 0.0, 1.0);
  double u2 = u * u;
  double u3 = u2 * u;

  // Cubic uniform B-spline basis functions
  const double b0 = (1.0 - 3.0 * u + 3.0 * u2 - u3) / 6.0;
  const double b1 = (4.0 - 6.0 * u2 + 3.0 * u3) / 6.0;
  const double b2 = (1.0 + 3.0 * u + 3.0 * u2 - 3.0 * u3) / 6.0;
  const double b3 = u3 / 6.0;

  // First derivative wrt u
  const double db0 = (-3.0 + 6.0 * u - 3.0 * u2) / 6.0;
  const double db1 = (-12.0 * u + 9.0 * u2) / 6.0;
  const double db2 = (3.0 + 6.0 * u - 9.0 * u2) / 6.0;
  const double db3 = (3.0 * u2) / 6.0;

  // Second derivative wrt u
  const double ddb0 = (6.0 - 6.0 * u) / 6.0;
  const double ddb1 = (-12.0 + 18.0 * u) / 6.0;
  const double ddb2 = (6.0 - 18.0 * u) / 6.0;
  const double ddb3 = (6.0 * u) / 6.0;

  const Vec3& q0 = ctrl_pts_[seg];
  const Vec3& q1 = ctrl_pts_[seg + 1];
  const Vec3& q2 = ctrl_pts_[seg + 2];
  const Vec3& q3 = ctrl_pts_[seg + 3];

  pos = q0 * b0 + q1 * b1 + q2 * b2 + q3 * b3;
  const double inv_dt = 1.0 / knot_dt_;
  const double inv_dt2 = inv_dt * inv_dt;
  vel = (q0 * db0 + q1 * db1 + q2 * db2 + q3 * db3) * inv_dt;
  acc = (q0 * ddb0 + q1 * ddb1 + q2 * ddb2 + q3 * ddb3) * inv_dt2;
}

StateSetpoint UniformBSplinePrimitive::evaluate(double t_rel_s, const VehicleState& current_state) const {
  StateSetpoint sp;
  // Evaluate reference position/velocity with small lookahead for smooth tracking
  const double t_ref = std::min(total_duration_s_, t_rel_s + 0.08);
  evaluatePVA(t_ref, sp.pos_ned, sp.vel_ned, sp.acc_ned);

  // Compute lookahead tangent for camera-aligned yaw
  Vec3 p_la, v_la, a_la;
  const double t_yaw = std::min(total_duration_s_, t_rel_s + yaw_lookahead_s_);
  evaluatePVA(t_yaw, p_la, v_la, a_la);

  Vec2 dir_xy = (p_la.xy() - current_state.pos_ned.xy());
  double cur_yaw = current_state.q_wxyz.toEulerZYX().z;
  if (dir_xy.norm() > 0.18) {
    sp.yaw_rad = std::atan2(dir_xy.y, dir_xy.x);
  } else if (v_la.xy().norm() > 0.20) {
    sp.yaw_rad = std::atan2(v_la.y, v_la.x);
  } else {
    sp.yaw_rad = cur_yaw;
  }
  sp.yawspeed_rad_s = clamp(2.1 * wrapPi(sp.yaw_rad - cur_yaw), -1.047, 1.047);
  return sp;
}

bool UniformBSplinePrimitive::isCompleted(double t_rel_s, const VehicleState& current_state) const {
  if (ctrl_pts_.empty()) return true;
  if (t_rel_s >= total_duration_s_) return true;
  Vec3 goal = ctrl_pts_.back();
  return (t_rel_s >= total_duration_s_ - 0.15) && ((goal - current_state.pos_ned).norm() < 0.22);
}

UniformBSplinePrimitive UniformBSplinePrimitive::fromWaypoints(const std::vector<Vec3>& waypoints,
                                                               const Vec3& init_vel_ned,
                                                               const Vec3& end_vel_ned,
                                                               double desired_speed_ms,
                                                               double knot_dt) {
  if (waypoints.empty()) {
    return UniformBSplinePrimitive({}, knot_dt);
  }
  if (waypoints.size() == 1) {
    std::vector<Vec3> q(6, waypoints[0]);
    return UniformBSplinePrimitive(q, knot_dt);
  }

  // Compute curvature-aware local speed along each polyline segment so the B-spline
  // naturally decelerates into tight 90-deg turns and accelerates on straightaways.
  const std::size_t m = waypoints.size();
  std::vector<double> vertex_speed_scale(m, 1.0);
  for (std::size_t i = 1; i + 1 < m; ++i) {
    Vec2 d0 = (waypoints[i] - waypoints[i - 1]).xy().normalized();
    Vec2 d1 = (waypoints[i + 1] - waypoints[i]).xy().normalized();
    double cos_ang = clamp(d0.dot(d1), -1.0, 1.0);
    double turn_rad = std::acos(cos_ang);
    double scale = clamp(1.0 - 0.65 * (turn_rad / (0.5 * kPi)), 0.36, 1.0);
    vertex_speed_scale[i] = std::min(vertex_speed_scale[i], scale);
    vertex_speed_scale[i - 1] = std::min(vertex_speed_scale[i - 1], std::max(0.42, scale));
    vertex_speed_scale[i + 1] = std::min(vertex_speed_scale[i + 1], std::max(0.48, scale));
  }
  vertex_speed_scale.back() = 0.35;

  std::vector<Vec3> samples;
  samples.push_back(waypoints.front());
  for (std::size_t i = 0; i + 1 < m; ++i) {
    Vec3 a = waypoints[i];
    Vec3 b = waypoints[i + 1];
    double seg_len = (b - a).norm();
    if (seg_len < 1e-4) continue;
    double seg_speed = desired_speed_ms * 0.5 * (vertex_speed_scale[i] + vertex_speed_scale[i + 1]);
    double step_m = std::max(0.06, seg_speed * knot_dt);
    int n_sub = std::max(1, static_cast<int>(std::ceil(seg_len / step_m)));
    for (int k = 1; k <= n_sub; ++k) {
      double alpha = static_cast<double>(k) / static_cast<double>(n_sub);
      samples.push_back(a + (b - a) * alpha);
    }
  }

  // Construct clamped control points Q:
  // At u=0 on segment 0: p(0) = (q0 + 4 q1 + q2)/6 = p_start, v(0) = (q2 - q0)/(2 dt) = v_init, a(0) = (q0 - 2 q1 + q2)/dt^2 = 0
  // -> q1 = p_start, q0 = p_start - v_init * dt, q2 = p_start + v_init * dt
  std::vector<Vec3> q;
  q.reserve(samples.size() + 2);
  const Vec3 p_start = samples.front();
  const Vec3 p_end = samples.back();
  q.push_back(p_start - init_vel_ned * knot_dt);
  for (const auto& pt : samples) {
    q.push_back(pt);
  }
  q.push_back(p_end + end_vel_ned * knot_dt);

  return UniformBSplinePrimitive(q, knot_dt);
}

// ============================================================================
// GatePassPrimitive
// ============================================================================
GatePassPrimitive::GatePassPrimitive(const GateSpec& gate, const Vec3& refined_center_ned,
                                     double pass_speed_ms, double post_margin_m)
    : gate_(gate),
      center_ned_(refined_center_ned),
      pass_speed_ms_(pass_speed_ms),
      post_margin_m_(post_margin_m) {}

StateSetpoint GatePassPrimitive::evaluate(double /*t_rel_s*/, const VehicleState& current_state) const {
  StateSetpoint sp;
  const Vec2 post_xy = center_ned_.xy() + gate_.normal_ned * post_margin_m_;
  const double target_z = clamp(center_ned_.z, -gate_.alt_max_m, -gate_.alt_min_m);
  sp.pos_ned = {post_xy.x, post_xy.y, target_z};

  Vec3 err = sp.pos_ned - current_state.pos_ned;
  Vec2 dir_xy = err.xy().normalized();
  double v_xy = std::min(pass_speed_ms_, std::max(1.2, 2.2 * err.xy().norm()));
  sp.vel_ned = {
      dir_xy.x * v_xy,
      dir_xy.y * v_xy,
      clamp(2.4 * err.z, -1.2, 1.2),
  };

  sp.yaw_rad = std::atan2(gate_.normal_ned.y, gate_.normal_ned.x);
  double cur_yaw = current_state.q_wxyz.toEulerZYX().z;
  sp.yawspeed_rad_s = clamp(2.0 * wrapPi(sp.yaw_rad - cur_yaw), -1.047, 1.047);
  return sp;
}

bool GatePassPrimitive::isCompleted(double t_rel_s, const VehicleState& current_state) const {
  Vec2 rel = current_state.pos_ned.xy() - gate_.A_ned;
  double d = rel.dot(gate_.normal_ned);
  double u = rel.dot(gate_.tangent_ned);
  bool past_line = (d >= 0.19) && (u >= -0.05) && (u <= gate_.length_m + 0.05);
  return past_line || (t_rel_s >= 2.5);
}

// ============================================================================
// PrecisionLandPrimitive
// ============================================================================
PrecisionLandPrimitive::PrecisionLandPrimitive(const Vec2& target_xy_ned, double capture_radius_m,
                                               double approach_alt_m, double descent_rate_ms,
                                               double touchdown_alt_m)
    : target_xy_ned_(target_xy_ned),
      capture_radius_m_(capture_radius_m),
      approach_alt_m_(approach_alt_m),
      descent_rate_ms_(descent_rate_ms),
      touchdown_alt_m_(touchdown_alt_m) {}

StateSetpoint PrecisionLandPrimitive::evaluate(double /*t_rel_s*/, const VehicleState& current_state) const {
  StateSetpoint sp;
  const Vec2 err_xy = target_xy_ned_ - current_state.pos_ned.xy();
  const double dist_xy = err_xy.norm();
  const double cur_alt = -current_state.pos_ned.z;
  const double speed_xy = current_state.vel_ned.xy().norm();

  // Horizontal centering with velocity damping so touchdown horizontal speed is well below 1.0 m/s
  const double kp_xy = (dist_xy > capture_radius_m_) ? 1.8 : 1.5;
  const double max_vxy = (cur_alt < 0.30) ? 0.45 : ((dist_xy > capture_radius_m_) ? 1.8 : 0.65);
  Vec2 v_xy = err_xy * kp_xy - current_state.vel_ned.xy() * 0.35;
  if (v_xy.norm() > max_vxy) {
    v_xy = v_xy.normalized() * max_vxy;
  }

  double vz_cmd = 0.0;
  if (dist_xy > capture_radius_m_ || speed_xy > 0.72) {
    // Hold/settle at approach altitude until centered and slow over the pad
    double target_z = -std::max(approach_alt_m_, 0.35);
    vz_cmd = clamp(1.8 * (target_z - current_state.pos_ned.z), -0.8, 0.50);
    sp.pos_ned = {current_state.pos_ned.x, current_state.pos_ned.y, current_state.pos_ned.z};
  } else {
    // Inside pad capture cylinder and slow: descend gently at 0.55 m/s so |vz| < 1.0 m/s is guaranteed
    vz_cmd = std::min(descent_rate_ms_, 0.58);  // +z is DOWN in NED
    sp.pos_ned = {current_state.pos_ned.x, current_state.pos_ned.y, current_state.pos_ned.z};
  }

  sp.vel_ned = {v_xy.x, v_xy.y, vz_cmd};
  double cur_yaw = current_state.q_wxyz.toEulerZYX().z;
  sp.yaw_rad = (dist_xy > 0.30) ? std::atan2(err_xy.y, err_xy.x) : cur_yaw;
  sp.yawspeed_rad_s = clamp(1.5 * wrapPi(sp.yaw_rad - cur_yaw), -0.8, 0.8);
  return sp;
}

bool PrecisionLandPrimitive::isCompleted(double /*t_rel_s*/, const VehicleState& current_state) const {
  double dist_xy = (target_xy_ned_ - current_state.pos_ned.xy()).norm();
  double alt = -current_state.pos_ned.z;
  return (dist_xy <= capture_radius_m_ + 0.25) && (alt <= touchdown_alt_m_ || current_state.landed);
}

}  // namespace ego_planner
