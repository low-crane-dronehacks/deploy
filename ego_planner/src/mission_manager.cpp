#include "ego_planner/mission_manager.hpp"

#include <algorithm>
#include <cmath>

namespace ego_planner {

namespace {

bool pointInPoly(const Vec2& pt, const std::vector<Vec2>& poly) {
  bool inside = false;
  const std::size_t n = poly.size();
  for (std::size_t i = 0, j = n - 1; i < n; j = i++) {
    const double xi = poly[i].x, yi = poly[i].y;
    const double xj = poly[j].x, yj = poly[j].y;
    const bool intersect = ((yi > pt.y) != (yj > pt.y)) &&
                           (pt.x < (xj - xi) * (pt.y - yi) / ((yj - yi) + 1e-12) + xi);
    if (intersect) inside = !inside;
  }
  return inside;
}

double yawFromState(const VehicleState& st) { return st.q_wxyz.toEulerZYX().z; }

}  // namespace

RunCourseLeg::RunCourseLeg(const MissionConfig& cfg, const std::vector<GateSpec>& gates)
    : cfg_(cfg), gates_(gates) {}

int RunCourseLeg::detectSectorCheckpoint(const Vec2& xy_ned) {
  static const std::vector<Vec2> kSec0 = {
      {2.58, -3.05}, {2.58, -1.97}, {-1.83, -1.51}, {-1.83, -3.05}};
  static const std::vector<Vec2> kSec1 = {
      {3.05, -3.05}, {3.05, 0.305}, {0.42, 0.305},   {0.42, -0.61},  {1.97, -0.61},
      {1.97, -1.10}, {-0.61, -0.84}, {-0.61, -3.05}, {-1.83, -3.05}, {-1.83, -1.51},
      {2.58, -1.97}, {2.58, -3.05}};
  static const std::vector<Vec2> kSec2 = {
      {3.05, 0.305}, {3.05, 3.05}, {-0.305, 3.05}, {-0.305, 1.97}, {0.42, 1.97}, {0.42, 0.305}};
  static const std::vector<Vec2> kSec3 = {
      {-0.305, 3.05}, {-3.05, 3.05},  {-3.05, 1.49}, {-1.83, 1.49}, {-1.83, 0.57},
      {-3.05, 0.57},  {-3.05, -0.20}, {-0.13, -0.84}, {1.97, -1.10}, {1.97, -0.61},
      {0.42, -0.61},  {0.42, 1.97},   {-0.305, 1.97}};

  if (pointInPoly(xy_ned, kSec0)) return 0;
  if (pointInPoly(xy_ned, kSec1)) return 1;
  if (pointInPoly(xy_ned, kSec2)) return 2;
  if (pointInPoly(xy_ned, kSec3)) return 3;
  return 4;
}

void RunCourseLeg::reset(const VehicleState& state, const GridMap& grid) {
  leg_done_ = false;
  has_prev_xy_ = true;
  prev_xy_ = state.pos_ned.xy();
  last_replan_t_ = -1e9;
  last_grid_updates_ = -1;
  prim_start_t_ = -1.0;

  const Vec2 start_xy = state.pos_ned.xy();
  next_cp_ = detectSectorCheckpoint(start_xy);

  if (next_cp_ == 0) {
    waypoint_stage_ = 0;
  } else if (next_cp_ == 1) {
    waypoint_stage_ = 4;
  } else if (next_cp_ == 2) {
    waypoint_stage_ = 9;
  } else if (next_cp_ == 3) {
    waypoint_stage_ = 15;
  } else {
    waypoint_stage_ = 21;
  }

  const double alt = -state.pos_ned.z;
  const double cur_yaw = yawFromState(state);
  if (alt < 0.45 && next_cp_ == 0 && (start_xy - cfg_.launch_mark_ned).norm() < 0.65) {
    sub_phase_ = SubPhase::kTakeoff;
    active_primitive_ = std::make_shared<TakeoffPrimitive>(
        state.pos_ned, cfg_.cruise_alt_m, cur_yaw, 0.0, 0.90);
  } else if (next_cp_ >= 4 && (start_xy - cfg_.pad_center_ned).norm() < 0.70) {
    sub_phase_ = SubPhase::kLandOnPad;
    active_primitive_ = std::make_shared<PrecisionLandPrimitive>(cfg_.pad_center_ned);
  } else {
    sub_phase_ = SubPhase::kNavigateGates;
    replanBSpline(0.0, state, grid, BSplineOptimizer{});
  }
}

void RunCourseLeg::checkGateCrossings(const Vec2& prev_xy, const Vec2& cur_xy) {
  if (next_cp_ >= 4) return;

  auto testGate = [&](const GateSpec& g) -> bool {
    const Vec2 rel_cur = cur_xy - g.A_ned;
    const Vec2 rel_prev = prev_xy - g.A_ned;
    const double d_cur = rel_cur.dot(g.normal_ned) - cfg_.cp_margin_m;
    const double d_prev = rel_prev.dot(g.normal_ned) - cfg_.cp_margin_m;
    const double u_cur = rel_cur.dot(g.tangent_ned);
    return (d_prev < 0.0 && d_cur >= 0.0 && u_cur >= -0.10 && u_cur <= g.length_m + 0.10);
  };

  if (next_cp_ == 0 && gates_.size() > 0 && testGate(gates_[0])) {
    next_cp_ = 1;
    waypoint_stage_ = std::max(waypoint_stage_, 3);
  }
  if (next_cp_ == 1 && gates_.size() > 1 && testGate(gates_[1])) {
    next_cp_ = 2;
    waypoint_stage_ = std::max(waypoint_stage_, 9);
  }
  if (next_cp_ == 2 && gates_.size() > 2 && testGate(gates_[2])) {
    next_cp_ = 3;
    waypoint_stage_ = std::max(waypoint_stage_, 14);
  }
  if (next_cp_ == 3 && gates_.size() > 4) {
    if (testGate(gates_[3]) || testGate(gates_[4])) {
      next_cp_ = 4;
      waypoint_stage_ = std::max(waypoint_stage_, 20);
    }
  }
}

std::vector<Vec3> RunCourseLeg::buildRemainingAnchors(const VehicleState& /*state*/,
                                                      const GridMap& grid) const {
  const double z_cruise = -cfg_.cruise_alt_m;

  const Vec3 cp1_center =
      (gates_.size() > 1) ? grid.refineGateCenter(gates_[1], 0.24) : Vec3{2.5146, 0.3048, z_cruise};
  const Vec3 cp2_center =
      (gates_.size() > 2) ? grid.refineGateCenter(gates_[2], 0.24) : Vec3{-0.3048, 2.2400, -0.80};
  const GateSpec& cp3_gate =
      (cfg_.prefer_window && gates_.size() > 4) ? gates_[4] : gates_[std::min<std::size_t>(3, gates_.size() - 1)];
  const Vec3 cp3_center = grid.refineGateCenter(cp3_gate, 0.22);

  const Vec2 n1 = (gates_.size() > 1) ? gates_[1].normal_ned : Vec2{0.0, 1.0};
  const Vec2 n2 = (gates_.size() > 2) ? gates_[2].normal_ned : Vec2{-1.0, 0.0};
  const Vec2 n3 = cp3_gate.normal_ned;

  const std::vector<Vec3> all_anchors = {
      Vec3{1.85, -2.50, z_cruise},
      Vec3{2.45, -2.45, z_cruise},
      Vec3{2.77, -2.25, z_cruise},
      Vec3{2.75, -1.72, z_cruise},
      Vec3{2.35, -1.42, z_cruise},
      Vec3{1.05, -1.38, z_cruise},
      Vec3{1.05, -0.45, z_cruise},
      Vec3{cp1_center.x, cp1_center.y - 0.35 * n1.y, z_cruise},
      Vec3{cp1_center.x, cp1_center.y + 0.35 * n1.y, z_cruise},
      Vec3{2.32, 1.50, z_cruise},
      Vec3{2.05, 2.25, -0.79},
      Vec3{1.15, 2.28, -0.80},
      Vec3{0.25, 2.28, -0.80},
      Vec3{cp2_center.x + 0.30 * n2.x, 2.24, -0.80},
      Vec3{-1.20, 1.85, z_cruise},
      Vec3{-1.75, 1.40, z_cruise},
      Vec3{-1.85, 0.55, z_cruise},
      Vec3{cp3_center.x - 0.35 * n3.x, cp3_center.y - 0.35 * n3.y, -0.78},
      cp3_center,
      Vec3{cp3_center.x + 0.38 * n3.x, cp3_center.y + 0.38 * n3.y, -0.78},
      Vec3{-1.45, -1.35, -0.72},
      Vec3{cfg_.pad_center_ned.x, cfg_.pad_center_ned.y, -0.50},
  };

  std::vector<Vec3> rem;
  const int start_idx = std::clamp(waypoint_stage_, 0, static_cast<int>(all_anchors.size()) - 1);
  for (std::size_t i = static_cast<std::size_t>(start_idx); i < all_anchors.size(); ++i) {
    Vec3 a = all_anchors[i];
    const bool is_gate_anchor = (i == 2 || i == 3 || i == 8 || i == 13 || i == 18 || i == 19);
    if (!is_gate_anchor) {
      Vec3 grad;
      const double d = grid.getDistanceAndGradient(a, grad);
      if (d < 0.30) {
        a = a + grad * (0.32 - d);
      }
    }
    rem.push_back(a);
  }
  return rem;
}

void RunCourseLeg::replanBSpline(double t_s, const VehicleState& state, const GridMap& grid,
                                 const BSplineOptimizer& opt) {
  const std::vector<Vec3> anchors = buildRemainingAnchors(state, grid);
  Vec3 goal_vel{0.0, 0.0, 0.0};
  active_spline_ = opt.planAndOptimizeTrajectory(grid, state.pos_ned, state.vel_ned, anchors,
                                                 goal_vel, cfg_.desired_speed_ms);
  prim_start_t_ = 0.0;  // Used as monotonic spline cursor t_cursor_s in kNavigateGates
  last_replan_t_ = t_s;
  last_grid_updates_ = grid.liveUpdatesCount();
  active_primitive_ = std::make_shared<UniformBSplinePrimitive>(active_spline_);
}

StateSetpoint RunCourseLeg::step(double t_s, double dt_s, const VehicleState& state,
                                 const GridMap& grid, const BSplineOptimizer& opt) {
  (void)dt_s;
  if (prim_start_t_ < 0.0) {
    prim_start_t_ = t_s;
  }
  const Vec2 cur_xy = state.pos_ned.xy();
  if (has_prev_xy_) {
    checkGateCrossings(prev_xy_, cur_xy);
  }
  prev_xy_ = cur_xy;
  has_prev_xy_ = true;

  const std::vector<Vec3> rem_anchors = buildRemainingAnchors(state, grid);
  if (!rem_anchors.empty()) {
    const double dist_first = (state.pos_ned.xy() - rem_anchors.front().xy()).norm();
    const bool gated_wait =
        (waypoint_stage_ == 2 && next_cp_ < 1) ||
        (waypoint_stage_ == 8 && next_cp_ < 2) ||
        (waypoint_stage_ == 13 && next_cp_ < 3) ||
        (waypoint_stage_ == 19 && next_cp_ < 4);
    if (!gated_wait && dist_first < 0.38 && waypoint_stage_ < 21) {
      waypoint_stage_++;
    }
  }

  if (sub_phase_ == SubPhase::kTakeoff) {
    const double t_rel = t_s - prim_start_t_;
    if (active_primitive_ && !active_primitive_->isCompleted(t_rel, state)) {
      return active_primitive_->evaluate(t_rel, state);
    }
    sub_phase_ = SubPhase::kNavigateGates;
    replanBSpline(t_s, state, grid, opt);
  }

  if (sub_phase_ == SubPhase::kNavigateGates) {
    const double dist_to_pad = (cur_xy - cfg_.pad_center_ned).norm();
    if (next_cp_ >= 4 && dist_to_pad < 0.45) {
      sub_phase_ = SubPhase::kLandOnPad;
      prim_start_t_ = t_s;
      active_primitive_ = std::make_shared<PrecisionLandPrimitive>(cfg_.pad_center_ned);
    } else {
      if (last_replan_t_ < -1e8 || active_spline_.duration() <= 0.05) {
        replanBSpline(t_s, state, grid, opt);
      }

      // Check if live depth updates revealed an obstacle along the upcoming 1.4s of the spline
      if (grid.liveUpdatesCount() != last_grid_updates_ && (t_s - last_replan_t_) >= cfg_.replan_period_s) {
        bool path_unsafe = false;
        const double t_cur = prim_start_t_;
        const double t_end = std::min(active_spline_.duration(), t_cur + 1.40);
        for (double tc = t_cur + 0.05; tc <= t_end; tc += 0.08) {
          Vec3 p_c, v_c, a_c;
          active_spline_.evaluatePVA(tc, p_c, v_c, a_c);
          if (grid.getDistance(p_c) < 0.22) {
            path_unsafe = true;
            break;
          }
        }
        last_grid_updates_ = grid.liveUpdatesCount();
        if (path_unsafe) {
          replanBSpline(t_s, state, grid, opt);
        }
      }

      // Project current vehicle position onto the B-spline in a forward window [t_cursor, t_cursor + 1.2s]
      const double dur = active_spline_.duration();
      const double t_search_lo = std::clamp(prim_start_t_, 0.0, dur);
      const double t_search_hi = std::min(dur, t_search_lo + 1.20);
      double best_t = t_search_lo;
      double best_d2 = 1e18;
      for (double tc = t_search_lo; tc <= t_search_hi; tc += 0.02) {
        Vec3 p_c, v_c, a_c;
        active_spline_.evaluatePVA(tc, p_c, v_c, a_c);
        const double d2 = (p_c - state.pos_ned).normSq();
        if (d2 < best_d2) {
          best_d2 = d2;
          best_t = tc;
        }
      }
      prim_start_t_ = std::max(prim_start_t_, best_t);

      // Evaluate lookahead setpoint on the B-spline
      StateSetpoint sp = active_spline_.evaluate(prim_start_t_ + 0.14, state);

      // Reactive + predictive ESDF safety push if vehicle or 0.18s lookahead approaches any wall/obstacle
      Vec3 grad;
      const double d_obs = grid.getDistanceAndGradient(state.pos_ned, grad);
      if (d_obs < 0.34) {
        const double push = 3.0 * (0.34 - d_obs);
        sp.vel_ned = sp.vel_ned + grad * push;
        sp.pos_ned = sp.pos_ned + grad * (0.34 - d_obs);
      }
      const Vec3 p_pred = state.pos_ned + state.vel_ned * 0.18;
      Vec3 grad_pred;
      const double d_pred = grid.getDistanceAndGradient(p_pred, grad_pred);
      if (d_pred < 0.32) {
        sp.vel_ned = sp.vel_ned + grad_pred * (2.6 * (0.32 - d_pred));
      }
      return sp;
    }
  }

  if (sub_phase_ == SubPhase::kLandOnPad) {
    const double t_rel = t_s - prim_start_t_;
    if (active_primitive_ && active_primitive_->isCompleted(t_rel, state)) {
      sub_phase_ = SubPhase::kFinished;
      leg_done_ = true;
    }
    if (active_primitive_) {
      return active_primitive_->evaluate(t_rel, state);
    }
  }

  StateSetpoint done_sp;
  done_sp.pos_ned = Vec3{cfg_.pad_center_ned.x, cfg_.pad_center_ned.y, 0.0};
  done_sp.vel_ned = Vec3{0.0, 0.0, 0.35};
  done_sp.yaw_rad = yawFromState(state);
  return done_sp;
}

// ============================================================================
// ReturnHomeLeg
// ============================================================================

ReturnHomeLeg::ReturnHomeLeg(const MissionConfig& cfg) : cfg_(cfg) {}

void ReturnHomeLeg::reset(const VehicleState& state, const GridMap& grid) {
  (void)grid;
  sub_phase_ = SubPhase::kPadHold;
  phase_start_t_ = 0.0;
  last_replan_t_ = -1e9;
  leg_done_ = false;
  active_primitive_ = std::make_shared<HoldPrimitive>(
      state.pos_ned, yawFromState(state), cfg_.return_hold_s);
}

StateSetpoint ReturnHomeLeg::step(double t_s, double dt_s, const VehicleState& state,
                                  const GridMap& grid, const BSplineOptimizer& opt) {
  (void)dt_s;
  if (phase_start_t_ == 0.0) phase_start_t_ = t_s;

  if (sub_phase_ == SubPhase::kPadHold) {
    if ((t_s - phase_start_t_) >= cfg_.return_hold_s) {
      sub_phase_ = SubPhase::kClimbAndReturn;
      phase_start_t_ = t_s;
      last_replan_t_ = -1e9;
    } else {
      return active_primitive_->evaluate(t_s - phase_start_t_, state);
    }
  }

  if (sub_phase_ == SubPhase::kClimbAndReturn) {
    const double dist_mark = (state.pos_ned.xy() - cfg_.launch_mark_ned).norm();
    if (dist_mark < 0.38) {
      sub_phase_ = SubPhase::kLandAtMark;
      phase_start_t_ = t_s;
      active_primitive_ = std::make_shared<PrecisionLandPrimitive>(cfg_.launch_mark_ned);
    } else {
      if ((t_s - last_replan_t_) >= cfg_.replan_period_s) {
        std::vector<Vec3> anchors = {
            Vec3{state.pos_ned.x, state.pos_ned.y, -2.05},
            Vec3{0.05, -2.28, -2.05},
            Vec3{cfg_.launch_mark_ned.x, cfg_.launch_mark_ned.y, -1.95},
            Vec3{cfg_.launch_mark_ned.x, cfg_.launch_mark_ned.y, -0.60},
        };
        active_spline_ = opt.planAndOptimizeTrajectory(grid, state.pos_ned, state.vel_ned, anchors,
                                                       Vec3{0.0, 0.0, 0.0}, 2.5);
        last_replan_t_ = t_s;
      }
      return active_spline_.evaluate((t_s - last_replan_t_) + 0.08, state);
    }
  }

  if (sub_phase_ == SubPhase::kLandAtMark) {
    const double t_rel = t_s - phase_start_t_;
    if (active_primitive_ && active_primitive_->isCompleted(t_rel, state)) {
      sub_phase_ = SubPhase::kFinished;
      leg_done_ = true;
    }
    if (active_primitive_) {
      return active_primitive_->evaluate(t_rel, state);
    }
  }

  StateSetpoint sp;
  sp.pos_ned = Vec3{cfg_.launch_mark_ned.x, cfg_.launch_mark_ned.y, 0.0};
  sp.vel_ned = Vec3{0.0, 0.0, 0.30};
  return sp;
}

// ============================================================================
// MissionManager
// ============================================================================

std::vector<GateSpec> MissionManager::defaultGates() {
  GateSpec g0{};
  g0.index = 0;
  g0.name = "corridor_end";
  g0.checkpoint = 0;
  g0.is_window = false;
  g0.A_ned = Vec2{2.58445, -3.0480};
  g0.B_ned = Vec2{2.58445, -1.97485};
  g0.tangent_ned = Vec2{0.0, 1.0};
  g0.normal_ned = Vec2{1.0, 0.0};
  g0.length_m = 1.07315;
  g0.center_ned = Vec3{2.58445, -2.5114, -0.78};
  g0.alt_min_m = 0.45;
  g0.alt_max_m = 1.35;
  g0.alt_nominal_m = 0.78;

  GateSpec g1{};
  g1.index = 1;
  g1.name = "gate_a";
  g1.checkpoint = 1;
  g1.is_window = false;
  g1.A_ned = Vec2{1.97485, 0.3048};
  g1.B_ned = Vec2{3.0480, 0.3048};
  g1.tangent_ned = Vec2{1.0, 0.0};
  g1.normal_ned = Vec2{0.0, 1.0};
  g1.length_m = 1.07315;
  g1.center_ned = Vec3{2.5114, 0.3048, -0.78};
  g1.alt_min_m = 0.45;
  g1.alt_max_m = 1.35;
  g1.alt_nominal_m = 0.78;

  GateSpec g2{};
  g2.index = 2;
  g2.name = "tunnel_exit";
  g2.checkpoint = 2;
  g2.is_window = false;
  g2.A_ned = Vec2{-0.3048, 3.0480};
  g2.B_ned = Vec2{-0.3048, 1.97485};
  g2.tangent_ned = Vec2{0.0, -1.0};
  g2.normal_ned = Vec2{-1.0, 0.0};
  g2.length_m = 1.07315;
  g2.center_ned = Vec3{-0.3048, 2.2400, -0.80};
  g2.alt_min_m = 0.45;
  g2.alt_max_m = 1.25;
  g2.alt_nominal_m = 0.80;

  GateSpec g3{};
  g3.index = 3;
  g3.name = "gate_b";
  g3.checkpoint = 3;
  g3.is_window = false;
  g3.A_ned = Vec2{-0.13335, -0.8382};
  g3.B_ned = Vec2{-1.9812, -0.4318};
  g3.tangent_ned = Vec2{-0.976642, 0.214875};
  g3.normal_ned = Vec2{-0.214875, -0.976642};
  g3.length_m = 1.892074;
  g3.center_ned = Vec3{-1.0573, -0.6350, -0.78};
  g3.alt_min_m = 0.45;
  g3.alt_max_m = 1.35;
  g3.alt_nominal_m = 0.78;

  GateSpec g4{};
  g4.index = 4;
  g4.name = "window";
  g4.checkpoint = 3;
  g4.is_window = true;
  g4.A_ned = Vec2{-1.9812, -0.4318};
  g4.B_ned = Vec2{-3.0480, -0.19685};
  g4.tangent_ned = Vec2{-0.976621, 0.214969};
  g4.normal_ned = Vec2{0.34202, -0.939693};
  g4.length_m = 1.092346;
  g4.center_ned = Vec3{-2.2830, -0.4750, -0.78};
  g4.alt_min_m = 0.62;
  g4.alt_max_m = 0.92;
  g4.alt_nominal_m = 0.78;

  return {g0, g1, g2, g3, g4};
}

MissionManager::MissionManager(const MissionConfig& cfg)
    : cfg_(cfg), gates_(defaultGates()) {
  if (!cfg_.prior_bin_path.empty()) {
    grid_.loadPriorBinary(cfg_.prior_bin_path);
  }
  reset();
}

void MissionManager::reset() {
  grid_.resetToPrior();
  legs_.clear();
  legs_.push_back(std::make_unique<RunCourseLeg>(cfg_, gates_));
  if (cfg_.enable_return_leg) {
    legs_.push_back(std::make_unique<ReturnHomeLeg>(cfg_));
  }
  active_leg_idx_ = 0;
  phase_ = Phase::kWait;
  why_ = "starting";
  result_.clear();
  t_phase_start_ = -1.0;
  last_step_t_ = -1.0;
  depth_t_ = -1e9;
  health_t_ = -1e9;
  healthy_ = false;
  airborne_ = false;
  leg_initialized_ = false;
  home_ned_ = Vec3{0.0, 0.0, 0.0};
  arena_offset_ned_ = Vec3{0.0, 0.0, 0.0};
  events_.clear();
}

void MissionManager::setPhase(double t_s, Phase new_phase, const std::string& why) {
  if (new_phase == phase_ && why == why_) return;
  if (phase_ == Phase::kLand && new_phase == Phase::kDone && result_.empty()) {
    result_ = why_;
  }
  phase_ = new_phase;
  why_ = why;
  t_phase_start_ = t_s;
  const char* pstr = "WAIT";
  if (phase_ == Phase::kArm) pstr = "ARM";
  if (phase_ == Phase::kFly) pstr = "FLY";
  if (phase_ == Phase::kLand) pstr = "LAND";
  if (phase_ == Phase::kDone) pstr = "DONE";
  events_.push_back(std::string(pstr) + ":" + why);
}

void MissionManager::onDepthTimestamp(double t_s) { depth_t_ = t_s; }

void MissionManager::onDepthFrame(double t_s, const float* depth_norm, int width, int height,
                                  double hfov_deg, double vfov_deg, double far_clip,
                                  const Vec3& pos_ned, const Quat& q_wxyz, int stride) {
  depth_t_ = t_s;
  grid_.updateFromDepth(depth_norm, width, height, hfov_deg, vfov_deg, far_clip,
                        pos_ned + arena_offset_ned_, q_wxyz, stride);
}

void MissionManager::onEstimatorHealth(double t_s, bool healthy) {
  health_t_ = t_s;
  healthy_ = healthy;
}

std::array<double, 4> MissionManager::velocityToBodyAction(
    const Vec3& vel_ned, double yawspeed_rad_s, const Quat& q_wxyz) const {
  const double yaw = q_wxyz.toEulerZYX().z;
  const double cy = std::cos(yaw);
  const double sy = std::sin(yaw);
  const double v_fwd = cy * vel_ned.x + sy * vel_ned.y;
  const double v_right = -sy * vel_ned.x + cy * vel_ned.y;
  const double v_down = vel_ned.z;

  const double a_fwd = std::clamp(v_fwd / cfg_.v_max[0], -1.0, 1.0);
  const double a_right = std::clamp(v_right / cfg_.v_max[1], -1.0, 1.0);
  const double a_down = std::clamp(v_down / cfg_.v_max[2], -1.0, 1.0);
  const double a_yaw = std::clamp(yawspeed_rad_s / cfg_.yaw_rate_max, -1.0, 1.0);
  return {a_fwd, a_right, a_down, a_yaw};
}

ControlOutput MissionManager::step(double t_s, const VehicleState& state) {
  if (t_phase_start_ < 0.0) {
    t_phase_start_ = t_s;
  }
  const double dt_s = (last_step_t_ >= 0.0) ? std::max(1e-3, t_s - last_step_t_) : cfg_.control_dt_s;
  last_step_t_ = t_s;

  const double dt_phase = t_s - t_phase_start_;
  const double depth_age = t_s - depth_t_;
  const double health_age = t_s - health_t_;
  const bool depth_fresh = depth_age <= cfg_.depth_stale_s;
  const bool health_fresh = healthy_ && (health_age <= cfg_.health_bad_land_s);
  const bool is_offboard = (state.nav_state == 14);

  ControlOutput out{};
  out.body_action = {0.0, 0.0, 0.0, 0.0};

  if (phase_ == Phase::kDone) {
    out.mission_phase = "DONE";
    out.why = why_;
    out.result = result_;
    return out;
  }

  if (phase_ == Phase::kLand) {
    if (!state.armed || state.landed) {
      setPhase(t_s, Phase::kDone, "landed");
    } else {
      out.command_flags |= kCmdLand;
    }
    out.mission_phase = (phase_ == Phase::kDone) ? "DONE" : "LAND";
    out.why = why_;
    out.result = result_;
    return out;
  }

  if (phase_ == Phase::kWait) {
    out.stream_velocity = true;
    out.vel_ned = Vec3{0.0, 0.0, 0.0};
    out.yawspeed_rad_s = 0.0;
    if (!depth_fresh) {
      setPhase(t_s, Phase::kWait, "no_depth");
    } else if (!state.pos_valid || !health_fresh) {
      setPhase(t_s, Phase::kWait, "estimator_unhealthy");
    } else if (!cfg_.arm_authorized) {
      setPhase(t_s, Phase::kWait, "ready_arm_disabled");
    } else if (dt_phase < cfg_.stream_lead_s) {
      setPhase(t_s, Phase::kWait, "streaming_setpoints");
    } else {
      home_ned_ = state.pos_ned;
      setPhase(t_s, Phase::kArm, "requesting_offboard_and_arm");
    }
    out.mission_phase = (phase_ == Phase::kArm) ? "ARM" : "WAIT";
    out.why = why_;
    return out;
  }

  if (phase_ == Phase::kArm) {
    out.stream_velocity = true;
    out.vel_ned = Vec3{0.0, 0.0, 0.0};
    out.yawspeed_rad_s = 0.0;
    if (!depth_fresh || !state.pos_valid || !health_fresh) {
      setPhase(t_s, Phase::kWait, "preconditions_lost");
      out.mission_phase = "WAIT";
      out.why = why_;
      return out;
    }
    if (state.armed && is_offboard) {
      home_ned_ = state.pos_ned;
      if ((state.pos_ned.xy() - cfg_.launch_mark_ned).norm() > 0.45 &&
          state.pos_ned.xy().norm() < 0.40) {
        arena_offset_ned_ = Vec3{cfg_.launch_mark_ned.x - state.pos_ned.x,
                                 cfg_.launch_mark_ned.y - state.pos_ned.y,
                                 -0.08 - state.pos_ned.z};
      } else {
        arena_offset_ned_ = Vec3{0.0, 0.0, 0.0};
      }
      airborne_ = false;
      leg_initialized_ = false;
      setPhase(t_s, Phase::kFly, "flying");
    } else if (dt_phase > cfg_.arm_wait_s) {
      setPhase(t_s, Phase::kDone, "arm_timeout");
      out.mission_phase = "DONE";
      out.why = why_;
      return out;
    } else {
      if (!is_offboard) out.command_flags |= kCmdOffboard;
      if (!state.armed) out.command_flags |= kCmdArm;
      out.mission_phase = "ARM";
      out.why = why_;
      return out;
    }
  }

  // Phase::kFly
  if (!state.armed && airborne_) {
    setPhase(t_s, Phase::kDone, "disarmed_in_flight");
    out.mission_phase = "DONE";
    out.why = why_;
    return out;
  }
  if (!is_offboard) {
    setPhase(t_s, Phase::kLand, "offboard_lost_pilot_takeover");
    out.command_flags |= kCmdLand;
    out.mission_phase = "LAND";
    out.why = why_;
    return out;
  }
  if (!state.pos_valid || !health_fresh) {
    setPhase(t_s, Phase::kLand, "estimator_unhealthy");
    out.command_flags |= kCmdLand;
    out.mission_phase = "LAND";
    out.why = why_;
    return out;
  }
  if (depth_age > cfg_.depth_lost_land_s) {
    setPhase(t_s, Phase::kLand, "depth_lost");
    out.command_flags |= kCmdLand;
    out.mission_phase = "LAND";
    out.why = why_;
    return out;
  }
  if (dt_phase > cfg_.max_flight_s) {
    setPhase(t_s, Phase::kLand, "flight_timeout");
    out.command_flags |= kCmdLand;
    out.mission_phase = "LAND";
    out.why = why_;
    return out;
  }

  const Vec3 rel = state.pos_ned - home_ned_;
  const double dist_home = std::hypot(rel.x, rel.y);
  const double alt_home = -rel.z;
  if (alt_home > cfg_.airborne_alt_m || -state.pos_ned.z > cfg_.airborne_alt_m) {
    airborne_ = true;
  }
  if (dist_home > cfg_.geofence_m || -state.pos_ned.z > cfg_.geofence_m) {
    setPhase(t_s, Phase::kLand, "geofence");
    out.command_flags |= kCmdLand;
    out.mission_phase = "LAND";
    out.why = why_;
    return out;
  }

  out.stream_velocity = true;
  if (!depth_fresh) {
    out.vel_ned = Vec3{0.0, 0.0, 0.0};
    out.yawspeed_rad_s = 0.0;
    out.body_action = {0.0, 0.0, 0.0, 0.0};
    out.mission_phase = "FLY";
    out.why = "depth_stale_hold";
    return out;
  }

  if (active_leg_idx_ >= legs_.size()) {
    setPhase(t_s, Phase::kLand, "policy_land");
    out.command_flags |= kCmdLand;
    out.mission_phase = "LAND";
    out.why = why_;
    return out;
  }

  VehicleState arena_state = state;
  arena_state.pos_ned = state.pos_ned + arena_offset_ned_;

  MissionLeg& leg = *legs_[active_leg_idx_];
  if (!leg_initialized_) {
    leg.reset(arena_state, grid_);
    leg_initialized_ = true;
  }

  const StateSetpoint sp = leg.step(t_s, dt_s, arena_state, grid_, optimizer_);
  out.active_leg = leg.legName();
  out.active_primitive = leg.activePrimitiveName();
  out.next_checkpoint = leg.nextCheckpointIndex();

  if (leg.isFinished(arena_state)) {
    active_leg_idx_++;
    leg_initialized_ = false;
    if (active_leg_idx_ >= legs_.size()) {
      setPhase(t_s, Phase::kLand, "policy_land");
      out.command_flags |= kCmdLand;
      out.mission_phase = "LAND";
      out.why = why_;
      out.vel_ned = Vec3{0.0, 0.0, 0.35};
      out.body_action = velocityToBodyAction(out.vel_ned, 0.0, state.q_wxyz);
      return out;
    }
  }

  const Vec3 pos_err = sp.pos_ned - arena_state.pos_ned;
  const Vec3 vel_err = sp.vel_ned - state.vel_ned;
  const double kp_xy = 2.35;
  const double kd_xy = 0.55;
  const double ka_xy = 0.32;  // 1 / K_vel feedforward acceleration gain
  const double kp_z = 2.50;
  const double kd_z = 0.45;
  Vec3 cmd_vel{
      sp.vel_ned.x + kp_xy * pos_err.x + kd_xy * vel_err.x + ka_xy * sp.acc_ned.x,
      sp.vel_ned.y + kp_xy * pos_err.y + kd_xy * vel_err.y + ka_xy * sp.acc_ned.y,
      sp.vel_ned.z + kp_z * pos_err.z + kd_z * vel_err.z + 0.25 * sp.acc_ned.z,
  };

  // Exponential Control Barrier Function (CBF) near outer net walls (|N|, |E| <= 2.86m)
  if (arena_state.pos_ned.x > 2.05) {
    const double v_lim = 2.0 * std::max(0.0, 2.85 - arena_state.pos_ned.x);
    if (cmd_vel.x > v_lim || state.vel_ned.x > v_lim) {
      cmd_vel.x = std::min(cmd_vel.x, v_lim - 0.50 * std::max(0.0, state.vel_ned.x - v_lim));
    }
  } else if (arena_state.pos_ned.x < -2.05) {
    const double v_lim = -2.0 * std::max(0.0, 2.85 + arena_state.pos_ned.x);
    if (cmd_vel.x < v_lim || state.vel_ned.x < v_lim) {
      cmd_vel.x = std::max(cmd_vel.x, v_lim - 0.50 * std::min(0.0, state.vel_ned.x - v_lim));
    }
  }
  if (arena_state.pos_ned.y > 2.05) {
    const double v_lim = 2.0 * std::max(0.0, 2.85 - arena_state.pos_ned.y);
    if (cmd_vel.y > v_lim || state.vel_ned.y > v_lim) {
      cmd_vel.y = std::min(cmd_vel.y, v_lim - 0.50 * std::max(0.0, state.vel_ned.y - v_lim));
    }
  } else if (arena_state.pos_ned.y < -2.05) {
    const double v_lim = -2.0 * std::max(0.0, 2.85 + arena_state.pos_ned.y);
    if (cmd_vel.y < v_lim || state.vel_ned.y < v_lim) {
      cmd_vel.y = std::max(cmd_vel.y, v_lim - 0.50 * std::min(0.0, state.vel_ned.y - v_lim));
    }
  }

  const double cur_yaw = yawFromState(state);
  const double yaw_err = wrapPi(sp.yaw_rad - cur_yaw);
  const double fov_scale =
      std::clamp(1.0 - 0.35 * std::max(0.0, std::abs(yaw_err) - 0.45), 0.45, 1.0);
  cmd_vel.x *= fov_scale;
  cmd_vel.y *= fov_scale;

  const double v_xy = std::hypot(cmd_vel.x, cmd_vel.y);
  const double max_xy = cfg_.v_max[0];
  if (v_xy > max_xy && v_xy > 1e-6) {
    cmd_vel.x *= max_xy / v_xy;
    cmd_vel.y *= max_xy / v_xy;
  }
  cmd_vel.z = std::clamp(cmd_vel.z, -cfg_.v_max[2], cfg_.v_max[2]);

  const double cmd_yawspeed =
      std::clamp(sp.yawspeed_rad_s + 2.2 * yaw_err, -cfg_.yaw_rate_max, cfg_.yaw_rate_max);

  out.pos_ned = sp.pos_ned;
  out.vel_ned = cmd_vel;
  out.acc_ned = sp.acc_ned;
  out.yaw_rad = sp.yaw_rad;
  out.yawspeed_rad_s = cmd_yawspeed;
  out.body_action = velocityToBodyAction(cmd_vel, cmd_yawspeed, state.q_wxyz);
  out.mission_phase = "FLY";
  out.why = why_;
  out.result = result_;
  return out;
}

}  // namespace ego_planner
