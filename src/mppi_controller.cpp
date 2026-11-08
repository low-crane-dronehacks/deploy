#include "maze_planning/mppi_controller.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace maze_planning {

MPPIController::MPPIController(const MPPIConfig& cfg,
                               const QuadPlantParams& plant,
                               std::shared_ptr<const VoxelCostmap> costmap)
    : cfg_(cfg), plant_(plant), costmap_(std::move(costmap)) {
  reset();
}

void MPPIController::reset() {
  u_mean_.assign(static_cast<std::size_t>(cfg_.horizon_steps), MPPIControl{});
  flat_noise_.resize(static_cast<std::size_t>(cfg_.num_samples * cfg_.horizon_steps));
  costs_.assign(static_cast<std::size_t>(cfg_.num_samples), 0.0);
  weights_.assign(static_cast<std::size_t>(cfg_.num_samples), 0.0);
  horizon_refs_.resize(static_cast<std::size_t>(cfg_.horizon_steps));
  horizon_safe_.assign(static_cast<std::size_t>(cfg_.horizon_steps), cfg_.safe_distance_m);
  path_frames_.assign(static_cast<std::size_t>(cfg_.horizon_steps), Eigen::Matrix3d::Identity());
  optimal_rollout_pos_.clear();
  optimal_rollout_yaw_.clear();
  candidates_ = MPPICandidateSet{};
  consecutive_stall_ticks_ = 0;
  last_solve_ms_ = 0.0;
  last_min_cost_ = 0.0;
}

void MPPIController::onEstimatorUpdate(
    double stamp_s,
    const Eigen::Vector3d& pos_ned,
    const Eigen::Quaterniond& q_ned_frd,
    const Eigen::Vector3d& vel_ned,
    std::shared_ptr<const isam2_depth::DistanceGrid> distance_grid) {
  std::lock_guard<std::mutex> lock(est_mutex_);
  latest_est_state_.timestamp_s = stamp_s;
  latest_est_state_.pos_odom = pos_ned;
  latest_est_state_.q_odom_base = q_ned_frd;
  latest_est_state_.vel_odom = vel_ned;
  latest_est_state_.tracking_valid = true;
  isam2_grid_ = std::move(distance_grid);
}

void MPPIController::setLocalDistanceGrid(std::shared_ptr<const isam2_depth::DistanceGrid> distance_grid) {
  std::lock_guard<std::mutex> lock(est_mutex_);
  isam2_grid_ = std::move(distance_grid);
}

MPPIControl MPPIController::clampControl(const MPPIControl& u) const {
  MPPIControl c = u;
  const double v_xy = std::hypot(c.vel_cmd_ned.x(), c.vel_cmd_ned.y());
  if (v_xy > cfg_.v_max_xy) {
    const double s = cfg_.v_max_xy / v_xy;
    c.vel_cmd_ned.x() *= s;
    c.vel_cmd_ned.y() *= s;
  }
  c.vel_cmd_ned.z() = std::clamp(c.vel_cmd_ned.z(), -cfg_.v_max_z, cfg_.v_max_z);
  const double v = c.vel_cmd_ned.norm();
  if (v > cfg_.v_max_norm) c.vel_cmd_ned *= cfg_.v_max_norm / v;
  c.yaw_rate_cmd_rad_s = std::clamp(c.yaw_rate_cmd_rad_s, -cfg_.yaw_rate_max, cfg_.yaw_rate_max);
  return c;
}

double MPPIController::queryClearanceOdom(const Eigen::Vector3d& p_odom) const {
  const Eigen::Vector3d p_map = T_map_odom_.apply(p_odom);
  double d = costmap_ ? costmap_->clearance(p_map) : 2.5;
  if (active_grid_ && active_grid_->inside(p_odom)) {
    d = std::min(d, static_cast<double>(active_grid_->distance(p_odom)));
  }
  return d;
}

void MPPIController::smoothControlSequence() {
  const int T = static_cast<int>(u_mean_.size());
  if (T < 3) return;
  std::vector<MPPIControl> filtered = u_mean_;
  for (int t = 1; t < T - 1; ++t) {
    filtered[t].vel_cmd_ned =
        0.22 * u_mean_[t - 1].vel_cmd_ned + 0.56 * u_mean_[t].vel_cmd_ned + 0.22 * u_mean_[t + 1].vel_cmd_ned;
    filtered[t].yaw_rate_cmd_rad_s = 0.22 * u_mean_[t - 1].yaw_rate_cmd_rad_s +
                                     0.56 * u_mean_[t].yaw_rate_cmd_rad_s +
                                     0.22 * u_mean_[t + 1].yaw_rate_cmd_rad_s;
    filtered[t] = clampControl(filtered[t]);
  }
  u_mean_ = std::move(filtered);
}

bool MPPIController::checkCorridorBlockedAhead(
    const VehicleKinematics& state_odom,
    const PathPoint& lookahead_pt,
    Eigen::Vector3d& out_obstacle_map) const {
  const Eigen::Vector3d diff = lookahead_pt.pos - state_odom.pos_odom;
  const double dist = std::hypot(diff.x(), diff.y());
  if (dist < 0.25) return false;

  const Eigen::Vector3d fwd(diff.x() / dist, diff.y() / dist, 0.0);
  const Eigen::Vector3d right(-fwd.y(), fwd.x(), 0.0);

  // Check if centerline ahead has a lethal obstacle AND both left/right lateral detours (<0.68m)
  // have clearance < drone_radius_m (Section 6 FMEA: Pillar Dead-Center narrow corridor block)
  for (double s = 0.35; s <= std::min(1.15, dist); s += 0.16) {
    const Eigen::Vector3d center_odom = state_odom.pos_odom + fwd * s;
    if (queryClearanceOdom(center_odom) < cfg_.lethal_distance_m) {
      bool left_clear = false;
      bool right_clear = false;
      for (double lat = 0.16; lat <= 0.68; lat += 0.05) {
        if (queryClearanceOdom(center_odom - right * lat) >= cfg_.drone_radius_m) {
          left_clear = true;
        }
        if (queryClearanceOdom(center_odom + right * lat) >= cfg_.drone_radius_m) {
          right_clear = true;
        }
      }
      if (!left_clear && !right_clear) {
        out_obstacle_map = T_map_odom_.apply(center_odom);
        return true;
      }
    }
  }
  return false;
}

TrajectorySetpointCmd MPPIController::computeCommand(
    const VehicleKinematics& state_odom,
    const PlannedPath& local_ref_odom) {
  const auto t_start = std::chrono::steady_clock::now();
  TrajectorySetpointCmd cmd{};
  cmd.timestamp_s = state_odom.timestamp_s;
  {
    // The estimator may swap the grid from another thread; sample one consistent grid for this cycle.
    std::lock_guard<std::mutex> lock(est_mutex_);
    active_grid_ = isam2_grid_;
  }

  if (local_ref_odom.empty()) {
    cmd.pos_ned = state_odom.pos_odom;
    cmd.yaw_rad = state_odom.yaw();
    return cmd;
  }

  const int K = cfg_.num_samples;
  const int T = cfg_.horizon_steps;
  if (static_cast<int>(u_mean_.size()) != T) {
    u_mean_.assign(static_cast<std::size_t>(T), MPPIControl{});
  }
  if (static_cast<int>(flat_noise_.size()) != K * T) {
    flat_noise_.resize(static_cast<std::size_t>(K * T));
    costs_.resize(static_cast<std::size_t>(K));
    weights_.resize(static_cast<std::size_t>(K));
    horizon_refs_.resize(static_cast<std::size_t>(T));
  }
  path_frames_.resize(static_cast<std::size_t>(T));

  // 1. Pre-sample O(1) horizon reference targets along local_ref_odom
  const int start_idx = local_ref_odom.findClosestIndex(state_odom.pos_odom, 0);
  const double t0_ref = local_ref_odom.points[start_idx].t_s;
  for (int t = 0; t < T; ++t) {
    horizon_refs_[t] = local_ref_odom.sampleByTime(t0_ref + (t + 1) * cfg_.dt_s);
  }
  horizon_safe_.assign(static_cast<std::size_t>(T), cfg_.safe_distance_m);
  if (cfg_.adaptive_safe_distance) {
    const double floor_safe = cfg_.lethal_distance_m + cfg_.adaptive_safe_margin_m;
    for (int t = 0; t < T; ++t) {
      const double c_ref = queryClearanceOdom(horizon_refs_[t].pos);
      horizon_safe_[t] = std::clamp(c_ref, floor_safe, cfg_.safe_distance_m);
    }
  }

  const double cur_yaw = state_odom.yaw();
  const double cy0 = std::cos(cur_yaw);
  const double sy0 = std::sin(cur_yaw);

  // Path frame per horizon step: along = horizontal reference tangent (from its velocity, else its yaw, else
  // the current heading), cross = along rotated +90 deg about NED down (the right-hand side), down = +z.
  for (int t = 0; t < T; ++t) {
    const PathPoint& ref = horizon_refs_[t];
    Eigen::Vector2d a(ref.vel.x(), ref.vel.y());
    if (a.norm() < 0.05) a = Eigen::Vector2d(std::cos(ref.yaw_rad), std::sin(ref.yaw_rad));
    if (!a.allFinite() || a.norm() < 1e-6) a = Eigen::Vector2d(cy0, sy0);
    a.normalize();
    Eigen::Matrix3d F;
    F.col(0) = Eigen::Vector3d(a.x(), a.y(), 0.0);
    F.col(1) = Eigen::Vector3d(-a.y(), a.x(), 0.0);
    F.col(2) = Eigen::Vector3d(0.0, 0.0, 1.0);
    path_frames_[t] = F;
  }

  // 2. Shift the warm-start mean sequence by 1 step (world frame: still valid after the vehicle yawed) and
  // blend it with the nominal reference-tracking prior. The yaw prior steers toward the reference heading from
  // the heading the mean sequence itself predicts at that step, not from the current heading at every step.
  for (int t = 0; t < T - 1; ++t) {
    u_mean_[t] = u_mean_[t + 1];
  }
  u_mean_[T - 1] = u_mean_[std::max(0, T - 2)];

  const double alpha = std::clamp(cfg_.alpha_prior, 0.0, 1.0);
  double yaw_pred = cur_yaw;
  for (int t = 0; t < T; ++t) {
    const PathPoint& ref = horizon_refs_[t];
    const Eigen::Vector3d pos_err =
        ref.pos - (state_odom.pos_odom + state_odom.vel_odom * ((t + 1) * cfg_.dt_s));
    MPPIControl u_nom{};
    u_nom.vel_cmd_ned = Eigen::Vector3d(ref.vel.x() + 1.45 * pos_err.x(),
                                        ref.vel.y() + 1.45 * pos_err.y(),
                                        ref.vel.z() + 1.80 * pos_err.z());
    u_nom.yaw_rate_cmd_rad_s = 1.8 * wrapPi(ref.yaw_rad - yaw_pred);
    u_nom = clampControl(u_nom);

    u_mean_[t].vel_cmd_ned = (1.0 - alpha) * u_mean_[t].vel_cmd_ned + alpha * u_nom.vel_cmd_ned;
    u_mean_[t].yaw_rate_cmd_rad_s =
        (1.0 - alpha) * u_mean_[t].yaw_rate_cmd_rad_s + alpha * u_nom.yaw_rate_cmd_rad_s;
    u_mean_[t] = clampControl(u_mean_[t]);
    yaw_pred = wrapPi(yaw_pred + u_mean_[t].yaw_rate_cmd_rad_s * cfg_.dt_s);
  }

  // 3. Sample K temporally correlated noise sequences (path frame) & evaluate rollouts through the quad.py
  // plant in parallel
  const QuadState x0 = plant_.fromNed(state_odom.pos_odom, state_odom.vel_odom, state_odom.q_odom_base,
                                      state_odom.omega_body);

  // Candidate rollouts kept for visualization: every `cand_stride`-th sample, so k = 0 (the noise-free
  // mean sequence) is always one of them.
  const int n_cand = std::clamp(cfg_.viz_num_candidates, 0, K);
  const int cand_stride = (n_cand > 0) ? std::max(1, K / n_cand) : 0;
  candidates_.num_candidates = n_cand;
  candidates_.points_per_candidate = T + 1;
  candidates_.positions.resize(static_cast<std::size_t>(n_cand * (T + 1)));
  candidates_.costs.resize(static_cast<std::size_t>(n_cand));

  const double beta = std::clamp(cfg_.noise_corr_beta, 0.0, 0.95);
  const double inv_sqrt = std::sqrt(std::max(1e-4, 1.0 - beta * beta));
  const double cos_fov = std::cos(std::clamp(cfg_.fov_half_angle_rad, 0.0, M_PI));

  #pragma omp parallel
  {
    int tid = 0;
#ifdef _OPENMP
    tid = omp_get_thread_num();
#endif
    std::mt19937 local_rng(static_cast<uint32_t>(
        918273u + 7919u * static_cast<uint32_t>(tid) +
        static_cast<uint32_t>(state_odom.timestamp_s * 1000.0)));
    std::normal_distribution<double> n01(0.0, 1.0);

    #pragma omp for schedule(static)
    for (int k = 0; k < K; ++k) {
      QuadState st = x0;
      MPPIControl prev_u = u_mean_[0];
      const int cand_slot = (cand_stride > 0 && k % cand_stride == 0 && k / cand_stride < n_cand)
                                ? k / cand_stride
                                : -1;
      Eigen::Vector3d* cand_pos = (cand_slot >= 0)
          ? &candidates_.positions[static_cast<std::size_t>(cand_slot * (T + 1))]
          : nullptr;
      if (cand_pos) cand_pos[0] = state_odom.pos_odom;
      double na = 0.0, nc = 0.0, nd = 0.0, yr_n = 0.0;
      double total_cost = 0.0;

      // Bimodal cross-track bias on 35% of samples so MPPI naturally splits left/right of the PATH around
      // dead-center corridor pillars (Section 6 FMEA: Pillar Dead-Center)
      double lateral_split_bias = 0.0;
      if (k >= (13 * K) / 20) {
        lateral_split_bias = ((k & 1) == 0) ? cfg_.lateral_split_m_s : -cfg_.lateral_split_m_s;
      }

      for (int t = 0; t < T; ++t) {
        if (k == 0) {
          na = 0.0; nc = 0.0; nd = 0.0; yr_n = 0.0;
        } else {
          na = beta * na + inv_sqrt * n01(local_rng) * cfg_.sigma_vel.x();
          nc = beta * nc + inv_sqrt * n01(local_rng) * cfg_.sigma_vel.y();
          nd = beta * nd + inv_sqrt * n01(local_rng) * cfg_.sigma_vel.z();
          yr_n = beta * yr_n + inv_sqrt * n01(local_rng) * cfg_.sigma_yaw_rate;
        }

        MPPIControl u_t{};
        u_t.vel_cmd_ned = u_mean_[t].vel_cmd_ned +
                          path_frames_[t] * Eigen::Vector3d(na, nc + lateral_split_bias, nd);
        u_t.yaw_rate_cmd_rad_s = u_mean_[t].yaw_rate_cmd_rad_s + yr_n;
        u_t = clampControl(u_t);

        // Store effective realized perturbation in time-major order (t * K + k) for cache-coherent reduction
        MPPIControl noise_t{};
        noise_t.vel_cmd_ned = u_t.vel_cmd_ned - u_mean_[t].vel_cmd_ned;
        noise_t.yaw_rate_cmd_rad_s = u_t.yaw_rate_cmd_rad_s - u_mean_[t].yaw_rate_cmd_rad_s;
        flat_noise_[static_cast<std::size_t>(t * K + k)] = noise_t;

        plant_.stepWorld(st, u_t.vel_cmd_ned, u_t.yaw_rate_cmd_rad_s, cfg_.dt_s);
        const Eigen::Vector3d pos = st.posNed();
        const Eigen::Vector3d vel = st.velNed();
        const double yaw = st.yawNed();
        if (cand_pos) cand_pos[t + 1] = pos;
        const PathPoint& ref = horizon_refs_[t];

        // Critic 1: PathFollowCritic (3D position tracking error)
        const Eigen::Vector3d pos_err = pos - ref.pos;
        const double path_follow_cost =
            pos_err.x() * pos_err.x() + pos_err.y() * pos_err.y() + 1.8 * pos_err.z() * pos_err.z();
        total_cost += cfg_.w_path_follow * path_follow_cost * cfg_.dt_s;

        // Critic 2: PathAlignCritic (velocity vector alignment)
        const Eigen::Vector3d vel_err = vel - ref.vel;
        total_cost += cfg_.w_path_align * vel_err.squaredNorm() * cfg_.dt_s;

        // Critic 3: ObstaclesCritic (GroundTruthMap static walls + STVL unmodeled beams + iSAM2 grid)
        const double d_obs = queryClearanceOdom(pos);
        if (d_obs <= cfg_.lethal_distance_m) {
          total_cost += cfg_.w_obstacles * (22.0 + 160.0 * (cfg_.lethal_distance_m - d_obs));
        } else if (d_obs < horizon_safe_[t]) {
          const double diff = horizon_safe_[t] - d_obs;
          total_cost += cfg_.w_obstacles * (0.65 * diff * diff +
                        0.12 * (diff * diff * diff) / std::max(0.03, d_obs - cfg_.lethal_distance_m));
        }

        // Altitude & outer-net geofence barrier, in the MAP (arena) frame: rollouts are in odom, whose origin is the
        // launch mark on SITL / the drone. Evaluated in odom, the "net" at 2.65 sat at arena E = 0.135, right in
        // front of gate A (every 2026-11-08 SITL flight stalled there); sim_loopback has map == odom and hid it.
        const Eigen::Vector3d pos_map = T_map_odom_.apply(pos);
        const double alt = -pos_map.z();
        if (alt < 0.20) {
          const double d_alt = 0.20 - alt;
          total_cost += 220.0 * d_alt * d_alt;
        } else if (alt > 1.62) {
          const double d_alt = alt - 1.62;
          total_cost += 220.0 * d_alt * d_alt;
        }
        const double max_xy = std::max(std::abs(pos_map.x()), std::abs(pos_map.y()));
        if (max_xy > 2.65) {
          const double d_net = max_xy - 2.65;
          total_cost += 180.0 * d_net * d_net;
        }

        // Critic 4: OrientationCritic (keep OAK-D-S2 depth camera aligned with corridor tangent)
        const double yaw_err = wrapPi(yaw - ref.yaw_rad);
        total_cost += cfg_.w_orientation * yaw_err * yaw_err * cfg_.dt_s;

        // Critic 5: FovCritic (horizontal motion must stay inside the camera's view of the rollout's heading)
        const double speed_xy = std::hypot(vel.x(), vel.y());
        if (cfg_.w_fov > 0.0 && speed_xy > cfg_.fov_min_speed_m_s) {
          const double c = (vel.x() * std::cos(yaw) + vel.y() * std::sin(yaw)) / speed_xy;
          if (c < cos_fov) {
            const double excess = std::acos(std::clamp(c, -1.0, 1.0)) - cfg_.fov_half_angle_rad;
            total_cost += cfg_.w_fov * excess * excess * speed_xy * cfg_.dt_s;
          }
        }

        // Critic 6: ControlRateCritic (penalize control slew rate)
        if (t > 0) {
          const Eigen::Vector3d du = u_t.vel_cmd_ned - prev_u.vel_cmd_ned;
          const double dyr = u_t.yaw_rate_cmd_rad_s - prev_u.yaw_rate_cmd_rad_s;
          total_cost += cfg_.w_control_rate * (du.squaredNorm() + 0.5 * dyr * dyr) * cfg_.dt_s;
        }
        prev_u = u_t;

        // Critic 7: GoalCritic (terminal horizon progress)
        if (t == T - 1) {
          total_cost += cfg_.w_goal * (pos_err.squaredNorm() + 0.25 * vel_err.squaredNorm());
        }
      }
      costs_[k] = total_cost;
    }
  }

  // 4. Softmax importance weights & expectation update
  const double min_cost = *std::min_element(costs_.begin(), costs_.end());
  if (n_cand > 0) {
    for (int c = 0; c < n_cand; ++c) {
      candidates_.costs[static_cast<std::size_t>(c)] = costs_[static_cast<std::size_t>(c * cand_stride)];
    }
    const auto [lo, hi] = std::minmax_element(candidates_.costs.begin(), candidates_.costs.end());
    candidates_.min_cost = *lo;
    candidates_.max_cost = *hi;
  }
  last_min_cost_ = min_cost;
  const double inv_lambda = 1.0 / std::max(1e-4, cfg_.lambda);

  double weight_sum = 0.0;
  for (int k = 0; k < K; ++k) {
    const double w = std::exp(-inv_lambda * (costs_[k] - min_cost));
    weights_[k] = w;
    weight_sum += w;
  }
  if (weight_sum < 1e-12) weight_sum = 1.0;

  for (int t = 0; t < T; ++t) {
    Eigen::Vector3d delta_v = Eigen::Vector3d::Zero();
    double delta_yr = 0.0;
    const std::size_t row_offset = static_cast<std::size_t>(t * K);
    for (int k = 0; k < K; ++k) {
      const double wk = weights_[k] / weight_sum;
      const MPPIControl& nt = flat_noise_[row_offset + static_cast<std::size_t>(k)];
      delta_v += wk * nt.vel_cmd_ned;
      delta_yr += wk * nt.yaw_rate_cmd_rad_s;
    }
    u_mean_[t].vel_cmd_ned += delta_v;
    u_mean_[t].yaw_rate_cmd_rad_s += delta_yr;
    u_mean_[t] = clampControl(u_mean_[t]);
  }

  smoothControlSequence();

  // 5. Roll out optimal mean trajectory for visualization & feedforward setpoint extraction
  optimal_rollout_pos_.resize(static_cast<std::size_t>(T + 1));
  optimal_rollout_yaw_.resize(static_cast<std::size_t>(T + 1));
  QuadState opt_st = x0;
  optimal_rollout_pos_[0] = state_odom.pos_odom;
  optimal_rollout_yaw_[0] = cur_yaw;
  QuadState step1_st = x0;
  Eigen::Vector3d step1_acc = Eigen::Vector3d::Zero();
  for (int t = 0; t < T; ++t) {
    const Eigen::Vector3d v_before = opt_st.velNed();
    plant_.stepWorld(opt_st, u_mean_[t].vel_cmd_ned, u_mean_[t].yaw_rate_cmd_rad_s, cfg_.dt_s);
    optimal_rollout_pos_[t + 1] = opt_st.posNed();
    optimal_rollout_yaw_[t + 1] = opt_st.yawNed();
    if (t == 0) {
      step1_st = opt_st;
      step1_acc = (opt_st.velNed() - v_before) / cfg_.dt_s;
    }
  }

  // 6. Check FMEA Pillar Dead-Center / Narrow Corridor Blockage
  Eigen::Vector3d blocked_obs_map = Eigen::Vector3d::Zero();
  const PathPoint& target_ref = horizon_refs_[std::min(T - 1, 5)];
  const bool corridor_blocked =
      checkCorridorBlockedAhead(state_odom, target_ref, blocked_obs_map);
  if (corridor_blocked || min_cost > cfg_.stall_cost_threshold) {
    ++consecutive_stall_ticks_;
  } else {
    consecutive_stall_ticks_ = 0;
  }

  if (corridor_blocked || consecutive_stall_ticks_ >= 6) {
    consecutive_stall_ticks_ = 0;
    cmd.obstacle_hold = true;
    cmd.request_global_replan = true;
    cmd.blocked_obstacle_pos = corridor_blocked
        ? blocked_obs_map
        : T_map_odom_.apply(state_odom.pos_odom +
                            Eigen::Vector3d(std::cos(cur_yaw), std::sin(cur_yaw), 0.0) * 0.65);
    cmd.pos_ned = state_odom.pos_odom;
    cmd.vel_ned = Eigen::Vector3d::Zero();
    cmd.acc_ned = Eigen::Vector3d::Zero();
    cmd.yaw_rad = cur_yaw;
    cmd.yawspeed_rad_s = 0.0;
    cmd.body_action = {0.0, 0.0, 0.0, 0.0};
  } else {
    const MPPIControl& u0 = u_mean_[0];
    Eigen::Vector3d v_cmd_odom = u0.vel_cmd_ned;

    // Exponential Control Barrier Function (CBF) near outer arena net walls (|x|, |y| <= 2.86m), in the MAP
    // (arena) frame: position and velocity go odom -> map, are clamped, and the velocity comes back. In odom
    // (origin = launch mark on SITL / the drone) it clamped east velocity from arena E = -0.46 and stopped it past
    // E = 0.34 (gate A), and stopped southward motion past arena N = -1.94 (the window).
    const Eigen::Vector3d p_map = T_map_odom_.apply(state_odom.pos_odom);
    Eigen::Vector3d v_map = T_map_odom_.rotate(v_cmd_odom);
    if (p_map.x() > 2.05) {
      const double v_lim = 2.0 * std::max(0.0, 2.85 - p_map.x());
      if (v_map.x() > v_lim) v_map.x() = v_lim;
    } else if (p_map.x() < -2.05) {
      const double v_lim = -2.0 * std::max(0.0, 2.85 + p_map.x());
      if (v_map.x() < v_lim) v_map.x() = v_lim;
    }
    if (p_map.y() > 2.05) {
      const double v_lim = 2.0 * std::max(0.0, 2.85 - p_map.y());
      if (v_map.y() > v_lim) v_map.y() = v_lim;
    } else if (p_map.y() < -2.05) {
      const double v_lim = -2.0 * std::max(0.0, 2.85 + p_map.y());
      if (v_map.y() < v_lim) v_map.y() = v_lim;
    }
    v_cmd_odom = T_map_odom_.unrotate(v_map);

    cmd.pos_ned = step1_st.posNed();
    cmd.vel_ned = v_cmd_odom;
    cmd.acc_ned = step1_acc;
    cmd.yaw_rad = step1_st.yawNed();
    cmd.yawspeed_rad_s = u0.yaw_rate_cmd_rad_s;

    const double v_fwd_body = cy0 * v_cmd_odom.x() + sy0 * v_cmd_odom.y();
    const double v_right_body = -sy0 * v_cmd_odom.x() + cy0 * v_cmd_odom.y();
    cmd.body_action[0] = std::clamp(v_fwd_body / 5.0, -1.0, 1.0);
    cmd.body_action[1] = std::clamp(v_right_body / 3.0, -1.0, 1.0);
    cmd.body_action[2] = std::clamp(v_cmd_odom.z() / 2.0, -1.0, 1.0);
    cmd.body_action[3] = std::clamp(u0.yaw_rate_cmd_rad_s / (M_PI / 3.0), -1.0, 1.0);
  }

  const auto t_end = std::chrono::steady_clock::now();
  last_solve_ms_ = std::chrono::duration<double, std::milli>(t_end - t_start).count();
  cmd.min_rollout_cost = last_min_cost_;
  cmd.solve_time_ms = last_solve_ms_;
  return cmd;
}

}  // namespace maze_planning
