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
                               const MultirotorPlantParams& plant,
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
  optimal_rollout_pos_.clear();
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
  c.vel_cmd_yaw_aligned.x() =
      std::clamp(c.vel_cmd_yaw_aligned.x(), -plant_.v_max_xy, plant_.v_max_xy);
  c.vel_cmd_yaw_aligned.y() =
      std::clamp(c.vel_cmd_yaw_aligned.y(), -plant_.v_max_xy, plant_.v_max_xy);
  const double v_xy = std::hypot(c.vel_cmd_yaw_aligned.x(), c.vel_cmd_yaw_aligned.y());
  if (v_xy > plant_.v_max_xy) {
    const double s = plant_.v_max_xy / v_xy;
    c.vel_cmd_yaw_aligned.x() *= s;
    c.vel_cmd_yaw_aligned.y() *= s;
  }
  c.vel_cmd_yaw_aligned.z() =
      std::clamp(c.vel_cmd_yaw_aligned.z(), -plant_.v_max_z, plant_.v_max_z);
  c.yaw_rate_cmd_rad_s =
      std::clamp(c.yaw_rate_cmd_rad_s, -plant_.yaw_rate_max, plant_.yaw_rate_max);
  return c;
}

MultirotorRolloutState MPPIController::stepPlant(const MultirotorRolloutState& st,
                                                 const MPPIControl& u,
                                                 double dt) const {
  MultirotorRolloutState next = st;
  const double cy = std::cos(st.yaw_rad);
  const double sy = std::sin(st.yaw_rad);

  // Rotate yaw-aligned velocity command [v_fwd, v_right, v_down] into odom/NED frame
  const Eigen::Vector3d v_des_odom(
      cy * u.vel_cmd_yaw_aligned.x() - sy * u.vel_cmd_yaw_aligned.y(),
      sy * u.vel_cmd_yaw_aligned.x() + cy * u.vel_cmd_yaw_aligned.y(),
      u.vel_cmd_yaw_aligned.z());

  // Lee geometric velocity tracking + aerodynamic drag + first-order motor lag
  const Eigen::Vector3d a_des(
      plant_.k_vel_xy * (v_des_odom.x() - st.vel_odom.x()) - plant_.drag_coeff_xy * st.vel_odom.x(),
      plant_.k_vel_xy * (v_des_odom.y() - st.vel_odom.y()) - plant_.drag_coeff_xy * st.vel_odom.y(),
      plant_.k_vel_z * (v_des_odom.z() - st.vel_odom.z()));

  const double alpha_lag = dt / (plant_.motor_tau_s + dt);
  next.acc_odom = (1.0 - alpha_lag) * st.acc_odom + alpha_lag * a_des;
  next.vel_odom = st.vel_odom + next.acc_odom * dt;
  next.pos_odom = st.pos_odom + 0.5 * (st.vel_odom + next.vel_odom) * dt;
  next.yaw_rad = wrapPi(st.yaw_rad + u.yaw_rate_cmd_rad_s * dt);
  return next;
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
    filtered[t].vel_cmd_yaw_aligned =
        0.22 * u_mean_[t - 1].vel_cmd_yaw_aligned +
        0.56 * u_mean_[t].vel_cmd_yaw_aligned +
        0.22 * u_mean_[t + 1].vel_cmd_yaw_aligned;
    filtered[t].yaw_rate_cmd_rad_s =
        0.22 * u_mean_[t - 1].yaw_rate_cmd_rad_s +
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

  // 2. Shift warm-start mean sequence by 1 step and blend with nominal reference velocity prior
  for (int t = 0; t < T - 1; ++t) {
    u_mean_[t] = u_mean_[t + 1];
  }
  u_mean_[T - 1] = u_mean_[std::max(0, T - 2)];

  const double cur_yaw = state_odom.yaw();
  const double cy0 = std::cos(cur_yaw);
  const double sy0 = std::sin(cur_yaw);
  const double alpha = std::clamp(cfg_.alpha_prior, 0.0, 1.0);

  for (int t = 0; t < T; ++t) {
    const PathPoint& ref = horizon_refs_[t];
    const Eigen::Vector3d pos_err =
        ref.pos - (state_odom.pos_odom + state_odom.vel_odom * ((t + 1) * cfg_.dt_s));
    const Eigen::Vector3d v_prior_odom(
        ref.vel.x() + 1.45 * pos_err.x(),
        ref.vel.y() + 1.45 * pos_err.y(),
        ref.vel.z() + 1.80 * pos_err.z());

    MPPIControl u_nom{};
    u_nom.vel_cmd_yaw_aligned = Eigen::Vector3d(
        cy0 * v_prior_odom.x() + sy0 * v_prior_odom.y(),
        -sy0 * v_prior_odom.x() + cy0 * v_prior_odom.y(),
        v_prior_odom.z());
    u_nom.yaw_rate_cmd_rad_s =
        std::clamp(1.8 * wrapPi(ref.yaw_rad - cur_yaw), -plant_.yaw_rate_max, plant_.yaw_rate_max);
    u_nom = clampControl(u_nom);

    u_mean_[t].vel_cmd_yaw_aligned =
        (1.0 - alpha) * u_mean_[t].vel_cmd_yaw_aligned + alpha * u_nom.vel_cmd_yaw_aligned;
    u_mean_[t].yaw_rate_cmd_rad_s =
        (1.0 - alpha) * u_mean_[t].yaw_rate_cmd_rad_s + alpha * u_nom.yaw_rate_cmd_rad_s;
    u_mean_[t] = clampControl(u_mean_[t]);
  }

  // 3. Sample K = 2048 temporally correlated noise sequences & evaluate 6-DoF rollouts in parallel
  MultirotorRolloutState x0{};
  x0.pos_odom = state_odom.pos_odom;
  x0.vel_odom = state_odom.vel_odom;
  x0.acc_odom = Eigen::Vector3d::Zero();
  x0.yaw_rad = cur_yaw;

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
      MultirotorRolloutState st = x0;
      MPPIControl prev_u = u_mean_[0];
      const int cand_slot = (cand_stride > 0 && k % cand_stride == 0 && k / cand_stride < n_cand)
                                ? k / cand_stride
                                : -1;
      Eigen::Vector3d* cand_pos = (cand_slot >= 0)
          ? &candidates_.positions[static_cast<std::size_t>(cand_slot * (T + 1))]
          : nullptr;
      if (cand_pos) cand_pos[0] = x0.pos_odom;
      double vx_n = 0.0, vy_n = 0.0, vz_n = 0.0, yr_n = 0.0;
      double total_cost = 0.0;

      // Bimodal lateral bias on 35% of samples so MPPI naturally splits left/right around
      // dead-center corridor pillars (Section 6 FMEA: Pillar Dead-Center)
      double lateral_split_bias = 0.0;
      if (k >= (13 * K) / 20) {
        lateral_split_bias = ((k & 1) == 0) ? 0.95 : -0.95;
      }

      for (int t = 0; t < T; ++t) {
        if (k == 0) {
          vx_n = 0.0; vy_n = 0.0; vz_n = 0.0; yr_n = 0.0;
        } else {
          vx_n = beta * vx_n + inv_sqrt * n01(local_rng) * cfg_.sigma_vel.x();
          vy_n = beta * vy_n + inv_sqrt * n01(local_rng) * cfg_.sigma_vel.y();
          vz_n = beta * vz_n + inv_sqrt * n01(local_rng) * cfg_.sigma_vel.z();
          yr_n = beta * yr_n + inv_sqrt * n01(local_rng) * cfg_.sigma_yaw_rate;
        }

        MPPIControl noise_t{};
        noise_t.vel_cmd_yaw_aligned = Eigen::Vector3d(vx_n, vy_n + lateral_split_bias, vz_n);
        noise_t.yaw_rate_cmd_rad_s = yr_n;

        MPPIControl u_t{};
        u_t.vel_cmd_yaw_aligned =
            u_mean_[t].vel_cmd_yaw_aligned + noise_t.vel_cmd_yaw_aligned;
        u_t.yaw_rate_cmd_rad_s =
            u_mean_[t].yaw_rate_cmd_rad_s + noise_t.yaw_rate_cmd_rad_s;
        u_t = clampControl(u_t);

        // Store effective realized perturbation in time-major order (t * K + k) for cache-coherent reduction
        noise_t.vel_cmd_yaw_aligned =
            u_t.vel_cmd_yaw_aligned - u_mean_[t].vel_cmd_yaw_aligned;
        noise_t.yaw_rate_cmd_rad_s =
            u_t.yaw_rate_cmd_rad_s - u_mean_[t].yaw_rate_cmd_rad_s;
        flat_noise_[static_cast<std::size_t>(t * K + k)] = noise_t;

        st = stepPlant(st, u_t, cfg_.dt_s);
        if (cand_pos) cand_pos[t + 1] = st.pos_odom;
        const PathPoint& ref = horizon_refs_[t];

        // Critic 1: PathFollowCritic (3D position tracking error)
        const Eigen::Vector3d pos_err = st.pos_odom - ref.pos;
        const double path_follow_cost =
            pos_err.x() * pos_err.x() + pos_err.y() * pos_err.y() + 1.8 * pos_err.z() * pos_err.z();
        total_cost += cfg_.w_path_follow * path_follow_cost * cfg_.dt_s;

        // Critic 2: PathAlignCritic (velocity vector alignment)
        const Eigen::Vector3d vel_err = st.vel_odom - ref.vel;
        total_cost += cfg_.w_path_align * vel_err.squaredNorm() * cfg_.dt_s;

        // Critic 3: ObstaclesCritic (GroundTruthMap static walls + STVL unmodeled beams + iSAM2 grid)
        const double d_obs = queryClearanceOdom(st.pos_odom);
        if (d_obs <= cfg_.lethal_distance_m) {
          total_cost += cfg_.w_obstacles * (22.0 + 160.0 * (cfg_.lethal_distance_m - d_obs));
        } else if (d_obs < horizon_safe_[t]) {
          const double diff = horizon_safe_[t] - d_obs;
          total_cost += cfg_.w_obstacles * (0.65 * diff * diff +
                        0.12 * (diff * diff * diff) / std::max(0.03, d_obs - cfg_.lethal_distance_m));
        }

        // Altitude & outer-net geofence barrier
        const double alt = -st.pos_odom.z();
        if (alt < 0.20) {
          const double d_alt = 0.20 - alt;
          total_cost += 220.0 * d_alt * d_alt;
        } else if (alt > 1.62) {
          const double d_alt = alt - 1.62;
          total_cost += 220.0 * d_alt * d_alt;
        }
        const double max_xy = std::max(std::abs(st.pos_odom.x()), std::abs(st.pos_odom.y()));
        if (max_xy > 2.65) {
          const double d_net = max_xy - 2.65;
          total_cost += 180.0 * d_net * d_net;
        }

        // Critic 4: OrientationCritic (keep OAK-D-S2 depth camera aligned with corridor tangent)
        const double yaw_err = wrapPi(st.yaw_rad - ref.yaw_rad);
        total_cost += cfg_.w_orientation * yaw_err * yaw_err * cfg_.dt_s;

        // Critic 5: ControlRateCritic (penalize control slew rate)
        if (t > 0) {
          const Eigen::Vector3d du = u_t.vel_cmd_yaw_aligned - prev_u.vel_cmd_yaw_aligned;
          const double dyr = u_t.yaw_rate_cmd_rad_s - prev_u.yaw_rate_cmd_rad_s;
          total_cost += cfg_.w_control_rate * (du.squaredNorm() + 0.5 * dyr * dyr) * cfg_.dt_s;
        }
        prev_u = u_t;

        // Critic 6: GoalCritic (terminal horizon progress)
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
      delta_v += wk * nt.vel_cmd_yaw_aligned;
      delta_yr += wk * nt.yaw_rate_cmd_rad_s;
    }
    u_mean_[t].vel_cmd_yaw_aligned += delta_v;
    u_mean_[t].yaw_rate_cmd_rad_s += delta_yr;
    u_mean_[t] = clampControl(u_mean_[t]);
  }

  smoothControlSequence();

  // 5. Roll out optimal mean trajectory for visualization & feedforward setpoint extraction
  optimal_rollout_pos_.resize(static_cast<std::size_t>(T + 1));
  MultirotorRolloutState opt_st = x0;
  optimal_rollout_pos_[0] = opt_st.pos_odom;
  MultirotorRolloutState step1_st = x0;
  for (int t = 0; t < T; ++t) {
    opt_st = stepPlant(opt_st, u_mean_[t], cfg_.dt_s);
    optimal_rollout_pos_[t + 1] = opt_st.pos_odom;
    if (t == 0) step1_st = opt_st;
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
    Eigen::Vector3d v_cmd_odom(
        cy0 * u0.vel_cmd_yaw_aligned.x() - sy0 * u0.vel_cmd_yaw_aligned.y(),
        sy0 * u0.vel_cmd_yaw_aligned.x() + cy0 * u0.vel_cmd_yaw_aligned.y(),
        u0.vel_cmd_yaw_aligned.z());

    // Exponential Control Barrier Function (CBF) near outer arena net walls (|x|, |y| <= 2.86m)
    if (state_odom.pos_odom.x() > 2.05) {
      const double v_lim = 2.0 * std::max(0.0, 2.85 - state_odom.pos_odom.x());
      if (v_cmd_odom.x() > v_lim) v_cmd_odom.x() = v_lim;
    } else if (state_odom.pos_odom.x() < -2.05) {
      const double v_lim = -2.0 * std::max(0.0, 2.85 + state_odom.pos_odom.x());
      if (v_cmd_odom.x() < v_lim) v_cmd_odom.x() = v_lim;
    }
    if (state_odom.pos_odom.y() > 2.05) {
      const double v_lim = 2.0 * std::max(0.0, 2.85 - state_odom.pos_odom.y());
      if (v_cmd_odom.y() > v_lim) v_cmd_odom.y() = v_lim;
    } else if (state_odom.pos_odom.y() < -2.05) {
      const double v_lim = -2.0 * std::max(0.0, 2.85 + state_odom.pos_odom.y());
      if (v_cmd_odom.y() < v_lim) v_cmd_odom.y() = v_lim;
    }

    cmd.pos_ned = step1_st.pos_odom;
    cmd.vel_ned = v_cmd_odom;
    cmd.acc_ned = step1_st.acc_odom;
    cmd.yaw_rad = step1_st.yaw_rad;
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
