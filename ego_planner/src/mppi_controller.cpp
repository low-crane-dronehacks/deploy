#include "ego_planner/mppi_controller.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace ego_planner {

MPPIController::MPPIController(const MPPIConfig& cfg, const QuadrotorParams& quad_params)
    : cfg_(cfg), dynamics_(quad_params), rng_(42) {
  reset();
}

void MPPIController::reset() {
  u_mean_.assign(cfg_.horizon_steps, QuadrotorAction{});
}

void MPPIController::sampleNoise(std::vector<std::vector<QuadrotorAction>>& noise_trajectories) {
  noise_trajectories.resize(cfg_.num_samples);
  for (int k = 0; k < cfg_.num_samples; ++k) {
    noise_trajectories[k].resize(cfg_.horizon_steps);
    for (int t = 0; t < cfg_.horizon_steps; ++t) {
      noise_trajectories[k][t].vel_cmd_yaw_aligned = Vec3{
          dist_normal_(rng_) * cfg_.sigma_vel.x,
          dist_normal_(rng_) * cfg_.sigma_vel.y,
          dist_normal_(rng_) * cfg_.sigma_vel.z,
      };
      noise_trajectories[k][t].yaw_rate_cmd_rad_s = dist_normal_(rng_) * cfg_.sigma_yaw_rate;
    }
  }
}

double MPPIController::evaluateTrajectoryCost(const QuadrotorState& initial_state,
                                             const std::vector<QuadrotorAction>& action_sequence,
                                             const GlobalTrajectory& ref_traj,
                                             const GridMap& grid_map) const {
  QuadrotorState state = initial_state;
  double total_cost = 0.0;

  for (size_t t = 0; t < action_sequence.size(); ++t) {
    state = dynamics_.step(state, action_sequence[t], cfg_.dt);

    // 1. Reference path tracking cost
    if (!ref_traj.empty()) {
      double arc_s = 0.0;
      GlobalTrajectoryPoint target_pt = ref_traj.projectPoint(state.pos_ned, arc_s);
      double dist_err_sq = (state.pos_ned - target_pt.pos_ned).normSq();
      total_cost += cfg_.w_track * dist_err_sq;

      // Desired velocity alignment
      double vel_err_sq = (state.vel_ned - target_pt.vel_ned).normSq();
      total_cost += cfg_.w_vel * vel_err_sq;
    }

    // 2. Obstacle clearance cost from continuous ESDF
    double obs_cost = grid_map.getObstacleCost(state.pos_ned, cfg_.safe_distance_m);
    total_cost += cfg_.w_obs * obs_cost;
  }

  // 3. Terminal state cost
  if (!ref_traj.empty()) {
    double arc_s = 0.0;
    GlobalTrajectoryPoint target_pt = ref_traj.projectPoint(state.pos_ned, arc_s);
    total_cost += cfg_.w_terminal * (state.pos_ned - target_pt.pos_ned).normSq();
  }

  return total_cost;
}

ControlOutput MPPIController::computeCommand(const VehicleState& state,
                                             const GlobalTrajectory& ref_traj,
                                             const GridMap& grid_map,
                                             double /*dt*/) {
  ControlOutput out;
  out.stream_velocity = true;
  out.mission_phase = "fly";
  out.active_primitive = "mppi";

  // Build initial QuadrotorState from VehicleState
  QuadrotorState quad_state;
  quad_state.pos_ned = state.pos_ned;
  quad_state.vel_ned = state.vel_ned;
  quad_state.q_wxyz = state.q_wxyz;

  // Warm-start shift: u_t <- u_{t+1}
  if (!u_mean_.empty()) {
    for (size_t t = 0; t + 1 < u_mean_.size(); ++t) {
      u_mean_[t] = u_mean_[t + 1];
    }
  }

  // Sample K stochastic noise trajectories
  std::vector<std::vector<QuadrotorAction>> noise;
  sampleNoise(noise);

  // Evaluate rollout costs S_k
  std::vector<double> costs(cfg_.num_samples, 0.0);
  double min_cost = 1e12;

  for (int k = 0; k < cfg_.num_samples; ++k) {
    std::vector<QuadrotorAction> sample_actions = u_mean_;
    for (int t = 0; t < cfg_.horizon_steps; ++t) {
      sample_actions[t].vel_cmd_yaw_aligned += noise[k][t].vel_cmd_yaw_aligned;
      sample_actions[t].yaw_rate_cmd_rad_s += noise[k][t].yaw_rate_cmd_rad_s;
    }
    costs[k] = evaluateTrajectoryCost(quad_state, sample_actions, ref_traj, grid_map);
    if (costs[k] < min_cost) {
      min_cost = costs[k];
    }
  }

  // Numerically stable softmax weights: w_k = exp(-1/lambda * (S_k - min_cost))
  std::vector<double> weights(cfg_.num_samples, 0.0);
  double sum_weights = 0.0;
  double inv_lambda = 1.0 / std::max(cfg_.lambda, 1e-4);

  for (int k = 0; k < cfg_.num_samples; ++k) {
    weights[k] = std::exp(-inv_lambda * (costs[k] - min_cost));
    sum_weights += weights[k];
  }

  if (sum_weights > 1e-12) {
    for (int k = 0; k < cfg_.num_samples; ++k) {
      weights[k] /= sum_weights;
    }

    // Expectation update: u_mean_t <- u_mean_t + sum_k w_k * noise_{k, t}
    for (int t = 0; t < cfg_.horizon_steps; ++t) {
      Vec3 delta_vel{0.0, 0.0, 0.0};
      double delta_yaw = 0.0;
      for (int k = 0; k < cfg_.num_samples; ++k) {
        delta_vel += noise[k][t].vel_cmd_yaw_aligned * weights[k];
        delta_yaw += noise[k][t].yaw_rate_cmd_rad_s * weights[k];
      }
      u_mean_[t].vel_cmd_yaw_aligned += delta_vel;
      u_mean_[t].yaw_rate_cmd_rad_s += delta_yaw;
    }
  }

  // First optimal action
  QuadrotorAction u0 = u_mean_.front();

  // Convert yaw-aligned velocity command to NED world velocity setpoint
  Vec3 rpy = state.q_wxyz.toEulerZYX();
  double yaw = rpy.z;
  double cy = std::cos(yaw);
  double sy = std::sin(yaw);

  out.vel_ned = Vec3{
      cy * u0.vel_cmd_yaw_aligned.x - sy * u0.vel_cmd_yaw_aligned.y,
      sy * u0.vel_cmd_yaw_aligned.x + cy * u0.vel_cmd_yaw_aligned.y,
      u0.vel_cmd_yaw_aligned.z,
  };
  out.yawspeed_rad_s = u0.yaw_rate_cmd_rad_s;
  out.yaw_rad = wrapPi(yaw + out.yawspeed_rad_s * 0.1);

  // Normalized action [-1, 1] for sim / policy interface
  out.body_action[0] = clamp(u0.vel_cmd_yaw_aligned.x / 5.0, -1.0, 1.0);
  out.body_action[1] = clamp(u0.vel_cmd_yaw_aligned.y / 3.0, -1.0, 1.0);
  out.body_action[2] = clamp(u0.vel_cmd_yaw_aligned.z / 2.0, -1.0, 1.0);
  out.body_action[3] = clamp(u0.yaw_rate_cmd_rad_s / (kPi / 3.0), -1.0, 1.0);

  return out;
}

}  // namespace ego_planner
