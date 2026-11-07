#pragma once

#include <random>
#include <string>
#include <vector>

#include "ego_planner/local_controller_interface.hpp"
#include "ego_planner/quadrotor_dynamics.hpp"

namespace ego_planner {

/**
 * @brief Configuration parameters for Model Predictive Path Integral (MPPI) Controller.
 */
struct MPPIConfig {
  int num_samples{512};                 // K: number of stochastic rollouts (2048 on GPU, 256-512 on CPU)
  int horizon_steps{25};                // T: lookahead steps (25 * 0.05s = 1.25s)
  double dt{0.05};                      // rollout timestep (s)
  double lambda{0.10};                  // temperature parameter for softmax expectation
  
  // Exploration noise standard deviations (velocity commands in m/s, yaw rate in rad/s)
  Vec3 sigma_vel{0.4, 0.4, 0.25};
  double sigma_yaw_rate{0.35};

  // Cost weights
  double w_track{12.0};                 // distance to reference trajectory
  double w_vel{2.5};                    // error against nominal forward cruise speed
  double w_obs{45.0};                   // obstacle clearance penalty from ESDF
  double w_terminal{15.0};              // terminal distance to horizon goal
  double safe_distance_m{0.25};         // safety inflation margin
};

/**
 * @brief MPPI Local Controller for agile quadrotor tracking & dynamic obstacle avoidance.
 * 
 * Formulates optimal control as an expectation over stochastic forward trajectories:
 *   u*_t = u_t + \sum_k w_k * \epsilon_{k, t}
 *   w_k = exp(-1/lambda * (S_k - min_j S_j)) / \sum_m exp(-1/lambda * (S_m - min_j S_j))
 * 
 * Utilizes QuadrotorDynamics for realistic plant integration and GridMap for O(1) continuous obstacle costs.
 */
class MPPIController : public LocalControllerBase {
 public:
  explicit MPPIController(const MPPIConfig& cfg = MPPIConfig{},
                          const QuadrotorParams& quad_params = QuadrotorParams{});

  std::string name() const override { return "mppi"; }

  ControlOutput computeCommand(const VehicleState& state,
                               const GlobalTrajectory& ref_traj,
                               const GridMap& grid_map,
                               double dt) override;

  void reset() override;

  // MPPI core routines exposed for unit tests & parallelization
  void sampleNoise(std::vector<std::vector<QuadrotorAction>>& noise_trajectories);
  double evaluateTrajectoryCost(const QuadrotorState& initial_state,
                                const std::vector<QuadrotorAction>& action_sequence,
                                const GlobalTrajectory& ref_traj,
                                const GridMap& grid_map) const;

  const MPPIConfig& config() const { return cfg_; }
  const std::vector<QuadrotorAction>& currentSequence() const { return u_mean_; }

 private:
  MPPIConfig cfg_;
  QuadrotorDynamics dynamics_;
  std::vector<QuadrotorAction> u_mean_; // size: horizon_steps
  std::mt19937 rng_;
  std::normal_distribution<double> dist_normal_{0.0, 1.0};
};

}  // namespace ego_planner
