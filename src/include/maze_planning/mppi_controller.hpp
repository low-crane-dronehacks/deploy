#pragma once

#include <array>
#include <memory>
#include <mutex>
#include <random>
#include <vector>

#include "isam2_depth/distance_grid.hpp"
#ifdef HAVE_GTSAM
#include "isam2_depth/mppi_interface.hpp"
#endif
#include "maze_planning/groundtruth_map.hpp"
#include "maze_planning/types.hpp"
#include "maze_planning/voxel_costmap.hpp"

namespace maze_planning {

/**
 * @brief Control action sampled at each horizon step t in the yaw-aligned frame.
 */
struct MPPIControl {
  Eigen::Vector3d vel_cmd_yaw_aligned{Eigen::Vector3d::Zero()}; // [v_fwd, v_right, v_down] m/s
  double yaw_rate_cmd_rad_s{0.0};
};

/**
 * @brief 6-DoF Multirotor State & Plant Parameters for MPPI Forward Rollouts.
 */
struct MultirotorPlantParams {
  double mass_kg{1.56};
  double gravity_m_s2{9.81};
  double k_vel_xy{3.6};        // horizontal velocity tracking bandwidth (1/s)
  double k_vel_z{4.0};         // vertical velocity tracking bandwidth (1/s)
  double motor_tau_s{0.030};   // first-order motor/attitude lag (30 ms)
  double drag_coeff_xy{0.12};  // aerodynamic drag coefficient
  double v_max_xy{4.5};
  double v_max_z{2.0};
  double yaw_rate_max{1.20};
};

struct MultirotorRolloutState {
  Eigen::Vector3d pos_odom{Eigen::Vector3d::Zero()};
  Eigen::Vector3d vel_odom{Eigen::Vector3d::Zero()};
  Eigen::Vector3d acc_odom{Eigen::Vector3d::Zero()};
  double yaw_rad{0.0};
};

/**
 * @brief Configuration for Nav2-Style Critic-Based MPPI Controller (Section 2.2, 2.3, 3.4, WP-3 Task 4).
 */
struct MPPIConfig {
  int num_samples{2048};            // Section 2.2 & 3.4: 2048 stochastic rollouts per control cycle
  int horizon_steps{16};            // T = 16 steps * 0.05s = 0.80s lookahead
  double dt_s{0.05};                // rollout step (20 Hz internal dynamics / 50 Hz control loop)
  double lambda{0.16};              // softmax temperature parameter
  double alpha_prior{0.35};         // blend weight toward nominal path tangent velocity prior
  double noise_corr_beta{0.65};     // AR(1) colored noise temporal correlation

  // Exploration standard deviations (including bimodal lateral exploration for Pillar Dead-Center splitting)
  Eigen::Vector3d sigma_vel{0.45, 0.58, 0.18};
  double sigma_yaw_rate{0.30};

  // Nav2-MPPI Critic Weights
  double w_path_follow{16.0};       // PathFollowCritic: 3D position tracking error
  double w_path_align{4.0};         // PathAlignCritic: velocity tangent alignment
  double w_obstacles{110.0};        // ObstaclesCritic: static maze walls + dynamic unmodeled beams/pillars
  double w_goal{22.0};              // GoalCritic: terminal horizon progress
  double w_orientation{3.5};        // OrientationCritic: camera FOV heading alignment
  double w_control_rate{1.4};       // ControlRateCritic: control slew-rate smoothness

  double safe_distance_m{0.42};     // obstacle comfort inflation radius (m)
  double lethal_distance_m{0.21};   // hard collision radius (m)
  double drone_radius_m{0.21};      // physical collision radius for FMEA Pillar Dead-Center check
  double stall_cost_threshold{220.0};
};

/**
 * @brief Nav2-Style Critic-Based MPPI Controller (2048 rollouts @ 50 Hz in C++).
 *
 * Implements:
 *   - 2048 parallel 6-DoF rollouts per cycle via OpenMP multithreading
 *   - Nav2-MPPI modular critics: PathFollowCritic, PathAlignCritic, ObstaclesCritic,
 *     GoalCritic, OrientationCritic, and ControlRateCritic
 *   - Savitzky-Golay causal control sequence smoothing
 *   - FMEA Pillar Dead-Center handling (Section 6): splits rollout distribution left/right around
 *     unmodeled support beams, or triggers OBSTACLE_HOLD + global RRT* replan if both sides are narrower
 *     than drone_radius_m
 *   - Direct in-process integration with isam2_depth::MppiStateSink (`onEstimate(const MppiInput&)`)
 */
class MPPIController
#ifdef HAVE_GTSAM
    : public isam2_depth::MppiStateSink
#endif
{
 public:
  explicit MPPIController(const MPPIConfig& cfg = MPPIConfig{},
                          const MultirotorPlantParams& plant = MultirotorPlantParams{},
                          std::shared_ptr<const VoxelCostmap> costmap = nullptr);

  void setCostmap(std::shared_ptr<const VoxelCostmap> costmap) { costmap_ = std::move(costmap); }
  void setOdomToMapTransform(const SE3Transform& T_map_odom) { T_map_odom_ = T_map_odom; }

#ifdef HAVE_GTSAM
  /**
   * @brief Zero-copy in-process hook called by isam2_depth::Isam2DepthNode at the PX4 rate.
   */
  void onEstimate(const isam2_depth::MppiInput& in) override;
#endif

  /**
   * @brief Update in-process state and DistanceGrid from estimator (usable with or without GTSAM).
   */
  void onEstimatorUpdate(double stamp_s,
                         const Eigen::Vector3d& pos_ned,
                         const Eigen::Quaterniond& q_ned_frd,
                         const Eigen::Vector3d& vel_ned,
                         std::shared_ptr<const isam2_depth::DistanceGrid> distance_grid);

  /**
   * @brief Execute one 50 Hz MPPI optimization cycle (2048 rollouts over T=16 horizon steps)
   * tracking local_ref_odom and avoiding static + unmodeled dynamic obstacles.
   */
  TrajectorySetpointCmd computeCommand(const VehicleKinematics& state_odom,
                                       const PlannedPath& local_ref_odom);

  void reset();

  const MPPIConfig& config() const { return cfg_; }
  const std::vector<MPPIControl>& meanControlSequence() const { return u_mean_; }
  const std::vector<Eigen::Vector3d>& lastOptimalRollout() const { return optimal_rollout_pos_; }
  double lastSolveTimeMs() const { return last_solve_ms_; }
  double lastMinCost() const { return last_min_cost_; }

 private:
  MultirotorRolloutState stepPlant(const MultirotorRolloutState& st,
                                   const MPPIControl& u,
                                   double dt) const;

  double queryClearanceOdom(const Eigen::Vector3d& p_odom) const;
  MPPIControl clampControl(const MPPIControl& u) const;
  void smoothControlSequence();
  bool checkCorridorBlockedAhead(const VehicleKinematics& state_odom,
                                 const PathPoint& lookahead_pt,
                                 Eigen::Vector3d& out_obstacle_map) const;

  MPPIConfig cfg_;
  MultirotorPlantParams plant_;
  std::shared_ptr<const VoxelCostmap> costmap_;
  std::shared_ptr<const isam2_depth::DistanceGrid> isam2_grid_;
  SE3Transform T_map_odom_{};

  std::vector<MPPIControl> u_mean_;
  std::vector<MPPIControl> flat_noise_;
  std::vector<double> costs_;
  std::vector<double> weights_;
  std::vector<PathPoint> horizon_refs_;
  std::vector<Eigen::Vector3d> optimal_rollout_pos_;
  double last_solve_ms_{0.0};
  double last_min_cost_{0.0};
  int consecutive_stall_ticks_{0};
  mutable std::mutex est_mutex_;
  VehicleKinematics latest_est_state_{};
};

}  // namespace maze_planning
