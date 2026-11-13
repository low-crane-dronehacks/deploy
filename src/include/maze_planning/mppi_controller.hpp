#pragma once

#include <array>
#include <memory>
#include <mutex>
#include <random>
#include <vector>

#include "isam2_depth/distance_grid.hpp"
#include "maze_planning/groundtruth_map.hpp"
#include "maze_planning/quad_plant.hpp"
#include "maze_planning/types.hpp"
#include "maze_planning/voxel_costmap.hpp"

namespace maze_planning {

/**
 * @brief Control action sampled at each horizon step: a world-frame (odom NED) velocity setpoint and NED yaw
 * rate, which is what PX4's Offboard velocity mode and skybeta's flight_interface take. The rollout plant
 * (QuadPlant, quad.py's Lee controller) converts it to its yaw-aligned command with the rollout's own heading.
 * Being world-frame, the warm start stays valid when the vehicle yaws between cycles.
 */
struct MPPIControl {
  Eigen::Vector3d vel_cmd_ned{Eigen::Vector3d::Zero()};   // m/s, odom NED
  double yaw_rate_cmd_rad_s{0.0};                         // NED (positive = clockwise from above)
};

/**
 * @brief Configuration for Nav2-Style Critic-Based MPPI Controller (Section 2.2, 2.3, 3.4, WP-3 Task 4).
 */
struct MPPIConfig {
  int num_samples{1024};            // rollouts per cycle (quad.py plant: ~5x the cost of the old point-mass one)
  int horizon_steps{16};            // T = 16 steps * 0.05s = 0.80s lookahead
  double dt_s{0.05};                // rollout control step; the plant substeps at QuadPlantParams::physics_dt_s
  double lambda{0.16};              // softmax temperature parameter
  double alpha_prior{0.35};         // blend weight toward nominal path tangent velocity prior
  double noise_corr_beta{0.65};     // AR(1) colored noise temporal correlation

  // Exploration standard deviations in the PATH frame of each horizon step: [along-track, cross-track, down]
  // (the reference tangent, not the body: a heading off the path must not tilt the exploration). The bimodal
  // lateral split for Pillar Dead-Center is cross-track too.
  Eigen::Vector3d sigma_vel{0.45, 0.58, 0.18};
  double sigma_yaw_rate{0.30};
  double lateral_split_m_s{0.95};   // +- cross-track bias on the last 35% of samples (0 = off)

  // Command limits (world frame). skybeta's flight_interface scales any twist whose 3-D norm exceeds its
  // max_speed_mps (1.5) back to 1.5, which would bend MPPI's command off the trajectory it optimised, so MPPI
  // limits the norm itself. Conservative defaults; raise together with flight_interface's max_speed_mps.
  // The plant itself caps yaw rate at QuadPlantParams::max_yaw_rate_rad_s.
  double v_max_norm{1.5};
  double v_max_xy{1.5};
  double v_max_z{1.0};
  double yaw_rate_max{M_PI / 3.0};  // quad.py LEE_MAX_YAW_RATE

  // Nav2-MPPI Critic Weights
  double w_path_follow{16.0};       // PathFollowCritic: 3D position tracking error
  double w_path_align{4.0};         // PathAlignCritic: velocity tangent alignment
  double w_obstacles{110.0};        // ObstaclesCritic: static maze walls + dynamic unmodeled beams/pillars
  double w_goal{22.0};              // GoalCritic: terminal horizon progress
  double w_orientation{3.5};        // OrientationCritic: heading toward the reference yaw (path tangent)
  // FovCritic: the depth camera looks along body x (74.7 deg HFOV). Penalise horizontal motion more than
  // fov_half_angle_rad off the rollout's own heading, so MPPI does not fly where the camera cannot see (and
  // where iSAM2 gets no depth to correct the map). Gated below fov_min_speed_m_s, where direction is noise.
  double w_fov{8.0};
  double fov_half_angle_rad{0.52};  // ~30 deg, inside the camera's 37 deg half-FOV
  double fov_min_speed_m_s{0.25};
  double w_control_rate{1.4};       // ControlRateCritic: control slew-rate smoothness

  double safe_distance_m{0.42};     // obstacle comfort inflation radius (m)
  // Never ask for more comfort than the reference itself has: per horizon step the comfort radius is
  // clamp(clearance at the reference point, lethal + adaptive_margin, safe_distance). Without it MPPI refuses a
  // planned pass that is legitimately tight (corridor_end needs ~0.24 m to clear its checkpoint) and stalls.
  bool adaptive_safe_distance{true};
  double adaptive_safe_margin_m{0.02};
  double lethal_distance_m{0.21};   // hard collision radius (m)
  double drone_radius_m{0.21};      // physical collision radius for FMEA Pillar Dead-Center check
  double stall_cost_threshold{850.0};
  // When true and an active iSAM2 DistanceGrid is present (built from PriorEsdfMap = retained CAD priors U
  // new/shifted obstacles, with stale CAD priors carved away), evaluate obstacle clearance inside the grid from
  // the iSAM2 DistanceGrid directly instead of clamping against un-carved GroundTruthMap static walls.
  bool trust_isam2_prior_grid{false};

  // Visualization: keep this many sampled rollouts (evenly strided over the K samples) per cycle so a node
  // can publish them. 0 = keep none (no extra cost in the rollout loop).
  int viz_num_candidates{0};
};

/**
 * @brief Sampled MPPI rollouts kept for visualization: num_candidates trajectories of horizon_steps + 1
 * positions each (odom frame, x0 first), with their total rollout cost.
 */
struct MPPICandidateSet {
  int num_candidates{0};
  int points_per_candidate{0};
  std::vector<Eigen::Vector3d> positions;  // candidate-major: positions[c * points_per_candidate + t]
  std::vector<double> costs;               // costs[c]
  double min_cost{0.0};
  double max_cost{0.0};
};

/**
 * @brief Nav2-Style Critic-Based MPPI Controller (2048 rollouts @ 50 Hz in C++).
 *
 * Implements:
 *   - K parallel rollouts per cycle (OpenMP) through QuadPlant, the C++ port of dronehacks-sim quad.py
 *     (Lee velocity controller -> rotor allocation -> motor lag -> rigid body)
 *   - Nav2-MPPI modular critics: PathFollowCritic, PathAlignCritic, ObstaclesCritic,
 *     GoalCritic, OrientationCritic, and ControlRateCritic
 *   - Savitzky-Golay causal control sequence smoothing
 *   - FMEA Pillar Dead-Center handling (Section 6): splits rollout distribution left/right around
 *     unmodeled support beams, or triggers OBSTACLE_HOLD + global RRT* replan if both sides are narrower
 *     than drone_radius_m
 *   - Optional local distance field from an estimator (odom frame), e.g. iSAM2's DistanceGrid, through
 *     onEstimatorUpdate() / setLocalDistanceGrid(). No GTSAM dependency here: the in-process iSAM2 hook is
 *     isam2_depth/mppi_sink_adapter.hpp.
 */
class MPPIController {
 public:
  explicit MPPIController(const MPPIConfig& cfg = MPPIConfig{},
                          const QuadPlantParams& plant = QuadPlantParams{},
                          std::shared_ptr<const VoxelCostmap> costmap = nullptr);

  void setCostmap(std::shared_ptr<const VoxelCostmap> costmap) { costmap_ = std::move(costmap); }
  void setOdomToMapTransform(const SE3Transform& T_map_odom) { T_map_odom_ = T_map_odom; }

  /**
   * @brief Update in-process state and DistanceGrid from estimator (usable with or without GTSAM).
   */
  void onEstimatorUpdate(double stamp_s,
                         const Eigen::Vector3d& pos_ned,
                         const Eigen::Quaterniond& q_ned_frd,
                         const Eigen::Vector3d& vel_ned,
                         std::shared_ptr<const isam2_depth::DistanceGrid> distance_grid);

  /**
   * @brief Replace only the local distance field (odom frame); nullptr disables it. Thread-safe.
   */
  void setLocalDistanceGrid(std::shared_ptr<const isam2_depth::DistanceGrid> distance_grid);

  /**
   * @brief Execute one MPPI optimization cycle (num_samples rollouts over horizon_steps) tracking
   * local_ref_odom and avoiding static + unmodeled dynamic obstacles.
   */
  TrajectorySetpointCmd computeCommand(const VehicleKinematics& state_odom,
                                       const PlannedPath& local_ref_odom);

  void reset();

  const MPPIConfig& config() const { return cfg_; }
  const std::vector<MPPIControl>& meanControlSequence() const { return u_mean_; }
  const std::vector<Eigen::Vector3d>& lastOptimalRollout() const { return optimal_rollout_pos_; }
  /** NED heading along the optimal rollout (same indexing as lastOptimalRollout()). */
  const std::vector<double>& lastOptimalRolloutYaw() const { return optimal_rollout_yaw_; }
  const QuadPlant& plant() const { return plant_; }
  const MPPICandidateSet& lastCandidates() const { return candidates_; }
  double lastSolveTimeMs() const { return last_solve_ms_; }
  double lastMinCost() const { return last_min_cost_; }

 private:
  double queryClearanceOdom(const Eigen::Vector3d& p_odom) const;
  double queryStaticClearanceOdom(const Eigen::Vector3d& p_odom) const;
  double queryDynamicClearanceOdom(const Eigen::Vector3d& p_odom) const;
  MPPIControl clampControl(const MPPIControl& u) const;
  void smoothControlSequence();
  bool checkCorridorBlockedAhead(const VehicleKinematics& state_odom,
                                 const PathPoint& lookahead_pt,
                                 Eigen::Vector3d& out_obstacle_map) const;

  MPPIConfig cfg_;
  QuadPlant plant_;
  std::shared_ptr<const VoxelCostmap> costmap_;
  std::shared_ptr<const isam2_depth::DistanceGrid> isam2_grid_;    // written by the estimator (under est_mutex_)
  std::shared_ptr<const isam2_depth::DistanceGrid> active_grid_;   // snapshot used for one computeCommand()
  SE3Transform T_map_odom_{};

  std::vector<MPPIControl> u_mean_;
  std::vector<MPPIControl> flat_noise_;
  std::vector<double> costs_;
  std::vector<double> weights_;
  std::vector<PathPoint> horizon_refs_;
  std::vector<double> horizon_safe_;   // comfort radius per horizon step (adaptive_safe_distance)
  std::vector<Eigen::Vector3d> optimal_rollout_pos_;
  std::vector<double> optimal_rollout_yaw_;
  std::vector<Eigen::Matrix3d> path_frames_;   // per horizon step: columns [along, cross, down] in odom NED
  MPPICandidateSet candidates_;
  double last_solve_ms_{0.0};
  double last_min_cost_{0.0};
  int consecutive_stall_ticks_{0};
  mutable std::mutex est_mutex_;
  VehicleKinematics latest_est_state_{};
};

}  // namespace maze_planning
