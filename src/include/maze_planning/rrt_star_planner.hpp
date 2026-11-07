#pragma once

#include <memory>
#include <random>
#include <vector>

#include "maze_planning/groundtruth_map.hpp"
#include "maze_planning/types.hpp"
#include "maze_planning/voxel_costmap.hpp"

namespace maze_planning {

/**
 * @brief Configuration for Informed RRT* Global Planner (Section 2.2, 2.3, 3.3, WP-3 Task 1).
 */
struct RRTStarConfig {
  int max_iterations{1200};
  int leg_iterations{480};               // iterations per gate-to-gate leg
  double step_size_m{0.32};              // max steering extension step (m)
  double goal_bias{0.18};                // probability of sampling exact goal
  double informed_bias{0.75};            // probability of prolate hyperspheroid sampling once solution exists
  double search_radius_gamma{1.65};      // asymptotic rewiring radius constant
  double max_search_radius_m{0.95};
  double robot_radius_m{0.21};           // collision inflation radius (m)
  double comfort_clearance_m{0.46};      // medial-axis comfort clearance (m)
  double comfort_weight{4.2};            // clearance penalty weight in edge cost
  double cruise_speed_m_s{2.60};         // nominal speed along smoothed B-spline path
  double bspline_dt_s{0.05};             // dense path discretization interval (20 Hz)
  Eigen::Vector3d bounds_min_ned{-2.86, -2.86, -1.60};
  Eigen::Vector3d bounds_max_ned{2.86, 2.86, -0.22};
};

struct RRTNode {
  Eigen::Vector3d pos_ned{Eigen::Vector3d::Zero()};
  int parent_idx{-1};
  double cost_from_start{0.0};
  double euclidean_from_start{0.0};
  std::vector<int> children;
};

/**
 * @brief Informed RRT* Global Planner operating on GroundTruthMap (non-ESDF CAD/mesh encoding).
 *
 * Implements:
 *   - Karaman & Frazzoli (2011) RRT* with Gammell et al. (2014) Informed RRT* ellipsoidal sampling
 *   - Medial-axis clearance-regularized edge cost & recursive subtree cost propagation on rewiring
 *   - Sphere-traced continuous collision validation against GroundTruthMap + VoxelCostmap
 *   - Line-of-sight shortcutting, medial-axis waypoint centering, and clamped cubic B-spline smoothing
 *   - FMEA Pillar Dead-Center recovery: blockCorridorAndReplan() marks blocked edges and replans globally
 */
class RRTStarPlanner {
 public:
  explicit RRTStarPlanner(const RRTStarConfig& cfg = RRTStarConfig{},
                          std::shared_ptr<GroundTruthMap> gt_map = nullptr,
                          std::shared_ptr<const VoxelCostmap> costmap = nullptr);

  void setGroundTruthMap(std::shared_ptr<GroundTruthMap> gt_map) { gt_map_ = std::move(gt_map); }
  void setCostmap(std::shared_ptr<const VoxelCostmap> costmap) { costmap_ = std::move(costmap); }
  const GroundTruthMap& groundTruthMap() const { return *gt_map_; }
  GroundTruthMap& mutableGroundTruthMap() { return *gt_map_; }

  /**
   * @brief Plan a collision-free, clearance-optimized path between start_map and goal_map
   * on the GroundTruthMap encoding, smoothed with a clamped cubic B-spline (/plan/global_rrt).
   */
  bool plan(const Eigen::Vector3d& start_map,
            const Eigen::Vector3d& goal_map,
            PlannedPath& out_path,
            std::vector<Eigen::Vector3d>* out_raw_waypoints = nullptr,
            int max_iters_override = -1);

  /**
   * @brief Plan a complete multi-gate maze trajectory from start_map through a sequence of
   * anchor/gate waypoints to the finish pad in the map frame.
   */
  bool planMultiWaypoint(const Eigen::Vector3d& start_map,
                         const std::vector<Eigen::Vector3d>& anchor_waypoints_map,
                         PlannedPath& out_path,
                         std::vector<Eigen::Vector3d>* out_waypoints = nullptr);

  /**
   * @brief Plan the full DroneHacks 2026 maze course (Launch -> Gate 0 -> Gate 1 -> Gate 2 ->
   * Gate 3 -> Gate 4 Window -> Pad) using GroundTruthMap.
   */
  bool planFullCourse(const Eigen::Vector3d& start_map,
                      bool prefer_window,
                      bool include_return_leg,
                      PlannedPath& out_path);

  /**
   * @brief FMEA Pillar Dead-Center Recovery (Section 6):
   * Marks the corridor around blocked_pos_map as obstructed in GroundTruthMap and triggers an
   * immediate global RRT* replan from current_pos_map to goal_map (or remaining course).
   */
  bool blockCorridorAndReplan(const Eigen::Vector3d& current_pos_map,
                              const Eigen::Vector3d& blocked_pos_map,
                              double block_radius_m,
                              const std::vector<Eigen::Vector3d>& remaining_anchors_map,
                              PlannedPath& out_path);

  /**
   * @brief Smooth discrete waypoints using a clamped cubic B-spline with continuous velocity,
   * acceleration, arc-length s, and tangent yaw profiles (Section 3.3).
   */
  PlannedPath smoothWithCubicBSpline(const std::vector<Eigen::Vector3d>& waypoints,
                                     double desired_speed_m_s) const;

  // Exposed RRT* primitives for unit testing
  Eigen::Vector3d steer(const Eigen::Vector3d& from, const Eigen::Vector3d& to, double max_step) const;
  bool isEdgeValid(const Eigen::Vector3d& a, const Eigen::Vector3d& b, double radius_m,
                   double alt_min_m = 0.18, double alt_max_m = 1.62) const;
  std::vector<Eigen::Vector3d> shortcutAndCenterPath(const std::vector<Eigen::Vector3d>& raw_path,
                                                     double radius_m) const;

  std::size_t lastTreeSize() const { return last_tree_size_; }
  const RRTStarConfig& config() const { return cfg_; }

 private:
  Eigen::Vector3d sampleInformedState(const Eigen::Vector3d& start_ned,
                                      const Eigen::Vector3d& goal_ned,
                                      double c_best_euclid,
                                      const Eigen::Vector3d& box_lo,
                                      const Eigen::Vector3d& box_hi);
  double evaluateEdgeCost(const Eigen::Vector3d& from, const Eigen::Vector3d& to) const;
  void propagateSubtreeCost(std::vector<RRTNode>& tree, int root_idx) const;

  RRTStarConfig cfg_;
  std::shared_ptr<GroundTruthMap> gt_map_;
  std::shared_ptr<const VoxelCostmap> costmap_;
  std::size_t last_tree_size_{0};
  mutable std::mt19937 rng_;
  mutable std::uniform_real_distribution<double> dist_01_{0.0, 1.0};
  mutable std::normal_distribution<double> dist_norm_{0.0, 1.0};
};

}  // namespace maze_planning
