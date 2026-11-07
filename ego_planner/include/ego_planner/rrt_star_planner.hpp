#pragma once

#include <random>
#include <string>
#include <vector>

#include "ego_planner/global_planner_interface.hpp"

namespace ego_planner {

/**
 * @brief Configuration parameters for RRT* Global Planner.
 */
struct RRTStarConfig {
  int max_iterations{2000};
  double step_size_m{0.25};             // maximum steering distance per tree extension
  double goal_bias{0.15};               // probability of sampling exact goal [0, 1]
  double search_radius_gamma{1.5};      // asymptotic optimality radius constant: r = min(gamma * (log(N)/N)^(1/d), max_radius)
  double max_search_radius_m{0.80};
  double robot_radius_m{0.22};          // safety inflation radius (m)
  double min_clearance_m{0.20};
  Vec3 bounds_min_ned{-2.85, -2.85, -1.65}; // arena bounding box
  Vec3 bounds_max_ned{2.85, 2.85, -0.20};
};

/**
 * @brief Node representation in the RRT* tree.
 */
struct RRTNode {
  Vec3 pos_ned{0.0, 0.0, 0.0};
  int parent_idx{-1};
  double cost_from_start{0.0};
};

/**
 * @brief RRT* Global Trajectory Planner for 3D maze navigation.
 * 
 * Implements Karaman & Frazzoli (2011) Optimal Rapidly-exploring Random Trees.
 * Provides the integration scaffold for colleagues to complete the asymptotic rewiring loop.
 */
class RRTStarPlanner : public GlobalPlannerBase {
 public:
  explicit RRTStarPlanner(const RRTStarConfig& cfg = RRTStarConfig{});

  std::string name() const override { return "rrt_star"; }

  /**
   * @brief Search for the optimal collision-free path from start to goal.
   * Utilizes grid_map.isSegmentClear() for O(1) continuous collision checking.
   */
  bool plan(const Vec3& start_ned, const Vec3& goal_ned,
            const GridMap& grid_map,
            std::vector<Vec3>& out_waypoints) override;

  /**
   * @brief Convert piecewise linear RRT* path into smooth trajectory.
   */
  bool generateTrajectory(const std::vector<Vec3>& waypoints,
                          double cruise_speed_m_s,
                          GlobalTrajectory& out_trajectory) override;

  // RRT* core subroutines exposed for testing & extension
  Vec3 sampleState(const Vec3& goal_ned, double goal_bias);
  int findNearest(const std::vector<RRTNode>& tree, const Vec3& sample) const;
  Vec3 steer(const Vec3& from, const Vec3& to, double max_step) const;
  bool isEdgeCollisionFree(const Vec3& from, const Vec3& to,
                           const GridMap& grid_map, double radius_m) const;
  std::vector<int> findNearNodes(const std::vector<RRTNode>& tree,
                                 const Vec3& new_pos, double radius) const;
  int chooseBestParent(const std::vector<RRTNode>& tree,
                       const std::vector<int>& near_indices,
                       const Vec3& new_pos, const GridMap& grid_map,
                       double& out_min_cost) const;
  void rewire(std::vector<RRTNode>& tree, const std::vector<int>& near_indices,
              int new_node_idx, const GridMap& grid_map);

  const RRTStarConfig& config() const { return cfg_; }

 private:
  RRTStarConfig cfg_;
  std::mt19937 rng_;
  std::uniform_real_distribution<double> dist_01_{0.0, 1.0};
};

}  // namespace ego_planner
