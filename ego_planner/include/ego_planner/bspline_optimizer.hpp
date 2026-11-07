#pragma once

#include <vector>

#include "ego_planner/grid_map.hpp"
#include "ego_planner/math_types.hpp"
#include "ego_planner/motion_primitives.hpp"

namespace ego_planner {

struct OptimizerConfig {
  double min_clearance_m{0.24};
  double comfort_clearance_m{0.48};
  double comfort_weight{6.5};
  double safe_distance_m{0.42};
  double alt_min_m{0.45};
  double alt_max_m{1.30};
  double max_vel_ms{3.6};
  double max_acc_ms2{6.0};
  double knot_dt{0.12};
  double w_jerk{0.25};
  double w_acc{0.80};
  double w_collision{28.0};
  double w_altitude{18.0};
  double w_feasibility{2.5};
  double w_ref{2.0};
  int max_iterations{35};
  double step_size{0.0035};
};

class BSplineOptimizer {
 public:
  explicit BSplineOptimizer(const OptimizerConfig& cfg = OptimizerConfig{});

  // 26-connected clearance-penalized 3D A* path search on GridMap
  bool planPathAStar(const GridMap& grid, const Vec3& start_ned, const Vec3& goal_ned,
                     std::vector<Vec3>& out_path, double custom_min_clearance = -1.0,
                     double custom_alt_min = -1.0, double custom_alt_max = -1.0) const;

  // Line-of-sight pruning keeping safety clearance
  std::vector<Vec3> shortcutPath(const GridMap& grid, const std::vector<Vec3>& path,
                                 double clearance_m) const;

  // Optimize a cubic uniform B-spline trajectory over a sequence of anchor waypoints
  // (e.g., current state -> intermediate gate anchors -> horizon goal)
  UniformBSplinePrimitive planAndOptimizeTrajectory(
      const GridMap& grid,
      const Vec3& start_pos_ned,
      const Vec3& start_vel_ned,
      const std::vector<Vec3>& anchor_waypoints_ned,
      const Vec3& end_vel_ned,
      double desired_speed_ms = 3.2) const;

  // Optimize control points of an existing UniformBSplinePrimitive in-place
  void optimizeControlPoints(const GridMap& grid,
                             UniformBSplinePrimitive& spline,
                             const std::vector<int>& pinned_indices = {}) const;

  const OptimizerConfig& config() const { return cfg_; }
  void setConfig(const OptimizerConfig& cfg) { cfg_ = cfg; }

 private:
  OptimizerConfig cfg_;
};

}  // namespace ego_planner
