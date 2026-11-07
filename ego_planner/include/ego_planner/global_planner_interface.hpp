#pragma once

#include <memory>
#include <string>
#include <vector>

#include "ego_planner/bspline_optimizer.hpp"
#include "ego_planner/grid_map.hpp"
#include "ego_planner/math_types.hpp"

namespace ego_planner {

/**
 * @brief Point on a continuous global trajectory.
 */
struct GlobalTrajectoryPoint {
  Vec3 pos_ned{0.0, 0.0, 0.0};
  Vec3 vel_ned{0.0, 0.0, 0.0};
  Vec3 acc_ned{0.0, 0.0, 0.0};
  double yaw_rad{0.0};
  double yaw_rate_rad_s{0.0};
  double time_s{0.0};
  double arc_length_m{0.0};
  double curvature{0.0};
};

/**
 * @brief Continuous global trajectory representation (ideal reference for MPPI / tracker).
 */
struct GlobalTrajectory {
  std::vector<GlobalTrajectoryPoint> points;
  double total_length_m{0.0};
  double total_duration_s{0.0};

  bool empty() const { return points.empty(); }
  void clear() {
    points.clear();
    total_length_m = 0.0;
    total_duration_s = 0.0;
  }

  GlobalTrajectoryPoint evaluateAtTime(double t) const {
    if (points.empty()) return GlobalTrajectoryPoint{};
    if (t <= points.front().time_s) return points.front();
    if (t >= points.back().time_s) return points.back();

    // Binary search
    size_t lo = 0, hi = points.size() - 1;
    while (hi - lo > 1) {
      size_t mid = (lo + hi) / 2;
      if (points[mid].time_s <= t) lo = mid;
      else hi = mid;
    }
    double dt = points[hi].time_s - points[lo].time_s;
    double alpha = dt > 1e-6 ? (t - points[lo].time_s) / dt : 0.0;

    GlobalTrajectoryPoint p;
    p.pos_ned = points[lo].pos_ned * (1.0 - alpha) + points[hi].pos_ned * alpha;
    p.vel_ned = points[lo].vel_ned * (1.0 - alpha) + points[hi].vel_ned * alpha;
    p.acc_ned = points[lo].acc_ned * (1.0 - alpha) + points[hi].acc_ned * alpha;
    p.yaw_rad = wrapPi(points[lo].yaw_rad + alpha * wrapPi(points[hi].yaw_rad - points[lo].yaw_rad));
    p.yaw_rate_rad_s = points[lo].yaw_rate_rad_s * (1.0 - alpha) + points[hi].yaw_rate_rad_s * alpha;
    p.time_s = t;
    p.arc_length_m = points[lo].arc_length_m * (1.0 - alpha) + points[hi].arc_length_m * alpha;
    p.curvature = points[lo].curvature * (1.0 - alpha) + points[hi].curvature * alpha;
    return p;
  }

  GlobalTrajectoryPoint projectPoint(const Vec3& p, double& out_arc_s) const {
    if (points.empty()) {
      out_arc_s = 0.0;
      return GlobalTrajectoryPoint{};
    }
    double best_d_sq = 1e9;
    size_t best_idx = 0;
    for (size_t i = 0; i < points.size(); ++i) {
      double d_sq = (points[i].pos_ned - p).normSq();
      if (d_sq < best_d_sq) {
        best_d_sq = d_sq;
        best_idx = i;
      }
    }
    out_arc_s = points[best_idx].arc_length_m;
    return points[best_idx];
  }
};

/**
 * @brief Abstract Base Class for Global Path & Trajectory Planners.
 * Implement this interface to plug in RRT*, Informed RRT*, Kinodynamic RRT*, etc.
 */
class GlobalPlannerBase {
 public:
  virtual ~GlobalPlannerBase() = default;

  virtual std::string name() const = 0;

  /**
   * @brief Compute a collision-free geometric path from start to goal.
   * @param start_ned Start point in NED (m).
   * @param goal_ned Goal point in NED (m).
   * @param grid_map Continuous ESDF grid map for collision checking.
   * @param out_waypoints Output piecewise linear path.
   * @return true if a valid path was found.
   */
  virtual bool plan(const Vec3& start_ned, const Vec3& goal_ned,
                    const GridMap& grid_map,
                    std::vector<Vec3>& out_waypoints) = 0;

  /**
   * @brief Convert discrete waypoints into a continuous time-parameterized trajectory.
   * @param waypoints Collision-free path points.
   * @param cruise_speed_m_s Nominal speed along the trajectory.
   * @param out_trajectory Sampled continuous trajectory.
   * @return true if trajectory generation succeeded.
   */
  virtual bool generateTrajectory(const std::vector<Vec3>& waypoints,
                                  double cruise_speed_m_s,
                                  GlobalTrajectory& out_trajectory) = 0;
};

/**
 * @brief Production B-Spline Global Planner (A* + Gradient Descent Optimization).
 * Default working implementation.
 */
class BSplineGlobalPlanner : public GlobalPlannerBase {
 public:
  explicit BSplineGlobalPlanner(const OptimizerConfig& cfg = OptimizerConfig{})
      : optimizer_(cfg) {}

  std::string name() const override { return "bspline"; }

  bool plan(const Vec3& start_ned, const Vec3& goal_ned,
            const GridMap& grid_map,
            std::vector<Vec3>& out_waypoints) override {
    return optimizer_.planPathAStar(grid_map, start_ned, goal_ned, out_waypoints);
  }

  bool generateTrajectory(const std::vector<Vec3>& waypoints,
                          double cruise_speed_m_s,
                          GlobalTrajectory& out_trajectory) override;

  const BSplineOptimizer& optimizer() const { return optimizer_; }

 private:
  BSplineOptimizer optimizer_;
};

}  // namespace ego_planner
