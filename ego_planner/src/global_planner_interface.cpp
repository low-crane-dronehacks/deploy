#include "ego_planner/global_planner_interface.hpp"

#include <cmath>

namespace ego_planner {

bool BSplineGlobalPlanner::generateTrajectory(const std::vector<Vec3>& waypoints,
                                             double cruise_speed_m_s,
                                             GlobalTrajectory& out_trajectory) {
  out_trajectory.clear();
  if (waypoints.size() < 2) return false;

  // Build a sample-based trajectory from waypoints using knot interpolation
  double total_t = 0.0;
  double running_arc = 0.0;

  for (size_t i = 0; i < waypoints.size(); ++i) {
    GlobalTrajectoryPoint pt;
    pt.pos_ned = waypoints[i];
    pt.time_s = total_t;
    pt.arc_length_m = running_arc;

    if (i + 1 < waypoints.size()) {
      Vec3 segment = waypoints[i + 1] - waypoints[i];
      double seg_len = segment.norm();
      Vec3 dir = seg_len > 1e-6 ? (segment / seg_len) : Vec3{1.0, 0.0, 0.0};
      pt.vel_ned = dir * cruise_speed_m_s;
      pt.yaw_rad = std::atan2(dir.y, dir.x);

      double dt = seg_len / std::max(cruise_speed_m_s, 0.5);
      total_t += dt;
      running_arc += seg_len;
    } else {
      pt.vel_ned = Vec3{0.0, 0.0, 0.0};
      pt.yaw_rad = out_trajectory.points.empty() ? 0.0 : out_trajectory.points.back().yaw_rad;
    }

    out_trajectory.points.push_back(pt);
  }

  out_trajectory.total_duration_s = total_t;
  out_trajectory.total_length_m = running_arc;
  return true;
}

}  // namespace ego_planner
