#include "ego_planner/rrt_star_planner.hpp"

#include <algorithm>
#include <cmath>

namespace ego_planner {

RRTStarPlanner::RRTStarPlanner(const RRTStarConfig& cfg)
    : cfg_(cfg), rng_(1337) {}

Vec3 RRTStarPlanner::sampleState(const Vec3& goal_ned, double goal_bias) {
  if (dist_01_(rng_) < goal_bias) {
    return goal_ned;
  }
  double rx = dist_01_(rng_);
  double ry = dist_01_(rng_);
  double rz = dist_01_(rng_);

  return Vec3{
      cfg_.bounds_min_ned.x + rx * (cfg_.bounds_max_ned.x - cfg_.bounds_min_ned.x),
      cfg_.bounds_min_ned.y + ry * (cfg_.bounds_max_ned.y - cfg_.bounds_min_ned.y),
      cfg_.bounds_min_ned.z + rz * (cfg_.bounds_max_ned.z - cfg_.bounds_min_ned.z),
  };
}

int RRTStarPlanner::findNearest(const std::vector<RRTNode>& tree, const Vec3& sample) const {
  if (tree.empty()) return -1;
  double best_dist_sq = 1e9;
  int best_idx = -1;
  for (size_t i = 0; i < tree.size(); ++i) {
    double d_sq = (tree[i].pos_ned - sample).normSq();
    if (d_sq < best_dist_sq) {
      best_dist_sq = d_sq;
      best_idx = static_cast<int>(i);
    }
  }
  return best_idx;
}

Vec3 RRTStarPlanner::steer(const Vec3& from, const Vec3& to, double max_step) const {
  Vec3 dir = to - from;
  double d = dir.norm();
  if (d <= max_step || d < 1e-6) {
    return to;
  }
  return from + dir * (max_step / d);
}

bool RRTStarPlanner::isEdgeCollisionFree(const Vec3& from, const Vec3& to,
                                        const GridMap& grid_map, double radius_m) const {
  return grid_map.isSegmentClear(from, to, radius_m, 0.04);
}

std::vector<int> RRTStarPlanner::findNearNodes(const std::vector<RRTNode>& tree,
                                              const Vec3& new_pos, double radius) const {
  std::vector<int> near_indices;
  double r_sq = radius * radius;
  for (size_t i = 0; i < tree.size(); ++i) {
    if ((tree[i].pos_ned - new_pos).normSq() <= r_sq) {
      near_indices.push_back(static_cast<int>(i));
    }
  }
  return near_indices;
}

int RRTStarPlanner::chooseBestParent(const std::vector<RRTNode>& tree,
                                    const std::vector<int>& near_indices,
                                    const Vec3& new_pos, const GridMap& grid_map,
                                    double& out_min_cost) const {
  int best_parent = -1;
  double min_cost = 1e9;

  for (int idx : near_indices) {
    double edge_cost = (new_pos - tree[idx].pos_ned).norm();
    double total_cost = tree[idx].cost_from_start + edge_cost;
    if (total_cost < min_cost) {
      if (isEdgeCollisionFree(tree[idx].pos_ned, new_pos, grid_map, cfg_.robot_radius_m)) {
        min_cost = total_cost;
        best_parent = idx;
      }
    }
  }

  out_min_cost = min_cost;
  return best_parent;
}

void RRTStarPlanner::rewire(std::vector<RRTNode>& tree, const std::vector<int>& near_indices,
                           int new_node_idx, const GridMap& grid_map) {
  const auto& new_node = tree[new_node_idx];
  for (int idx : near_indices) {
    if (idx == new_node.parent_idx) continue;
    double potential_cost = new_node.cost_from_start + (tree[idx].pos_ned - new_node.pos_ned).norm();
    if (potential_cost < tree[idx].cost_from_start) {
      if (isEdgeCollisionFree(new_node.pos_ned, tree[idx].pos_ned, grid_map, cfg_.robot_radius_m)) {
        tree[idx].parent_idx = new_node_idx;
        tree[idx].cost_from_start = potential_cost;
      }
    }
  }
}

bool RRTStarPlanner::plan(const Vec3& start_ned, const Vec3& goal_ned,
                         const GridMap& grid_map,
                         std::vector<Vec3>& out_waypoints) {
  out_waypoints.clear();

  // Validate start and goal
  if (!grid_map.isCollisionFree(start_ned, cfg_.robot_radius_m)) return false;
  if (!grid_map.isCollisionFree(goal_ned, cfg_.robot_radius_m)) return false;

  // Direct line of sight check
  if (isEdgeCollisionFree(start_ned, goal_ned, grid_map, cfg_.robot_radius_m)) {
    out_waypoints = {start_ned, goal_ned};
    return true;
  }

  std::vector<RRTNode> tree;
  tree.reserve(cfg_.max_iterations + 1);
  tree.push_back(RRTNode{start_ned, -1, 0.0});

  int best_goal_idx = -1;
  double best_goal_cost = 1e9;
  double goal_reach_threshold = cfg_.step_size_m;

  for (int iter = 0; iter < cfg_.max_iterations; ++iter) {
    Vec3 sample = sampleState(goal_ned, cfg_.goal_bias);
    int nearest_idx = findNearest(tree, sample);
    if (nearest_idx < 0) continue;

    Vec3 new_pos = steer(tree[nearest_idx].pos_ned, sample, cfg_.step_size_m);
    if (!grid_map.isCollisionFree(new_pos, cfg_.robot_radius_m)) continue;

    // Asymptotically optimal neighborhood radius: r = min(gamma * (log(N)/N)^(1/3), r_max)
    double n = static_cast<double>(tree.size());
    double radius = std::min(cfg_.search_radius_gamma * std::cbrt(std::log(n + 1.0) / (n + 1.0)),
                             cfg_.max_search_radius_m);
    radius = std::max(radius, cfg_.step_size_m);

    std::vector<int> near_indices = findNearNodes(tree, new_pos, radius);
    if (near_indices.empty()) {
      near_indices.push_back(nearest_idx);
    }

    double min_cost = 0.0;
    int best_parent = chooseBestParent(tree, near_indices, new_pos, grid_map, min_cost);
    if (best_parent < 0) continue;

    int new_idx = static_cast<int>(tree.size());
    tree.push_back(RRTNode{new_pos, best_parent, min_cost});

    // Rewire near nodes
    rewire(tree, near_indices, new_idx, grid_map);

    // Check if goal reached
    double d_to_goal = (new_pos - goal_ned).norm();
    if (d_to_goal < goal_reach_threshold) {
      if (isEdgeCollisionFree(new_pos, goal_ned, grid_map, cfg_.robot_radius_m)) {
        double cost = min_cost + d_to_goal;
        if (cost < best_goal_cost) {
          best_goal_cost = cost;
          best_goal_idx = new_idx;
        }
      }
    }
  }

  if (best_goal_idx < 0) {
    return false;
  }

  // Trace back path
  out_waypoints.push_back(goal_ned);
  int curr = best_goal_idx;
  while (curr >= 0) {
    out_waypoints.push_back(tree[curr].pos_ned);
    curr = tree[curr].parent_idx;
  }
  std::reverse(out_waypoints.begin(), out_waypoints.end());
  return true;
}

bool RRTStarPlanner::generateTrajectory(const std::vector<Vec3>& waypoints,
                                       double cruise_speed_m_s,
                                       GlobalTrajectory& out_trajectory) {
  out_trajectory.clear();
  if (waypoints.size() < 2) return false;

  double current_time = 0.0;
  double current_arc = 0.0;

  for (size_t i = 0; i < waypoints.size(); ++i) {
    GlobalTrajectoryPoint pt;
    pt.pos_ned = waypoints[i];
    pt.arc_length_m = current_arc;
    pt.time_s = current_time;

    if (i + 1 < waypoints.size()) {
      Vec3 segment = waypoints[i + 1] - waypoints[i];
      double seg_len = segment.norm();
      Vec3 dir = seg_len > 1e-6 ? segment / seg_len : Vec3{1.0, 0.0, 0.0};
      pt.vel_ned = dir * cruise_speed_m_s;
      pt.yaw_rad = std::atan2(dir.y, dir.x);

      double seg_dt = seg_len / std::max(cruise_speed_m_s, 0.5);
      current_time += seg_dt;
      current_arc += seg_len;
    } else {
      pt.vel_ned = Vec3{0.0, 0.0, 0.0};
      pt.yaw_rad = out_trajectory.points.empty() ? 0.0 : out_trajectory.points.back().yaw_rad;
    }

    out_trajectory.points.push_back(pt);
  }

  out_trajectory.total_duration_s = current_time;
  out_trajectory.total_length_m = current_arc;
  return true;
}

}  // namespace ego_planner
