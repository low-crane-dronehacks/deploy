#include "maze_planning/rrt_star_planner.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace maze_planning {

RRTStarPlanner::RRTStarPlanner(const RRTStarConfig& cfg,
                               std::shared_ptr<GroundTruthMap> gt_map,
                               std::shared_ptr<const VoxelCostmap> costmap)
    : cfg_(cfg), gt_map_(std::move(gt_map)), costmap_(std::move(costmap)), rng_(1337) {
  if (!gt_map_) {
    gt_map_ = std::make_shared<GroundTruthMap>();
    if (!gt_map_->loadFromBinary("config/course_groundtruth.bin")) {
      if (!gt_map_->loadFromObj("sitl/world/course.obj")) {
        gt_map_->initDefaultCourse();
      }
    }
  }
}

Eigen::Vector3d RRTStarPlanner::steer(const Eigen::Vector3d& from,
                                      const Eigen::Vector3d& to,
                                      double max_step) const {
  const Eigen::Vector3d d = to - from;
  const double len = d.norm();
  if (len <= max_step || len < 1e-8) return to;
  return from + (d / len) * max_step;
}

bool RRTStarPlanner::isEdgeValid(const Eigen::Vector3d& a, const Eigen::Vector3d& b,
                                 double radius_m, double alt_min_m, double alt_max_m) const {
  if (!gt_map_->isSegmentValid(a, b, radius_m, 0.025, alt_min_m, alt_max_m)) return false;
  if (costmap_ && costmap_->numDynamicObstacles() > 0) {
    if (!costmap_->isSegmentClear(a, b, radius_m)) return false;
  }
  return true;
}

double RRTStarPlanner::evaluateEdgeCost(const Eigen::Vector3d& from,
                                        const Eigen::Vector3d& to) const {
  const Eigen::Vector3d ab = to - from;
  const double dist = ab.norm();
  if (dist < 1e-6) return 0.0;

  constexpr int kSamples = 4;
  double penalty_sum = 0.0;
  for (int i = 1; i <= kSamples; ++i) {
    const double u = static_cast<double>(i) / static_cast<double>(kSamples);
    const Eigen::Vector3d p = from + u * ab;
    double c = gt_map_->clearance(p);
    if (costmap_ && costmap_->numDynamicObstacles() > 0) {
      c = std::min(c, costmap_->clearance(p));
    }
    if (c < cfg_.comfort_clearance_m) {
      const double diff = cfg_.comfort_clearance_m - c;
      penalty_sum += diff * diff;
    }
  }
  return dist * (1.0 + cfg_.comfort_weight * (penalty_sum / kSamples));
}

void RRTStarPlanner::propagateSubtreeCost(std::vector<RRTNode>& tree, int root_idx) const {
  std::vector<int> stack = tree[root_idx].children;
  while (!stack.empty()) {
    const int curr = stack.back();
    stack.pop_back();
    const int p = tree[curr].parent_idx;
    if (p >= 0) {
      tree[curr].euclidean_from_start =
          tree[p].euclidean_from_start + (tree[curr].pos_ned - tree[p].pos_ned).norm();
      tree[curr].cost_from_start =
          tree[p].cost_from_start + evaluateEdgeCost(tree[p].pos_ned, tree[curr].pos_ned);
    }
    for (int ch : tree[curr].children) stack.push_back(ch);
  }
}

Eigen::Vector3d RRTStarPlanner::sampleInformedState(const Eigen::Vector3d& start_ned,
                                                    const Eigen::Vector3d& goal_ned,
                                                    double c_best_euclid,
                                                    const Eigen::Vector3d& box_lo,
                                                    const Eigen::Vector3d& box_hi) {
  if (dist_01_(rng_) < cfg_.goal_bias) return goal_ned;

  const Eigen::Vector3d diff = goal_ned - start_ned;
  const double c_min = diff.norm();

  if (std::isfinite(c_best_euclid) && c_best_euclid > c_min + 1e-3 &&
      dist_01_(rng_) < cfg_.informed_bias) {
    const Eigen::Vector3d center = 0.5 * (start_ned + goal_ned);
    const double a1 = 0.5 * c_best_euclid;
    const double r_perp = 0.5 * std::sqrt(std::max(0.0, c_best_euclid * c_best_euclid - c_min * c_min));

    // Uniform sample in 3D unit ball via normal direction + u^(1/3) radius
    double gx = dist_norm_(rng_), gy = dist_norm_(rng_), gz = dist_norm_(rng_);
    const double gn = std::max(1e-8, std::sqrt(gx * gx + gy * gy + gz * gz));
    const double rad = std::cbrt(dist_01_(rng_));
    gx = (gx / gn) * rad;
    gy = (gy / gn) * rad;
    gz = (gz / gn) * rad;

    const Eigen::Vector3d e1 = (c_min > 1e-6) ? (diff / c_min) : Eigen::Vector3d(1.0, 0.0, 0.0);
    const Eigen::Vector3d ref =
        (std::abs(e1.z()) < 0.9) ? Eigen::Vector3d(0.0, 0.0, 1.0) : Eigen::Vector3d(0.0, 1.0, 0.0);
    const Eigen::Vector3d e2 = e1.cross(ref).normalized();
    const Eigen::Vector3d e3 = e1.cross(e2);

    const Eigen::Vector3d x_ellip =
        center + e1 * (a1 * gx) + e2 * (r_perp * gy) + e3 * (r_perp * gz);
    return Eigen::Vector3d(
        std::clamp(x_ellip.x(), box_lo.x(), box_hi.x()),
        std::clamp(x_ellip.y(), box_lo.y(), box_hi.y()),
        std::clamp(x_ellip.z(), box_lo.z(), box_hi.z()));
  }

  return Eigen::Vector3d(
      box_lo.x() + dist_01_(rng_) * (box_hi.x() - box_lo.x()),
      box_lo.y() + dist_01_(rng_) * (box_hi.y() - box_lo.y()),
      box_lo.z() + dist_01_(rng_) * (box_hi.z() - box_lo.z()));
}

bool RRTStarPlanner::plan(const Eigen::Vector3d& start_map,
                          const Eigen::Vector3d& goal_map,
                          PlannedPath& out_path,
                          std::vector<Eigen::Vector3d>* out_raw_waypoints,
                          int max_iters_override) {
  std::vector<Eigen::Vector3d> wps;
  const double min_z = std::min(start_map.z(), goal_map.z());
  const double max_z = std::max(start_map.z(), goal_map.z());
  const double alt_min = std::max(0.16, std::min(-max_z - 0.12, 0.20));
  const double alt_max = std::min(1.65, std::max(-min_z + 0.25, 1.55));
  const double radius = cfg_.robot_radius_m;

  // Fast direct-line check when clearance is high everywhere along the segment
  if (isEdgeValid(start_map, goal_map, radius + 0.04, alt_min, alt_max)) {
    wps = {start_map, goal_map};
    wps = shortcutAndCenterPath(wps, radius);
    if (out_raw_waypoints) *out_raw_waypoints = wps;
    out_path = smoothWithCubicBSpline(wps, cfg_.cruise_speed_m_s);
    last_tree_size_ = 2;
    return true;
  }

  std::vector<RRTNode> tree;
  const int max_iters = (max_iters_override > 0) ? max_iters_override : cfg_.max_iterations;
  tree.reserve(static_cast<std::size_t>(max_iters) + 2);
  tree.push_back(RRTNode{start_map, -1, 0.0, 0.0, {}});

  const Eigen::Vector3d box_lo = cfg_.bounds_min_ned;
  const Eigen::Vector3d box_hi(
      cfg_.bounds_max_ned.x(),
      cfg_.bounds_max_ned.y(),
      std::max(cfg_.bounds_max_ned.z(), max_z + 0.08));

  int best_goal_idx = -1;
  double best_goal_cost = std::numeric_limits<double>::infinity();
  double best_goal_euclid = std::numeric_limits<double>::infinity();

  for (int iter = 0; iter < max_iters; ++iter) {
    const Eigen::Vector3d x_rand =
        sampleInformedState(start_map, goal_map, best_goal_euclid, box_lo, box_hi);

    // 1. Nearest neighbor in tree
    int nearest_idx = 0;
    double best_d2 = (tree[0].pos_ned - x_rand).squaredNorm();
    for (std::size_t i = 1; i < tree.size(); ++i) {
      const double d2 = (tree[i].pos_ned - x_rand).squaredNorm();
      if (d2 < best_d2) {
        best_d2 = d2;
        nearest_idx = static_cast<int>(i);
      }
    }

    const Eigen::Vector3d x_new = steer(tree[nearest_idx].pos_ned, x_rand, cfg_.step_size_m);
    if ((x_new - tree[nearest_idx].pos_ned).norm() < 1e-4) continue;
    if (!isEdgeValid(tree[nearest_idx].pos_ned, x_new, radius, alt_min, alt_max)) continue;

    // 2. Asymptotic neighborhood query: r_n = min(gamma * (log(n)/n)^(1/3), r_max)
    const double n_d = static_cast<double>(tree.size() + 1);
    const double r_rewire = std::min(
        cfg_.max_search_radius_m,
        std::max(cfg_.step_size_m * 1.4,
                 cfg_.search_radius_gamma * std::cbrt(std::log(n_d) / n_d)));
    const double r2 = r_rewire * r_rewire;

    std::vector<int> near_indices;
    near_indices.reserve(24);
    for (std::size_t i = 0; i < tree.size(); ++i) {
      if ((tree[i].pos_ned - x_new).squaredNorm() <= r2) {
        near_indices.push_back(static_cast<int>(i));
      }
    }

    // 3. Choose best parent minimizing clearance-regularized cost_from_start
    int best_parent = nearest_idx;
    double min_cost =
        tree[nearest_idx].cost_from_start + evaluateEdgeCost(tree[nearest_idx].pos_ned, x_new);
    for (int idx : near_indices) {
      if (idx == nearest_idx) continue;
      const double cand_cost =
          tree[idx].cost_from_start + evaluateEdgeCost(tree[idx].pos_ned, x_new);
      if (cand_cost + 1e-6 < min_cost) {
        if (isEdgeValid(tree[idx].pos_ned, x_new, radius, alt_min, alt_max)) {
          min_cost = cand_cost;
          best_parent = idx;
        }
      }
    }

    const int new_idx = static_cast<int>(tree.size());
    const double new_euclid =
        tree[best_parent].euclidean_from_start + (x_new - tree[best_parent].pos_ned).norm();
    tree.push_back(RRTNode{x_new, best_parent, min_cost, new_euclid, {}});
    tree[best_parent].children.push_back(new_idx);

    // 4. Rewire neighborhood through x_new if it lowers cost
    for (int idx : near_indices) {
      if (idx == best_parent || idx == 0) continue;
      const double cand_cost = min_cost + evaluateEdgeCost(x_new, tree[idx].pos_ned);
      if (cand_cost + 1e-5 < tree[idx].cost_from_start) {
        if (isEdgeValid(x_new, tree[idx].pos_ned, radius, alt_min, alt_max)) {
          const int old_p = tree[idx].parent_idx;
          if (old_p >= 0) {
            auto& ch = tree[old_p].children;
            ch.erase(std::remove(ch.begin(), ch.end(), idx), ch.end());
          }
          tree[idx].parent_idx = new_idx;
          tree[idx].cost_from_start = cand_cost;
          tree[idx].euclidean_from_start = new_euclid + (tree[idx].pos_ned - x_new).norm();
          tree[new_idx].children.push_back(idx);
          propagateSubtreeCost(tree, idx);
        }
      }
    }

    // 5. Try connecting x_new to goal_map
    const double dist_to_goal = (x_new - goal_map).norm();
    if (dist_to_goal <= cfg_.step_size_m * 2.2) {
      if (isEdgeValid(x_new, goal_map, radius, alt_min, alt_max)) {
        const double total_cost = min_cost + evaluateEdgeCost(x_new, goal_map);
        if (total_cost < best_goal_cost) {
          best_goal_cost = total_cost;
          best_goal_euclid = new_euclid + dist_to_goal;
          best_goal_idx = new_idx;
        }
      }
    }

    if (best_goal_idx >= 0 && iter >= 140 && (iter % 35 == 0)) {
      const double direct_d = (goal_map - start_map).norm();
      if (best_goal_euclid <= 1.28 * direct_d) break;
    }
  }

  last_tree_size_ = tree.size();

  // If exact connection to goal was not made, pick node closest to goal that is collision-free
  if (best_goal_idx < 0) {
    int closest_idx = 0;
    double closest_d = (tree[0].pos_ned - goal_map).norm();
    for (std::size_t i = 1; i < tree.size(); ++i) {
      const double d = (tree[i].pos_ned - goal_map).norm();
      if (d < closest_d) {
        closest_d = d;
        closest_idx = static_cast<int>(i);
      }
    }
    best_goal_idx = closest_idx;
  }

  std::vector<Eigen::Vector3d> rev;
  if (best_goal_cost < std::numeric_limits<double>::infinity() &&
      (tree[best_goal_idx].pos_ned - goal_map).norm() > 1e-3) {
    rev.push_back(goal_map);
  }
  int curr = best_goal_idx;
  while (curr >= 0) {
    rev.push_back(tree[curr].pos_ned);
    curr = tree[curr].parent_idx;
  }
  std::reverse(rev.begin(), rev.end());

  wps = shortcutAndCenterPath(rev, std::max(0.17, radius - 0.02));
  if (out_raw_waypoints) *out_raw_waypoints = wps;
  out_path = smoothWithCubicBSpline(wps, cfg_.cruise_speed_m_s);
  return (best_goal_cost < std::numeric_limits<double>::infinity()) || wps.size() >= 2;
}

std::vector<Eigen::Vector3d> RRTStarPlanner::shortcutAndCenterPath(
    const std::vector<Eigen::Vector3d>& raw_path,
    double radius_m) const {
  if (raw_path.size() <= 2) return raw_path;

  double min_z = raw_path.front().z(), max_z = raw_path.front().z();
  for (const auto& p : raw_path) {
    min_z = std::min(min_z, p.z());
    max_z = std::max(max_z, p.z());
  }
  const double alt_min = std::max(0.16, std::min(-max_z - 0.10, 0.20));
  const double alt_max = std::min(1.65, std::max(-min_z + 0.20, 1.55));

  // 1. Greedy line-of-sight shortcutting
  std::vector<Eigen::Vector3d> shortcut;
  shortcut.push_back(raw_path.front());
  std::size_t idx = 0;
  while (idx + 1 < raw_path.size()) {
    std::size_t next_idx = idx + 1;
    for (std::size_t j = raw_path.size() - 1; j > idx + 1; --j) {
      if ((raw_path[j] - raw_path[idx]).norm() > 2.2) continue;
      if (isEdgeValid(raw_path[idx], raw_path[j], radius_m + 0.03, alt_min, alt_max)) {
        next_idx = j;
        break;
      }
    }
    shortcut.push_back(raw_path[next_idx]);
    idx = next_idx;
  }

  // 2. Subdivide long edges and push interior vertices toward the corridor medial axis
  //    only when the resulting segments to the neighbors remain strictly valid.
  std::vector<Eigen::Vector3d> refined;
  refined.push_back(shortcut.front());
  for (std::size_t i = 0; i + 1 < shortcut.size(); ++i) {
    const Eigen::Vector3d a = refined.back();
    const Eigen::Vector3d b = shortcut[i + 1];
    const double seg_len = (b - a).norm();
    const int n_sub = std::max(1, static_cast<int>(std::ceil(seg_len / 0.35)));
    for (int k = 1; k <= n_sub; ++k) {
      Eigen::Vector3d pt = a + (b - a) * (static_cast<double>(k) / n_sub);
      if (i + 1 < shortcut.size() - 1 || k < n_sub) {
        for (int rep = 0; rep < 3; ++rep) {
          Eigen::Vector3d grad(0.0, 0.0, 0.0);
          const double d = costmap_ ? costmap_->clearanceAndGradient(pt, grad)
                                    : gt_map_->clearanceAndGradient(pt, grad);
          if (d >= cfg_.comfort_clearance_m) break;
          const double step = std::min(0.05, 0.50 * (cfg_.comfort_clearance_m - d));
          Eigen::Vector3d cand(pt.x() + grad.x() * step, pt.y() + grad.y() * step, pt.z());
          const double d_cand = costmap_ ? costmap_->clearance(cand) : gt_map_->clearance(cand);
          if (d_cand > d &&
              isEdgeValid(refined.back(), cand, radius_m, alt_min, alt_max) &&
              isEdgeValid(cand, b, radius_m, alt_min, alt_max)) {
            pt = cand;
          }
        }
      }
      refined.push_back(pt);
    }
  }
  return refined;
}

PlannedPath RRTStarPlanner::smoothWithCubicBSpline(
    const std::vector<Eigen::Vector3d>& waypoints,
    double desired_speed_m_s) const {
  PlannedPath out;
  out.frame_id = "map";
  if (waypoints.empty()) return out;
  if (waypoints.size() == 1) {
    out.points.push_back(PathPoint{waypoints[0], Eigen::Vector3d::Zero(),
                                   Eigen::Vector3d::Zero(), 0.0, 0.0, 0.0, 0.0});
    return out;
  }

  // Build clamped uniform cubic B-spline control points (p_-1, p_0, ..., p_M, p_M+1)
  std::vector<Eigen::Vector3d> ctrl;
  ctrl.reserve(waypoints.size() + 4);
  ctrl.push_back(waypoints.front());
  ctrl.push_back(waypoints.front());
  for (const auto& w : waypoints) ctrl.push_back(w);
  ctrl.push_back(waypoints.back());
  ctrl.push_back(waypoints.back());

  double raw_len = 0.0;
  for (std::size_t i = 0; i + 1 < waypoints.size(); ++i) {
    raw_len += (waypoints[i + 1] - waypoints[i]).norm();
  }
  const int num_spans = static_cast<int>(ctrl.size()) - 3;
  const double total_dur =
      std::max(0.25, raw_len / std::max(0.3, desired_speed_m_s));
  const double knot_dt = total_dur / std::max(1, num_spans);

  const int n_samples = std::max(2, static_cast<int>(std::ceil(total_dur / cfg_.bspline_dt_s)) + 1);
  out.points.reserve(static_cast<std::size_t>(n_samples));

  double accum_s = 0.0;
  double prev_yaw = 0.0;
  for (int k = 0; k < n_samples; ++k) {
    const double t = std::min(total_dur, k * cfg_.bspline_dt_s);
    const double u_global = std::clamp(t / knot_dt, 0.0, static_cast<double>(num_spans) - 1e-7);
    const int span = std::clamp(static_cast<int>(std::floor(u_global)), 0, num_spans - 1);
    const double u = u_global - span;
    const double u2 = u * u;
    const double u3 = u2 * u;

    // Cubic B-spline basis functions and derivatives
    const double b0 = (1.0 - 3.0 * u + 3.0 * u2 - u3) / 6.0;
    const double b1 = (4.0 - 6.0 * u2 + 3.0 * u3) / 6.0;
    const double b2 = (1.0 + 3.0 * u + 3.0 * u2 - 3.0 * u3) / 6.0;
    const double b3 = u3 / 6.0;

    const double db0 = (-3.0 + 6.0 * u - 3.0 * u2) / (6.0 * knot_dt);
    const double db1 = (-12.0 * u + 9.0 * u2) / (6.0 * knot_dt);
    const double db2 = (3.0 + 6.0 * u - 9.0 * u2) / (6.0 * knot_dt);
    const double db3 = (3.0 * u2) / (6.0 * knot_dt);

    const double ddb0 = (6.0 - 6.0 * u) / (6.0 * knot_dt * knot_dt);
    const double ddb1 = (-12.0 + 18.0 * u) / (6.0 * knot_dt * knot_dt);
    const double ddb2 = (6.0 - 18.0 * u) / (6.0 * knot_dt * knot_dt);
    const double ddb3 = (6.0 * u) / (6.0 * knot_dt * knot_dt);

    const Eigen::Vector3d& p0 = ctrl[span];
    const Eigen::Vector3d& p1 = ctrl[span + 1];
    const Eigen::Vector3d& p2 = ctrl[span + 2];
    const Eigen::Vector3d& p3 = ctrl[span + 3];

    PathPoint pt{};
    pt.pos = b0 * p0 + b1 * p1 + b2 * p2 + b3 * p3;
    pt.vel = db0 * p0 + db1 * p1 + db2 * p2 + db3 * p3;
    pt.acc = ddb0 * p0 + ddb1 * p1 + ddb2 * p2 + ddb3 * p3;
    pt.t_s = t;

    if (k > 0) {
      accum_s += (pt.pos - out.points.back().pos).norm();
    }
    pt.s_m = accum_s;

    const double v_xy = std::hypot(pt.vel.x(), pt.vel.y());
    if (v_xy > 0.08) {
      pt.yaw_rad = std::atan2(pt.vel.y(), pt.vel.x());
    } else if (k == 0 && waypoints.size() >= 2) {
      const Eigen::Vector3d d0 = waypoints[1] - waypoints[0];
      pt.yaw_rad = std::atan2(d0.y(), d0.x());
    } else {
      pt.yaw_rad = prev_yaw;
    }
    if (k > 0) {
      pt.yawspeed_rad_s = wrapPi(pt.yaw_rad - prev_yaw) / std::max(1e-4, cfg_.bspline_dt_s);
    }
    prev_yaw = pt.yaw_rad;
    out.points.push_back(pt);
  }

  out.total_length_m = accum_s;
  out.total_duration_s = total_dur;
  return out;
}

bool RRTStarPlanner::planMultiWaypoint(
    const Eigen::Vector3d& start_map,
    const std::vector<Eigen::Vector3d>& anchor_waypoints_map,
    PlannedPath& out_path,
    std::vector<Eigen::Vector3d>* out_waypoints) {
  if (anchor_waypoints_map.empty()) return false;

  std::vector<Eigen::Vector3d> combined;
  combined.push_back(start_map);
  Eigen::Vector3d curr = start_map;

  for (const auto& anchor : anchor_waypoints_map) {
    if ((anchor - curr).norm() < 0.08) continue;
    PlannedPath leg_path;
    std::vector<Eigen::Vector3d> leg_wps;
    plan(curr, anchor, leg_path, &leg_wps, cfg_.leg_iterations);
    for (std::size_t i = 1; i < leg_wps.size(); ++i) {
      combined.push_back(leg_wps[i]);
    }
    curr = combined.back();
  }

  if (out_waypoints) *out_waypoints = combined;
  out_path = smoothWithCubicBSpline(combined, cfg_.cruise_speed_m_s);
  return !out_path.empty();
}

bool RRTStarPlanner::planFullCourse(const Eigen::Vector3d& start_map,
                                    bool prefer_window,
                                    bool include_return_leg,
                                    PlannedPath& out_path) {
  const double z_cruise = -0.78;
  // Refine gate centers using GroundTruthMap clearance
  const auto& gates = gt_map_->defaultGates();
  Eigen::Vector3d g0 = (gates.size() > 0) ? gt_map_->refineGateCenter(gates[0])
                                          : Eigen::Vector3d(2.55, -2.21, z_cruise);
  Eigen::Vector3d g1 = (gates.size() > 1) ? gt_map_->refineGateCenter(gates[1])
                                          : Eigen::Vector3d(1.78, -1.03, z_cruise);
  Eigen::Vector3d g2 = (gates.size() > 2) ? gt_map_->refineGateCenter(gates[2])
                                          : Eigen::Vector3d(1.36, 0.85, z_cruise);
  Eigen::Vector3d g3 = (gates.size() > 3) ? gt_map_->refineGateCenter(gates[3])
                                          : Eigen::Vector3d(-1.05, -0.02, z_cruise);
  Eigen::Vector3d g4 = (gates.size() > 4) ? gt_map_->refineGateCenter(gates[4])
                                          : Eigen::Vector3d(-2.283, -0.475, z_cruise);
  g0.z() = z_cruise;
  g1.z() = z_cruise;
  g2.z() = z_cruise;
  g3.z() = z_cruise;
  g4.z() = z_cruise;

  // Check if the window aperture is blocked (e.g. by FMEA corridor blockage); if so, route via lower door
  bool use_window = prefer_window && gt_map_->clearance(g4) > (cfg_.robot_radius_m + 0.03);
  std::vector<Eigen::Vector3d> anchors = {
      Eigen::Vector3d(2.45, -2.42, z_cruise),
      g0,
      Eigen::Vector3d(2.55, -1.40, z_cruise),
      Eigen::Vector3d(1.15, -1.35, z_cruise),
      g1,
      Eigen::Vector3d(1.15, -0.25, z_cruise),
      Eigen::Vector3d(2.45, -0.25, z_cruise),
      g2,
      Eigen::Vector3d(2.45, 0.95, z_cruise),
      Eigen::Vector3d(0.95, 1.05, z_cruise),
      Eigen::Vector3d(0.85, 2.38, z_cruise),
      g3,
      Eigen::Vector3d(-0.35, 2.10, z_cruise),
      Eigen::Vector3d(-1.75, 0.90, z_cruise),
  };
  if (use_window) {
    anchors.push_back(Eigen::Vector3d(-2.15, -0.05, z_cruise));
    anchors.push_back(g4);
    anchors.push_back(Eigen::Vector3d(-2.35, -1.00, z_cruise));
  } else {
    anchors.push_back(Eigen::Vector3d(-1.20, -0.55, z_cruise));
    anchors.push_back(Eigen::Vector3d(-1.20, -1.55, z_cruise));
  }
  const Eigen::Vector3d pad_hover(
      gt_map_->padCenterNed().x(), gt_map_->padCenterNed().y(), z_cruise);
  anchors.push_back(pad_hover);

  if (include_return_leg) {
    anchors.push_back(Eigen::Vector3d(-2.35, -0.95, z_cruise));
    anchors.push_back(g4);
    anchors.push_back(Eigen::Vector3d(-2.12, 0.00, z_cruise));
    anchors.push_back(g3);
    anchors.push_back(Eigen::Vector3d(-0.35, 0.85, z_cruise));
    anchors.push_back(Eigen::Vector3d(0.75, 1.15, z_cruise));
    anchors.push_back(g2);
    anchors.push_back(Eigen::Vector3d(1.75, 0.05, z_cruise));
    anchors.push_back(g1);
    anchors.push_back(Eigen::Vector3d(2.30, -0.50, z_cruise));
    anchors.push_back(Eigen::Vector3d(2.58, -1.45, z_cruise));
    anchors.push_back(g0);
    anchors.push_back(Eigen::Vector3d(
        gt_map_->launchMarkNed().x(), gt_map_->launchMarkNed().y(), z_cruise));
  }

  return planMultiWaypoint(start_map, anchors, out_path);
}

bool RRTStarPlanner::blockCorridorAndReplan(
    const Eigen::Vector3d& current_pos_map,
    const Eigen::Vector3d& blocked_pos_map,
    double block_radius_m,
    const std::vector<Eigen::Vector3d>& remaining_anchors_map,
    PlannedPath& out_path) {
  gt_map_->markCorridorBlocked(blocked_pos_map, block_radius_m);
  if (remaining_anchors_map.empty()) return false;
  return planMultiWaypoint(current_pos_map, remaining_anchors_map, out_path);
}

}  // namespace maze_planning
