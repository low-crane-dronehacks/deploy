#include "ego_planner/bspline_optimizer.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <queue>
#include <unordered_set>

namespace ego_planner {

namespace {

struct AStarNode {
  double f{0.0};
  double g{0.0};
  int idx{0};
  bool operator>(const AStarNode& o) const { return f > o.f; }
};

}  // namespace

BSplineOptimizer::BSplineOptimizer(const OptimizerConfig& cfg) : cfg_(cfg) {}

bool BSplineOptimizer::planPathAStar(const GridMap& grid, const Vec3& start_ned, const Vec3& goal_ned,
                                     std::vector<Vec3>& out_path, double custom_min_clearance,
                                     double custom_alt_min, double custom_alt_max) const {
  out_path.clear();
  const double min_clr = (custom_min_clearance > 0.0) ? custom_min_clearance : cfg_.min_clearance_m;
  const double alt_min = (custom_alt_min > 0.0) ? custom_alt_min : cfg_.alt_min_m;
  const double alt_max = (custom_alt_max > 0.0) ? custom_alt_max : cfg_.alt_max_m;

  int sx = 0, sy = 0, sz = 0;
  int gx = 0, gy = 0, gz = 0;
  grid.posToIndex(start_ned, sx, sy, sz);
  grid.posToIndex(goal_ned, gx, gy, gz);

  const int nx = grid.nx();
  const int ny = grid.ny();
  const int nz = grid.nz();
  const int total = nx * ny * nz;
  const double res = grid.resolution();

  auto to_coords = [&](int flat, int& ix, int& iy, int& iz) {
    iz = flat % nz;
    int rem = flat / nz;
    iy = rem % ny;
    ix = rem / ny;
  };

  // If start or goal voxel has slightly less clearance than min_clr, allow snapping to nearest free voxel within 0.35m
  auto snap_to_free = [&](int& ix, int& iy, int& iz, double req_clr) {
    if (grid.voxelDistance(ix, iy, iz) >= req_clr) return;
    int best_x = ix, best_y = iy, best_z = iz;
    double best_score = -1e9;
    for (int dx = -6; dx <= 6; ++dx) {
      for (int dy = -6; dy <= 6; ++dy) {
        for (int dz = -4; dz <= 4; ++dz) {
          int cx = ix + dx, cy = iy + dy, cz = iz + dz;
          if (!grid.inBounds(cx, cy, cz)) continue;
          Vec3 p = grid.indexToPos(cx, cy, cz);
          double alt = -p.z;
          if (alt < alt_min - 0.08 || alt > alt_max + 0.08) continue;
          double d = grid.voxelDistance(cx, cy, cz);
          if (d >= req_clr) {
            double dist_pen = std::sqrt(static_cast<double>(dx * dx + dy * dy + dz * dz)) * res;
            double score = d - 1.5 * dist_pen;
            if (score > best_score) {
              best_score = score;
              best_x = cx;
              best_y = cy;
              best_z = cz;
            }
          }
        }
      }
    }
    ix = best_x;
    iy = best_y;
    iz = best_z;
  };

  snap_to_free(sx, sy, sz, std::min(min_clr, 0.21));
  snap_to_free(gx, gy, gz, std::min(min_clr, 0.21));

  const int start_flat = grid.flatIndex(sx, sy, sz);
  const int goal_flat = grid.flatIndex(gx, gy, gz);
  if (start_flat == goal_flat) {
    out_path = {start_ned, goal_ned};
    return true;
  }

  std::vector<double> best_g(total, std::numeric_limits<double>::infinity());
  std::vector<int> parent(total, -1);
  std::priority_queue<AStarNode, std::vector<AStarNode>, std::greater<AStarNode>> open;

  auto heuristic = [&](int ix, int iy, int iz) {
    double dx = static_cast<double>(ix - gx);
    double dy = static_cast<double>(iy - gy);
    double dz = static_cast<double>(iz - gz);
    return std::sqrt(dx * dx + dy * dy + dz * dz) * res;
  };

  best_g[start_flat] = 0.0;
  open.push({heuristic(sx, sy, sz), 0.0, start_flat});

  bool found = false;
  while (!open.empty()) {
    AStarNode cur = open.top();
    open.pop();
    if (cur.idx == goal_flat) {
      found = true;
      break;
    }
    if (cur.g > best_g[cur.idx] + 1e-9) {
      continue;
    }

    int cx = 0, cy = 0, cz = 0;
    to_coords(cur.idx, cx, cy, cz);
    const double cur_clr = grid.voxelDistance(cx, cy, cz);
    const double cur_pen =
        1.0 + cfg_.comfort_weight *
                  std::pow(clamp((cfg_.comfort_clearance_m - cur_clr) / cfg_.comfort_clearance_m, 0.0, 1.0), 2.0);

    for (int dx = -1; dx <= 1; ++dx) {
      for (int dy = -1; dy <= 1; ++dy) {
        for (int dz = -1; dz <= 1; ++dz) {
          if (dx == 0 && dy == 0 && dz == 0) continue;
          int nx_i = cx + dx;
          int ny_i = cy + dy;
          int nz_i = cz + dz;
          if (!grid.inBounds(nx_i, ny_i, nz_i)) continue;

          Vec3 np = grid.indexToPos(nx_i, ny_i, nz_i);
          double alt = -np.z;
          if (alt < alt_min || alt > alt_max) continue;
          if (std::fabs(np.x) > 2.86 || std::fabs(np.y) > 2.86) continue;

          int nflat = grid.flatIndex(nx_i, ny_i, nz_i);
          double nclr = grid.voxelDistance(nx_i, ny_i, nz_i);
          if (nclr < min_clr && nflat != goal_flat) continue;

          double step_len = std::sqrt(static_cast<double>(dx * dx + dy * dy + dz * dz)) * res;
          // Mild vertical change cost so paths stay level unless needed
          if (dz != 0) step_len *= 1.15;
          double n_pen =
              1.0 + cfg_.comfort_weight *
                        std::pow(clamp((cfg_.comfort_clearance_m - nclr) / cfg_.comfort_clearance_m, 0.0, 1.0), 2.0);
          double ng = cur.g + step_len * 0.5 * (cur_pen + n_pen);
          if (ng < best_g[nflat]) {
            best_g[nflat] = ng;
            parent[nflat] = cur.idx;
            open.push({ng + heuristic(nx_i, ny_i, nz_i), ng, nflat});
          }
        }
      }
    }
  }

  if (!found) {
    return false;
  }

  std::vector<Vec3> rev;
  int cur = goal_flat;
  while (cur != -1) {
    int ix = 0, iy = 0, iz = 0;
    to_coords(cur, ix, iy, iz);
    rev.push_back(grid.indexToPos(ix, iy, iz));
    if (cur == start_flat) break;
    cur = parent[cur];
  }
  std::reverse(rev.begin(), rev.end());
  rev.front() = start_ned;
  rev.back() = goal_ned;
  out_path = std::move(rev);
  return true;
}

std::vector<Vec3> BSplineOptimizer::shortcutPath(const GridMap& grid, const std::vector<Vec3>& path,
                                                 double clearance_m) const {
  if (path.size() <= 2) return path;
  std::vector<Vec3> out;
  out.push_back(path.front());
  std::size_t i = 0;
  while (i < path.size() - 1) {
    std::size_t j = path.size() - 1;
    // Limit max shortcut span to 1.4m so control points have dense guidance around walls/corners
    while (j > i + 1) {
      double seg_len = (path[j] - path[i]).norm();
      if (seg_len <= 1.4 && grid.isSegmentClear(path[i], path[j], clearance_m, 0.025)) {
        break;
      }
      --j;
    }
    out.push_back(path[j]);
    i = j;
  }
  return out;
}

UniformBSplinePrimitive BSplineOptimizer::planAndOptimizeTrajectory(
    const GridMap& grid,
    const Vec3& start_pos_ned,
    const Vec3& start_vel_ned,
    const std::vector<Vec3>& anchor_waypoints_ned,
    const Vec3& end_vel_ned,
    double desired_speed_ms) const {
  std::vector<Vec3> full_polyline;
  full_polyline.push_back(start_pos_ned);

  Vec3 cur = start_pos_ned;
  for (std::size_t idx = 0; idx < anchor_waypoints_ned.size(); ++idx) {
    const Vec3& target = anchor_waypoints_ned[idx];
    if ((target - cur).norm() < 0.06) continue;
    if (grid.isSegmentClear(cur, target, 0.215, 0.03)) {
      full_polyline.push_back(target);
    } else {
      std::vector<Vec3> seg_path;
      bool ok = planPathAStar(grid, cur, target, seg_path, cfg_.min_clearance_m);
      if (!ok) {
        ok = planPathAStar(grid, cur, target, seg_path, 0.205);
      }
      if (ok && seg_path.size() >= 2) {
        auto pruned = shortcutPath(grid, seg_path, std::max(0.24, cfg_.min_clearance_m));
        for (std::size_t k = 1; k < pruned.size(); ++k) {
          full_polyline.push_back(pruned[k]);
        }
      } else {
        full_polyline.push_back(target);
      }
    }
    cur = target;
  }

  Vec3 tangent_init = start_vel_ned;
  if (full_polyline.size() >= 2) {
    tangent_init = (full_polyline[1] - full_polyline[0]).normalized() * (0.65 * desired_speed_ms);
  }
  UniformBSplinePrimitive spline = UniformBSplinePrimitive::fromWaypoints(
      full_polyline, tangent_init, end_vel_ned, desired_speed_ms, cfg_.knot_dt);

  // Identify control points nearest to anchor waypoints so gate crossings remain strongly anchored
  const auto& q = spline.controlPoints();
  std::vector<int> pinned;
  for (std::size_t a = 0; a + 1 < anchor_waypoints_ned.size(); ++a) {
    const Vec3& wp = anchor_waypoints_ned[a];
    int best_idx = -1;
    double best_d = 1e9;
    for (int i = 2; i < static_cast<int>(q.size()) - 2; ++i) {
      double d = (q[i] - wp).norm();
      if (d < best_d) {
        best_d = d;
        best_idx = i;
      }
    }
    if (best_idx >= 2 && best_d < 0.35) {
      pinned.push_back(best_idx);
    }
  }

  optimizeControlPoints(grid, spline, pinned);
  return spline;
}

void BSplineOptimizer::optimizeControlPoints(const GridMap& grid,
                                             UniformBSplinePrimitive& spline,
                                             const std::vector<int>& pinned_indices) const {
  auto& q = spline.controlPointsMutable();
  const int n = static_cast<int>(q.size());
  if (n < 6) return;

  const std::vector<Vec3> q_ref = q;
  std::vector<bool> is_anchor(n, false);
  for (int idx : pinned_indices) {
    if (idx >= 0 && idx < n) is_anchor[idx] = true;
  }

  std::vector<Vec3> grad(n, {0.0, 0.0, 0.0});
  std::vector<Vec3> vel_m(n, {0.0, 0.0, 0.0});
  const double dt = spline.knotDt();
  const double inv_dt = 1.0 / dt;

  for (int iter = 0; iter < cfg_.max_iterations; ++iter) {
    std::fill(grad.begin(), grad.end(), Vec3{0.0, 0.0, 0.0});

    // 1. Jerk smoothness: sum ||q_{i+3} - 3 q_{i+2} + 3 q_{i+1} - q_i||^2
    for (int i = 0; i <= n - 4; ++i) {
      Vec3 j = q[i + 3] - q[i + 2] * 3.0 + q[i + 1] * 3.0 - q[i];
      Vec3 gj = j * (2.0 * cfg_.w_jerk);
      grad[i] -= gj;
      grad[i + 1] += gj * 3.0;
      grad[i + 2] -= gj * 3.0;
      grad[i + 3] += gj;
    }

    // 2. Acceleration smoothness: sum ||q_{i+2} - 2 q_{i+1} + q_i||^2
    for (int i = 0; i <= n - 3; ++i) {
      Vec3 a = q[i + 2] - q[i + 1] * 2.0 + q[i];
      Vec3 ga = a * (2.0 * cfg_.w_acc);
      grad[i] += ga;
      grad[i + 1] -= ga * 2.0;
      grad[i + 2] += ga;
    }

    // 3. ESDF Obstacle Collision Repulsion + Altitude Bounds + Reference Anchor
    for (int i = 2; i <= n - 3; ++i) {
      Vec3 grad_d;
      double dist = grid.getDistanceAndGradient(q[i], grad_d);
      if (dist < cfg_.safe_distance_m) {
        double pen = cfg_.safe_distance_m - dist;
        // d/dq [ w_coll * (safe - dist)^3 ] = -3 * w_coll * pen^2 * grad_d
        grad[i] -= grad_d * (3.0 * cfg_.w_collision * pen * pen);
      }

      // Also check midpoint between q[i] and q[i+1] so thin walls cannot slip between control points
      Vec3 mid = (q[i] + q[i + 1]) * 0.5;
      Vec3 grad_mid;
      double dist_mid = grid.getDistanceAndGradient(mid, grad_mid);
      if (dist_mid < cfg_.safe_distance_m) {
        double pen = cfg_.safe_distance_m - dist_mid;
        Vec3 g_m = grad_mid * (-1.5 * cfg_.w_collision * pen * pen);
        grad[i] += g_m;
        if (i + 1 <= n - 3) grad[i + 1] += g_m;
      }

      // Altitude bounds (NED: alt = -z)
      double alt = -q[i].z;
      if (alt < cfg_.alt_min_m) {
        double pen = cfg_.alt_min_m - alt;
        // Increase alt -> decrease q[i].z
        grad[i].z += 2.0 * cfg_.w_altitude * pen;
      } else if (alt > cfg_.alt_max_m) {
        double pen = alt - cfg_.alt_max_m;
        // Decrease alt -> increase q[i].z
        grad[i].z -= 2.0 * cfg_.w_altitude * pen;
      }

      // Reference path & gate anchor spring
      double w_r = is_anchor[i] ? (cfg_.w_ref * 6.0) : cfg_.w_ref;
      grad[i] += (q[i] - q_ref[i]) * (2.0 * w_r);
    }

    // 4. Dynamic Feasibility (Velocity & Acceleration soft bounds)
    for (int i = 1; i <= n - 3; ++i) {
      Vec3 v = (q[i + 1] - q[i]) * inv_dt;
      double v_norm = v.norm();
      if (v_norm > cfg_.max_vel_ms) {
        double ex = v_norm - cfg_.max_vel_ms;
        Vec3 gv = v.normalized() * (2.0 * cfg_.w_feasibility * ex * inv_dt);
        if (i >= 2) grad[i] -= gv;
        if (i + 1 <= n - 3) grad[i + 1] += gv;
      }
    }

    // Update interior control points with overdamped step clamping (no high-freq ringing)
    for (int i = 2; i <= n - 3; ++i) {
      vel_m[i] = vel_m[i] * 0.20 - grad[i] * cfg_.step_size;
      double step_norm = vel_m[i].norm();
      if (step_norm > 0.015) {
        vel_m[i] = vel_m[i] * (0.015 / step_norm);
      }
      Vec3 cand = q[i] + vel_m[i];
      if (is_anchor[i] && (cand - q_ref[i]).norm() > 0.04) {
        cand = q_ref[i] + (cand - q_ref[i]).normalized() * 0.04;
      }
      // Never step into a lower-clearance cell below min_clearance_m
      if (grid.getDistance(cand) >= std::min(cfg_.min_clearance_m, grid.getDistance(q[i]) - 0.005)) {
        q[i] = cand;
      }
    }
  }
}

}  // namespace ego_planner
