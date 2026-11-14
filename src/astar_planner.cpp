#include "maze_planning/astar_planner.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <queue>
#include <utility>

namespace maze_planning {

namespace {

double cross2(const Eigen::Vector2d& a, const Eigen::Vector2d& b) { return a.x() * b.y() - a.y() * b.x(); }

}  // namespace

AStarPlanner::AStarPlanner(const AStarConfig& cfg, std::shared_ptr<const GroundTruthMap> map)
    : cfg_(cfg), map_(std::move(map)) {
  n_ = static_cast<int>(std::lround(2.0 * cfg_.arena_half_m / cfg_.resolution_m)) + 1;
  rebuildClearance();
}

int AStarPlanner::cellOf(const Eigen::Vector2d& p) const {
  const int i = std::clamp(static_cast<int>(std::lround((p.x() + cfg_.arena_half_m) / cfg_.resolution_m)), 0, n_ - 1);
  const int j = std::clamp(static_cast<int>(std::lround((p.y() + cfg_.arena_half_m) / cfg_.resolution_m)), 0, n_ - 1);
  return i * n_ + j;
}

Eigen::Vector2d AStarPlanner::posOf(int cell) const {
  return Eigen::Vector2d(-cfg_.arena_half_m + cfg_.resolution_m * (cell / n_),
                         -cfg_.arena_half_m + cfg_.resolution_m * (cell % n_));
}

bool AStarPlanner::inAltitudeZone(const Eigen::Vector2d& p) const {
  if (alt_zones_.empty()) return true;
  for (const auto& [a, b] : alt_zones_) {
    const Eigen::Vector2d ab = b - a;
    const double t = std::clamp((p - a).dot(ab) / std::max(1e-12, ab.squaredNorm()), 0.0, 1.0);
    if ((p - (a + t * ab)).norm() <= alt_zone_radius_m_) return true;
  }
  return false;
}

void AStarPlanner::rebuildClearance() {
  rebuildClearance(nullptr, 0.04);
}

void AStarPlanner::rebuildClearance(const isam2_depth::DistanceGrid* adapted_grid_map, double voxel_half_diag_m) {
  const std::size_t total = static_cast<std::size_t>(n_) * n_;
  clr_.assign(total, 0.0);
  opt_z_.assign(total, cfg_.flight_z_ned);
  const double net = cfg_.arena_half_m - cfg_.net_margin_m;
  for (int c = 0; c < n_ * n_; ++c) {
    const Eigen::Vector2d p = posOf(c);
    const Eigen::Vector3d p3(p.x(), p.y(), cfg_.flight_z_ned);
    const double d_net = net - std::max(std::abs(p.x()), std::abs(p.y()));   // exact inside a square
    double d_obs = map_ ? map_->clearance(p3) : 1e9;
    double best_z = cfg_.flight_z_ned;
    if (adapted_grid_map && adapted_grid_map->inside(p3)) {
      const auto query_adapted = [&](double z_test) {
        const Eigen::Vector3d q(p.x(), p.y(), z_test);
        if (!adapted_grid_map->inside(q)) return 0.0;
        double d = std::max(0.0, static_cast<double>(adapted_grid_map->distance(q)) - voxel_half_diag_m);
        if (map_) {
          for (const auto& blk : map_->blockedCorridors()) {
            d = std::min(d, (q - blk.center_ned).norm() - blk.radius_m);
          }
        }
        return d;
      };
      d_obs = query_adapted(cfg_.flight_z_ned);
      // If nominal flight altitude is tight/blocked (e.g. a physical window cut higher than the CAD window),
      // search safe flight altitudes z_ned in [-1.42, -0.55] m with a light altitude-deviation tie-breaker so
      // open corridors stay at flight_z_ned while elevated window centers are chosen at maximum 3D clearance!
      if (d_obs < cfg_.clearance_m + 0.06 && inAltitudeZone(p)) {
        double best_score = d_obs;
        for (double z_test = -1.42; z_test <= -0.55 + 1e-6; z_test += 0.04) {
          const double d_z = query_adapted(z_test);
          const double score = d_z - 0.018 * std::abs(z_test - cfg_.flight_z_ned);
          if (score > best_score && d_z > d_obs + 0.015) {
            best_score = score;
            d_obs = d_z;
            best_z = z_test;
          }
        }
      }
    }
    clr_[static_cast<std::size_t>(c)] = std::min(d_net, d_obs);
    opt_z_[static_cast<std::size_t>(c)] = best_z;
  }
}

std::vector<int> AStarPlanner::searchRegion(int start, const std::function<bool(int)>& is_goal,
                                            const std::function<double(int)>& h,
                                            const std::function<double(int)>& terminal,
                                            const std::function<bool(int)>& allowed, std::size_t* expanded) const {
  if (!allowed(start)) return {};
  const std::size_t total = static_cast<std::size_t>(n_) * n_;
  const int kSuper = static_cast<int>(total);   // virtual goal: goal cell -> super costs terminal(cell)
  std::vector<double> g(total + 1, std::numeric_limits<double>::infinity());
  std::vector<int> parent(total + 1, -1);
  std::vector<char> closed(total + 1, 0);
  using Entry = std::pair<double, int>;
  std::priority_queue<Entry, std::vector<Entry>, std::greater<Entry>> pq;
  g[static_cast<std::size_t>(start)] = 0.0;
  pq.push({h(start), start});
  static const int kDi[8] = {-1, 1, 0, 0, -1, -1, 1, 1};
  static const int kDj[8] = {0, 0, -1, 1, -1, 1, -1, 1};
  static const double kW[8] = {1.0, 1.0, 1.0, 1.0, M_SQRT2, M_SQRT2, M_SQRT2, M_SQRT2};
  std::size_t n_exp = 0;
  while (!pq.empty()) {
    const int c = pq.top().second;
    pq.pop();
    if (closed[static_cast<std::size_t>(c)]) continue;
    closed[static_cast<std::size_t>(c)] = 1;
    if (c == kSuper) break;
    ++n_exp;
    if (is_goal(c)) {
      const double cost = g[static_cast<std::size_t>(c)] + terminal(c);
      if (cost < g[total]) {
        g[total] = cost;
        parent[total] = c;
        pq.push({cost, kSuper});
      }
    }
    const int ci = c / n_, cj = c % n_;
    for (int k = 0; k < 8; ++k) {
      const int mi = ci + kDi[k], mj = cj + kDj[k];
      if (mi < 0 || mj < 0 || mi >= n_ || mj >= n_) continue;
      const int m = mi * n_ + mj;
      if (closed[static_cast<std::size_t>(m)] || !allowed(m)) continue;
      // Prefer the middle of corridors and apply a strong soft barrier below 0.30 m so tight openings (windows,
      // narrow gates) are threaded dead-center instead of grazing a jamb to shave a few cm of path length.
      const double cm = clr_[static_cast<std::size_t>(m)];
      const double tight = std::max(0.0, 0.30 - cm);
      const double cost = g[static_cast<std::size_t>(c)] +
                          kW[k] * (1.0 + cfg_.centre_weight * std::max(0.0, cfg_.centre_dist_m - cm) +
                                   28.0 * tight * tight);
      if (cost < g[static_cast<std::size_t>(m)]) {
        g[static_cast<std::size_t>(m)] = cost;
        parent[static_cast<std::size_t>(m)] = c;
        pq.push({cost + h(m), m});
      }
    }
  }
  if (expanded) *expanded += n_exp;
  if (!closed[total]) return {};
  std::vector<int> path;
  for (int c = parent[total]; c >= 0; c = parent[static_cast<std::size_t>(c)]) path.push_back(c);
  std::reverse(path.begin(), path.end());
  return path;
}

std::vector<int> AStarPlanner::search(int start, int goal, double r_min, std::size_t* expanded, double r_near,
                                      double near_radius_m) const {
  const double near_cells = near_radius_m / cfg_.resolution_m;
  const auto cdist = [&](int a, int b) {
    return std::hypot(static_cast<double>(a / n_ - b / n_), static_cast<double>(a % n_ - b % n_));
  };
  const auto allowed = [&](int c) {
    const double cl = clr_[static_cast<std::size_t>(c)];
    if (cl >= r_min) return true;
    return r_near >= 0.0 && cl >= r_near && (cdist(c, start) <= near_cells || cdist(c, goal) <= near_cells);
  };
  if (!allowed(goal)) return {};
  return searchRegion(start, [&](int c) { return c == goal; }, [&](int c) { return cdist(c, goal); },
                      [](int) { return 0.0; }, allowed, expanded);
}

std::vector<int> AStarPlanner::searchGate(int start, const PlanGoal& gate, double d_lo, double d_hi, double r_min,
                                          double r_near, const Eigen::Vector2d* next_xy,
                                          std::size_t* expanded) const {
  const Eigen::Vector2d n = gate.normal.normalized();
  const Eigen::Vector2d t_full = gate.seg_b - gate.seg_a;
  const double len = t_full.norm();
  if (len < 1e-9) return {};
  const Eigen::Vector2d t = t_full / len;
  const double res = cfg_.resolution_m;
  const double near_m = cfg_.snap_radius_m;
  // Distance (m) from a point to the band {u in [0, len], d in [d_lo, d_hi]} in the gate's (u, d) frame.
  const auto band_dist = [&](const Eigen::Vector2d& p) {
    const Eigen::Vector2d r = p - gate.seg_a;
    const double u = r.dot(t), d = r.dot(n);
    const double du = std::max({0.0, -u, u - len});
    const double dd = std::max({0.0, d_lo - d, d - d_hi});
    return std::hypot(du, dd);
  };
  const auto is_goal = [&](int c) { return band_dist(posOf(c)) <= 0.5 * res; };
  const auto allowed = [&](int c) {
    const double cl = clr_[static_cast<std::size_t>(c)];
    if (cl >= r_min) return true;
    if (r_near < 0.0 || cl < r_near) return false;
    return (posOf(c) - posOf(start)).norm() <= near_m || band_dist(posOf(c)) <= near_m;
  };
  // Terminal: straight-line distance to the next goal, so the gate is crossed where the next leg is short.
  const auto terminal = [&](int c) { return next_xy ? (posOf(c) - *next_xy).norm() / res : 0.0; };
  return searchRegion(start, is_goal, [&](int c) { return band_dist(posOf(c)) / res; }, terminal, allowed,
                      expanded);
}

// dronehacks-sim env.py: d = (p - A) . n - cp_margin goes from < 0 to >= 0 with p within the line's extent.
bool AStarPlanner::clears(const PlanGoal& g, const std::vector<Eigen::Vector2d>& xy) const {
  const Eigen::Vector2d n = g.normal.normalized();
  const Eigen::Vector2d t = g.seg_b - g.seg_a;
  const double len = t.norm();
  if (len < 1e-9) return false;
  for (std::size_t i = 1; i < xy.size(); ++i) {
    const double d0 = (xy[i - 1] - g.seg_a).dot(n) - cfg_.cp_margin_m;
    const double d1 = (xy[i] - g.seg_a).dot(n) - cfg_.cp_margin_m;
    const double u = (xy[i] - g.seg_a).dot(t / len);
    if (d0 < 0.0 && d1 >= 0.0 && u >= 0.0 && u <= len) return true;
  }
  return false;
}

int AStarPlanner::snap(int cell, double r_min) const {
  if (clr_[static_cast<std::size_t>(cell)] >= r_min) return cell;
  // Nearest cell (then the clearest one) with enough clearance within snap_radius_m.
  const int r = static_cast<int>(std::ceil(cfg_.snap_radius_m / cfg_.resolution_m));
  const int ci = cell / n_, cj = cell % n_;
  int best = -1;
  double best_d2 = std::numeric_limits<double>::infinity(), best_clr = -1.0;
  for (int di = -r; di <= r; ++di) {
    for (int dj = -r; dj <= r; ++dj) {
      const int i = ci + di, j = cj + dj;
      if (i < 0 || j < 0 || i >= n_ || j >= n_) continue;
      const double d2 = di * di + dj * dj;
      if (d2 > r * r) continue;
      const double c = clr_[static_cast<std::size_t>(i * n_ + j)];
      if (c < r_min) continue;
      if (d2 < best_d2 || (d2 == best_d2 && c > best_clr)) {
        best = i * n_ + j;
        best_d2 = d2;
        best_clr = c;
      }
    }
  }
  return best;
}

bool AStarPlanner::freeLine(const Eigen::Vector2d& a, const Eigen::Vector2d& b, double r_min) const {
  const int n = std::max(2, static_cast<int>((b - a).norm() / (0.5 * cfg_.resolution_m)));
  for (int k = 0; k <= n; ++k) {
    if (clr_[static_cast<std::size_t>(cellOf(a + (b - a) * (static_cast<double>(k) / n)))] < r_min) return false;
  }
  return true;
}

AStarPlanResult AStarPlanner::plan(const Eigen::Vector3d& start_map, const std::vector<PlanGoal>& goals,
                                   double start_speed_m_s) const {
  AStarPlanResult res;
  const double r_req = cfg_.clearance_m;
  const double r_fb = cfg_.clearance_m * cfg_.clearance_fallback_ratio;
  const Eigen::Vector2d start_xy = start_map.head<2>();

  const int s_cell = snap(cellOf(start_xy), r_fb);
  if (s_cell < 0 || goals.empty()) return res;

  // 1. A* goal to goal (the script's two-level clearance), skipping unreachable goals. A gate is not a point:
  //    any cell checkpoint_overshoot..max past its line, within its extent, clears it; the search picks the one
  //    that keeps start -> gate -> next goal short (so corridor_end is taken at its corner, not as a hairpin).
  std::vector<int> cells = {s_cell};
  std::vector<int> goal_raw_idx(goals.size(), -1);
  std::vector<Eigen::Vector2d> reached(goals.size(), Eigen::Vector2d::Zero());
  int cur = s_cell;
  const double d_min = cfg_.checkpoint_overshoot_m, d_max = cfg_.checkpoint_overshoot_max_m;
  for (std::size_t gi = 0; gi < goals.size(); ++gi) {
    const PlanGoal& goal = goals[gi];
    std::vector<int> seg;
    if (goal.has_segment && goal.normal.norm() > 1e-9) {
      const Eigen::Vector2d next = (gi + 1 < goals.size()) ? goals[gi + 1].center.head<2>() : goal.center.head<2>();
      const Eigen::Vector2d* next_p = (gi + 1 < goals.size()) ? &next : nullptr;
      // Deep band first (margin for MPPI rounding the corner), then the whole band, then fallback clearance,
      // then snap to the nearest clear opening if the gate shifted physically in the real arena.
      seg = searchGate(cur, goal, std::max(d_min, d_max - 0.05), d_max + 0.05, r_req, r_fb, next_p, &res.expanded);
      if (seg.empty()) seg = searchGate(cur, goal, d_min, d_max + 0.05, r_req, r_fb, next_p, &res.expanded);
      if (seg.empty()) seg = searchGate(cur, goal, d_min, d_max + 0.05, r_fb, -1.0, next_p, &res.expanded);
      if (seg.empty()) {
        const Eigen::Vector2d target_xy = goal.center.head<2>() + goal.normal.normalized() * d_min;
        const int exact = cellOf(target_xy);
        const int g_req = snap(exact, r_req);
        if (g_req >= 0) seg = search(cur, g_req, r_req, &res.expanded);
        if (seg.empty()) {
          const int g_fb = snap(exact, r_fb);
          if (g_fb >= 0) seg = search(cur, g_fb, r_fb, &res.expanded);
        }
      }
    } else {
      const int exact = cellOf(goal.center.head<2>());
      seg = search(cur, exact, r_req, &res.expanded, r_fb, cfg_.snap_radius_m);
      if (seg.empty()) {
        const int g_req = snap(exact, r_req);
        if (g_req >= 0) seg = search(cur, g_req, r_req, &res.expanded);
      }
      if (seg.empty()) {
        const int g_fb = snap(exact, r_fb);
        if (g_fb >= 0) seg = search(cur, g_fb, r_fb, &res.expanded);
      }
    }
    if (seg.empty()) {
      res.skipped.push_back(gi);
      continue;
    }
    cells.insert(cells.end(), seg.begin() + 1, seg.end());
    goal_raw_idx[gi] = static_cast<int>(cells.size()) - 1;
    reached[gi] = posOf(seg.back());
    cur = seg.back();
  }
  if (cells.size() < 2) return res;

  std::vector<Eigen::Vector2d> raw;
  raw.reserve(cells.size() + 1);
  for (int c : cells) raw.push_back(posOf(c));
  // Start exactly at the vehicle when it is within a cell of the start cell (it normally is).
  int shift = 0;
  if ((raw.front() - start_xy).norm() <= cfg_.resolution_m) {
    raw.front() = start_xy;
  } else {
    raw.insert(raw.begin(), start_xy);
    shift = 1;
  }
  // Pinned vertices: start, every reached goal, end. Neither the shortcut nor Chaikin removes them.
  std::vector<char> pinned_raw(raw.size(), 0);
  pinned_raw.front() = pinned_raw.back() = 1;
  for (int idx : goal_raw_idx) {
    if (idx >= 0) pinned_raw[static_cast<std::size_t>(idx + shift)] = 1;
  }

  // 2. Line-of-sight shortcut, never past a pinned vertex.
  std::vector<Eigen::Vector2d> poly = {raw.front()};
  std::vector<char> pinned = {1};
  for (std::size_t i = 0; i + 1 < raw.size();) {
    std::size_t lim = i + 1;
    while (lim + 1 < raw.size() && !pinned_raw[lim]) ++lim;   // next pinned vertex
    std::size_t j = lim;
    while (j > i + 1 && !freeLine(raw[i], raw[j], r_req)) --j;
    poly.push_back(raw[j]);
    pinned.push_back(pinned_raw[j]);
    i = j;
  }

  // 3. Chaikin corner cutting; pinned vertices stay (the corner at a gate is shortened, not cut).
  for (int it = 0; it < cfg_.chaikin_iterations && poly.size() > 2; ++it) {
    std::vector<Eigen::Vector2d> q = {poly.front()};
    std::vector<char> qp = {1};
    for (std::size_t i = 0; i + 1 < poly.size(); ++i) {
      q.push_back(0.75 * poly[i] + 0.25 * poly[i + 1]);
      q.push_back(0.25 * poly[i] + 0.75 * poly[i + 1]);
      qp.push_back(0);
      qp.push_back(0);
      if (pinned[i + 1] && i + 2 < poly.size()) {
        q.push_back(poly[i + 1]);
        qp.push_back(1);
      }
    }
    q.push_back(poly.back());
    qp.push_back(1);
    poly = std::move(q);
    pinned = std::move(qp);
  }

  // 4. Uniform resample.
  std::vector<double> sp(1, 0.0);
  for (std::size_t i = 1; i < poly.size(); ++i) sp.push_back(sp.back() + (poly[i] - poly[i - 1]).norm());
  std::vector<Eigen::Vector2d> xy;
  std::vector<double> s;
  std::size_t j = 0;
  for (std::size_t k_s = 0;; ++k_s) {
    const double u = static_cast<double>(k_s) * cfg_.resample_ds_m;
    if (u >= sp.back()) break;
    while (j + 2 < sp.size() && sp[j + 1] < u) ++j;
    const double a = (u - sp[j]) / std::max(1e-12, sp[j + 1] - sp[j]);
    xy.push_back((1.0 - a) * poly[j] + a * poly[j + 1]);
    s.push_back(u);
  }
  xy.push_back(poly.back());
  s.push_back(sp.back());
  const std::size_t n = xy.size();
  if (n < 2) return res;

  // Sample index of each reached goal (in order; -1 = skipped).
  std::vector<long> goal_idx(goals.size(), -1);
  {
    std::size_t from = 0;
    for (std::size_t gi = 0; gi < goals.size(); ++gi) {
      if (goal_raw_idx[gi] < 0) continue;
      const Eigen::Vector2d target = reached[gi];
      std::size_t best = from;
      double best_d = std::numeric_limits<double>::infinity();
      for (std::size_t i = from; i < n; ++i) {
        const double d = (xy[i] - target).norm();
        if (d < best_d) {
          best_d = d;
          best = i;
        }
      }
      goal_idx[gi] = static_cast<long>(best);
      from = best;
    }
  }

  // 5. Speed profile: curvature from the turning angle over a chord, smoothed; then acceleration limits.
  const int k = std::max(1, cfg_.curvature_smooth_half_window);
  std::vector<double> kappa(n, 0.0);
  for (std::size_t i = 0; i < n; ++i) {
    const std::size_t i0 = (i >= static_cast<std::size_t>(k)) ? i - k : 0;
    const std::size_t i1 = std::min(n - 1, i + k);
    if (i0 == i || i1 == i) continue;
    const Eigen::Vector2d a = xy[i] - xy[i0], b = xy[i1] - xy[i];
    const double ds = 0.5 * (a.norm() + b.norm());
    if (ds < 1e-9) continue;
    kappa[i] = std::abs(std::atan2(cross2(a, b), a.dot(b))) / ds;
  }
  std::vector<double> kappa_s(n, 0.0);
  for (std::size_t i = 0; i < n; ++i) {
    double acc = 0.0;
    int cnt = 0;
    for (int d = -k; d <= k; ++d) {
      const long m = static_cast<long>(i) + d;
      if (m < 0 || m >= static_cast<long>(n)) continue;
      acc += kappa[static_cast<std::size_t>(m)];
      ++cnt;
    }
    kappa_s[i] = acc / std::max(1, cnt);
  }
  std::vector<double> v(n);
  for (std::size_t i = 0; i < n; ++i) {
    v[i] = std::min(cfg_.cruise_speed_m_s, std::sqrt(cfg_.lat_acc_m_s2 / std::max(kappa_s[i], 1e-6)));
    if (cfg_.yaw_rate_max_rad_s > 0.0) v[i] = std::min(v[i], cfg_.yaw_rate_max_rad_s / std::max(kappa_s[i], 1e-6));
  }
  if (cfg_.gate_apex_speed_m_s > 0.0) {
    for (std::size_t gi = 0; gi < goals.size(); ++gi) {
      if (goal_idx[gi] >= 0 && goals[gi].has_segment) {
        double& vi = v[static_cast<std::size_t>(goal_idx[gi])];
        vi = std::min(vi, cfg_.gate_apex_speed_m_s);
      }
    }
  }
  v.front() = std::min(v.front(), std::max(0.0, start_speed_m_s));
  v.back() = std::min(v.back(), std::max(0.0, cfg_.end_speed_m_s));
  for (std::size_t i = 1; i < n; ++i) {
    v[i] = std::min(v[i], std::sqrt(v[i - 1] * v[i - 1] + 2.0 * cfg_.acc_m_s2 * (s[i] - s[i - 1])));
  }
  for (std::size_t i = n - 1; i-- > 0;) {
    v[i] = std::min(v[i], std::sqrt(v[i + 1] * v[i + 1] + 2.0 * cfg_.acc_m_s2 * (s[i + 1] - s[i])));
  }

  // 6. Timed PlannedPath (smooth z_ned profile when 3D adapted clearance opens an elevated window).
  std::vector<double> z_prof(n, cfg_.flight_z_ned);
  if (opt_z_.size() == clr_.size()) {
    std::vector<double> z_raw(n, cfg_.flight_z_ned);
    bool has_alt_dev = false;
    for (std::size_t i = 0; i < n; ++i) {
      z_raw[i] = opt_z_[static_cast<std::size_t>(cellOf(xy[i]))];
      if (std::abs(z_raw[i] - cfg_.flight_z_ned) > 1e-3) has_alt_dev = true;
    }
    if (has_alt_dev || std::abs(start_map.z() - cfg_.flight_z_ned) > 0.05) {
      z_raw.front() = std::clamp(start_map.z(), -1.45, -0.45);
      // Widen elevated window segments along arc length (+-0.60 m = 30 samples) so the drone climbs cleanly
      // before reaching the window sill and descends only after clearing the window frame!
      std::vector<double> z_dil = z_raw;
      const int hw_dil = 30;
      for (std::size_t i = 0; i < n; ++i) {
        double z_pick = z_raw[i];
        for (int d = -hw_dil; d <= hw_dil; ++d) {
          const long m = static_cast<long>(i) + d;
          if (m < 0 || m >= static_cast<long>(n)) continue;
          if (std::abs(z_raw[static_cast<std::size_t>(m)] - cfg_.flight_z_ned) >
              std::abs(z_pick - cfg_.flight_z_ned)) {
            z_pick = z_raw[static_cast<std::size_t>(m)];
          }
        }
        z_dil[i] = z_pick;
      }
      z_dil.front() = z_raw.front();
      z_dil.back() = cfg_.flight_z_ned;
      // Two-pass box smooth (~0.50 m window) for C1-smooth vertical velocity reference
      std::vector<double> z_tmp = z_dil;
      const int hw_sm = 14;
      for (int pass = 0; pass < 2; ++pass) {
        for (std::size_t i = 1; i + 1 < n; ++i) {
          double sum = 0.0;
          int cnt = 0;
          for (int d = -hw_sm; d <= hw_sm; ++d) {
            const long m = std::clamp<long>(static_cast<long>(i) + d, 0, static_cast<long>(n) - 1);
            sum += z_tmp[static_cast<std::size_t>(m)];
            ++cnt;
          }
          z_prof[i] = sum / std::max(1, cnt);
        }
        z_prof.front() = z_dil.front();
        z_prof.back() = z_dil.back();
        z_tmp = z_prof;
      }
    }
  }

  PlannedPath& out = res.path;
  out.frame_id = "map";
  out.points.resize(n);
  double t = 0.0;
  res.min_clearance_m = std::numeric_limits<double>::infinity();
  for (std::size_t i = 0; i < n; ++i) {
    if (i > 0) t += 2.0 * (s[i] - s[i - 1]) / std::max(0.02, v[i - 1] + v[i]);
    const std::size_t i0 = i > 0 ? i - 1 : 0;
    const std::size_t i1 = std::min(n - 1, i + 1);
    const Eigen::Vector2d d = xy[i1] - xy[i0];
    const Eigen::Vector2d tan = d.norm() > 1e-9 ? Eigen::Vector2d(d / d.norm()) : Eigen::Vector2d(1.0, 0.0);
    const double ds_span = std::max(1e-6, s[i1] - s[i0]);
    const double dz_ds = (z_prof[i1] - z_prof[i0]) / ds_span;
    PathPoint& p = out.points[i];
    p.pos = Eigen::Vector3d(xy[i].x(), xy[i].y(), z_prof[i]);
    p.vel = Eigen::Vector3d(tan.x() * v[i], tan.y() * v[i], std::clamp(dz_ds * v[i], -0.8, 0.8));
    p.yaw_rad = std::atan2(tan.y(), tan.x());
    p.s_m = s[i];
    p.t_s = t;
    res.min_clearance_m = std::min(res.min_clearance_m, clearanceAt(xy[i]));
  }
  for (std::size_t i = 0; i < n; ++i) {
    const std::size_t a = i > 0 ? i - 1 : 0, b = std::min(n - 1, i + 1);
    const double dt = out.points[b].t_s - out.points[a].t_s;
    if (dt > 1e-9) {
      out.points[i].acc = (out.points[b].vel - out.points[a].vel) / dt;
      out.points[i].yawspeed_rad_s = wrapPi(out.points[b].yaw_rad - out.points[a].yaw_rad) / dt;
    }
  }
  out.total_length_m = s.back();
  out.total_duration_s = t;

  // 7. Where each goal sits along the path (in order), and whether the path clears each gate.
  res.goal_s_m.assign(goals.size(), 0.0);
  double prev_s = 0.0;
  for (std::size_t gi = 0; gi < goals.size(); ++gi) {
    if (goal_idx[gi] >= 0) prev_s = s[static_cast<std::size_t>(goal_idx[gi])];
    res.goal_s_m[gi] = prev_s;
    if (goal_idx[gi] >= 0 && goals[gi].has_segment && !clears(goals[gi], xy)) res.not_crossed.push_back(gi);
  }
  res.ok = true;
  return res;
}

}  // namespace maze_planning
