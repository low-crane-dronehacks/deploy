#pragma once
// Grid A* global planner at flight height, ported from src/test/spoof_px4_depth.py (astar / plan_route /
// trajectory) for flight:
//   clearance grid (GroundTruthMap slice at flight height + arena net) -> 8-connected A* goal to goal, cost
//   w * (1 + centre_weight * max(0, centre_dist - clearance)) -> line-of-sight shortcut -> Chaikin corner
//   cutting -> uniform resample -> speed profile (lateral- and longitudinal-acceleration limits) -> timed
//   PlannedPath (map frame, NED).
// Differences from the spoof script, all for flight:
//   - a gate goal is the band 0.18-0.35 m past the gate line, across its whole extent, not its centre point. The
//     competition clears a checkpoint only when the centre is cp_margin (0.15 m) past the line, anywhere within
//     its extent (dronehacks-sim env.py). The search reaches the band where start -> band -> next goal is
//     shortest (a one-step look-ahead), and the reached cell is pinned through the shortcut and the corner
//     cutting (the script's shortcut + Chaikin turn 0.23 m before corridor_end's line; aiming at its centre
//     makes a 128 deg hairpin MPPI rounds off);
//   - goals are moved only if their cell is blocked, and only within snap_radius_m (the script moves every goal
//     up to 0.8 m to the clearest spot nearby, which can leave the gate line);
//   - each gate is checked with the scoring rule (centre goes from < cp_margin to >= cp_margin past the line,
//     within its extent);
//   - the speed profile starts at the vehicle's current speed (a replan in flight does not restart from rest).
// Deterministic: the same map and goals always give the same path.

#include <cstddef>
#include <functional>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "maze_planning/groundtruth_map.hpp"
#include "maze_planning/types.hpp"

namespace maze_planning {

struct AStarConfig {
  double resolution_m{0.05};
  double arena_half_m{3.0};           // config/arena_prior.json arena.arena_half_m
  double net_margin_m{0.0};           // extra margin to the outer net (GroundTruthMap already models it; the
                                      // spoof script needs 0.1 because a scan has no net)
  double flight_z_ned{-0.80};         // the plan is 2D at this height
  double clearance_m{0.30};           // required clearance to walls (drone radius 0.2 + margin)
  double clearance_fallback_ratio{0.75};
  double centre_dist_m{0.60};         // prefer the middle of corridors narrower than 2x this
  double centre_weight{1.5};
  double snap_radius_m{0.30};         // max distance a blocked goal / start is moved to a free cell
  int chaikin_iterations{4};
  double resample_ds_m{0.02};
  double cruise_speed_m_s{2.60};
  double lat_acc_m_s2{3.0};
  double acc_m_s2{2.0};
  double end_speed_m_s{0.0};          // stop at the last goal (pad hover)
  int curvature_smooth_half_window{5};
  double cp_margin_m{0.15};           // arena_prior.json arena.cp_margin_m (scoring)
  double checkpoint_overshoot_m{0.18};    // a gate is reached in the band this far past its line (cp_margin + 3 cm)
  double checkpoint_overshoot_max_m{0.30};// ... up to this; the deep part of the band is tried first, because MPPI
                                          // rounds pinned corners by a few cm
  double gate_apex_speed_m_s{0.0};        // > 0: cap the reference speed where a gate's band is reached
};

// A goal the path must reach (map frame, NED). With a segment it is a gate: the line A-B, crossed along
// `normal` (arena_prior.json normal_ned, the direction of travel).
struct PlanGoal {
  std::string name;
  Eigen::Vector3d center{Eigen::Vector3d::Zero()};
  bool has_segment{false};
  Eigen::Vector2d seg_a{Eigen::Vector2d::Zero()};
  Eigen::Vector2d seg_b{Eigen::Vector2d::Zero()};
  Eigen::Vector2d normal{Eigen::Vector2d::Zero()};
};

struct AStarPlanResult {
  bool ok{false};                       // a path exists through every goal that was not skipped
  PlannedPath path;
  std::vector<double> goal_s_m;         // arc length of each goal along path (skipped: previous goal's s)
  std::vector<std::size_t> skipped;     // goals no path reaches (the script's behaviour: skip, keep going)
  std::vector<std::size_t> not_crossed; // gates the final path does not clear (scoring rule, cp_margin)
  double min_clearance_m{0.0};          // along the final path
  std::size_t expanded{0};              // A* expansions, all legs
};

class AStarPlanner {
 public:
  explicit AStarPlanner(const AStarConfig& cfg, std::shared_ptr<const GroundTruthMap> map);

  // Recompute the clearance grid (after the map changes, e.g. GroundTruthMap::markCorridorBlocked).
  void rebuildClearance();

  // start -> goals in order. start_speed_m_s seeds the speed profile (0 on the ground / at a hover).
  AStarPlanResult plan(const Eigen::Vector3d& start_map, const std::vector<PlanGoal>& goals,
                       double start_speed_m_s = 0.0) const;

  // Grid search between two cells (row-major index). Cells need clearance r_min, except within near_radius_m
  // of the start or goal cell, where r_near is enough (a gate's pass-through point can be tighter than the
  // corridor: only its neighbourhood is relaxed, not the whole leg). Empty if unreachable.
  std::vector<int> search(int start_cell, int goal_cell, double r_min, std::size_t* expanded = nullptr,
                          double r_near = -1.0, double near_radius_m = 0.0) const;

  int cellOf(const Eigen::Vector2d& p) const;
  Eigen::Vector2d posOf(int cell) const;
  double clearanceAt(const Eigen::Vector2d& p) const { return clr_[static_cast<std::size_t>(cellOf(p))]; }
  int gridSize() const { return n_; }
  const AStarConfig& config() const { return cfg_; }

 private:
  int snap(int cell, double r_min) const;
  // A* from start to any goal cell, with terminal(cell) added on arrival (a virtual super-goal).
  std::vector<int> searchRegion(int start, const std::function<bool(int)>& is_goal,
                                const std::function<double(int)>& h, const std::function<double(int)>& terminal,
                                const std::function<bool(int)>& allowed, std::size_t* expanded) const;
  // A* into a gate's band [d_lo, d_hi] past its line; r_near is enough near the start and the band.
  std::vector<int> searchGate(int start, const PlanGoal& gate, double d_lo, double d_hi, double r_min, double r_near,
                              const Eigen::Vector2d* next_xy, std::size_t* expanded) const;
  bool clears(const PlanGoal& g, const std::vector<Eigen::Vector2d>& xy) const;
  bool freeLine(const Eigen::Vector2d& a, const Eigen::Vector2d& b, double r_min) const;

  AStarConfig cfg_;
  std::shared_ptr<const GroundTruthMap> map_;
  int n_{0};
  std::vector<double> clr_;   // clearance per cell (m), row-major [i * n + j], i along x (north)
};

}  // namespace maze_planning
