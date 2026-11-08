#pragma once
// Tracks how far along a multi-anchor global path the vehicle is, so a replan can start from the current
// pose and continue through the anchors that are still ahead (never back through ones already passed).
//
// Shared by rrt_star_planner_node and the C API; any global planner that knows where its anchors / gates
// sit along its path can use it.

#include <cstddef>
#include <vector>

#include "maze_planning/types.hpp"

namespace maze_planning {

class CourseProgress {
 public:
  explicit CourseProgress(double pass_margin_m = 0.15) : pass_margin_m_(pass_margin_m) {}

  // path_map: the path being flown. anchor_s_m[i]: arc length of anchor (first_anchor_idx + i) on that path.
  void reset(const PlannedPath& path_map, std::vector<double> anchor_s_m, std::size_t first_anchor_idx) {
    path_ = path_map;
    anchor_s_m_ = std::move(anchor_s_m);
    first_anchor_idx_ = first_anchor_idx;
    passed_in_path_ = 0;
    progress_idx_ = 0;
    progress_s_m_ = 0.0;
  }

  // Feed the current position (map frame). Returns the index of the next anchor not yet passed.
  std::size_t update(const Eigen::Vector3d& pos_map) {
    if (path_.empty()) return nextAnchor();
    // Forward-windowed search: progress only moves forward along the path.
    progress_idx_ = path_.findClosestIndex(pos_map, progress_idx_);
    progress_s_m_ = path_.points[static_cast<std::size_t>(progress_idx_)].s_m;
    while (passed_in_path_ < anchor_s_m_.size() &&
           progress_s_m_ >= anchor_s_m_[passed_in_path_] - pass_margin_m_) {
      ++passed_in_path_;
    }
    return nextAnchor();
  }

  std::size_t nextAnchor() const { return first_anchor_idx_ + passed_in_path_; }
  double progressS() const { return progress_s_m_; }
  bool hasPath() const { return !path_.empty(); }

 private:
  double pass_margin_m_;
  PlannedPath path_;
  std::vector<double> anchor_s_m_;
  std::size_t first_anchor_idx_{0};
  std::size_t passed_in_path_{0};
  int progress_idx_{0};
  double progress_s_m_{0.0};
};

}  // namespace maze_planning
