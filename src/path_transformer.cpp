#include "maze_planning/path_transformer.hpp"

#include <algorithm>
#include <cmath>

namespace maze_planning {

PathTransformer::PathTransformer(const PathTransformerConfig& cfg) : cfg_(cfg) {
  T_map_odom_ = SE3Transform{};
  T_odom_map_ = T_map_odom_.inverse();
}

void PathTransformer::setGlobalPathMap(const PlannedPath& path_map) {
  global_path_map_ = path_map;
  progress_idx_ = 0;
  current_s_m_ = 0.0;
}

void PathTransformer::setMapToOdomTransform(const SE3Transform& T_map_odom) {
  T_map_odom_ = T_map_odom;
  T_odom_map_ = T_map_odom.inverse();
}

PathPoint PathTransformer::transformPointMapToOdom(const PathPoint& pt_map) const {
  PathPoint out = pt_map;
  out.pos = T_odom_map_.apply(pt_map.pos);
  out.vel = T_odom_map_.rotate(pt_map.vel);
  out.acc = T_odom_map_.rotate(pt_map.acc);
  out.yaw_rad = wrapPi(pt_map.yaw_rad + T_odom_map_.yaw());
  return out;
}

void PathTransformer::updateProgress(const Eigen::Vector3d& pos_odom) {
  if (global_path_map_.empty()) return;
  const Eigen::Vector3d pos_map = T_map_odom_.apply(pos_odom);
  progress_idx_ = global_path_map_.findClosestIndex(pos_map, progress_idx_);
  current_s_m_ = global_path_map_.points[progress_idx_].s_m;
}

PlannedPath PathTransformer::extractLocalReferenceOdom(const VehicleKinematics& state_odom) {
  PlannedPath local_odom;
  local_odom.frame_id = "odom";
  local_odom.stamp_s = state_odom.timestamp_s;
  if (global_path_map_.empty()) return local_odom;

  updateProgress(state_odom.pos_odom);

  const double s_start = current_s_m_;
  const double s_end =
      std::min(global_path_map_.total_length_m, s_start + cfg_.local_window_length_m);
  const int n_pts =
      std::max(2, static_cast<int>(std::ceil((s_end - s_start) / cfg_.sample_ds_m)) + 1);
  local_odom.points.reserve(static_cast<std::size_t>(n_pts));

  double prev_s = 0.0;
  double prev_t = 0.0;
  for (int i = 0; i < n_pts; ++i) {
    const double s_query =
        std::min(s_end, s_start + i * cfg_.sample_ds_m);
    PathPoint pt_map = global_path_map_.sampleByArcLength(s_query);
    PathPoint pt_odom = transformPointMapToOdom(pt_map);
    if (i == 0) {
      prev_s = pt_odom.s_m;
      prev_t = pt_odom.t_s;
    }
    pt_odom.s_m = std::max(0.0, pt_odom.s_m - prev_s);
    pt_odom.t_s = std::max(0.0, pt_odom.t_s - prev_t);
    local_odom.points.push_back(pt_odom);
  }

  local_odom.total_length_m = local_odom.points.back().s_m;
  local_odom.total_duration_s = local_odom.points.back().t_s;
  return local_odom;
}

std::array<Eigen::Vector3d, 4> PathTransformer::computeRLLookaheadsBody(
    const VehicleKinematics& state_odom,
    std::array<float, 12>* out_flat_12) {
  std::array<Eigen::Vector3d, 4> lookaheads_body;
  lookaheads_body.fill(Eigen::Vector3d::Zero());

  if (!global_path_map_.empty()) {
    updateProgress(state_odom.pos_odom);
    const Eigen::Quaterniond q_body_odom = state_odom.q_odom_base.conjugate();
    for (std::size_t k = 0; k < 4; ++k) {
      const double s_target = std::min(
          global_path_map_.total_length_m,
          current_s_m_ + cfg_.rl_lookahead_distances_m[k]);
      const PathPoint pt_map = global_path_map_.sampleByArcLength(s_target);
      const Eigen::Vector3d p_path_odom = T_odom_map_.apply(pt_map.pos);
      // Section 4.1 Eq: Delta p_k = R_B^odom * (p_path,k^odom - p_drone^odom)
      lookaheads_body[k] = q_body_odom * (p_path_odom - state_odom.pos_odom);
    }
  }

  if (out_flat_12 != nullptr) {
    for (std::size_t k = 0; k < 4; ++k) {
      (*out_flat_12)[3 * k + 0] = static_cast<float>(lookaheads_body[k].x());
      (*out_flat_12)[3 * k + 1] = static_cast<float>(lookaheads_body[k].y());
      (*out_flat_12)[3 * k + 2] = static_cast<float>(lookaheads_body[k].z());
    }
  }
  return lookaheads_body;
}

}  // namespace maze_planning
