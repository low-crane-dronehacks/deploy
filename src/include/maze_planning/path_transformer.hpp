#pragma once

#include <array>
#include <vector>

#include "maze_planning/types.hpp"

namespace maze_planning {

struct PathTransformerConfig {
  double local_window_length_m{4.50};  // sliding lookahead window length in odom frame (m)
  double sample_ds_m{0.10};            // spatial sampling step along local reference path (m)
  std::array<double, 4> rl_lookahead_distances_m{0.5, 1.0, 2.0, 3.5}; // Section 4.1: d = [0.5, 1.0, 2.0, 3.5] m
};

/**
 * @brief Path Transformer (Section 2.2, 2.3, 4.1, WP-3 Task 2, Section 6 FMEA).
 *
 * Responsibilities:
 *   1. Ingests the global RRT* path (/plan/global_rrt) in the `map` frame and the latest
 *      `map -> odom` transform T_map_odom (broadcast by iSAM2 at 20 Hz).
 *   2. Transforms the sliding lookahead window of the RRT* path into the continuous,
 *      jump-free `odom` frame (`T_odom_map = T_map_odom.inverse()`) and outputs `/plan/local_reference`
 *      at 20 Hz for the MPPI controller.
 *   3. Extracts N = 4 future lookahead coordinates spaced at d = [0.5, 1.0, 2.0, 3.5] m along the
 *      RRT* path and transforms them into the vehicle body frame B:
 *        Delta p_k = R_B^odom * (p_path,k^odom - p_drone^odom) in R^3  (k = 1..4)
 *      for `/plan/rl_lookaheads` at 50 Hz (Section 4.1).
 */
class PathTransformer {
 public:
  explicit PathTransformer(const PathTransformerConfig& cfg = PathTransformerConfig{});

  /**
   * @brief Update the global RRT* path in the `map` frame (/plan/global_rrt).
   */
  void setGlobalPathMap(const PlannedPath& path_map);

  /**
   * @brief Update the global correction transform T_map_odom (from iSAM2 / TF2).
   */
  void setMapToOdomTransform(const SE3Transform& T_map_odom);

  /**
   * @brief Extract the sliding local reference path in the `odom` frame (/plan/local_reference)
   * starting from the vehicle's monotonic projection along the global RRT* path.
   */
  PlannedPath extractLocalReferenceOdom(const VehicleKinematics& state_odom);

  /**
   * @brief Compute the N = 4 relative lookahead vectors in the body frame B (/plan/rl_lookaheads):
   *   Delta p_k = R_B^odom (p_path,k^odom - p_drone^odom), k in {1..4} at d = [0.5, 1.0, 2.0, 3.5] m.
   * Also returns the flattened 12-float array for std_msgs/Float32MultiArray.
   */
  std::array<Eigen::Vector3d, 4> computeRLLookaheadsBody(
      const VehicleKinematics& state_odom,
      std::array<float, 12>* out_flat_12 = nullptr);

  void resetProgress() { progress_idx_ = 0; current_s_m_ = 0.0; }
  bool hasGlobalPath() const { return !global_path_map_.empty(); }
  double currentArcLengthM() const { return current_s_m_; }
  double remainingDistanceM() const {
    return std::max(0.0, global_path_map_.total_length_m - current_s_m_);
  }
  const SE3Transform& mapToOdomTransform() const { return T_map_odom_; }
  const SE3Transform& odomToMapTransform() const { return T_odom_map_; }

 private:
  PathPoint transformPointMapToOdom(const PathPoint& pt_map) const;
  void updateProgress(const Eigen::Vector3d& pos_odom);

  PathTransformerConfig cfg_;
  PlannedPath global_path_map_;
  SE3Transform T_map_odom_{};
  SE3Transform T_odom_map_{};
  int progress_idx_{0};
  double current_s_m_{0.0};
};

}  // namespace maze_planning
