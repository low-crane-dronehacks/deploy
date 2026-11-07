// What the estimator hands to an MPPI controller (the controller itself lives elsewhere).
#pragma once

#include <cstdint>
#include <map>
#include <memory>

#include <Eigen/Core>
#include <gtsam/geometry/Pose3.h>

#include "isam2_depth/distance_grid.hpp"

namespace isam2_depth {

struct MppiInput {
  double stamp = 0;                       // s, the instant the state refers to
  uint64_t keyframe = 0;                  // newest keyframe folded into the estimate
  gtsam::Pose3 T_world_body;              // world NED <- body FRD (rollout start pose)
  Eigen::Vector3d v_world = Eigen::Vector3d::Zero();   // NED m/s (rollout start velocity)
  Eigen::Matrix<double, 6, 6> pose_cov = Eigen::Matrix<double, 6, 6>::Identity();  // body tangent [rot, trans]
  Eigen::Matrix3d vel_cov = Eigen::Matrix3d::Identity();
  // Obstacle distance field in world NED, consistent with T_world_body (rebuilt at every keyframe from the
  // smoothed poses). Shared, immutable: an MPPI thread may keep sampling an old one while a new one is built.
  std::shared_ptr<const DistanceGrid> distance;
  std::map<int, Eigen::Vector3d> landmarks;   // world NED, e.g. the landing-pad ring (goal), if observed
};

// Implement this in the MPPI and register it with the node; onEstimate is called on the estimator's thread.
class MppiStateSink {
 public:
  virtual ~MppiStateSink() = default;
  virtual void onEstimate(const MppiInput& in) = 0;
};

}  // namespace isam2_depth
