// Shared types. Frames (same as policynav/obs.py): world = PX4 local NED, body = FRD, camera = optical
// (x right, y down, z forward). Poses are gtsam::Pose3 T_a_b (maps points in b to a).
#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <vector>

#include <Eigen/Core>
#include <gtsam/geometry/Pose3.h>

namespace isam2_depth {

struct PinholeCamera {
  double fx = 0, fy = 0, cx = 0, cy = 0;
  int width = 0, height = 0;

  // The camera whose pixel (u, v) is the s x s block starting at (s*u, s*v) of the full image, i.e. centred
  // on full-image pixel s*u + (s-1)/2 (OpenCV convention: x = (u - cx) / fx).
  PinholeCamera decimated(int s) const {
    PinholeCamera c;
    c.fx = fx / s; c.fy = fy / s;
    c.cx = (cx - 0.5 * (s - 1)) / s; c.cy = (cy - 0.5 * (s - 1)) / s;
    c.width = (width + s - 1) / s; c.height = (height + s - 1) / s;
    return c;
  }
};

// Metric depth, row-major. NaN = invalid (no stereo match); +inf = nothing within range (a simulator's miss).
struct DepthFrame {
  double stamp = 0;  // s
  int width = 0, height = 0;
  std::vector<float> depth_m;
};

// PX4 estimate at one instant (vehicle_local_position_v1 + vehicle_attitude), interpolated to `stamp`.
struct OdomSample {
  double stamp = 0;
  gtsam::Pose3 T_world_body;
  Eigen::Vector3d v_world = Eigen::Vector3d::Zero();
};

// A 3D point of a known landmark (e.g. the landing-pad ring centre) measured from depth, in the body frame.
struct LandmarkObs {
  int id = 0;
  Eigen::Vector3d p_body = Eigen::Vector3d::Zero();
  double sigma = 0.1;  // m, isotropic
};

}  // namespace isam2_depth
