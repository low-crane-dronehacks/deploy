// Depth -> organised point cloud with normals, and projective point-to-plane ICP between two depth frames.
// The ICP result (relative pose + information matrix) becomes a BetweenFactor in the iSAM2 graph.
#pragma once

#include <cmath>
#include <optional>
#include <vector>

#include <Eigen/Core>
#include <gtsam/geometry/Pose3.h>

#include "isam2_depth/types.hpp"

namespace isam2_depth {

struct CloudParams {
  int stride = 4;            // decimation by block mean: 640x400 -> 160x100
  int normal_radius = 2;     // normals: plane fit over a (2r+1)^2 window of decimated pixels
  double min_range = 0.35;   // m, OAK-D S2 minimum range
  double max_range = 6.0;    // m, stereo noise grows ~z^2; the arena is 6.1 m across
  double max_normal_jump = 0.05;  // m + 3% of depth, per pixel: a larger depth step is an edge
};

// Organised cloud in the camera optical frame (one entry per decimated pixel; invalid = NaN).
struct OrganisedCloud {
  PinholeCamera cam;  // decimated intrinsics
  std::vector<Eigen::Vector3f> points;
  std::vector<Eigen::Vector3f> normals;
  int n_valid = 0;    // pixels with a point and a normal

  bool valid(int i) const { return std::isfinite(normals[i].x()); }
};

OrganisedCloud makeCloud(const DepthFrame& depth, const PinholeCamera& full_cam, const CloudParams& p);

struct IcpParams {
  int max_iterations = 30;
  double converged_rot = 1e-4, converged_trans = 1e-4;   // rad, m: step size that ends the iterations
  int point_step = 2;            // use every n-th valid source point
  // Correspondence gates. Point-to-point must stay loose: floor / ceiling are seen at grazing angles, where a few
  // cm of vertical offset moves the projective match tens of cm along the surface.
  double max_corr_dist = 0.5;    // m, point-to-point
  double max_plane_dist = 0.1;   // m, point-to-plane
  double min_normal_cos = 0.8;   // normals of a correspondence agree within ~37 deg
  double huber_k = 0.03;         // m, point-to-plane residual where Huber starts
  // per-point noise: sigma = sigma0 + sigma_z2 * z^2 (stereo: fx 421, baseline 7.5 cm, ~0.25 px disparity noise)
  double sigma0 = 0.01, sigma_z2 = 0.008;
  // Point-to-plane information assumes independent points; it is wildly overconfident. Divide it by this.
  double info_deflation = 100.0;
  int min_inliers = 300;
  double min_inlier_ratio = 0.3;
  double min_inlier_keep = 0.8;  // reject if the solution keeps fewer matches than this x those at the guess
  // weak prior toward the initial guess (see alignProjective); not included in IcpResult::info
  double prior_sigma_rot = 0.05, prior_sigma_trans = 0.1;   // rad, m
};

struct IcpResult {
  gtsam::Pose3 T_target_source;          // camera_target <- camera_source
  Eigen::Matrix<double, 6, 6> info;      // information on the right tangent [rot, trans] of T_target_source
  int inliers = 0;
  double inlier_ratio = 0, rmse = 0;
  int iterations = 0;
};

// Gauss-Newton on SE(3) starting from `init`; nullopt when the alignment is not trustworthy.
std::optional<IcpResult> alignProjective(const OrganisedCloud& target, const OrganisedCloud& source,
                                         const gtsam::Pose3& init, const IcpParams& p);

}  // namespace isam2_depth
