// CAD-derived 3D Closest-Point + Surface-Normal ESDF prior ("CADP" v1) for DepthIsam2.
//
// Built offline from `dronehacks-course-v3-obj.zip` (`scripts/build_cad_prior.py`). Each grid cell stores
// the exact closest point on the CAD triangle mesh (0.1 mm int16 offset from the cell centre) and its
// analytical triangle normal (int8 unit vector). Online, keyframe depth clouds are registered against this
// field with normal-gated, Cauchy M-estimated point-to-plane Gauss-Newton so that faithful CAD structures
// (barriers, walls, tunnel, boulders, floor) anchor global VIO drift while as-built differences or
// unmodelled obstacles in a real course scan are smoothly rejected.
#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <gtsam/geometry/Pose3.h>

#include "isam2_depth/depth_icp.hpp"

namespace isam2_depth {

class CadPriorEsdf {
 public:
  struct Cell {
    int16_t dx, dy, dz;       // (q_closest - cell_centre) in units of 0.1 mm (1e-4 m)
    int8_t nx, ny, nz, flags; // unit surface normal * 127
  };
  static_assert(sizeof(Cell) == 10, "CADP v1 cell layout must be 10 bytes");

  static std::shared_ptr<const CadPriorEsdf> load(const std::string& path, std::string* err = nullptr);

  bool query(const Eigen::Vector3d& p_ned, Eigen::Vector3d& q_closest, Eigen::Vector3d& unit_normal) const;

  Eigen::Vector3i shape() const { return {nx_, ny_, nz_}; }
  double resolution() const { return res_; }
  const Eigen::Vector3d& origin() const { return lo_; }

 private:
  int nx_ = 0, ny_ = 0, nz_ = 0;
  double res_ = 0.05;
  Eigen::Vector3d lo_ = Eigen::Vector3d::Zero();
  std::vector<Cell> cells_;
};

struct CadPriorParams {
  int max_iterations = 15;
  double converged_rot = 1e-4, converged_trans = 1e-4;
  int point_step = 3;
  double max_corr_dist = 0.14;    // m: gates out as-built shifted/unmodelled walls (>14 cm from CAD)
  double max_plane_dist = 0.10;   // m: point-to-plane gate
  double min_normal_cos = 0.90;   // |nu_cad . n_depth| >= cos(~25 deg) rejects tilted as-built walls
  double cauchy_k = 0.04;         // m: redescending Cauchy scale for as-built tolerance
  double max_rmse = 0.048;        // m: reject alignment when matched surfaces have systematic offset
  double sigma0 = 0.015, sigma_z2 = 0.008;
  double info_deflation = 900.0;  // gentle global drift anchor; never overrides tight local ICP
  int min_inliers = 140;
  double min_inlier_ratio = 0.30; // require >=30% of visible scene to agree with CAD
  double min_inlier_keep = 0.85;
  double prior_sigma_rot = 0.05, prior_sigma_trans = 0.10;  // MAP regulariser (excluded from returned info)
  double gate_trans = 0.16;       // m max correction relative to graph prediction
  double gate_rot_deg = 5.0;
};

struct CadAlignResult {
  gtsam::Pose3 T_world_body;
  Eigen::Matrix<double, 6, 6> info;  // right-tangent information on X(k) [rot, trans]
  int inliers = 0;
  double inlier_ratio = 0.0, rmse = 0.0;
  int iterations = 0;
};

std::optional<CadAlignResult> alignToCadPrior(const OrganisedCloud& cloud,
                                              const gtsam::Pose3& T_body_cam,
                                              const gtsam::Pose3& T_world_body_init,
                                              const CadPriorEsdf& prior,
                                              const CadPriorParams& p);

}  // namespace isam2_depth
