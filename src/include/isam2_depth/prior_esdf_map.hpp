// Arena-wide 3D log-odds occupancy + Euclidean distance field seeded at t = 0 from the Course v3 CAD prior ESDF
// (`CadPriorEsdf`), and updated online from iSAM2-smoothed keyframe depth rays:
//   - Free-space ray traversal (-l_free) carves away stale CAD priors where the real arena is empty space.
//   - Surface depth hits (+l_occ) reinforce confirmed priors and populate newly discovered / shifted obstacles.
// Floor returns (-z < 0.16 m) and outer safety-net / room boundaries (|x|, |y| > 2.92 m) lie outside the voxel
// volume so rays hitting the floor behind an absent CAD obstacle still carve through that obstacle's voxels
// without ever marking the floor itself as an obstacle.
#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include <Eigen/Core>
#include <gtsam/geometry/Pose3.h>

#include "isam2_depth/cad_prior.hpp"
#include "isam2_depth/distance_grid.hpp"
#include "isam2_depth/types.hpp"

namespace isam2_depth {

struct PriorEsdfParams {
  double resolution = 0.08;                        // m, voxel size
  Eigen::Vector3d min_ned{-2.92, -2.92, -1.96};    // m, excludes perimeter net (3.05 m) and ceiling
  Eigen::Vector3d max_ned{ 2.92,  2.92, -0.16};    // m, excludes floor slab (-z < 0.16 m)
  float l_prior_occ = 1.50f;                       // initial log-odds on CAD obstacle voxels
  float l_prior_free = -1.00f;                     // initial log-odds in CAD free space
  float l_free = -0.28f;                           // log-odds decrement per free-space ray traversal
  float l_occ = 0.65f;                             // log-odds increment per finite surface hit
  float l_min = -4.0f;
  float l_max = 4.0f;
  float new_occ_thresh = 0.20f;                    // log-odds threshold for newUpdatedObstacles()
  float cleared_thresh = -0.20f;                   // log-odds threshold for clearedEmptySpace()
  double min_range = 0.35;                         // m
  double max_occ_range = 5.5;                      // m, finite hits beyond this carve free space only
  double raycast_max_range = 6.0;                  // m, max free-space carving length
  int pixel_stride = 8;                            // 640x400 -> 80x50 rays per keyframe
};

class PriorEsdfMap {
 public:
  struct RaySample {
    Eigen::Vector3f dir_cam;   // unit vector in camera optical frame
    float range_m = 0.0f;
    bool surface_hit = false;
  };

  explicit PriorEsdfMap(const PriorEsdfParams& p = {}, const CadPriorEsdf* cad_prior = nullptr);

  void setCadPrior(const CadPriorEsdf* cad_prior);
  void resetToPrior();

  std::vector<RaySample> extractRays(const DepthFrame& depth, const PinholeCamera& cam) const;
  void integrateRays(const std::vector<RaySample>& rays, const gtsam::Pose3& T_world_cam);

  // Active CAD prior voxel centres: seeded from CAD at t = 0 and still occupied (log_odds > 0).
  std::vector<Eigen::Vector3f> priors() const;

  // Newly discovered / shifted real-world obstacle voxel centres (!prior_occ && log_odds > new_occ_thresh).
  std::vector<Eigen::Vector3f> newUpdatedObstacles() const;

  // Prebaked CAD prior voxel centres cleared by pass-through depth rays (prior_occ && log_odds < cleared_thresh).
  std::vector<Eigen::Vector3f> clearedEmptySpace() const;

  // All currently occupied obstacle voxel centres (priors U newUpdatedObstacles).
  std::vector<Eigen::Vector3f> activeObstacles() const;

  // Exact 3D Euclidean Distance Field centred on `center` built from activeObstacles().
  std::shared_ptr<const DistanceGrid> buildDistanceGrid(const Eigen::Vector3d& center,
                                                        const GridParams& grid_params) const;

  Eigen::Vector3i shape() const { return {nx_, ny_, nz_}; }
  double resolution() const { return p_.resolution; }

 private:
  int flatIndex(int ix, int iy, int iz) const { return (iz * ny_ + iy) * nx_ + ix; }
  Eigen::Vector3f voxelCenter(int ix, int iy, int iz) const;

  PriorEsdfParams p_;
  int nx_ = 0, ny_ = 0, nz_ = 0;
  Eigen::Vector3d origin_ = Eigen::Vector3d::Zero();
  std::vector<uint8_t> prior_occ_;
  std::vector<float> prior_seed_;
  std::vector<float> log_odds_;
};

}  // namespace isam2_depth
