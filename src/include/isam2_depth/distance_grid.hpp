// Dense local Euclidean distance field (world NED) built from the smoothed keyframe clouds: what an MPPI cost
// samples (O(1) per rollout point). Occupied = a voxel with >= min_hits depth points.
#pragma once

#include <vector>

#include <Eigen/Core>

namespace isam2_depth {

struct GridParams {
  double resolution = 0.1;                        // m
  Eigen::Vector3d half_extent{5.0, 5.0, 2.0};     // m around the vehicle (x, y, z NED)
  int min_hits = 2;                               // stereo speckle rejection
  float max_distance = 3.0f;                      // m, distances saturate here
};

class DistanceGrid {
 public:
  DistanceGrid() = default;
  // `points` in world NED; the grid is centred (and voxel-aligned) at `center`.
  DistanceGrid(const std::vector<Eigen::Vector3f>& points, const Eigen::Vector3d& center, const GridParams& p);

  // Distance to the nearest occupied voxel centre (m), saturated at max_distance. Outside the grid the map
  // knows nothing: returns `outside` (default: saturated = free; an MPPI should bound its rollouts itself).
  float distance(const Eigen::Vector3d& p_world, float outside = -1.0f) const;
  bool inside(const Eigen::Vector3d& p_world) const;
  bool occupied(const Eigen::Vector3d& p_world) const { return inside(p_world) && distance(p_world) <= 0.0f; }

  const Eigen::Vector3d& origin() const { return origin_; }   // corner of voxel (0, 0, 0)
  double resolution() const { return res_; }
  Eigen::Vector3i size() const { return {nx_, ny_, nz_}; }
  const std::vector<float>& data() const { return dist_; }   // x fastest
  std::vector<Eigen::Vector3f> occupiedCentres() const;

 private:
  int index(const Eigen::Vector3d& p, bool* ok) const;
  Eigen::Vector3d origin_ = Eigen::Vector3d::Zero();
  double res_ = 0.1;
  float max_dist_ = 3.0f;
  int nx_ = 0, ny_ = 0, nz_ = 0;
  std::vector<float> dist_;
};

}  // namespace isam2_depth
