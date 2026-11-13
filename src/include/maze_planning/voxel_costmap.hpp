#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "isam2_depth/distance_grid.hpp"
#include "maze_planning/groundtruth_map.hpp"
#include "maze_planning/types.hpp"

namespace maze_planning {

struct VoxelCostmapConfig {
  double resolution_m{0.08};                     // local dynamic voxel resolution (m)
  Eigen::Vector3d half_extent_m{4.0, 4.0, 1.8};  // local dynamic grid window around vehicle
  double max_sensor_range_m{3.50};               // disparity clamping z_max <= 3.5 m (Section 2.3 & 3.1)
  double floor_crop_alt_m{0.14};                 // ignore floor returns below 14 cm altitude
  double ceiling_crop_alt_m{1.65};               // ignore ceiling returns above 1.65 m altitude
  double dynamic_decay_time_s{4.0};              // STVL temporal decay for dynamic voxels (s)
  int min_cluster_hits{2};                       // speckle filter threshold
  double max_distance_m{2.5};                    // saturation distance for dynamic EDT
};

/**
 * @brief Spatio-Temporal Voxel Layer (STVL) / Local Costmap (Section 2.2, 2.3, WP-3 Task 3, Section 6 FMEA).
 *
 * Combines:
 *   1. Hard-locked static maze walls from GroundTruthMap (disparity punch-through on netting can NEVER clear
 *      voxels inside the prior static maze wall bounding boxes, Section 6 FMEA).
 *   2. Dynamic / unmodeled interior obstacles (support beams, vertical poles, unexpected clutter) updated from
 *      clamped stereo point clouds (/oak/points/clamped, z_max <= 3.5 m) with ray-clearing of dynamic voxels
 *      and temporal decay, solved into an O(1) 3D Euclidean Distance Transform (EDT).
 *   3. Optional zero-copy in-process DistanceGrid from isam2_depth (DepthIsam2::latest().distance).
 */
class VoxelCostmap {
 public:
  explicit VoxelCostmap(const VoxelCostmapConfig& cfg = VoxelCostmapConfig{},
                        std::shared_ptr<const GroundTruthMap> gt_map = nullptr);

  void setGroundTruthMap(std::shared_ptr<const GroundTruthMap> gt_map) {
    gt_map_ = std::move(gt_map);
  }
  const VoxelCostmapConfig& config() const { return cfg_; }
  const GroundTruthMap& groundTruthMap() const { return *gt_map_; }

  /**
   * @brief Update the in-process iSAM2 DistanceGrid pointer (from MppiInput::distance).
   */
  void setIsam2DistanceGrid(std::shared_ptr<const isam2_depth::DistanceGrid> grid) {
    isam2_grid_ = std::move(grid);
  }

  /**
   * @brief Ingest a clamped 3D point cloud (/oak/points/clamped) in the odom/map frame.
   * Performs disparity clamping (z <= 3.5m), floor/ceiling cropping, ray-clearing of dynamic
   * voxels (while preserving hard-locked static maze walls!), and rebuilds the local 3D EDT.
   */
  void updateFromClampedPointCloud(double stamp_s,
                                   const Eigen::Vector3d& sensor_origin_map,
                                   const std::vector<Eigen::Vector3d>& points_map);

  /**
   * @brief Inject a synthetic unmodeled vertical pole/pillar obstacle (for Milestone 3 & FMEA testing).
   */
  void injectUnmodeledPillar(const Eigen::Vector2d& center_xy_map, double radius_m = 0.10,
                             double height_m = 1.70, double stamp_s = 0.0);

  /**
   * @brief Clear all dynamic (non-static-wall) obstacles.
   */
  void clearDynamicObstacles();

  /**
   * @brief Combined O(1) obstacle clearance (m) at a point in map/odom frame:
   * min(GroundTruthMap static wall clearance, STVL dynamic obstacle clearance, iSAM2 DistanceGrid).
   */
  double clearance(const Eigen::Vector3d& p_map) const;

  /**
   * @brief Clearance (m) to dynamic obstacles only (STVL dynamic EDT and iSAM2 DistanceGrid).
   */
  double dynamicClearance(const Eigen::Vector3d& p_map) const;

  /**
   * @brief Combined clearance and unit spatial gradient nabla d(p).
   */
  double clearanceAndGradient(const Eigen::Vector3d& p_map, Eigen::Vector3d& grad_out) const;

  /**
   * @brief Smooth quadratic + barrier obstacle cost for MPPI rollout evaluation.
   */
  double obstacleCost(const Eigen::Vector3d& p_map, double safe_dist_m = 0.34,
                      double lethal_dist_m = 0.18) const;

  /**
   * @brief Check continuous segment validity against both static maze walls and dynamic obstacles.
   */
  bool isSegmentClear(const Eigen::Vector3d& a_map, const Eigen::Vector3d& b_map,
                      double radius_m = 0.21) const;

  /**
   * @brief Export a 2D OccupancyGrid slice (values: 0 = free, 100 = occupied static/dynamic)
   * at a given flight altitude for publishing on /costmap/local_voxels.
   */
  void exportOccupancyGrid2D(double slice_z_ned, double res_m,
                             int& out_nx, int& out_ny,
                             Eigen::Vector2d& out_origin_xy,
                             std::vector<int8_t>& out_data) const;

  /**
   * @brief Return 3D centers of active dynamic obstacle voxels.
   */
  std::vector<Eigen::Vector3d> activeDynamicObstaclePoints() const;

  std::size_t numDynamicObstacles() const { return dynamic_voxels_.size(); }

 private:
  struct DynamicVoxel {
    Eigen::Vector3i key{Eigen::Vector3i::Zero()};
    Eigen::Vector3d center_map{Eigen::Vector3d::Zero()};
    int hits{1};
    double last_seen_s{0.0};
  };

  void rebuildDynamicDistanceGrid(const Eigen::Vector3d& center_map);

  VoxelCostmapConfig cfg_;
  std::shared_ptr<const GroundTruthMap> gt_map_;
  std::shared_ptr<const isam2_depth::DistanceGrid> dynamic_edt_;
  std::shared_ptr<const isam2_depth::DistanceGrid> isam2_grid_;
  std::vector<DynamicVoxel> dynamic_voxels_;
  Eigen::Vector3d last_center_map_{Eigen::Vector3d::Zero()};
};

}  // namespace maze_planning
