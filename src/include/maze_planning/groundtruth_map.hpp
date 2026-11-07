#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "maze_planning/types.hpp"

namespace maze_planning {

/**
 * @brief Analytical 3D Oriented Bounding Box (OBB) slab representing a static maze wall
 * or tunnel structural rib in the map frame (NED).
 */
struct OBBSlabPrimitive {
  Eigen::Vector2d center_xy{Eigen::Vector2d::Zero()};
  Eigen::Vector2d axis_u_xy{Eigen::Vector2d(1.0, 0.0)};
  Eigen::Vector2d axis_v_xy{Eigen::Vector2d(0.0, 1.0)};
  double half_len_m{0.5};
  double half_thick_m{0.0508};
  double z_min_ned{-1.7272};
  double z_max_ned{0.0};

  double signedDistance(const Eigen::Vector3d& p_ned, Eigen::Vector3d* closest_pt_out = nullptr) const;
};

/**
 * @brief Analytical 3D Semi-Ellipsoid primitive representing a floor boulder in the map frame.
 */
struct BoulderPrimitive {
  Eigen::Vector3d center_ned{Eigen::Vector3d::Zero()};
  Eigen::Vector3d radii_m{Eigen::Vector3d(0.2286, 0.2286, 0.5334)};

  double signedDistance(const Eigen::Vector3d& p_ned, Eigen::Vector3d* closest_pt_out = nullptr) const;
};

/**
 * @brief Analytical 3D Tunnel Archway primitive (curved arch roof + side walls).
 * Tunnel spans x in [-0.3048, 1.2192], y in [1.5748, 3.0480] (center y = 2.3114, arch radius 0.728 m).
 */
struct TunnelArchPrimitive {
  Eigen::Vector2d entry_xy{Eigen::Vector2d(1.2192, 2.3114)};
  Eigen::Vector2d exit_xy{Eigen::Vector2d(-0.3048, 2.3114)};
  Eigen::Vector2d axis_u_xy{Eigen::Vector2d(-1.0, 0.0)};
  Eigen::Vector2d axis_v_xy{Eigen::Vector2d(0.0, -1.0)};
  double length_m{1.5240};
  double half_width_m{0.7280};
  double arch_top_m{1.8542};
  double shell_thick_m{0.0172};
};

/**
 * @brief Surface triangle for exact BVH closest-point and ray-intersection queries.
 */
struct SurfaceTriangle {
  Eigen::Vector3d v0{Eigen::Vector3d::Zero()};
  Eigen::Vector3d v1{Eigen::Vector3d::Zero()};
  Eigen::Vector3d v2{Eigen::Vector3d::Zero()};
  Eigen::Vector3d normal{Eigen::Vector3d(0.0, 0.0, -1.0)};
  Eigen::Vector3d centroid{Eigen::Vector3d::Zero()};
};

/**
 * @brief Flat AABB Bounding Volume Hierarchy (BVH) node.
 */
struct BVHNode {
  Eigen::Vector3d bbox_min{Eigen::Vector3d::Zero()};
  Eigen::Vector3d bbox_max{Eigen::Vector3d::Zero()};
  int left_child{-1};
  int right_child{-1};
  int tri_start{0};
  int tri_count{0};

  bool isLeaf() const { return tri_count > 0; }
  double minDistanceSqToPoint(const Eigen::Vector3d& p) const;
};

/**
 * @brief Blocked corridor sphere registered dynamically when an unmodeled pillar blocks a narrow passage
 * (Section 6 FMEA: Pillar Dead-Center recovery -> mark edge blocked in RRT* and trigger global replan).
 */
struct BlockedCorridorSphere {
  Eigen::Vector3d center_ned{Eigen::Vector3d::Zero()};
  double radius_m{0.35};
};

/**
 * @brief New Non-ESDF GroundTruth Map Encoding ("CGTM" v1) for RRT* Global Planning & Static Wall Locking.
 *
 * Does NOT use a pre-made ESDF map. Encodes the static CAD/mesh maze map as:
 *   1. Analytical 3D OBB wall & tunnel slabs (exact signed distance & normal gradients),
 *   2. Analytical 3D semi-ellipsoid boulder primitives,
 *   3. Surface triangle mesh indexed in a flat AABB Bounding Volume Hierarchy (BVH),
 *   4. 1-bit-per-voxel hard-locked static wall occupancy bitset (uint64_t words, zero ESDF distances),
 *   5. Dynamic corridor blockage spheres for FMEA Pillar Dead-Center global replanning.
 */
class GroundTruthMap {
 public:
  GroundTruthMap();

  /**
   * @brief Load the CGTM v1 binary groundtruth map (config/course_groundtruth.bin).
   */
  bool loadFromBinary(const std::string& path);

  /**
   * @brief Parse Wavefront OBJ mesh (e.g. sitl/world/course.obj) directly and build BVH + bitset.
   */
  bool loadFromObj(const std::string& obj_path);

  /**
   * @brief Initialize default analytical DroneHacks 2026 maze geometry.
   */
  void initDefaultCourse();

  /**
   * @brief Exact clearance distance (m) to the nearest static maze obstacle or blocked corridor.
   */
  double clearance(const Eigen::Vector3d& p_ned) const;

  /**
   * @brief Exact clearance distance (m) and unit outward gradient nabla d(p).
   */
  double clearanceAndGradient(const Eigen::Vector3d& p_ned, Eigen::Vector3d& grad_out) const;

  /**
   * @brief O(1) check whether a 3D point lies inside a hard-locked static maze wall voxel.
   * Used by VoxelCostmap to prevent stereo disparity punch-through on netting from clearing static walls.
   */
  bool isStaticWallOccupied(const Eigen::Vector3d& p_ned) const;

  /**
   * @brief Sphere-traced continuous segment collision check between a_ned and b_ned.
   * Exploits the Lipschitz-1 property of Euclidean clearance for fast, exact edge validation in RRT*.
   */
  bool isSegmentValid(const Eigen::Vector3d& a_ned, const Eigen::Vector3d& b_ned,
                      double radius_m = 0.20, double min_step_m = 0.025,
                      double alt_min_m = 0.18, double alt_max_m = 1.62) const;

  /**
   * @brief Smooth quadratic obstacle cost for MPPI rollouts:
   * c(p) = 0.5 * (safe_dist - d)^2 if d < safe_dist else 0.0
   */
  double getObstacleCost(const Eigen::Vector3d& p_ned, double safe_dist_m = 0.32) const {
    const double d = clearance(p_ned);
    if (d < safe_dist_m) {
      const double diff = safe_dist_m - d;
      return 0.5 * diff * diff;
    }
    return 0.0;
  }

  /**
   * @brief Refine gate/window crossing center to maximize clearance between posts.
   */
  Eigen::Vector3d refineGateCenter(const GateAperture& gate, double margin_from_posts_m = 0.24) const;

  /**
   * @brief FMEA Pillar Dead-Center recovery: mark a corridor sphere as blocked for RRT* replanning.
   */
  void markCorridorBlocked(const Eigen::Vector3d& center_ned, double radius_m = 0.38);
  void clearBlockedCorridors() { blocked_corridors_.clear(); }
  const std::vector<BlockedCorridorSphere>& blockedCorridors() const { return blocked_corridors_; }

  bool isLoaded() const { return binary_loaded_ || obj_loaded_; }
  std::size_t numOBBSlabs() const { return obb_slabs_.size(); }
  std::size_t numBoulders() const { return boulders_.size(); }
  std::size_t numTriangles() const { return triangles_.size(); }
  std::size_t numBVHNodes() const { return bvh_nodes_.size(); }
  const std::vector<OBBSlabPrimitive>& obbSlabs() const { return obb_slabs_; }
  const std::vector<BoulderPrimitive>& boulders() const { return boulders_; }
  const TunnelArchPrimitive& tunnelArch() const { return tunnel_; }
  const std::vector<SurfaceTriangle>& triangles() const { return triangles_; }
  const std::vector<GateAperture>& defaultGates() const { return default_gates_; }
  Eigen::Vector3d launchMarkNed() const { return launch_mark_ned_; }
  Eigen::Vector3d padCenterNed() const { return pad_center_ned_; }

 private:
  static OBBSlabPrimitive makeOBB(double ax, double ay, double bx, double by,
                                  double half_thick, double z_min, double z_max);
  void buildBVH();
  int buildBVHRecursive(std::vector<int>& tri_indices, int start, int end);
  double queryAnalytical(const Eigen::Vector3d& p_ned, Eigen::Vector3d& closest_pt_out) const;
  void queryBVH(int node_idx, const Eigen::Vector3d& p_ned,
                double& best_dist_sq, Eigen::Vector3d& best_closest_pt) const;
  static Eigen::Vector3d closestPointOnTriangle(const Eigen::Vector3d& p, const SurfaceTriangle& tri);

  double arena_half_m_{2.99};
  double wall_top_m_{1.7272};

  std::vector<OBBSlabPrimitive> obb_slabs_;
  std::vector<BoulderPrimitive> boulders_;
  TunnelArchPrimitive tunnel_;

  std::vector<SurfaceTriangle> triangles_;
  std::vector<BVHNode> bvh_nodes_;

  int bit_nx_{123};
  int bit_ny_{123};
  int bit_nz_{62};
  double bit_res_{0.05};
  Eigen::Vector3d bit_origin_{-3.05, -3.05, -3.05};
  std::vector<uint64_t> occ_bitset_;

  std::vector<BlockedCorridorSphere> blocked_corridors_;
  std::vector<GateAperture> default_gates_;
  Eigen::Vector3d launch_mark_ned_{0.9144, -2.5146, -0.08};
  Eigen::Vector3d pad_center_ned_{-0.7925, -2.0422, -0.08};

  bool binary_loaded_{false};
  bool obj_loaded_{false};
};

}  // namespace maze_planning
