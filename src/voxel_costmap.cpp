#include "maze_planning/voxel_costmap.hpp"

#include <algorithm>
#include <cmath>

namespace maze_planning {

VoxelCostmap::VoxelCostmap(const VoxelCostmapConfig& cfg,
                           std::shared_ptr<const GroundTruthMap> gt_map)
    : cfg_(cfg), gt_map_(std::move(gt_map)) {
  if (!gt_map_) {
    gt_map_ = std::make_shared<GroundTruthMap>();
  }
}

void VoxelCostmap::updateFromClampedPointCloud(
    double stamp_s,
    const Eigen::Vector3d& sensor_origin_map,
    const std::vector<Eigen::Vector3d>& points_map) {
  last_center_map_ = sensor_origin_map;

  // 1. Temporal decay of expired dynamic voxels
  dynamic_voxels_.erase(
      std::remove_if(dynamic_voxels_.begin(), dynamic_voxels_.end(),
                     [&](const DynamicVoxel& v) {
                       return (stamp_s - v.last_seen_s) > cfg_.dynamic_decay_time_s;
                     }),
      dynamic_voxels_.end());

  // 2. Ray-clear dynamic voxels along sensor rays that pass through them prior to the hit point.
  // CRITICAL FMEA PROTECTION (Disparity Punch-Through):
  // Static maze walls in gt_map_ are permanently hard-locked and can NEVER be cleared by depth rays!
  if (!dynamic_voxels_.empty() && !points_map.empty()) {
    const std::size_t stride = std::max<std::size_t>(1, points_map.size() / 64);
    for (std::size_t i = 0; i < points_map.size(); i += stride) {
      const Eigen::Vector3d& pt = points_map[i];
      const Eigen::Vector3d ray = pt - sensor_origin_map;
      const double ray_len = std::min(ray.norm(), cfg_.max_sensor_range_m);
      if (ray_len < 0.25) continue;
      const Eigen::Vector3d dir = ray / ray.norm();

      dynamic_voxels_.erase(
          std::remove_if(dynamic_voxels_.begin(), dynamic_voxels_.end(),
                         [&](const DynamicVoxel& v) {
                           const Eigen::Vector3d rel = v.center_map - sensor_origin_map;
                           const double proj = rel.dot(dir);
                           if (proj <= 0.15 || proj >= ray_len - 0.14) return false;
                           const double perp_sq = (rel - proj * dir).squaredNorm();
                           return perp_sq < (0.85 * cfg_.resolution_m * cfg_.resolution_m);
                         }),
          dynamic_voxels_.end());
    }
  }

  // 3. Insert clamped dynamic obstacle hits (ignoring points already inside static walls or floor/ceiling)
  for (const auto& pt : points_map) {
    if (!pt.allFinite()) continue;
    const double dist = (pt - sensor_origin_map).norm();
    // Disparity clamping: z_max <= 3.5 m
    if (dist < 0.15 || dist > cfg_.max_sensor_range_m) continue;
    const double alt = -pt.z();
    if (alt < cfg_.floor_crop_alt_m || alt > cfg_.ceiling_crop_alt_m) continue;

    // Ignore returns that are already part of a known static maze wall
    if (gt_map_ && gt_map_->clearance(pt) < 0.08) continue;

    const Eigen::Vector3i key =
        (pt / cfg_.resolution_m).array().floor().cast<int>();
    bool found = false;
    for (auto& v : dynamic_voxels_) {
      if (v.key == key) {
        v.hits = std::min(v.hits + 1, 50);
        v.last_seen_s = stamp_s;
        found = true;
        break;
      }
    }
    if (!found) {
      DynamicVoxel nv{};
      nv.key = key;
      nv.center_map = (key.cast<double>() + Eigen::Vector3d::Constant(0.5)) * cfg_.resolution_m;
      nv.hits = 1;
      nv.last_seen_s = stamp_s;
      dynamic_voxels_.push_back(nv);
    }
  }

  rebuildDynamicDistanceGrid(sensor_origin_map);
}

void VoxelCostmap::injectUnmodeledPillar(const Eigen::Vector2d& center_xy_map,
                                         double radius_m,
                                         double height_m,
                                         double stamp_s) {
  const double r_step = 0.5 * cfg_.resolution_m;
  for (double z = -height_m; z <= -cfg_.floor_crop_alt_m; z += cfg_.resolution_m) {
    for (double dx = -radius_m; dx <= radius_m; dx += r_step) {
      for (double dy = -radius_m; dy <= radius_m; dy += r_step) {
        if (dx * dx + dy * dy > radius_m * radius_m) continue;
        const Eigen::Vector3d pt(center_xy_map.x() + dx, center_xy_map.y() + dy, z);
        const Eigen::Vector3i key =
            (pt / cfg_.resolution_m).array().floor().cast<int>();
        bool exists = false;
        for (auto& v : dynamic_voxels_) {
          if (v.key == key) {
            v.hits = std::max(v.hits, cfg_.min_cluster_hits + 2);
            v.last_seen_s = stamp_s;
            exists = true;
            break;
          }
        }
        if (!exists) {
          DynamicVoxel nv{};
          nv.key = key;
          nv.center_map = (key.cast<double>() + Eigen::Vector3d::Constant(0.5)) * cfg_.resolution_m;
          nv.hits = cfg_.min_cluster_hits + 2;
          nv.last_seen_s = stamp_s;
          dynamic_voxels_.push_back(nv);
        }
      }
    }
  }
  rebuildDynamicDistanceGrid(Eigen::Vector3d(center_xy_map.x(), center_xy_map.y(), -0.80));
}

void VoxelCostmap::clearDynamicObstacles() {
  dynamic_voxels_.clear();
  dynamic_edt_.reset();
}

void VoxelCostmap::rebuildDynamicDistanceGrid(const Eigen::Vector3d& center_map) {
  std::vector<Eigen::Vector3f> pts;
  pts.reserve(dynamic_voxels_.size());
  for (const auto& v : dynamic_voxels_) {
    if (v.hits >= cfg_.min_cluster_hits) {
      for (int h = 0; h < cfg_.min_cluster_hits; ++h) {
        pts.push_back(v.center_map.cast<float>());
      }
    }
  }
  if (pts.empty()) {
    dynamic_edt_.reset();
    return;
  }
  isam2_depth::GridParams gp{};
  gp.resolution = cfg_.resolution_m;
  gp.half_extent = cfg_.half_extent_m;
  gp.min_hits = cfg_.min_cluster_hits;
  gp.max_distance = static_cast<float>(cfg_.max_distance_m);
  dynamic_edt_ = std::make_shared<const isam2_depth::DistanceGrid>(pts, center_map, gp);
}

double VoxelCostmap::dynamicClearance(const Eigen::Vector3d& p_map) const {
  double d = cfg_.max_distance_m;
  if (dynamic_edt_ && dynamic_edt_->inside(p_map)) {
    d = std::min(d, static_cast<double>(dynamic_edt_->distance(p_map)));
  }
  if (isam2_grid_ && isam2_grid_->inside(p_map)) {
    d = std::min(d, static_cast<double>(isam2_grid_->distance(p_map)));
  }
  return d;
}

double VoxelCostmap::clearance(const Eigen::Vector3d& p_map) const {
  const double d_gt = gt_map_ ? gt_map_->clearance(p_map) : 2.5;
  return std::min(d_gt, dynamicClearance(p_map));
}

double VoxelCostmap::clearanceAndGradient(const Eigen::Vector3d& p_map,
                                          Eigen::Vector3d& grad_out) const {
  Eigen::Vector3d g_gt(1.0, 0.0, 0.0);
  const double d_gt = gt_map_ ? gt_map_->clearanceAndGradient(p_map, g_gt) : 2.5;
  const double d_dyn = dynamicClearance(p_map);

  if (d_gt <= d_dyn) {
    grad_out = g_gt;
    return d_gt;
  }
  const double eps = cfg_.resolution_m;
  const double dx = clearance(p_map + Eigen::Vector3d(eps, 0.0, 0.0)) -
                    clearance(p_map - Eigen::Vector3d(eps, 0.0, 0.0));
  const double dy = clearance(p_map + Eigen::Vector3d(0.0, eps, 0.0)) -
                    clearance(p_map - Eigen::Vector3d(0.0, eps, 0.0));
  const double dz = clearance(p_map + Eigen::Vector3d(0.0, 0.0, eps)) -
                    clearance(p_map - Eigen::Vector3d(0.0, 0.0, eps));
  const Eigen::Vector3d g(dx, dy, dz);
  const double gn = g.norm();
  grad_out = (gn > 1e-9) ? (g / gn) : g_gt;
  return d_dyn;
}

double VoxelCostmap::obstacleCost(const Eigen::Vector3d& p_map, double safe_dist_m,
                                  double lethal_dist_m) const {
  const double d = clearance(p_map);
  if (d <= lethal_dist_m) {
    const double pen = lethal_dist_m - d;
    return 18.0 + 120.0 * pen;
  }
  if (d < safe_dist_m) {
    const double diff = safe_dist_m - d;
    return 0.5 * diff * diff + 0.08 * (diff * diff * diff) / std::max(0.04, d - lethal_dist_m);
  }
  return 0.0;
}

bool VoxelCostmap::isSegmentClear(const Eigen::Vector3d& a_map, const Eigen::Vector3d& b_map,
                                  double radius_m) const {
  if (gt_map_ && !gt_map_->isSegmentValid(a_map, b_map, radius_m)) return false;
  if (!dynamic_edt_ && !isam2_grid_) return true;

  const Eigen::Vector3d ab = b_map - a_map;
  const double len = ab.norm();
  if (len < 1e-6) return clearance(a_map) >= radius_m;
  const Eigen::Vector3d dir = ab / len;
  const double step = 0.5 * cfg_.resolution_m;
  for (double s = 0.0; s <= len; s += step) {
    if (clearance(a_map + dir * s) < radius_m) return false;
  }
  return clearance(b_map) >= radius_m;
}

void VoxelCostmap::exportOccupancyGrid2D(double slice_z_ned, double res_m,
                                         int& out_nx, int& out_ny,
                                         Eigen::Vector2d& out_origin_xy,
                                         std::vector<int8_t>& out_data) const {
  out_origin_xy = Eigen::Vector2d(-3.0, -3.0);
  out_nx = static_cast<int>(std::round(6.0 / res_m));
  out_ny = static_cast<int>(std::round(6.0 / res_m));
  out_data.assign(static_cast<std::size_t>(out_nx * out_ny), 0);

  for (int iy = 0; iy < out_ny; ++iy) {
    const double y = out_origin_xy.y() + (iy + 0.5) * res_m;
    for (int ix = 0; ix < out_nx; ++ix) {
      const double x = out_origin_xy.x() + (ix + 0.5) * res_m;
      const Eigen::Vector3d p(x, y, slice_z_ned);
      if ((gt_map_ && gt_map_->isStaticWallOccupied(p)) || clearance(p) <= 0.5 * res_m) {
        out_data[static_cast<std::size_t>(iy * out_nx + ix)] = 100;
      }
    }
  }
}

std::vector<Eigen::Vector3d> VoxelCostmap::activeDynamicObstaclePoints() const {
  std::vector<Eigen::Vector3d> out;
  for (const auto& v : dynamic_voxels_) {
    if (v.hits >= cfg_.min_cluster_hits) out.push_back(v.center_map);
  }
  return out;
}

}  // namespace maze_planning
