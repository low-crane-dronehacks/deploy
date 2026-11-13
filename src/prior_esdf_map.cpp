#include "isam2_depth/prior_esdf_map.hpp"

#include <algorithm>
#include <cmath>

namespace isam2_depth {

PriorEsdfMap::PriorEsdfMap(const PriorEsdfParams& p, const CadPriorEsdf* cad_prior) : p_(p), origin_(p.min_ned) {
  nx_ = std::max(1, static_cast<int>(std::round((p_.max_ned.x() - p_.min_ned.x()) / p_.resolution)));
  ny_ = std::max(1, static_cast<int>(std::round((p_.max_ned.y() - p_.min_ned.y()) / p_.resolution)));
  nz_ = std::max(1, static_cast<int>(std::round((p_.max_ned.z() - p_.min_ned.z()) / p_.resolution)));
  setCadPrior(cad_prior);
}

Eigen::Vector3f PriorEsdfMap::voxelCenter(int ix, int iy, int iz) const {
  return Eigen::Vector3f(static_cast<float>(origin_.x() + (ix + 0.5) * p_.resolution),
                         static_cast<float>(origin_.y() + (iy + 0.5) * p_.resolution),
                         static_cast<float>(origin_.z() + (iz + 0.5) * p_.resolution));
}

void PriorEsdfMap::setCadPrior(const CadPriorEsdf* cad_prior) {
  const size_t total = static_cast<size_t>(nx_) * ny_ * nz_;
  prior_occ_.assign(total, 0);
  prior_seed_.assign(total, p_.l_prior_free);

  if (cad_prior) {
    // Half-diagonal of a voxel (~0.62 * res) captures thin CAD walls cleanly without over-thickening.
    const double occ_thresh = 0.62 * p_.resolution;
    for (int iz = 0; iz < nz_; ++iz) {
      for (int iy = 0; iy < ny_; ++iy) {
        for (int ix = 0; ix < nx_; ++ix) {
          const Eigen::Vector3d c = voxelCenter(ix, iy, iz).cast<double>();
          Eigen::Vector3d q_closest, n_unit;
          if (!cad_prior->query(c, q_closest, n_unit)) continue;
          // Exclude outer perimeter safety-net panels (|x| or |y| >= 3.00 m) and floor plane (z >= -0.10 m)
          if (std::abs(q_closest.x()) >= 2.99 || std::abs(q_closest.y()) >= 2.99 || q_closest.z() >= -0.10) continue;
          if ((c - q_closest).norm() <= occ_thresh) {
            const int idx = flatIndex(ix, iy, iz);
            prior_occ_[idx] = 1;
            prior_seed_[idx] = p_.l_prior_occ;
          }
        }
      }
    }
  }
  resetToPrior();
}

void PriorEsdfMap::resetToPrior() {
  log_odds_ = prior_seed_;
}

std::vector<PriorEsdfMap::RaySample> PriorEsdfMap::extractRays(const DepthFrame& depth,
                                                               const PinholeCamera& cam) const {
  std::vector<RaySample> out;
  if (depth.width != cam.width || depth.height != cam.height) return out;
  const int step = std::max(1, p_.pixel_stride);
  out.reserve(static_cast<size_t>((depth.width / step + 1) * (depth.height / step + 1)));

  for (int v = step / 2; v < depth.height; v += step) {
    for (int u = step / 2; u < depth.width; u += step) {
      const float z = depth.depth_m[static_cast<size_t>(v) * depth.width + u];
      if (std::isnan(z)) continue;
      Eigen::Vector3f dir(static_cast<float>((u - cam.cx) / cam.fx),
                          static_cast<float>((v - cam.cy) / cam.fy),
                          1.0f);
      const float norm = dir.norm();
      if (!(norm > 0.0f)) continue;
      dir /= norm;

      if (std::isinf(z)) {
        out.push_back({dir, static_cast<float>(p_.raycast_max_range), false});
      } else if (z >= static_cast<float>(p_.min_range)) {
        const float range = z * norm;   // Euclidean ray distance along unit `dir`
        const bool is_hit = (z <= static_cast<float>(p_.max_occ_range));
        out.push_back({dir, range, is_hit});
      }
    }
  }
  return out;
}

void PriorEsdfMap::integrateRays(const std::vector<RaySample>& rays, const gtsam::Pose3& T_world_cam) {
  if (rays.empty()) return;
  const Eigen::Vector3d o = T_world_cam.translation();
  const Eigen::Matrix3d R_wc = T_world_cam.rotation().matrix();
  const double res = p_.resolution;
  const double inv_res = 1.0 / res;
  const Eigen::Vector3d box_max = origin_ + res * Eigen::Vector3d(nx_, ny_, nz_);

  for (const auto& ray : rays) {
    const Eigen::Vector3d d = R_wc * ray.dir_cam.cast<double>();
    const double t0 = 0.20;
    const double free_margin = ray.surface_hit ? (0.85 * res) : 0.0;
    const double t1 = std::min(static_cast<double>(ray.range_m) - free_margin, p_.raycast_max_range);

    if (t1 > t0) {
      double t_enter = t0;
      double t_exit = t1;
      bool hit_box = true;
      for (int a = 0; a < 3; ++a) {
        if (std::abs(d[a]) < 1e-9) {
          if (o[a] < origin_[a] || o[a] >= box_max[a]) {
            hit_box = false;
            break;
          }
        } else {
          const double inv_d = 1.0 / d[a];
          double ta = (origin_[a] - o[a]) * inv_d;
          double tb = (box_max[a] - o[a]) * inv_d;
          if (ta > tb) std::swap(ta, tb);
          t_enter = std::max(t_enter, ta);
          t_exit = std::min(t_exit, tb);
          if (t_enter >= t_exit) {
            hit_box = false;
            break;
          }
        }
      }

      if (hit_box && t_enter < t_exit) {
        const Eigen::Vector3d p_start = o + (t_enter + 1e-6) * d;
        int ix = std::clamp(static_cast<int>(std::floor((p_start.x() - origin_.x()) * inv_res)), 0, nx_ - 1);
        int iy = std::clamp(static_cast<int>(std::floor((p_start.y() - origin_.y()) * inv_res)), 0, ny_ - 1);
        int iz = std::clamp(static_cast<int>(std::floor((p_start.z() - origin_.z()) * inv_res)), 0, nz_ - 1);

        const int step_x = d.x() >= 0.0 ? 1 : -1;
        const int step_y = d.y() >= 0.0 ? 1 : -1;
        const int step_z = d.z() >= 0.0 ? 1 : -1;

        const double next_bx = origin_.x() + (d.x() >= 0.0 ? (ix + 1) : ix) * res;
        const double next_by = origin_.y() + (d.y() >= 0.0 ? (iy + 1) : iy) * res;
        const double next_bz = origin_.z() + (d.z() >= 0.0 ? (iz + 1) : iz) * res;

        double t_max_x = std::abs(d.x()) < 1e-9 ? 1e30 : (next_bx - o.x()) / d.x();
        double t_max_y = std::abs(d.y()) < 1e-9 ? 1e30 : (next_by - o.y()) / d.y();
        double t_max_z = std::abs(d.z()) < 1e-9 ? 1e30 : (next_bz - o.z()) / d.z();

        const double t_dx = std::abs(d.x()) < 1e-9 ? 1e30 : res / std::abs(d.x());
        const double t_dy = std::abs(d.y()) < 1e-9 ? 1e30 : res / std::abs(d.y());
        const double t_dz = std::abs(d.z()) < 1e-9 ? 1e30 : res / std::abs(d.z());

        double t_cur = t_enter;
        while (t_cur < t_exit && ix >= 0 && ix < nx_ && iy >= 0 && iy < ny_ && iz >= 0 && iz < nz_) {
          const int idx = flatIndex(ix, iy, iz);
          log_odds_[idx] = std::max(p_.l_min, log_odds_[idx] + p_.l_free);
          if (t_max_x < t_max_y) {
            if (t_max_x < t_max_z) {
              t_cur = t_max_x;
              t_max_x += t_dx;
              ix += step_x;
            } else {
              t_cur = t_max_z;
              t_max_z += t_dz;
              iz += step_z;
            }
          } else {
            if (t_max_y < t_max_z) {
              t_cur = t_max_y;
              t_max_y += t_dy;
              iy += step_y;
            } else {
              t_cur = t_max_z;
              t_max_z += t_dz;
              iz += step_z;
            }
          }
        }
      }
    }

    if (ray.surface_hit) {
      const Eigen::Vector3d p_hit = o + static_cast<double>(ray.range_m) * d;
      const int hx = static_cast<int>(std::floor((p_hit.x() - origin_.x()) * inv_res));
      const int hy = static_cast<int>(std::floor((p_hit.y() - origin_.y()) * inv_res));
      const int hz = static_cast<int>(std::floor((p_hit.z() - origin_.z()) * inv_res));
      if (hx >= 0 && hx < nx_ && hy >= 0 && hy < ny_ && hz >= 0 && hz < nz_) {
        const int hidx = flatIndex(hx, hy, hz);
        log_odds_[hidx] = std::min(p_.l_max, log_odds_[hidx] + p_.l_occ);
      }
    }
  }
}

std::vector<Eigen::Vector3f> PriorEsdfMap::priors() const {
  std::vector<Eigen::Vector3f> out;
  for (int iz = 0; iz < nz_; ++iz)
    for (int iy = 0; iy < ny_; ++iy)
      for (int ix = 0; ix < nx_; ++ix) {
        const int idx = flatIndex(ix, iy, iz);
        if (prior_occ_[idx] && log_odds_[idx] > 0.0f) out.push_back(voxelCenter(ix, iy, iz));
      }
  return out;
}

std::vector<Eigen::Vector3f> PriorEsdfMap::newUpdatedObstacles() const {
  std::vector<Eigen::Vector3f> out;
  for (int iz = 0; iz < nz_; ++iz)
    for (int iy = 0; iy < ny_; ++iy)
      for (int ix = 0; ix < nx_; ++ix) {
        const int idx = flatIndex(ix, iy, iz);
        if (!prior_occ_[idx] && log_odds_[idx] > p_.new_occ_thresh) out.push_back(voxelCenter(ix, iy, iz));
      }
  return out;
}

std::vector<Eigen::Vector3f> PriorEsdfMap::clearedEmptySpace() const {
  std::vector<Eigen::Vector3f> out;
  for (int iz = 0; iz < nz_; ++iz)
    for (int iy = 0; iy < ny_; ++iy)
      for (int ix = 0; ix < nx_; ++ix) {
        const int idx = flatIndex(ix, iy, iz);
        if (prior_occ_[idx] && log_odds_[idx] < p_.cleared_thresh) out.push_back(voxelCenter(ix, iy, iz));
      }
  return out;
}

std::vector<Eigen::Vector3f> PriorEsdfMap::activeObstacles() const {
  std::vector<Eigen::Vector3f> out;
  for (int iz = 0; iz < nz_; ++iz)
    for (int iy = 0; iy < ny_; ++iy)
      for (int ix = 0; ix < nx_; ++ix) {
        const int idx = flatIndex(ix, iy, iz);
        const bool occ = prior_occ_[idx] ? (log_odds_[idx] > 0.0f) : (log_odds_[idx] > p_.new_occ_thresh);
        if (occ) out.push_back(voxelCenter(ix, iy, iz));
      }
  return out;
}

std::shared_ptr<const DistanceGrid> PriorEsdfMap::buildDistanceGrid(const Eigen::Vector3d& center,
                                                                    const GridParams& grid_params) const {
  GridParams gp = grid_params;
  gp.min_hits = 1;   // activeObstacles() is already log-odds filtered (speckle rejected)
  return std::make_shared<const DistanceGrid>(activeObstacles(), center, gp);
}

}  // namespace isam2_depth
