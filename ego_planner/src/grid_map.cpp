#include "ego_planner/grid_map.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>

namespace ego_planner {

namespace {

constexpr float kInfSq = 1e12f;

// 1D squared Euclidean Distance Transform (Felzenszwalb & Huttenlocher, 2012)
void edt1d(const float* f, float* d, int n, std::vector<int>& v, std::vector<float>& z) {
  int k = 0;
  v[0] = 0;
  z[0] = -1e20f;
  z[1] = +1e20f;
  for (int q = 1; q < n; ++q) {
    float s = ((f[q] + static_cast<float>(q * q)) - (f[v[k]] + static_cast<float>(v[k] * v[k]))) /
              static_cast<float>(2 * q - 2 * v[k]);
    while (s <= z[k]) {
      --k;
      s = ((f[q] + static_cast<float>(q * q)) - (f[v[k]] + static_cast<float>(v[k] * v[k]))) /
          static_cast<float>(2 * q - 2 * v[k]);
    }
    ++k;
    v[k] = q;
    z[k] = s;
    z[k + 1] = +1e20f;
  }
  k = 0;
  for (int q = 0; q < n; ++q) {
    while (z[k + 1] < static_cast<float>(q)) {
      ++k;
    }
    int dx = q - v[k];
    d[q] = static_cast<float>(dx * dx) + f[v[k]];
  }
}

}  // namespace

GridMap::GridMap(const GridMapConfig& cfg) : cfg_(cfg) {
  nx_ = static_cast<int>(std::llround((cfg_.hi_ned.x - cfg_.lo_ned.x) / cfg_.resolution_m)) + 1;
  ny_ = static_cast<int>(std::llround((cfg_.hi_ned.y - cfg_.lo_ned.y) / cfg_.resolution_m)) + 1;
  nz_ = static_cast<int>(std::llround((cfg_.hi_ned.z - cfg_.lo_ned.z) / cfg_.resolution_m)) + 1;
  const std::size_t total = static_cast<std::size_t>(nx_) * ny_ * nz_;
  prior_dist_.assign(total, 1.5f);
  prior_log_odds_.assign(total, cfg_.l_prior_free);
  log_odds_.assign(total, cfg_.l_prior_free);
  esdf_.assign(total, 1.5f);
}

bool GridMap::loadPriorBinary(const std::string& bin_path) {
  std::ifstream in(bin_path, std::ios::binary);
  if (!in.is_open()) {
    return false;
  }
  char magic[4]{};
  uint32_t version = 0;
  int32_t nx = 0, ny = 0, nz = 0;
  float res = 0.0f, lox = 0.0f, loy = 0.0f, loz = 0.0f;
  in.read(magic, 4);
  in.read(reinterpret_cast<char*>(&version), sizeof(uint32_t));
  in.read(reinterpret_cast<char*>(&nx), sizeof(int32_t));
  in.read(reinterpret_cast<char*>(&ny), sizeof(int32_t));
  in.read(reinterpret_cast<char*>(&nz), sizeof(int32_t));
  in.read(reinterpret_cast<char*>(&res), sizeof(float));
  in.read(reinterpret_cast<char*>(&lox), sizeof(float));
  in.read(reinterpret_cast<char*>(&loy), sizeof(float));
  in.read(reinterpret_cast<char*>(&loz), sizeof(float));
  if (!in || std::memcmp(magic, "EGOP", 4) != 0 || version != 1 || nx <= 0 || ny <= 0 || nz <= 0) {
    return false;
  }

  nx_ = nx;
  ny_ = ny;
  nz_ = nz;
  cfg_.resolution_m = static_cast<double>(res);
  cfg_.lo_ned = {static_cast<double>(lox), static_cast<double>(loy), static_cast<double>(loz)};
  cfg_.hi_ned = {
      cfg_.lo_ned.x + (nx_ - 1) * cfg_.resolution_m,
      cfg_.lo_ned.y + (ny_ - 1) * cfg_.resolution_m,
      cfg_.lo_ned.z + (nz_ - 1) * cfg_.resolution_m,
  };

  const std::size_t total = static_cast<std::size_t>(nx_) * ny_ * nz_;
  std::vector<uint8_t> raw(total);
  in.read(reinterpret_cast<char*>(raw.data()), static_cast<std::streamsize>(total));
  if (!in) {
    return false;
  }

  prior_dist_.resize(total);
  prior_log_odds_.resize(total);
  for (std::size_t i = 0; i < total; ++i) {
    float d = static_cast<float>(raw[i]) / 200.0f;
    prior_dist_[i] = d;
    prior_log_odds_[i] = (d <= static_cast<float>(cfg_.prior_occ_threshold_m))
                             ? cfg_.l_prior_occ
                             : cfg_.l_prior_free;
  }
  prior_loaded_ = true;
  resetToPrior();
  return true;
}

void GridMap::resetToPrior() {
  log_odds_ = prior_log_odds_;
  esdf_ = prior_dist_;
  live_updates_ = 0;
}

void GridMap::recomputeESDF() {
  const std::size_t total = static_cast<std::size_t>(nx_) * ny_ * nz_;
  std::vector<float> sq_dist(total, kInfSq);
  bool any_occ = false;
  for (std::size_t i = 0; i < total; ++i) {
    if (log_odds_[i] >= 0.0f) {
      sq_dist[i] = 0.0f;
      any_occ = true;
    }
  }
  if (!any_occ) {
    std::fill(esdf_.begin(), esdf_.end(), 2.0f);
    return;
  }

  const int max_dim = std::max({nx_, ny_, nz_});
  std::vector<int> v(max_dim + 1);
  std::vector<float> z(max_dim + 2);
  std::vector<float> f(max_dim);
  std::vector<float> d(max_dim);

  // Pass 1: along Z
  for (int ix = 0; ix < nx_; ++ix) {
    for (int iy = 0; iy < ny_; ++iy) {
      int base = (ix * ny_ + iy) * nz_;
      edt1d(&sq_dist[base], d.data(), nz_, v, z);
      for (int iz = 0; iz < nz_; ++iz) {
        sq_dist[base + iz] = d[iz];
      }
    }
  }

  // Pass 2: along Y
  for (int ix = 0; ix < nx_; ++ix) {
    for (int iz = 0; iz < nz_; ++iz) {
      for (int iy = 0; iy < ny_; ++iy) {
        f[iy] = sq_dist[(ix * ny_ + iy) * nz_ + iz];
      }
      edt1d(f.data(), d.data(), ny_, v, z);
      for (int iy = 0; iy < ny_; ++iy) {
        sq_dist[(ix * ny_ + iy) * nz_ + iz] = d[iy];
      }
    }
  }

  // Pass 3: along X + convert to metres
  const float res = static_cast<float>(cfg_.resolution_m);
  for (int iy = 0; iy < ny_; ++iy) {
    for (int iz = 0; iz < nz_; ++iz) {
      for (int ix = 0; ix < nx_; ++ix) {
        f[ix] = sq_dist[(ix * ny_ + iy) * nz_ + iz];
      }
      edt1d(f.data(), d.data(), nx_, v, z);
      for (int ix = 0; ix < nx_; ++ix) {
        int idx = (ix * ny_ + iy) * nz_ + iz;
        float edt_m = std::sqrt(std::max(0.0f, d[ix])) * res;
        // Blend with exact sub-voxel prior distance when occupancy matches prior
        if (prior_loaded_ && (log_odds_[idx] >= 0.0f) == (prior_log_odds_[idx] >= 0.0f) &&
            std::fabs(edt_m - prior_dist_[idx]) < res) {
          esdf_[idx] = prior_dist_[idx];
        } else {
          esdf_[idx] = edt_m;
        }
      }
    }
  }
}

void GridMap::updateFromDepth(const float* depth_norm, int width, int height,
                              double hfov_deg, double vfov_deg, double far_clip,
                              const Vec3& pos_ned, const Quat& q_wxyz, int stride) {
  if (!depth_norm || width <= 0 || height <= 0) return;

  const double fx = 0.5 * width / std::tan(hfov_deg * (kPi / 180.0) * 0.5);
  const double fy = 0.5 * height / std::tan(vfov_deg * (kPi / 180.0) * 0.5);
  const Mat3 R_wb = q_wxyz.toRotationMatrix();

  bool changed = false;
  const int step = std::max(1, stride);
  const double max_free_ray = std::min(far_clip, 3.2);
  const double ray_step = cfg_.resolution_m * 0.85;

  for (int v = step / 2; v < height; v += step) {
    const double cv = (static_cast<double>(v) + 0.5) - 0.5 * height;
    const double bz = cv / fy;
    for (int u = step / 2; u < width; u += step) {
      const float dn = depth_norm[v * width + u];
      if (!std::isfinite(dn) || dn <= 0.015f) {
        // Invalid stereo return or near-clip: never clear or mark
        continue;
      }
      const double cu = (static_cast<double>(u) + 0.5) - 0.5 * width;
      const double by = cu / fx;
      const Vec3 dir_b = Vec3{1.0, by, bz}.normalized();
      const Vec3 dir_w = R_wb * dir_b;

      // Convert camera-plane z-depth (dn * far_clip) to range t along ray
      const double z_cam = static_cast<double>(dn) * far_clip;
      const double body_len = std::sqrt(1.0 + by * by + bz * bz);
      const double t_hit = z_cam * body_len;

      // 1. Free-space clearing along ray up to t_hit - margin
      const double t_clear_end = std::min(max_free_ray, t_hit - 0.15);
      for (double t = 0.22; t < t_clear_end; t += ray_step) {
        Vec3 p = pos_ned + dir_w * t;
        // Only clear within flight altitude band [0.18m, wall_top - 0.08m]
        double alt = -p.z;
        if (alt < 0.18 || alt > cfg_.wall_top_m - 0.08) continue;
        int ix = static_cast<int>(std::llround((p.x - cfg_.lo_ned.x) / cfg_.resolution_m));
        int iy = static_cast<int>(std::llround((p.y - cfg_.lo_ned.y) / cfg_.resolution_m));
        int iz = static_cast<int>(std::llround((p.z - cfg_.lo_ned.z) / cfg_.resolution_m));
        if (!inBounds(ix, iy, iz)) continue;
        int idx = flatIndex(ix, iy, iz);
        bool was_occ = log_odds_[idx] >= 0.0f;
        log_odds_[idx] = std::max(cfg_.l_min, log_odds_[idx] - cfg_.l_free);
        if (was_occ && log_odds_[idx] < 0.0f) {
          changed = true;
        }
      }

      // 2. Occupied hit update if within sensor range and not floor/net/ceiling
      if (dn < 0.98f && t_hit >= 0.20 && t_hit <= 3.6) {
        Vec3 p_hit = pos_ned + dir_w * t_hit;
        double hit_alt = -p_hit.z;
        // Ignore floor (< 0.14 m), high ceiling (> wall_top), and outer net boundary
        if (hit_alt >= 0.14 && hit_alt <= cfg_.wall_top_m &&
            std::fabs(p_hit.x) < cfg_.arena_half_m - 0.10 &&
            std::fabs(p_hit.y) < cfg_.arena_half_m - 0.10) {
          int ix = static_cast<int>(std::llround((p_hit.x - cfg_.lo_ned.x) / cfg_.resolution_m));
          int iy = static_cast<int>(std::llround((p_hit.y - cfg_.lo_ned.y) / cfg_.resolution_m));
          int iz = static_cast<int>(std::llround((p_hit.z - cfg_.lo_ned.z) / cfg_.resolution_m));
          if (inBounds(ix, iy, iz)) {
            int idx = flatIndex(ix, iy, iz);
            // Only mark as newly occupied if not already within 7.5 cm of a known prior wall
            // (prevents 5cm voxel quantization from inflating nominal walls, while capturing perturbed walls)
            if (!prior_loaded_ || prior_dist_[idx] > 0.075f) {
              bool was_occ = log_odds_[idx] >= 0.0f;
              log_odds_[idx] = std::min(cfg_.l_max, log_odds_[idx] + cfg_.l_occ);
              if (!was_occ && log_odds_[idx] >= 0.0f) {
                changed = true;
              }
            }
          }
        }
      }
    }
  }

  if (changed) {
    ++live_updates_;
    recomputeESDF();
  }
}

double GridMap::getDistance(const Vec3& p_ned) const {
  Vec3 g;
  return getDistanceAndGradient(p_ned, g);
}

double GridMap::getDistanceAndGradient(const Vec3& p_ned, Vec3& grad_out) const {
  const double gx = (p_ned.x - cfg_.lo_ned.x) / cfg_.resolution_m;
  const double gy = (p_ned.y - cfg_.lo_ned.y) / cfg_.resolution_m;
  const double gz = (p_ned.z - cfg_.lo_ned.z) / cfg_.resolution_m;

  int i0 = std::clamp(static_cast<int>(std::floor(gx)), 0, nx_ - 2);
  int j0 = std::clamp(static_cast<int>(std::floor(gy)), 0, ny_ - 2);
  int k0 = std::clamp(static_cast<int>(std::floor(gz)), 0, nz_ - 2);

  const double tx = clamp(gx - i0, 0.0, 1.0);
  const double ty = clamp(gy - j0, 0.0, 1.0);
  const double tz = clamp(gz - k0, 0.0, 1.0);

  double c[2][2][2];
  for (int dx = 0; dx < 2; ++dx) {
    for (int dy = 0; dy < 2; ++dy) {
      for (int dz = 0; dz < 2; ++dz) {
        c[dx][dy][dz] = static_cast<double>(esdf_[flatIndex(i0 + dx, j0 + dy, k0 + dz)]);
      }
    }
  }

  // Trilinear interpolation
  const double c00 = c[0][0][0] * (1.0 - tx) + c[1][0][0] * tx;
  const double c01 = c[0][0][1] * (1.0 - tx) + c[1][0][1] * tx;
  const double c10 = c[0][1][0] * (1.0 - tx) + c[1][1][0] * tx;
  const double c11 = c[0][1][1] * (1.0 - tx) + c[1][1][1] * tx;

  const double c0 = c00 * (1.0 - ty) + c10 * ty;
  const double c1 = c01 * (1.0 - ty) + c11 * ty;
  double dist = c0 * (1.0 - tz) + c1 * tz;

  // Analytical derivatives wrt (x, y, z)
  const double inv_res = 1.0 / cfg_.resolution_m;
  const double dcdx0 = (c[1][0][0] - c[0][0][0]) * (1.0 - ty) + (c[1][1][0] - c[0][1][0]) * ty;
  const double dcdx1 = (c[1][0][1] - c[0][0][1]) * (1.0 - ty) + (c[1][1][1] - c[0][1][1]) * ty;
  grad_out.x = (dcdx0 * (1.0 - tz) + dcdx1 * tz) * inv_res;
  grad_out.y = ((c10 - c00) * (1.0 - tz) + (c11 - c01) * tz) * inv_res;
  grad_out.z = (c1 - c0) * inv_res;

  // Also include arena net boundary distance when within 0.21 m of the net
  const double net_x = cfg_.arena_half_m - std::fabs(p_ned.x);
  const double net_y = cfg_.arena_half_m - std::fabs(p_ned.y);
  if (net_x < 0.21 && net_x < dist) {
    dist = net_x;
    grad_out = {p_ned.x >= 0.0 ? -1.0 : 1.0, 0.0, 0.0};
  }
  if (net_y < 0.21 && net_y < dist) {
    dist = net_y;
    grad_out = {0.0, p_ned.y >= 0.0 ? -1.0 : 1.0, 0.0};
  }
  const double nrm = grad_out.norm();
  if (nrm > 1e-6) {
    grad_out = grad_out / nrm;
  }
  return dist;
}

bool GridMap::isSegmentClear(const Vec3& a_ned, const Vec3& b_ned, double min_clearance_m, double step_m) const {
  const double len = (b_ned - a_ned).norm();
  const int n = std::max(2, static_cast<int>(std::ceil(len / step_m)) + 1);
  for (int i = 0; i < n; ++i) {
    double s = static_cast<double>(i) / static_cast<double>(n - 1);
    Vec3 p = a_ned + (b_ned - a_ned) * s;
    if (-p.z < 0.18 || -p.z > cfg_.wall_top_m - 0.10) return false;
    if (getDistance(p) < min_clearance_m) return false;
  }
  return true;
}

Vec3 GridMap::refineGateCenter(const GateSpec& gate, double margin_from_posts_m) const {
  const double u_lo = std::min(margin_from_posts_m, 0.45 * gate.length_m);
  const double u_hi = std::max(u_lo, gate.length_m - margin_from_posts_m);
  const double u_nom = (gate.center_ned.xy() - gate.A_ned).dot(gate.tangent_ned);

  Vec3 best_pt = gate.center_ned;
  double best_score = -1e9;

  const int nu = 17;
  const int nz = gate.is_window ? 9 : 5;
  for (int iu = 0; iu < nu; ++iu) {
    double u = (nu == 1) ? u_nom : (u_lo + (u_hi - u_lo) * static_cast<double>(iu) / (nu - 1));
    Vec2 xy = gate.A_ned + gate.tangent_ned * u;
    for (int iz = 0; iz < nz; ++iz) {
      double alt = (nz == 1) ? gate.alt_nominal_m
                             : (gate.alt_min_m + (gate.alt_max_m - gate.alt_min_m) * static_cast<double>(iz) / (nz - 1));
      Vec3 cand{xy.x, xy.y, -alt};
      // Evaluate minimum clearance across pre-gate, center, and post-gate points along normal
      Vec3 pre{xy.x - 0.18 * gate.normal_ned.x, xy.y - 0.18 * gate.normal_ned.y, -alt};
      Vec3 post{xy.x + 0.18 * gate.normal_ned.x, xy.y + 0.18 * gate.normal_ned.y, -alt};
      double c_mid = getDistance(cand);
      double c_pre = getDistance(pre);
      double c_post = getDistance(post);
      double min_c = std::min({c_mid, c_pre, c_post});
      // Mild regularization toward nominal center and altitude
      double reg = 0.08 * std::fabs(u - u_nom) + 0.10 * std::fabs(alt - gate.alt_nominal_m);
      double score = min_c - reg;
      if (score > best_score) {
        best_score = score;
        best_pt = cand;
      }
    }
  }
  return best_pt;
}

}  // namespace ego_planner
