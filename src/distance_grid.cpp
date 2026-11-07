#include "isam2_depth/distance_grid.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

namespace isam2_depth {
namespace {
constexpr double kFar = 1e20;

// Exact 1D squared Euclidean distance transform (Felzenszwalb & Huttenlocher 2012), in place over a strided line.
void edt1d(double* f, int n, int stride, std::vector<double>& d, std::vector<int>& v, std::vector<double>& z) {
  d.resize(n); v.resize(n); z.resize(n + 1);
  int k = 0;
  v[0] = 0;
  z[0] = -std::numeric_limits<double>::infinity();
  z[1] = std::numeric_limits<double>::infinity();
  auto F = [&](int q) { return f[static_cast<size_t>(q) * stride]; };
  for (int q = 1; q < n; ++q) {
    double s = ((F(q) + double(q) * q) - (F(v[k]) + double(v[k]) * v[k])) / (2.0 * q - 2.0 * v[k]);
    while (s <= z[k]) {
      --k;
      s = ((F(q) + double(q) * q) - (F(v[k]) + double(v[k]) * v[k])) / (2.0 * q - 2.0 * v[k]);
    }
    ++k;
    v[k] = q;
    z[k] = s;
    z[k + 1] = std::numeric_limits<double>::infinity();
  }
  k = 0;
  for (int q = 0; q < n; ++q) {
    while (z[k + 1] < q) ++k;
    d[q] = double(q - v[k]) * (q - v[k]) + F(v[k]);
  }
  for (int q = 0; q < n; ++q) f[static_cast<size_t>(q) * stride] = d[q];
}
}  // namespace

DistanceGrid::DistanceGrid(const std::vector<Eigen::Vector3f>& points, const Eigen::Vector3d& center,
                           const GridParams& p)
    : res_(p.resolution), max_dist_(p.max_distance) {
  nx_ = 2 * static_cast<int>(std::lround(p.half_extent.x() / res_));
  ny_ = 2 * static_cast<int>(std::lround(p.half_extent.y() / res_));
  nz_ = 2 * static_cast<int>(std::lround(p.half_extent.z() / res_));
  const Eigen::Vector3d n(nx_, ny_, nz_);
  origin_ = ((center / res_).array().floor() - (n / 2).array()).matrix() * res_;
  const size_t N = static_cast<size_t>(nx_) * ny_ * nz_;
  std::vector<uint16_t> hits(N, 0);
  bool ok;
  for (const auto& q : points) {
    const int i = index(q.cast<double>(), &ok);
    if (ok && hits[i] < 65535) ++hits[i];
  }
  std::vector<double> f(N);
  for (size_t i = 0; i < N; ++i) f[i] = hits[i] >= p.min_hits ? 0.0 : kFar;
  std::vector<double> d, z;
  std::vector<int> v;
  for (int k = 0; k < nz_; ++k)      // along x
    for (int j = 0; j < ny_; ++j) edt1d(&f[(static_cast<size_t>(k) * ny_ + j) * nx_], nx_, 1, d, v, z);
  for (int k = 0; k < nz_; ++k)      // along y
    for (int i = 0; i < nx_; ++i) edt1d(&f[static_cast<size_t>(k) * ny_ * nx_ + i], ny_, nx_, d, v, z);
  for (int j = 0; j < ny_; ++j)      // along z
    for (int i = 0; i < nx_; ++i) edt1d(&f[static_cast<size_t>(j) * nx_ + i], nz_, nx_ * ny_, d, v, z);
  dist_.resize(N);
  for (size_t i = 0; i < N; ++i)
    dist_[i] = static_cast<float>(std::min<double>(std::sqrt(f[i]) * res_, max_dist_));
}

int DistanceGrid::index(const Eigen::Vector3d& p, bool* ok) const {
  const Eigen::Vector3d c = (p - origin_) / res_;
  const int i = static_cast<int>(std::floor(c.x())), j = static_cast<int>(std::floor(c.y())),
            k = static_cast<int>(std::floor(c.z()));
  *ok = i >= 0 && j >= 0 && k >= 0 && i < nx_ && j < ny_ && k < nz_;
  return *ok ? (k * ny_ + j) * nx_ + i : -1;
}

bool DistanceGrid::inside(const Eigen::Vector3d& p) const {
  bool ok;
  index(p, &ok);
  return ok;
}

float DistanceGrid::distance(const Eigen::Vector3d& p, float outside) const {
  bool ok;
  const int i = index(p, &ok);
  if (!ok) return outside < 0 ? max_dist_ : outside;
  return dist_[i];
}

std::vector<Eigen::Vector3f> DistanceGrid::occupiedCentres() const {
  std::vector<Eigen::Vector3f> out;
  for (int k = 0; k < nz_; ++k)
    for (int j = 0; j < ny_; ++j)
      for (int i = 0; i < nx_; ++i)
        if (dist_[(static_cast<size_t>(k) * ny_ + j) * nx_ + i] <= 0.0f)
          out.emplace_back((origin_ + res_ * Eigen::Vector3d(i + 0.5, j + 0.5, k + 0.5)).cast<float>());
  return out;
}

}  // namespace isam2_depth
