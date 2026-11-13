#include "isam2_depth/cad_prior.hpp"

#include <cmath>
#include <cstring>
#include <fstream>

#include <Eigen/Cholesky>

namespace isam2_depth {

std::shared_ptr<const CadPriorEsdf> CadPriorEsdf::load(const std::string& path, std::string* err) {
  auto fail = [&](const std::string& msg) -> std::shared_ptr<const CadPriorEsdf> {
    if (err) *err = msg;
    return nullptr;
  };
  std::ifstream in(path, std::ios::binary);
  if (!in) return fail("cannot open " + path);

  char hdr[36];
  if (!in.read(hdr, sizeof hdr)) return fail("truncated header in " + path);
  if (std::memcmp(hdr, "CADP", 4) != 0) return fail("bad magic in " + path + " (expected CADP)");

  uint32_t ver = 0;
  int32_t nx = 0, ny = 0, nz = 0;
  float res = 0.f, lx = 0.f, ly = 0.f, lz = 0.f;
  std::memcpy(&ver, hdr + 4, 4);
  std::memcpy(&nx, hdr + 8, 4);
  std::memcpy(&ny, hdr + 12, 4);
  std::memcpy(&nz, hdr + 16, 4);
  std::memcpy(&res, hdr + 20, 4);
  std::memcpy(&lx, hdr + 24, 4);
  std::memcpy(&ly, hdr + 28, 4);
  std::memcpy(&lz, hdr + 32, 4);
  if (ver != 1 || nx <= 0 || ny <= 0 || nz <= 0 || !(res > 0.f)) {
    return fail("invalid CADP header parameters in " + path);
  }

  auto out = std::make_shared<CadPriorEsdf>();
  out->nx_ = nx;
  out->ny_ = ny;
  out->nz_ = nz;
  out->res_ = static_cast<double>(res);
  out->lo_ = Eigen::Vector3d(lx, ly, lz);
  const size_t total = static_cast<size_t>(nx) * static_cast<size_t>(ny) * static_cast<size_t>(nz);
  out->cells_.resize(total);
  if (!in.read(reinterpret_cast<char*>(out->cells_.data()), static_cast<std::streamsize>(total * sizeof(Cell)))) {
    return fail("truncated cell payload in " + path);
  }
  return out;
}

bool CadPriorEsdf::query(const Eigen::Vector3d& p_ned, Eigen::Vector3d& q_closest,
                         Eigen::Vector3d& unit_normal) const {
  const int ix = static_cast<int>(std::lround((p_ned.x() - lo_.x()) / res_));
  const int iy = static_cast<int>(std::lround((p_ned.y() - lo_.y()) / res_));
  const int iz = static_cast<int>(std::lround((p_ned.z() - lo_.z()) / res_));
  if (ix < 0 || ix >= nx_ || iy < 0 || iy >= ny_ || iz < 0 || iz >= nz_) return false;

  const size_t idx = (static_cast<size_t>(ix) * static_cast<size_t>(ny_) + static_cast<size_t>(iy)) *
                         static_cast<size_t>(nz_) +
                     static_cast<size_t>(iz);
  const Cell& c = cells_[idx];
  const Eigen::Vector3d centre(lo_.x() + ix * res_, lo_.y() + iy * res_, lo_.z() + iz * res_);
  q_closest = centre + 1e-4 * Eigen::Vector3d(c.dx, c.dy, c.dz);
  Eigen::Vector3d n(c.nx, c.ny, c.nz);
  const double nn = n.norm();
  if (nn < 1e-6) return false;
  unit_normal = n / nn;
  return true;
}

namespace {
struct BodySample {
  Eigen::Vector3d p_body;
  Eigen::Vector3d n_body;
  double z_cam;
};

struct CadLinearised {
  Eigen::Matrix<double, 6, 6> H = Eigen::Matrix<double, 6, 6>::Zero();
  Eigen::Matrix<double, 6, 1> g = Eigen::Matrix<double, 6, 1>::Zero();
  int tried = 0, inliers = 0;
  double sse = 0.0;
};

CadLinearised lineariseCad(const std::vector<BodySample>& samples, const gtsam::Pose3& T_wb,
                           const CadPriorEsdf& prior, const CadPriorParams& p) {
  CadLinearised L;
  const Eigen::Matrix3d R = T_wb.rotation().matrix();
  const Eigen::Vector3d t = T_wb.translation();
  const double inv_k2 = 1.0 / (p.cauchy_k * p.cauchy_k);

  for (const auto& s : samples) {
    ++L.tried;
    const Eigen::Vector3d pw = R * s.p_body + t;
    Eigen::Vector3d qc, nc;
    if (!prior.query(pw, qc, nc)) continue;
    if ((pw - qc).norm() > p.max_corr_dist) continue;

    const Eigen::Vector3d nw = R * s.n_body;
    const double ndot = nc.dot(nw);
    if (std::abs(ndot) < p.min_normal_cos) continue;
    const Eigen::Vector3d nu = (ndot >= 0.0) ? nc : -nc;

    const double r = nu.dot(pw - qc);
    if (std::abs(r) > p.max_plane_dist) continue;

    const double sigma = p.sigma0 + p.sigma_z2 * s.z_cam * s.z_cam;
    const double cauchy_w = 1.0 / (1.0 + r * r * inv_k2);
    const double w = cauchy_w / (sigma * sigma);

    const Eigen::Vector3d a = R.transpose() * nu;
    Eigen::Matrix<double, 6, 1> J;
    J << s.p_body.cross(a), a;
    L.H.noalias() += w * J * J.transpose();
    L.g.noalias() += w * J * r;
    L.sse += r * r;
    ++L.inliers;
  }
  return L;
}
}  // namespace

std::optional<CadAlignResult> alignToCadPrior(const OrganisedCloud& cloud,
                                              const gtsam::Pose3& T_body_cam,
                                              const gtsam::Pose3& T_world_body_init,
                                              const CadPriorEsdf& prior,
                                              const CadPriorParams& p) {
  if (cloud.n_valid < p.min_inliers) return std::nullopt;

  const Eigen::Matrix3d R_bc = T_body_cam.rotation().matrix();
  const Eigen::Vector3d t_bc = T_body_cam.translation();
  std::vector<BodySample> samples;
  samples.reserve(static_cast<size_t>(cloud.n_valid / std::max(1, p.point_step) + 1));
  int seen = 0;
  for (size_t i = 0; i < cloud.points.size(); ++i) {
    if (!cloud.valid(static_cast<int>(i)) || (seen++ % p.point_step) != 0) continue;
    const Eigen::Vector3d pc = cloud.points[i].cast<double>();
    const Eigen::Vector3d nc = cloud.normals[i].cast<double>();
    samples.push_back({R_bc * pc + t_bc, R_bc * nc, pc.z()});
  }
  if (static_cast<int>(samples.size()) < p.min_inliers) return std::nullopt;

  Eigen::Matrix<double, 6, 1> prior_info;
  prior_info << Eigen::Vector3d::Constant(1.0 / (p.prior_sigma_rot * p.prior_sigma_rot)),
      Eigen::Vector3d::Constant(1.0 / (p.prior_sigma_trans * p.prior_sigma_trans));

  gtsam::Pose3 T = T_world_body_init;
  int it = 0, inliers_at_init = 0;
  for (; it < p.max_iterations; ++it) {
    const CadLinearised L = lineariseCad(samples, T, prior, p);
    if (L.inliers < p.min_inliers) return std::nullopt;
    if (it == 0) inliers_at_init = L.inliers;
    const Eigen::Matrix<double, 6, 1> dev = gtsam::Pose3::Logmap(T_world_body_init.between(T));
    const Eigen::Matrix<double, 6, 6> H = L.H + Eigen::Matrix<double, 6, 6>(prior_info.asDiagonal());
    const Eigen::Matrix<double, 6, 1> g = L.g + prior_info.cwiseProduct(dev);
    const Eigen::Matrix<double, 6, 1> dx = -H.ldlt().solve(g);
    if (!dx.allFinite()) return std::nullopt;
    T = T * gtsam::Pose3::Expmap(dx);
    if (dx.head<3>().norm() < p.converged_rot && dx.tail<3>().norm() < p.converged_trans) {
      ++it;
      break;
    }
  }

  const CadLinearised L = lineariseCad(samples, T, prior, p);
  CadAlignResult res;
  res.T_world_body = T;
  res.inliers = L.inliers;
  res.inlier_ratio = L.tried ? static_cast<double>(L.inliers) / L.tried : 0.0;
  res.rmse = L.inliers ? std::sqrt(L.sse / L.inliers) : 0.0;
  res.iterations = it;
  res.info = L.H / p.info_deflation;

  if (L.inliers < p.min_inliers || res.inlier_ratio < p.min_inlier_ratio || res.rmse > p.max_rmse ||
      !res.info.allFinite() || L.inliers < p.min_inlier_keep * inliers_at_init) {
    return std::nullopt;
  }
  return res;
}

}  // namespace isam2_depth
