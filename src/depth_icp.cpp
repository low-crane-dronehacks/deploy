#include "isam2_depth/depth_icp.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>

#include <Eigen/Cholesky>
#include <Eigen/Eigenvalues>

namespace isam2_depth {
namespace {
const Eigen::Vector3f kNaN3 = Eigen::Vector3f::Constant(std::numeric_limits<float>::quiet_NaN());
}

OrganisedCloud makeCloud(const DepthFrame& d, const PinholeCamera& full, const CloudParams& p) {
  if (d.width != full.width || d.height != full.height ||
      d.depth_m.size() != static_cast<size_t>(d.width) * d.height)
    throw std::invalid_argument("makeCloud: depth image size does not match the camera");
  OrganisedCloud c;
  c.cam = full.decimated(p.stride);
  const int W = c.cam.width, H = c.cam.height;
  c.points.assign(W * H, kNaN3);
  c.normals.assign(W * H, kNaN3);
  // Block mean over the stride x stride source pixels of each decimated pixel (PinholeCamera::decimated).
  // Stereo noise ~ z^2 makes single pixels useless for normals; a block straddling a depth edge is dropped.
  for (int v = 0; v < H; ++v)
    for (int u = 0; u < W; ++u) {
      float sum = 0, lo = 1e9f, hi = -1e9f;
      int n = 0, total = 0;
      for (int dv = 0; dv < p.stride; ++dv)
        for (int du = 0; du < p.stride; ++du) {
          const int su = u * p.stride + du, sv = v * p.stride + dv;
          if (su >= d.width || sv >= d.height) continue;
          ++total;
          const float z = d.depth_m[sv * d.width + su];
          if (!std::isfinite(z) || z < p.min_range || z > p.max_range) continue;  // +inf (no return): no point
          sum += z; lo = std::min(lo, z); hi = std::max(hi, z);
          ++n;
        }
      if (n == 0 || n * 4 < total) continue;   // mostly invalid; also skips the unrendered pixels of a sparse image
      const float z = sum / n;
      if (hi - lo > static_cast<float>(p.max_normal_jump) + 0.03f * z) continue;
      c.points[v * W + u] = Eigen::Vector3f((u - c.cam.cx) / c.cam.fx * z, (v - c.cam.cy) / c.cam.fy * z, z);
    }
  // normals: least-squares plane through the (2r+1)^2 window, neighbours across a depth edge left out (a post
  // against a far wall); stereo noise at a few metres makes 2-point differences tens of degrees off
  const int r = p.normal_radius;
  const int min_n = (2 * r + 1) * (2 * r + 1) / 2;
  for (int v = r; v < H - r; ++v)
    for (int u = r; u < W - r; ++u) {
      const int i = v * W + u;
      const Eigen::Vector3f& P = c.points[i];
      if (!std::isfinite(P.z())) continue;
      const float jump = static_cast<float>(r * p.max_normal_jump) + 0.03f * r * P.z();
      Eigen::Vector3d sum = Eigen::Vector3d::Zero();
      Eigen::Matrix3d sq = Eigen::Matrix3d::Zero();
      int n = 0;
      for (int dv = -r; dv <= r; ++dv)
        for (int du = -r; du <= r; ++du) {
          const Eigen::Vector3f& Q = c.points[i + dv * W + du];
          if (!std::isfinite(Q.z()) || std::abs(Q.z() - P.z()) > jump) continue;
          const Eigen::Vector3d q = Q.cast<double>();
          sum += q;
          sq.noalias() += q * q.transpose();
          ++n;
        }
      if (n < min_n) continue;
      const Eigen::Vector3d mean = sum / n;
      const Eigen::Matrix3d cov = sq / n - mean * mean.transpose();
      Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es(cov);
      Eigen::Vector3f nrm = es.eigenvectors().col(0).cast<float>();   // smallest eigenvalue
      if (!nrm.allFinite()) continue;
      if (nrm.dot(P) > 0) nrm = -nrm;  // face the camera
      c.normals[i] = nrm;
      ++c.n_valid;
    }
  return c;
}

namespace {
struct Linearised {
  Eigen::Matrix<double, 6, 6> H = Eigen::Matrix<double, 6, 6>::Zero();
  Eigen::Matrix<double, 6, 1> g = Eigen::Matrix<double, 6, 1>::Zero();
  int tried = 0, inliers = 0;
  double sse = 0;
};

// Point-to-plane residuals r = n_t . (T p_s - q_t), with q_t the target point the source point projects onto.
// Right perturbation T Exp([w, v]): d(T p_s) = R(-[p_s]x w + v), so J = [p_s x a, a] with a = R^T n_t.
Linearised linearise(const OrganisedCloud& tgt, const OrganisedCloud& src, const gtsam::Pose3& T,
                     const IcpParams& p) {
  Linearised L;
  const Eigen::Matrix3d R = T.rotation().matrix();
  const Eigen::Vector3d t = T.translation();
  const PinholeCamera& k = tgt.cam;
  int seen = 0;
  for (size_t i = 0; i < src.points.size(); ++i) {
    if (!src.valid(static_cast<int>(i)) || (seen++ % p.point_step) != 0) continue;
    const Eigen::Vector3d ps = src.points[i].cast<double>();
    const Eigen::Vector3d pt = R * ps + t;
    ++L.tried;
    if (pt.z() < 0.05) continue;
    const int u = static_cast<int>(std::lround(k.fx * pt.x() / pt.z() + k.cx));
    const int v = static_cast<int>(std::lround(k.fy * pt.y() / pt.z() + k.cy));
    if (u < 0 || v < 0 || u >= k.width || v >= k.height) continue;
    const int j = v * k.width + u;
    if (!tgt.valid(j)) continue;
    const Eigen::Vector3d q = tgt.points[j].cast<double>(), nq = tgt.normals[j].cast<double>();
    if ((pt - q).norm() > p.max_corr_dist) continue;
    if (nq.dot(R * src.normals[i].cast<double>()) < p.min_normal_cos) continue;
    const double r = nq.dot(pt - q);
    if (std::abs(r) > p.max_plane_dist) continue;
    const double sigma = p.sigma0 + p.sigma_z2 * q.z() * q.z();
    const double huber = std::abs(r) <= p.huber_k ? 1.0 : p.huber_k / std::abs(r);
    const double w = huber / (sigma * sigma);
    const Eigen::Vector3d a = R.transpose() * nq;
    Eigen::Matrix<double, 6, 1> J;
    J << ps.cross(a), a;
    L.H.noalias() += w * J * J.transpose();
    L.g.noalias() += w * J * r;
    L.sse += r * r;
    ++L.inliers;
  }
  return L;
}
}  // namespace

std::optional<IcpResult> alignProjective(const OrganisedCloud& target, const OrganisedCloud& source,
                                         const gtsam::Pose3& init, const IcpParams& p) {
  if (target.n_valid < p.min_inliers || source.n_valid < p.min_inliers) return std::nullopt;
  // Weak prior on the deviation from `init` (MAP rather than ML): where the geometry observes nothing (vertical
  // translation with only vertical walls / posts in view, sliding along a long wall) the solution stays at the
  // prediction instead of wandering. Elsewhere the depth outweighs it by orders of magnitude. It is NOT part of
  // the returned information, so the graph does not count the odometry twice.
  Eigen::Matrix<double, 6, 1> prior_info;
  prior_info << Eigen::Vector3d::Constant(1.0 / (p.prior_sigma_rot * p.prior_sigma_rot)),
      Eigen::Vector3d::Constant(1.0 / (p.prior_sigma_trans * p.prior_sigma_trans));
  gtsam::Pose3 T = init;
  IcpResult res;
  int it = 0, inliers_at_init = 0;
  for (; it < p.max_iterations; ++it) {
    const Linearised L = linearise(target, source, T, p);
    if (L.inliers < p.min_inliers) return std::nullopt;
    if (it == 0) inliers_at_init = L.inliers;
    const Eigen::Matrix<double, 6, 1> dev = gtsam::Pose3::Logmap(init.between(T));
    const Eigen::Matrix<double, 6, 6> H = L.H + Eigen::Matrix<double, 6, 6>(prior_info.asDiagonal());
    const Eigen::Matrix<double, 6, 1> g = L.g + prior_info.cwiseProduct(dev);
    const Eigen::Matrix<double, 6, 1> dx = -H.ldlt().solve(g);
    if (!dx.allFinite()) return std::nullopt;
    T = T * gtsam::Pose3::Expmap(dx);
    if (dx.head<3>().norm() < p.converged_rot && dx.tail<3>().norm() < p.converged_trans) { ++it; break; }
  }
  const Linearised L = linearise(target, source, T, p);   // information at the solution
  res.T_target_source = T;
  res.inliers = L.inliers;
  res.inlier_ratio = L.tried ? static_cast<double>(L.inliers) / L.tried : 0.0;
  res.rmse = L.inliers ? std::sqrt(L.sse / L.inliers) : 0.0;
  res.iterations = it;
  res.info = L.H / p.info_deflation;
  // Trimmed ICP can lower its cost by gating points out: walls and posts run floor to ceiling, so a vertical
  // slide that drops the floor fits "perfectly". A solution that kept far fewer matches than the guess is one.
  if (L.inliers < p.min_inliers || res.inlier_ratio < p.min_inlier_ratio || !res.info.allFinite() ||
      L.inliers < p.min_inlier_keep * inliers_at_init)
    return std::nullopt;
  return res;
}

}  // namespace isam2_depth
