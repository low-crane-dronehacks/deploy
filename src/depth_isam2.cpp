#include "isam2_depth/depth_isam2.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <set>
#include <unordered_set>

#include <gtsam/inference/Symbol.h>
#include <gtsam/navigation/AttitudeFactor.h>
#include <gtsam/nonlinear/PriorFactor.h>
#include <gtsam/slam/BetweenFactor.h>

#include "isam2_depth/factors.hpp"

namespace isam2_depth {

using gtsam::symbol_shorthand::L;
using gtsam::symbol_shorthand::V;
using gtsam::symbol_shorthand::X;
namespace nm = gtsam::noiseModel;

namespace {
constexpr double kDeg = M_PI / 180.0;

double angle(const gtsam::Rot3& R) { return gtsam::Rot3::Logmap(R).norm(); }

gtsam::ISAM2Params isamParams(const Isam2Params& p) {
  gtsam::ISAM2Params ip;
  ip.optimizationParams = gtsam::ISAM2GaussNewtonParams(p.wildfire_threshold);
  ip.relinearizeThreshold = p.relinearize_threshold;
  ip.relinearizeSkip = p.relinearize_skip;
  ip.enableRelinearization = true;
  ip.factorization = gtsam::ISAM2Params::CHOLESKY;
  return ip;
}

// One point per occupied voxel (the first seen), so map density does not depend on range or decimation.
std::vector<Eigen::Vector3f> voxelDownsample(const std::vector<Eigen::Vector3f>& pts, double voxel) {
  struct H {
    size_t operator()(const Eigen::Vector3i& v) const {
      return (size_t(v.x()) * 73856093u) ^ (size_t(v.y()) * 19349663u) ^ (size_t(v.z()) * 83492791u);
    }
  };
  std::unordered_set<Eigen::Vector3i, H> seen;
  std::vector<Eigen::Vector3f> out;
  for (const auto& q : pts) {
    const Eigen::Vector3i key = (q / static_cast<float>(voxel)).array().floor().cast<int>();
    if (seen.insert(key).second) out.push_back(q);
  }
  return out;
}
}  // namespace

DepthIsam2::DepthIsam2(const Isam2Params& p, const PinholeCamera& cam, const gtsam::Pose3& T_body_cam)
    : p_(p), cam_(cam), T_bc_(T_body_cam), T_cb_(T_body_cam.inverse()), isam_(isamParams(p)) {}

bool DepthIsam2::addFrame(const DepthFrame& depth, const OdomSample& odom, const std::vector<LandmarkObs>& obs) {
  // ── keyframe? ──
  gtsam::Pose3 rel_odom;
  if (!kfs_.empty()) {
    const Keyframe& last = kfs_.back();
    rel_odom = last.odom.T_world_body.between(odom.T_world_body);
    const bool moved = rel_odom.translation().norm() >= p_.kf_min_translation ||
                       angle(rel_odom.rotation()) >= p_.kf_min_rotation_deg * kDeg;
    if (!moved && odom.stamp - last.stamp < p_.kf_max_interval && obs.empty()) return false;
    if (odom.stamp <= last.stamp) return false;   // out of order
  }
  using Clock = std::chrono::steady_clock;
  const auto ms = [](Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
  };
  const auto t0 = Clock::now();
  const size_t k = kfs_.size();

  Keyframe kf;
  kf.stamp = odom.stamp;
  kf.odom = odom;
  OrganisedCloud cloud = makeCloud(depth, cam_, p_.cloud);
  {
    std::vector<Eigen::Vector3f> body;
    body.reserve(cloud.points.size());
    const Eigen::Matrix3f R = T_bc_.rotation().matrix().cast<float>();
    const Eigen::Vector3f t = T_bc_.translation().cast<float>();
    for (const auto& q : cloud.points)
      if (std::isfinite(q.z())) body.push_back(R * q + t);
    kf.map_points = voxelDownsample(body, p_.map_voxel);
  }

  gtsam::NonlinearFactorGraph g;
  gtsam::Values init;
  const gtsam::Rot3& R_px4 = odom.T_world_body.rotation();
  gtsam::Pose3 X_init;
  if (k == 0) {
    X_init = odom.T_world_body;
    g.addPrior(X(0), X_init, nm::Diagonal::Sigmas((gtsam::Vector6() << gtsam::Vector3::Constant(p_.prior_sigma_rot),
                                                   gtsam::Vector3::Constant(p_.prior_sigma_trans)).finished()));
    g.addPrior(V(0), gtsam::Vector3(odom.v_world), nm::Isotropic::Sigma(3, p_.prior_sigma_vel));
    init.insert(V(0), gtsam::Vector3(odom.v_world));
  } else {
    const Keyframe& prev = kfs_[k - 1];
    const gtsam::Pose3 X_prev = estimate_.at<gtsam::Pose3>(X(k - 1));
    X_init = X_prev * rel_odom;
    // PX4 relative motion
    const double st = p_.odom_sigma_trans + p_.odom_sigma_trans_per_m * rel_odom.translation().norm();
    const double sr = p_.odom_sigma_rot + p_.odom_sigma_rot_per_rad * angle(rel_odom.rotation());
    g.emplace_shared<gtsam::BetweenFactor<gtsam::Pose3>>(
        X(k - 1), X(k), rel_odom,
        nm::Diagonal::Sigmas((gtsam::Vector6() << gtsam::Vector3::Constant(sr), gtsam::Vector3::Constant(st)).finished()));
    // velocity: PX4's, rotated into the graph frame (they differ by the smoothing correction)
    const gtsam::Rot3 R_corr = X_prev.rotation() * prev.odom.T_world_body.rotation().inverse();
    const gtsam::Vector3 v_init = R_corr.rotate(gtsam::Point3(odom.v_world));
    init.insert(V(k), v_init);
    g.emplace_shared<KinematicFactor>(X(k - 1), V(k - 1), X(k), V(k), odom.stamp - prev.stamp,
                                      nm::Isotropic::Sigma(3, p_.kin_sigma));
    // depth: chain ICP to the previous keyframe, then to older keyframes that overlap (revisits)
    const auto ti = Clock::now();
    if (prev.cloud && addIcpFactor(k - 1, k, cloud, rel_odom, g)) ++stats_.icp_ok;
    else ++stats_.icp_rejected;
    if (p_.extra_links > 0) {
      std::vector<std::pair<double, size_t>> near;
      for (size_t j = 0; j + p_.link_min_age < k; ++j) {
        if (!kfs_[j].cloud) continue;
        const gtsam::Pose3 rel = estimate_.at<gtsam::Pose3>(X(j)).between(X_init);
        const double d = rel.translation().norm();
        if (d < p_.link_max_dist && angle(rel.rotation()) < p_.link_max_angle_deg * kDeg) near.emplace_back(d, j);
      }
      std::sort(near.begin(), near.end());
      int added = 0;
      for (const auto& [d, j] : near) {
        if (added >= p_.extra_links) break;
        const gtsam::Pose3 rel = estimate_.at<gtsam::Pose3>(X(j)).between(X_init);
        if (addIcpFactor(j, k, cloud, rel, g)) { ++added; ++stats_.links; }
      }
    }
    stats_.icp_ms = ms(ti, Clock::now());
  }
  init.insert(X(k), X_init);
  g.emplace_shared<BodyVelocityFactor>(X(k), V(k), R_px4.unrotate(gtsam::Point3(odom.v_world)),
                                      nm::Isotropic::Sigma(3, p_.vel_sigma));
  // roll / pitch: body down axis expressed in NED, as PX4 estimates it (yaw-free)
  g.emplace_shared<gtsam::Pose3AttitudeFactor>(X(k), gtsam::Unit3(R_px4.rotate(gtsam::Point3(0, 0, 1))),
                                               nm::Isotropic::Sigma(2, p_.attitude_sigma), gtsam::Unit3(0, 0, 1));
  std::set<int> new_lm;
  for (const auto& o : obs) {
    const gtsam::Key l = L(o.id);
    if (!estimate_.exists(l) && new_lm.insert(o.id).second) init.insert(l, X_init.transformFrom(o.p_body));
    g.emplace_shared<BodyPointFactor>(X(k), l, o.p_body,
                                      nm::Robust::Create(nm::mEstimator::Huber::Create(1.345),
                                                         nm::Isotropic::Sigma(3, o.sigma)));
  }

  const auto tu = Clock::now();
  isam_.update(g, init);
  estimate_ = isam_.calculateEstimate();
  const auto to = Clock::now();
  stats_.isam_ms = ms(tu, to);

  kf.cloud = std::move(cloud);
  kfs_.push_back(std::move(kf));
  if (kfs_.size() > static_cast<size_t>(p_.keep_dense)) kfs_[kfs_.size() - 1 - p_.keep_dense].cloud.reset();
  rebuildOutput(k);
  stats_.output_ms = ms(to, Clock::now());
  stats_.keyframes = kfs_.size();
  stats_.factors = isam_.getFactorsUnsafe().size();
  stats_.last_update_ms = ms(t0, Clock::now());
  return true;
}

bool DepthIsam2::addIcpFactor(size_t j, size_t k, const OrganisedCloud& cloud_k, const gtsam::Pose3& predicted_rel,
                              gtsam::NonlinearFactorGraph& g) {
  // target = keyframe j, source = new keyframe k: T_cj_ck = T_cb * T_bj_bk * T_bc
  const gtsam::Pose3 init = T_cb_ * predicted_rel * T_bc_;
  const auto r = alignProjective(*kfs_[j].cloud, cloud_k, init, p_.icp);
  if (!r) return false;
  const gtsam::Pose3 rel = T_bc_ * r->T_target_source * T_cb_;
  const gtsam::Pose3 err = predicted_rel.between(rel);
  if (err.translation().norm() > p_.icp_gate_trans || angle(err.rotation()) > p_.icp_gate_rot_deg * kDeg)
    return false;
  // right-tangent change of frame: xi_body = Ad(T_bc) xi_cam  =>  info_body = Ad^-T info_cam Ad^-1
  const gtsam::Matrix6 Ad_inv = T_bc_.AdjointMap().inverse();
  gtsam::Matrix6 info = Ad_inv.transpose() * r->info * Ad_inv;
  info.diagonal().head<3>().array() += 1.0 / (p_.icp_sigma_max_rot * p_.icp_sigma_max_rot);
  info.diagonal().tail<3>().array() += 1.0 / (p_.icp_sigma_max_trans * p_.icp_sigma_max_trans);
  g.emplace_shared<gtsam::BetweenFactor<gtsam::Pose3>>(
      X(j), X(k), rel,
      nm::Robust::Create(nm::mEstimator::Huber::Create(1.345), nm::Gaussian::Information(info)));
  return true;
}

void DepthIsam2::rebuildOutput(size_t k) {
  MppiInput out;
  out.stamp = kfs_[k].stamp;
  out.keyframe = k;
  out.T_world_body = estimate_.at<gtsam::Pose3>(X(k));
  out.v_world = estimate_.at<gtsam::Vector3>(V(k));
  try {
    out.pose_cov = isam_.marginalCovariance(X(k));
    out.vel_cov = isam_.marginalCovariance(V(k));
  } catch (const std::exception&) {   // keep identity: "unknown"
  }
  const Eigen::Vector3d c = out.T_world_body.translation();
  std::vector<Eigen::Vector3f> world;
  for (size_t j = 0; j <= k; ++j) {
    const gtsam::Pose3 Xj = estimate_.at<gtsam::Pose3>(X(j));
    if ((Xj.translation() - c).norm() > p_.map_radius) continue;
    const Eigen::Matrix3f R = Xj.rotation().matrix().cast<float>();
    const Eigen::Vector3f t = Xj.translation().cast<float>();
    for (const auto& q : kfs_[j].map_points) world.push_back(R * q + t);
  }
  out.distance = std::make_shared<const DistanceGrid>(world, c, p_.grid);
  // Point3 and Vector3 are the same type: select landmarks by symbol, not by type
  for (const auto& kv : estimate_.extract<gtsam::Point3>(gtsam::Symbol::ChrTest('l')))
    out.landmarks[static_cast<int>(gtsam::Symbol(kv.first).index())] = kv.second;
  latest_ = std::move(out);
}

MppiInput DepthIsam2::predict(const OdomSample& now) const {
  MppiInput out = latest_;
  if (kfs_.empty()) return out;
  const OdomSample& ko = kfs_.back().odom;
  out.stamp = now.stamp;
  out.T_world_body = latest_.T_world_body * ko.T_world_body.between(now.T_world_body);
  const gtsam::Rot3 R_corr = latest_.T_world_body.rotation() * ko.T_world_body.rotation().inverse();
  out.v_world = R_corr.rotate(gtsam::Point3(now.v_world));
  return out;
}

}  // namespace isam2_depth
