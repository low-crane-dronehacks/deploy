// Custom factors (GTSAM 4.2 API: boost::optional<Matrix&> Jacobians; 4.3 changed it, see config/versions.yaml).
#pragma once

#include <gtsam/base/Vector.h>
#include <gtsam/geometry/Point3.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/nonlinear/NonlinearFactor.h>

namespace isam2_depth {

// Ties keyframe positions to velocities: p_j - p_i = dt (v_i + v_j) / 2 (trapezoidal, world frame).
class KinematicFactor : public gtsam::NoiseModelFactorN<gtsam::Pose3, gtsam::Vector3, gtsam::Pose3, gtsam::Vector3> {
  double dt_;

 public:
  using Base = gtsam::NoiseModelFactorN<gtsam::Pose3, gtsam::Vector3, gtsam::Pose3, gtsam::Vector3>;
  KinematicFactor(gtsam::Key xi, gtsam::Key vi, gtsam::Key xj, gtsam::Key vj, double dt,
                  const gtsam::SharedNoiseModel& model)
      : Base(model, xi, vi, xj, vj), dt_(dt) {}

  gtsam::Vector evaluateError(const gtsam::Pose3& xi, const gtsam::Vector3& vi, const gtsam::Pose3& xj,
                              const gtsam::Vector3& vj, boost::optional<gtsam::Matrix&> Hxi = boost::none,
                              boost::optional<gtsam::Matrix&> Hvi = boost::none,
                              boost::optional<gtsam::Matrix&> Hxj = boost::none,
                              boost::optional<gtsam::Matrix&> Hvj = boost::none) const override {
    gtsam::Matrix36 Ji, Jj;
    const gtsam::Point3 ti = xi.translation(Hxi ? &Ji : nullptr), tj = xj.translation(Hxj ? &Jj : nullptr);
    if (Hxi) *Hxi = -Ji;
    if (Hvi) *Hvi = -0.5 * dt_ * gtsam::Matrix3::Identity();
    if (Hxj) *Hxj = Jj;
    if (Hvj) *Hvj = -0.5 * dt_ * gtsam::Matrix3::Identity();
    return tj - ti - 0.5 * dt_ * (vi + vj);
  }

  gtsam::NonlinearFactor::shared_ptr clone() const override {
    return gtsam::NonlinearFactor::shared_ptr(new KinematicFactor(*this));
  }
};

// PX4's velocity expressed in the body frame: z = R_world_body^T v_world. Body-frame, so it stays valid when the
// graph's world frame has been corrected away from PX4's drifting one (yaw, mostly).
class BodyVelocityFactor : public gtsam::NoiseModelFactorN<gtsam::Pose3, gtsam::Vector3> {
  gtsam::Vector3 z_;

 public:
  using Base = gtsam::NoiseModelFactorN<gtsam::Pose3, gtsam::Vector3>;
  BodyVelocityFactor(gtsam::Key x, gtsam::Key v, const gtsam::Vector3& v_body, const gtsam::SharedNoiseModel& model)
      : Base(model, x, v), z_(v_body) {}

  gtsam::Vector evaluateError(const gtsam::Pose3& x, const gtsam::Vector3& v,
                              boost::optional<gtsam::Matrix&> Hx = boost::none,
                              boost::optional<gtsam::Matrix&> Hv = boost::none) const override {
    gtsam::Matrix3 Jr, Jv;
    const gtsam::Point3 vb = x.rotation().unrotate(gtsam::Point3(v), Hx ? &Jr : nullptr, Hv ? &Jv : nullptr);
    if (Hx) {   // Pose3 tangent is [rot, trans]; the body velocity does not depend on translation
      *Hx = gtsam::Matrix36::Zero();
      Hx->leftCols<3>() = Jr;
    }
    if (Hv) *Hv = Jv;
    return vb - z_;
  }

  gtsam::NonlinearFactor::shared_ptr clone() const override {
    return gtsam::NonlinearFactor::shared_ptr(new BodyVelocityFactor(*this));
  }
};

// A landmark (world point) measured in the body frame from depth: z = T_world_body^-1 * l.
class BodyPointFactor : public gtsam::NoiseModelFactorN<gtsam::Pose3, gtsam::Point3> {
  gtsam::Point3 z_;

 public:
  using Base = gtsam::NoiseModelFactorN<gtsam::Pose3, gtsam::Point3>;
  BodyPointFactor(gtsam::Key x, gtsam::Key l, const gtsam::Point3& z_body, const gtsam::SharedNoiseModel& model)
      : Base(model, x, l), z_(z_body) {}

  gtsam::Vector evaluateError(const gtsam::Pose3& x, const gtsam::Point3& l,
                              boost::optional<gtsam::Matrix&> Hx = boost::none,
                              boost::optional<gtsam::Matrix&> Hl = boost::none) const override {
    gtsam::Matrix36 Jx;
    gtsam::Matrix3 Jl;
    const gtsam::Point3 p = x.transformTo(l, Hx ? &Jx : nullptr, Hl ? &Jl : nullptr);
    if (Hx) *Hx = Jx;
    if (Hl) *Hl = Jl;
    return p - z_;
  }

  gtsam::NonlinearFactor::shared_ptr clone() const override {
    return gtsam::NonlinearFactor::shared_ptr(new BodyPointFactor(*this));
  }
};

}  // namespace isam2_depth
