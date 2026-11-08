// C++ port of dronehacks-sim quad.py (TorchQuad + LeeVelocityController), with PX4's velocity loop as the default
// outer loop. Comments name the quad.py / PX4 code each block mirrors; keep them in step (src/test/quad_reference.inc
// is generated from quad.py and checked with QuadPlantParams::quadPy()).
#include "maze_planning/quad_plant.hpp"

#include <algorithm>

#include <Eigen/SVD>

namespace maze_planning {

namespace {

// Z-up world <-> NED axis maps for attitude (quad.py _P, _Q): R_ned = P R_world Q. Both are involutions, so
// R_world = P R_ned Q as well.
const Eigen::Matrix3d& P() {
  static const Eigen::Matrix3d m = (Eigen::Matrix3d() << 0, 1, 0, 1, 0, 0, 0, 0, -1).finished();
  return m;
}
const Eigen::Matrix3d& Q() {
  static const Eigen::Matrix3d m = Eigen::Vector3d(1, -1, -1).asDiagonal();
  return m;
}

inline Eigen::Vector3d nedToWorld(const Eigen::Vector3d& v) { return {v.y(), v.x(), -v.z()}; }

}  // namespace

Eigen::Matrix3d QuadState::rotNed() const { return P() * q_w.toRotationMatrix() * Q(); }

double QuadState::yawNed() const {
  const Eigen::Matrix3d r = rotNed();
  return std::atan2(r(1, 0), r(0, 0));
}

QuadPlant::QuadPlant(const QuadPlantParams& p) : p_(p) {
  // TorchQuad.__init__: allocation rows Fz = 1, Tx = +y_i, Ty = -x_i, Tz = dir_i * ratio (Fx, Fy rows zero)
  alloc_.setZero();
  for (int i = 0; i < 4; ++i) {
    alloc_(2, i) = 1.0;
    alloc_(3, i) = p_.rotor_pos_flu[i].y();
    alloc_(4, i) = -p_.rotor_pos_flu[i].x();
    alloc_(5, i) = p_.rotor_dir[i] * p_.torque_to_thrust_ratio;
  }
  alloc_pinv_ = alloc_.completeOrthogonalDecomposition().pseudoInverse();
  hover_thrust_ = alloc_pinv_.col(2) * (p_.mass_kg * p_.gravity_m_s2);
  // _physics_step battery terms at a constant voltage: max thrust ~ (V/V_nom)^2, motor tau ~ V_nom/V
  const double vr = p_.batt_v / p_.batt_nom_v;
  thrust_max_eff_ = p_.thrust_max_n * vr * vr;
  const double tau_eff = p_.motor_tau_s / vr;
  motor_alpha_ = 1.0 - std::exp(-p_.physics_dt_s / tau_eff);
}

QuadState QuadPlant::fromNed(const Eigen::Vector3d& pos_ned, const Eigen::Vector3d& vel_ned,
                             const Eigen::Quaterniond& q_ned_frd, const Eigen::Vector3d& omega_frd) const {
  QuadState s;
  s.pos_w = nedToWorld(pos_ned);
  s.vel_w = nedToWorld(vel_ned);
  s.q_w = Eigen::Quaterniond(Eigen::Matrix3d(P() * q_ned_frd.normalized().toRotationMatrix() * Q())).normalized();
  s.omega_b = Q() * omega_frd;
  s.thrust = hover_thrust_;
  return s;
}

int QuadPlant::substeps(double duration_s) const {
  return std::max(1, static_cast<int>(std::lround(duration_s / p_.physics_dt_s)));
}

void QuadPlant::stepWorld(QuadState& s, const Eigen::Vector3d& vel_ned, double yaw_rate_ned,
                          double duration_s) const {
  const Eigen::Vector3d v_des_w = nedToWorld(vel_ned);
  const int n = substeps(duration_s);
  for (int i = 0; i < n; ++i) physicsStep(s, v_des_w, -yaw_rate_ned);
}

void QuadPlant::stepYawAligned(QuadState& s, const Eigen::Vector3d& v_frd, double yaw_rate_ned,
                               double duration_s) const {
  // step_control: NED yaw-aligned -> FLU yaw-aligned command; the controller rotates it by the current yaw
  const Eigen::Vector3d cmd_flu(v_frd.x(), -v_frd.y(), -v_frd.z());
  const int n = substeps(duration_s);
  for (int i = 0; i < n; ++i) {
    const Eigen::Matrix3d R = s.q_w.toRotationMatrix();
    const double yaw = std::atan2(R(1, 0), R(0, 0));
    const Eigen::Vector3d v_des_w(std::cos(yaw) * cmd_flu.x() - std::sin(yaw) * cmd_flu.y(),
                                  std::sin(yaw) * cmd_flu.x() + std::cos(yaw) * cmd_flu.y(), cmd_flu.z());
    physicsStep(s, v_des_w, -yaw_rate_ned);
  }
}

void QuadPlant::physicsStep(QuadState& s, const Eigen::Vector3d& v_des_w, double yaw_rate_flu) const {
  const double dt = p_.physics_dt_s;
  const double m = p_.mass_kg;
  const Eigen::Vector3d g(0.0, 0.0, -p_.gravity_m_s2);
  const Eigen::Vector3d& I = p_.inertia_diag;
  const Eigen::Matrix3d R = s.q_w.toRotationMatrix();               // body -> world
  const Eigen::Vector3d& w = s.omega_b;

  // ---- LeeVelocityController.compute ----------------------------------------------------------------------
  const double yaw = std::atan2(R(1, 0), R(0, 0));
  const double roll = std::atan2(R(2, 1), R(2, 2));
  const double pitch = std::asin(std::clamp(-R(2, 0), -1.0, 1.0));
  const Eigen::Vector3d vel_err = v_des_w - s.vel_w;
  Eigen::Vector3d acc_cmd;
  if (p_.velocity_loop == VelocityLoop::kQuadPyLee) {
    acc_cmd = p_.k_vel.cwiseProduct(vel_err);
  } else {
    // PX4 PositionControl::_velocityControl (v1.16): PID, D on the measured (low-passed) acceleration
    s.vel_int.z() = std::clamp(s.vel_int.z(), -p_.gravity_m_s2, p_.gravity_m_s2);
    const Eigen::Vector3d acc_sp = p_.px4_vel_p.cwiseProduct(vel_err) + s.vel_int - p_.px4_vel_d.cwiseProduct(s.acc_lp);
    // _accelerationControl with MPC_ACC_DECOUPLE: body z from (-a_x, -a_y, g), tilt-limited
    acc_cmd = acc_sp;
    const double a_xy = std::hypot(acc_sp.x(), acc_sp.y());
    const double a_xy_max = p_.gravity_m_s2 * std::tan(p_.px4_tilt_max_rad);
    if (a_xy > a_xy_max) acc_cmd.head<2>() *= a_xy_max / a_xy;
    // integrator: vertical clamp above; horizontal tracking anti-windup while the tilt limit saturates
    Eigen::Vector3d e_int = vel_err;
    if (a_xy > a_xy_max) {
      const double arw_gain = 2.0 / p_.px4_vel_p.x();
      e_int.head<2>() -= arw_gain * (acc_sp.head<2>() - acc_cmd.head<2>());
    }
    s.vel_int += p_.px4_vel_i.cwiseProduct(e_int) * p_.physics_dt_s;
  }
  const Eigen::Vector3d forces_w = (acc_cmd - g) * m;
  Eigen::Matrix<double, 6, 1> wrench = Eigen::Matrix<double, 6, 1>::Zero();
  wrench(2) = forces_w.dot(R.col(2));

  // desired attitude: thrust along the force, heading = current yaw
  const Eigen::Vector3d b3 = forces_w / (forces_w.norm() + 1e-12);
  const Eigen::Vector3d tmp(std::cos(yaw), std::sin(yaw), 0.0);
  Eigen::Vector3d b2 = b3.cross(tmp);
  b2 /= (b2.norm() + 1e-12);
  const Eigen::Vector3d b1 = b2.cross(b3);
  Eigen::Matrix3d R_des;
  R_des.col(0) = b1;
  R_des.col(1) = b2;
  R_des.col(2) = b3;

  // yaw-rate command -> body rates
  Eigen::Vector3d w_des(-std::sin(pitch) * yaw_rate_flu, std::sin(roll) * std::cos(pitch) * yaw_rate_flu,
                        std::cos(roll) * std::cos(pitch) * yaw_rate_flu);
  w_des.z() = std::clamp(w_des.z(), -p_.max_yaw_rate_rad_s, p_.max_yaw_rate_rad_s);

  // PD on SO(3) + Coriolis feedforward
  const Eigen::Matrix3d R_err = R.transpose() * R_des;
  const Eigen::Matrix3d skew = R_err.transpose() - R_err;
  const Eigen::Vector3d rot_err = 0.5 * Eigen::Vector3d(-skew(1, 2), skew(0, 2), -skew(0, 1));
  const Eigen::Vector3d Iw = I.cwiseProduct(w);
  wrench.tail<3>() = -p_.k_rot.cwiseProduct(rot_err) - p_.k_angvel.cwiseProduct(w - w_des) + w.cross(Iw);

  // ---- TorchQuad._physics_step ------------------------------------------------------------------------------
  const Eigen::Vector4d raw = (alloc_pinv_ * wrench).cwiseMax(0.0);
  const Eigen::Vector4d thrust_cmd = raw.cwiseMax(p_.thrust_min_n).cwiseMin(thrust_max_eff_);
  s.thrust += (thrust_cmd - s.thrust) * motor_alpha_;
  const Eigen::Matrix<double, 6, 1> w_b = alloc_ * s.thrust;

  const Eigen::Vector3d acc = R * w_b.head<3>() / m - p_.drag_linear * s.vel_w + g;
  const Eigen::Vector3d torque = w_b.tail<3>() - p_.drag_angular * Iw;
  const Eigen::Vector3d ang_acc = (torque - w.cross(Iw)).cwiseQuotient(I);

  const Eigen::Vector3d vel_prev = s.vel_w;
  s.vel_w += acc * dt;
  s.pos_w += s.vel_w * dt;
  if (p_.velocity_loop == VelocityLoop::kPx4Pid) {   // PX4's velocity derivative, low-passed at MPC_VELD_LP
    const double rc = 1.0 / (2.0 * M_PI * p_.px4_veld_lp_hz);
    s.acc_lp += (dt / (dt + rc)) * ((s.vel_w - vel_prev) / dt - s.acc_lp);
  }
  s.omega_b += ang_acc * dt;
  const double wn = s.omega_b.norm();
  const double theta = wn * dt;
  const Eigen::Vector3d axis = s.omega_b / std::max(wn, 1e-9);
  const Eigen::Quaterniond dq(std::cos(0.5 * theta), axis.x() * std::sin(0.5 * theta),
                              axis.y() * std::sin(0.5 * theta), axis.z() * std::sin(0.5 * theta));
  s.q_w = (s.q_w * dq).normalized();
}

}  // namespace maze_planning
