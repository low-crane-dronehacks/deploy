#include "ego_planner/quadrotor_dynamics.hpp"

#include <algorithm>
#include <cmath>

namespace ego_planner {

QuadrotorDynamics::QuadrotorDynamics(const QuadrotorParams& params)
    : params_(params), hover_thrust_(params.mass * params.gravity / 4.0) {}

void QuadrotorDynamics::computeLeeWrench(const QuadrotorAction& cmd,
                                        const QuadrotorState& state,
                                        std::array<double, 6>& wrench_out) const {
  wrench_out.fill(0.0);

  // Extract Euler yaw in NED from FRD quaternion
  Vec3 rpy = state.q_wxyz.toEulerZYX();
  double roll = rpy.x;
  double pitch = rpy.y;
  double yaw = rpy.z;

  // Rotate velocity command from yaw-aligned body frame to NED world frame
  double cy = std::cos(yaw);
  double sy = std::sin(yaw);
  Vec3 vel_des_ned{
      cy * cmd.vel_cmd_yaw_aligned.x - sy * cmd.vel_cmd_yaw_aligned.y,
      sy * cmd.vel_cmd_yaw_aligned.x + cy * cmd.vel_cmd_yaw_aligned.y,
      cmd.vel_cmd_yaw_aligned.z,
  };

  // Desired acceleration in NED (proportional velocity tracking)
  Vec3 vel_err = vel_des_ned - state.vel_ned;
  Vec3 acc_des{
      params_.k_vel.x * vel_err.x,
      params_.k_vel.y * vel_err.y,
      params_.k_vel.z * vel_err.z,
  };

  // Gravity vector in NED (+z is down)
  Vec3 g_ned{0.0, 0.0, params_.gravity};

  // Total desired force in NED: F = m * (a - g)
  Vec3 f_des_ned = (acc_des - g_ned) * params_.mass;

  // Rotation matrix from body FRD to NED
  Mat3 R = state.q_wxyz.toRotationMatrix();
  Vec3 b3{R.m[0][2], R.m[1][2], R.m[2][2]}; // body +z axis (down) in NED

  // Body z force: in FRD, downward force is f_des_ned . b3 (< 0 when producing lift)
  wrench_out[2] = f_des_ned.dot(b3);

  // Desired body z direction (opposes total thrust force vector)
  double f_norm = f_des_ned.norm();
  Vec3 b3_des = f_norm > 1e-6 ? (-f_des_ned / f_norm) : Vec3{0.0, 0.0, 1.0};

  // Construct desired body frame matching yaw
  Vec3 heading_vec{std::cos(yaw), std::sin(yaw), 0.0};
  Vec3 b2_des = b3_des.cross(heading_vec);
  double b2_norm = b2_des.norm();
  if (b2_norm > 1e-6) {
    b2_des = b2_des / b2_norm;
  } else {
    b2_des = Vec3{0.0, 1.0, 0.0};
  }
  Vec3 b1_des = b2_des.cross(b3_des);

  Mat3 R_des;
  R_des.m[0][0] = b1_des.x; R_des.m[0][1] = b2_des.x; R_des.m[0][2] = b3_des.x;
  R_des.m[1][0] = b1_des.y; R_des.m[1][1] = b2_des.y; R_des.m[1][2] = b3_des.y;
  R_des.m[2][0] = b1_des.z; R_des.m[2][1] = b2_des.z; R_des.m[2][2] = b3_des.z;

  // Attitude error on SO(3): e_R = 0.5 * (R_des^T * R - R^T * R_des)^\vee
  Mat3 R_err = R_des.transposed() * R;
  Vec3 rot_err{
      0.5 * (R_err.m[2][1] - R_err.m[1][2]),
      0.5 * (R_err.m[0][2] - R_err.m[2][0]),
      0.5 * (R_err.m[1][0] - R_err.m[0][1]),
  };

  // Desired angular rates from yaw-rate command
  double clamped_yaw_rate = clamp(cmd.yaw_rate_cmd_rad_s, -params_.max_yaw_rate_rad_s, params_.max_yaw_rate_rad_s);
  Vec3 omega_des{
      -std::sin(pitch) * clamped_yaw_rate,
      std::sin(roll) * std::cos(pitch) * clamped_yaw_rate,
      std::cos(roll) * std::cos(pitch) * clamped_yaw_rate,
  };

  Vec3 omega_err = state.omega_b - omega_des;

  // Coriolis torque: omega x (I * omega)
  Vec3 I_omega{
      params_.inertia_diag.x * state.omega_b.x,
      params_.inertia_diag.y * state.omega_b.y,
      params_.inertia_diag.z * state.omega_b.z,
  };
  Vec3 coriolis = state.omega_b.cross(I_omega);

  // PD torque feedback
  wrench_out[3] = -params_.k_rot.x * rot_err.x - params_.k_angvel.x * omega_err.x + coriolis.x;
  wrench_out[4] = -params_.k_rot.y * rot_err.y - params_.k_angvel.y * omega_err.y + coriolis.y;
  wrench_out[5] = -params_.k_rot.z * rot_err.z - params_.k_angvel.z * omega_err.z + coriolis.z;
}

void QuadrotorDynamics::allocateRotorThrusts(const std::array<double, 6>& wrench,
                                            std::array<double, 4>& desired_thrusts) const {
  // In FRD NED convention:
  // Fz is downward force (< 0 when producing lift)
  // Lift force = -wrench[2]
  double total_lift = -wrench[2];
  double tx = wrench[3];
  double ty = wrench[4];
  double tz = wrench[5];

  double d = params_.arm_m;
  double c_tau = params_.torque_to_thrust_ratio;

  // Allocation matrix inverse for quad-X configuration:
  // Rotor 0: (+d, -d) CW (-c_tau)
  // Rotor 1: (-d, -d) CCW (+c_tau)
  // Rotor 2: (-d, +d) CW (-c_tau)
  // Rotor 3: (+d, +d) CCW (+c_tau)
  double f_base = total_lift * 0.25;
  double f_roll = tx / (4.0 * d);
  double f_pitch = ty / (4.0 * d);
  double f_yaw = tz / (4.0 * c_tau);

  desired_thrusts[0] = f_base - f_roll + f_pitch - f_yaw;
  desired_thrusts[1] = f_base - f_roll - f_pitch + f_yaw;
  desired_thrusts[2] = f_base + f_roll - f_pitch - f_yaw;
  desired_thrusts[3] = f_base + f_roll + f_pitch + f_yaw;

  for (int i = 0; i < 4; ++i) {
    desired_thrusts[i] = clamp(desired_thrusts[i], params_.thrust_min_n, params_.thrust_max_n);
  }
}

QuadrotorState QuadrotorDynamics::step(const QuadrotorState& state,
                                      const QuadrotorAction& action,
                                      double dt) const {
  QuadrotorState next = state;

  // Substepping at 100 Hz (sub_dt = 0.01s)
  int n_substeps = std::max(1, static_cast<int>(std::round(dt / 0.01)));
  double sub_dt = dt / n_substeps;

  for (int step = 0; step < n_substeps; ++step) {
    // 1. Controller wrench
    std::array<double, 6> wrench;
    computeLeeWrench(action, next, wrench);

    // 2. Rotor allocation
    std::array<double, 4> des_thrusts;
    allocateRotorThrusts(wrench, des_thrusts);

    // 3. Motor lag (first-order filter)
    double alpha = sub_dt / (params_.motor_tau_s + sub_dt);
    for (int i = 0; i < 4; ++i) {
      next.motor_thrusts[i] += alpha * (des_thrusts[i] - next.motor_thrusts[i]);
      next.motor_thrusts[i] = clamp(next.motor_thrusts[i], params_.thrust_min_n, params_.thrust_max_n);
    }

    // 4. Actual wrench from rotor thrusts
    double d = params_.arm_m;
    double c_tau = params_.torque_to_thrust_ratio;
    double total_thrust = next.motor_thrusts[0] + next.motor_thrusts[1] +
                          next.motor_thrusts[2] + next.motor_thrusts[3];
    double actual_tx = d * (-next.motor_thrusts[0] - next.motor_thrusts[1] +
                             next.motor_thrusts[2] + next.motor_thrusts[3]);
    double actual_ty = d * (next.motor_thrusts[0] - next.motor_thrusts[1] -
                            next.motor_thrusts[2] + next.motor_thrusts[3]);
    double actual_tz = c_tau * (-next.motor_thrusts[0] + next.motor_thrusts[1] -
                                 next.motor_thrusts[2] + next.motor_thrusts[3]);

    // Body thrust force is along -z in FRD: F_body = [0, 0, -total_thrust]
    Mat3 R = next.q_wxyz.toRotationMatrix();
    Vec3 thrust_ned = R * Vec3{0.0, 0.0, -total_thrust};

    // Gravity in NED
    Vec3 gravity_ned{0.0, 0.0, params_.gravity * params_.mass};

    // Linear drag in NED: F_drag = -C_drag * m * vel
    Vec3 drag_ned = next.vel_ned * (-params_.drag_linear * params_.mass);

    // Total force & linear acceleration
    Vec3 f_total = thrust_ned + gravity_ned + drag_ned;
    Vec3 acc_ned = f_total / params_.mass;

    // Semi-implicit Euler for linear motion
    next.vel_ned += acc_ned * sub_dt;
    next.pos_ned += next.vel_ned * sub_dt;

    // Angular acceleration in body frame
    Vec3 angular_drag = next.omega_b * (-params_.drag_angular);
    Vec3 tau_total{
        actual_tx + angular_drag.x * params_.inertia_diag.x,
        actual_ty + angular_drag.y * params_.inertia_diag.y,
        actual_tz + angular_drag.z * params_.inertia_diag.z,
    };
    Vec3 alpha_body{
        tau_total.x / params_.inertia_diag.x,
        tau_total.y / params_.inertia_diag.y,
        tau_total.z / params_.inertia_diag.z,
    };

    next.omega_b += alpha_body * sub_dt;

    // Quaternion integration via exponential map
    Vec3 delta_rot = next.omega_b * (0.5 * sub_dt);
    double d_angle = delta_rot.norm();
    Quat dq;
    if (d_angle > 1e-8) {
      double sin_half = std::sin(d_angle);
      dq = Quat{std::cos(d_angle),
                delta_rot.x / d_angle * sin_half,
                delta_rot.y / d_angle * sin_half,
                delta_rot.z / d_angle * sin_half};
    } else {
      dq = Quat{1.0, delta_rot.x, delta_rot.y, delta_rot.z};
    }

    // Hamilton product: q_next = q * dq
    Quat q = next.q_wxyz;
    next.q_wxyz = Quat{
        q.w * dq.w - q.x * dq.x - q.y * dq.y - q.z * dq.z,
        q.w * dq.x + q.x * dq.w + q.y * dq.z - q.z * dq.y,
        q.w * dq.y - q.x * dq.z + q.y * dq.w + q.z * dq.x,
        q.w * dq.z + q.x * dq.y - q.y * dq.x + q.z * dq.w,
    };

    // Normalize quaternion
    double q_norm = std::sqrt(next.q_wxyz.w * next.q_wxyz.w + next.q_wxyz.x * next.q_wxyz.x +
                              next.q_wxyz.y * next.q_wxyz.y + next.q_wxyz.z * next.q_wxyz.z);
    if (q_norm > 1e-9) {
      next.q_wxyz.w /= q_norm;
      next.q_wxyz.x /= q_norm;
      next.q_wxyz.y /= q_norm;
      next.q_wxyz.z /= q_norm;
    }
  }

  return next;
}

}  // namespace ego_planner
