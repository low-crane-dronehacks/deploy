#pragma once

#include <array>
#include <cmath>
#include "ego_planner/math_types.hpp"

namespace ego_planner {

/**
 * @brief Quadrotor physical and controller parameters.
 * Matches Isaac Lab / AttentionSeeker TorchQuad specifications in dronehacks-sim.
 */
struct QuadrotorParams {
  double mass{1.5};                     // kg
  double gravity{9.81};                 // m/s^2
  Vec3 inertia_diag{0.012, 0.012, 0.020}; // kg*m^2 (Ixx, Iyy, Izz)
  double arm_m{0.13};                   // rotor moment arm, m
  double torque_to_thrust_ratio{0.07};  // rotor reaction torque coefficient
  double thrust_min_n{0.1};             // min thrust per rotor, N
  double thrust_max_n{10.0};            // max thrust per rotor, N (2.7:1 TWR)
  double motor_tau_s{0.030};            // first-order motor lag, s
  double drag_linear{0.25};             // linear aero drag coefficient (1/s)
  double drag_angular{0.05};            // angular aero drag coefficient (1/s)
  double max_yaw_rate_rad_s{kPi / 3.0}; // 60 deg/s

  // Lee velocity controller gains (ARL Robot 1 defaults)
  Vec3 k_vel{3.0, 3.0, 1.75};           // velocity error gain
  Vec3 k_rot{1.725, 1.725, 0.325};      // attitude error gain
  Vec3 k_angvel{0.45, 0.45, 0.0825};    // angular rate error gain
};

/**
 * @brief State representation for MPPI forward rollouts.
 * Represented in NED world frame and FRD body frame.
 */
struct QuadrotorState {
  Vec3 pos_ned{0.0, 0.0, 0.0};          // position in NED (m)
  Vec3 vel_ned{0.0, 0.0, 0.0};          // velocity in NED (m/s)
  Quat q_wxyz{1.0, 0.0, 0.0, 0.0};      // orientation (body FRD -> NED)
  Vec3 omega_b{0.0, 0.0, 0.0};          // body angular velocity in FRD (rad/s)
  std::array<double, 4> motor_thrusts{{3.68, 3.68, 3.68, 3.68}}; // N per rotor (hover = m*g/4)
};

/**
 * @brief Control input representation for MPPI.
 * Can be expressed either as velocity setpoints in yaw-aligned body frame,
 * or as body wrench / normalized action [-1, 1].
 */
struct QuadrotorAction {
  // Velocity commands [v_fwd, v_right, v_down] in yaw-aligned frame (m/s)
  Vec3 vel_cmd_yaw_aligned{0.0, 0.0, 0.0};
  double yaw_rate_cmd_rad_s{0.0};       // yaw rate command (rad/s)
};

/**
 * @brief Forward predictive quadrotor simulator for MPPI trajectory rollouts.
 * Ported directly from dronehacks/quad.py.
 */
class QuadrotorDynamics {
 public:
  explicit QuadrotorDynamics(const QuadrotorParams& params = QuadrotorParams{});

  /**
   * @brief Step the quadrotor forward by dt seconds.
   * Runs substepping (e.g. 100 Hz physics) to ensure stability.
   */
  QuadrotorState step(const QuadrotorState& state, const QuadrotorAction& action, double dt) const;

  /**
   * @brief Compute the Lee geometric velocity controller body wrench.
   * @param cmd Velocity command in yaw-aligned frame + yaw rate.
   * @param state Current vehicle state.
   * @param wrench_out [Fx, Fy, Fz, Tx, Ty, Tz] in body FRD frame.
   */
  void computeLeeWrench(const QuadrotorAction& cmd, const QuadrotorState& state,
                        std::array<double, 6>& wrench_out) const;

  /**
   * @brief Allocate body wrench to 4 motor thrusts.
   */
  void allocateRotorThrusts(const std::array<double, 6>& wrench,
                            std::array<double, 4>& desired_thrusts) const;

  const QuadrotorParams& params() const { return params_; }

 private:
  QuadrotorParams params_;
  double hover_thrust_{3.68};
};

}  // namespace ego_planner
