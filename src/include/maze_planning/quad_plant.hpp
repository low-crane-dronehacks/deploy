#pragma once

#include <array>
#include <cmath>

#include <Eigen/Core>
#include <Eigen/Geometry>

namespace maze_planning {

/** Which outer velocity loop produces the acceleration setpoint the Lee attitude/thrust stage then realises. */
enum class VelocityLoop {
  kPx4Pid,     // PX4 v1.16 PositionControl::_velocityControl with the fleet's MPC_* gains (what the drone runs)
  kQuadPyLee,  // quad.py's Lee velocity controller: acc = K_vel (v_des - v), no integral, no tilt limit
};

/**
 * @brief Physical + controller parameters. The airframe (mass, inertia, deadcat rotors, thrust range, motor lag,
 * drag) is dronehacks-sim `quad.py`'s; where quad.py draws a value per episode the default is the middle of its
 * range. The velocity loop defaults to PX4's (skybeta fleet params, px4/params/00_baseline.params), because the
 * drone flies on PX4's velocity controller, not quad.py's; QuadPlantParams::quadPy() selects quad.py's own loop
 * (the training plant, used to check this port against quad.py). Update together with quad.py / the fleet params.
 */
struct QuadPlantParams {
  VelocityLoop velocity_loop{VelocityLoop::kPx4Pid};
  // PX4 velocity PID (MPC_XY_VEL_{P,I,D}_ACC, MPC_Z_VEL_{P,I,D}_ACC), D on the 5 Hz low-passed velocity
  // derivative (MPC_VELD_LP), tilt limit MPC_TILTMAX_AIR (MPC_ACC_DECOUPLE = 1: tilt from horizontal acc vs g)
  Eigen::Vector3d px4_vel_p{1.8, 1.8, 4.0};
  Eigen::Vector3d px4_vel_i{0.4, 0.4, 2.0};
  Eigen::Vector3d px4_vel_d{0.2, 0.2, 0.0};
  double px4_veld_lp_hz{5.0};
  double px4_tilt_max_rad{M_PI / 4.0};

  /** quad.py's training plant as-is (Lee velocity loop): for checking the port against quad.py. */
  static QuadPlantParams quadPy() {
    QuadPlantParams p;
    p.velocity_loop = VelocityLoop::kQuadPyLee;
    return p;
  }

  double mass_kg{1.348};                                   // MASS
  Eigen::Vector3d inertia_diag{0.006401, 0.006743, 0.010085};  // I_DIAG, kg m^2 (FLU body)
  // ROTOR_POSITIONS_B (FLU body, metres) and ROTOR_DIRECTIONS: the deadcat frame
  std::array<Eigen::Vector3d, 4> rotor_pos_flu{{{+0.0921, -0.1596, 0.0},
                                                {-0.1384, -0.1351, 0.0},
                                                {-0.1384, +0.1351, 0.0},
                                                {+0.0921, +0.1596, 0.0}}};
  std::array<double, 4> rotor_dir{{-1.0, 1.0, -1.0, 1.0}};
  double torque_to_thrust_ratio{0.0116};                   // TORQUE_TO_THRUST_RATIO
  double thrust_min_n{0.15};                               // THRUST_RANGE at BATT_NOM_V
  double thrust_max_n{12.6};
  double motor_tau_s{0.030};                               // middle of MOTOR_TAU_RANGE (25-35 ms) at BATT_NOM_V
  double drag_linear{0.25};                                // C_DRAG, 1/s: F = -C_DRAG m v
  double drag_angular{0.05};                               // C_ANG, 1/s: tau = -C_ANG I w
  double max_yaw_rate_rad_s{M_PI / 3.0};                   // LEE_MAX_YAW_RATE (60 deg/s)
  // Lee gains: middle of K_VEL_RANGE / K_ROT_RANGE / K_ANGVEL_RANGE. k_vel is used only by kQuadPyLee; the
  // attitude gains are used by both loops (the PX4 attitude / rate loops are not modelled).
  Eigen::Vector3d k_vel{3.0, 3.0, 1.75};
  Eigen::Vector3d k_rot{1.725, 1.725, 0.325};
  Eigen::Vector3d k_angvel{0.45, 0.45, 0.0825};
  // Battery, simplified for planning: a constant terminal voltage. quad.py's thrust ceiling and motor lag scale
  // with it ((V/V_nom)^2 and V_nom/V); its sag transient (I*R_int against PX4's 0.8 s voltage filter) is left out,
  // which is steady-state exact and negligible over an MPPI horizon. 14.8 V = BATT_NOM_V (quad.py's reference).
  double batt_nom_v{14.8};
  double batt_v{14.8};
  double gravity_m_s2{9.81};
  double physics_dt_s{0.01};                               // quad.py physics_dt (100 Hz substeps)
};

/**
 * @brief Plant state, held in quad.py's own frames so the port stays line-for-line comparable:
 * Z-up world (x east, y north), FLU body, q = body -> world. Use the NED accessors / fromNed() at the boundary.
 */
struct QuadState {
  Eigen::Vector3d pos_w{Eigen::Vector3d::Zero()};
  Eigen::Vector3d vel_w{Eigen::Vector3d::Zero()};
  Eigen::Quaterniond q_w{Eigen::Quaterniond::Identity()};
  Eigen::Vector3d omega_b{Eigen::Vector3d::Zero()};        // FLU body rates
  Eigen::Vector4d thrust{Eigen::Vector4d::Zero()};         // applied rotor thrusts, N
  // PX4 velocity loop state (world Z-up axes): integrator and the low-passed velocity derivative. A state made
  // from an estimate (fromNed) starts both at zero: PX4's own integrator is not observable from outside.
  Eigen::Vector3d vel_int{Eigen::Vector3d::Zero()};
  Eigen::Vector3d acc_lp{Eigen::Vector3d::Zero()};

  Eigen::Vector3d posNed() const { return {pos_w.y(), pos_w.x(), -pos_w.z()}; }
  Eigen::Vector3d velNed() const { return {vel_w.y(), vel_w.x(), -vel_w.z()}; }
  /** Body FRD -> NED rotation. */
  Eigen::Matrix3d rotNed() const;
  double yawNed() const;
};

/**
 * @brief Multirotor plant for MPPI rollouts, ported from dronehacks-sim quad.py: velocity loop (PX4's PID by
 * default, or quad.py's Lee loop) -> Lee geometric attitude/thrust -> rotor allocation (pseudo-inverse, floored
 * at 0) -> per-rotor first-order thrust lag -> rigid body with linear and angular drag, semi-implicit Euler with a
 * quaternion exponential update at physics_dt_s.
 *
 * Deterministic: one vehicle at the parameters above (no per-episode randomisation, no ground contact).
 */
class QuadPlant {
 public:
  explicit QuadPlant(const QuadPlantParams& p = QuadPlantParams{});

  const QuadPlantParams& params() const { return p_; }

  /** Level-or-not state from PX4-convention inputs; rotors at the hover allocation. */
  QuadState fromNed(const Eigen::Vector3d& pos_ned, const Eigen::Vector3d& vel_ned,
                    const Eigen::Quaterniond& q_ned_frd,
                    const Eigen::Vector3d& omega_frd = Eigen::Vector3d::Zero()) const;

  /**
   * @brief Hold a world-frame command for `duration_s` (whole physics substeps): NED velocity + NED yaw rate
   * (positive = clockwise seen from above), what PX4 / skybeta's flight_interface take. Internally this is
   * quad.py's yaw-aligned command re-expressed with the current heading at every substep, which the Lee
   * controller rotates straight back, so the vehicle tracks exactly this world velocity.
   */
  void stepWorld(QuadState& s, const Eigen::Vector3d& vel_ned, double yaw_rate_ned, double duration_s) const;

  /**
   * @brief quad.py's own action convention (after scaling): [v_fwd, v_right, v_down] m/s in the yaw-aligned
   * frame (NED axes) + NED yaw rate, held for `duration_s`. TorchQuad.step_control with v_max = 1, yaw_rate_max = 1.
   */
  void stepYawAligned(QuadState& s, const Eigen::Vector3d& vel_fwd_right_down, double yaw_rate_ned,
                      double duration_s) const;

  int substeps(double duration_s) const;

 private:
  /** One physics step; v_des_w is the desired world (Z-up) velocity, yaw_rate_flu the FLU yaw-rate command. */
  void physicsStep(QuadState& s, const Eigen::Vector3d& v_des_w, double yaw_rate_flu) const;

  QuadPlantParams p_;
  Eigen::Matrix<double, 6, 4> alloc_;
  Eigen::Matrix<double, 4, 6> alloc_pinv_;
  Eigen::Vector4d hover_thrust_;
  double thrust_max_eff_{12.6};
  double motor_alpha_{0.3};
};

}  // namespace maze_planning
