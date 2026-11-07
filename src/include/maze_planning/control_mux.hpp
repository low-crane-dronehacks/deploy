#pragma once

#include <string>

#include "maze_planning/types.hpp"

namespace maze_planning {

enum class ControlPathway {
  kClassicalMPPI = 0,   // Routes /control/trajectory_setpoint -> /fmu/in/trajectory_setpoint
  kRLPolicy = 1,        // Routes /control/rates_setpoint -> /fmu/in/vehicle_rates_setpoint
  kSafetyHold = 2       // Fallback stop-and-hover / loiter hold
};

struct ControlMuxConfig {
  double max_rl_latency_ms{20.0};            // WP-5 Task 3: fallback if inference > 20 ms
  double max_rl_command_age_s{0.08};         // watchdog timeout on /control/rates_setpoint
  double max_roll_pitch_rate_rad_s{M_PI};    // Section 4.2: [-180, 180] deg/s
  double max_yaw_rate_rad_s{0.5 * M_PI};     // Section 4.2: [-90, 90] deg/s
  double min_thrust_norm{0.10};              // Section 4.2: [0.1, 0.9]
  double max_thrust_norm{0.90};
  double max_pos_variance{0.25};             // WP-4 Task 4 & FMEA VIO Tracking Loss threshold
  double collision_lookahead_s{0.30};        // WP-4 Task 4: impact prediction < 0.3s -> hold
};

struct MuxDecision {
  ControlPathway active_pathway{ControlPathway::kClassicalMPPI};
  bool publish_trajectory_setpoint{true};
  bool publish_rates_setpoint{false};
  TrajectorySetpointCmd trajectory_cmd{};
  RatesSetpointCmd rates_cmd{};
  bool fallback_tripped{false};
  std::string status_reason{"classical_mppi"};
};

/**
 * @brief Control Mode Multiplexer & Safety Interlock (Section 2.2, 2.3, WP-4, WP-5, Section 6 FMEA).
 *
 * Switches between Classical MPPI (`TrajectorySetpoint`) and RL Policy (`VehicleRatesSetpoint`),
 * with automatic watchdog fallback to Classical MPPI / Hold if RL inference exceeds 20 ms,
 * outputs violate flight envelope limits, VIO covariance spikes, or collision is predicted < 0.3s.
 */
class ControlMux {
 public:
  explicit ControlMux(const ControlMuxConfig& cfg = ControlMuxConfig{});

  void setRequestedPathway(ControlPathway pathway) { requested_pathway_ = pathway; }
  ControlPathway requestedPathway() const { return requested_pathway_; }

  void onClassicalCommand(const TrajectorySetpointCmd& cmd);
  void onRLRatesCommand(const RatesSetpointCmd& cmd);

  MuxDecision evaluate(double now_s,
                       const VehicleKinematics& state_odom,
                       double current_clearance_m);

 private:
  bool isRLCommandValid(double now_s, std::string& out_reason) const;

  ControlMuxConfig cfg_;
  ControlPathway requested_pathway_{ControlPathway::kClassicalMPPI};
  TrajectorySetpointCmd last_classical_cmd_{};
  RatesSetpointCmd last_rl_cmd_{};
  bool has_classical_{false};
  bool has_rl_{false};
};

}  // namespace maze_planning
