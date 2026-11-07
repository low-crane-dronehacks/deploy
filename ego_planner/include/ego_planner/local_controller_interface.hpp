#pragma once

#include <memory>
#include <string>

#include "ego_planner/global_planner_interface.hpp"
#include "ego_planner/grid_map.hpp"
#include "ego_planner/math_types.hpp"
#include "ego_planner/motion_primitives.hpp"

namespace ego_planner {

/**
 * @brief Abstract Base Class for Local Tracking Controllers.
 * Implement this interface to plug in MPPI, NMPC, Geometric Tracking, etc.
 */
class LocalControllerBase {
 public:
  virtual ~LocalControllerBase() = default;

  virtual std::string name() const = 0;

  /**
   * @brief Compute the local control command given vehicle state, reference trajectory, and ESDF.
   * @param state Current estimated vehicle state (NED).
   * @param ref_traj Global reference trajectory to track.
   * @param grid_map Continuous ESDF obstacle grid for reactive collision avoidance.
   * @param dt Sampling time delta since last control loop (s).
   * @return ControlOutput containing velocity setpoints and command flags.
   */
  virtual ControlOutput computeCommand(const VehicleState& state,
                                       const GlobalTrajectory& ref_traj,
                                       const GridMap& grid_map,
                                       double dt) = 0;

  virtual void reset() {}
};

/**
 * @brief Production Primitive Tracking Controller (EGO-Planner-v2 Arc Cursor + Reactive Push).
 * Default working implementation.
 */
class PrimitiveTracker : public LocalControllerBase {
 public:
  explicit PrimitiveTracker(double desired_speed_ms = 2.25)
      : desired_speed_(desired_speed_ms) {}

  std::string name() const override { return "primitive_tracker"; }

  ControlOutput computeCommand(const VehicleState& state,
                               const GlobalTrajectory& ref_traj,
                               const GridMap& grid_map,
                               double dt) override;

  void reset() override {
    cursor_arc_m_ = 0.0;
  }

  void setDesiredSpeed(double speed_ms) { desired_speed_ = speed_ms; }
  double desiredSpeed() const { return desired_speed_; }

 private:
  double desired_speed_{2.25};
  double cursor_arc_m_{0.0};
};

}  // namespace ego_planner
