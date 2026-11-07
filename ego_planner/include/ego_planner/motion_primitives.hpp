#pragma once

#include <memory>
#include <string>
#include <vector>

#include "ego_planner/math_types.hpp"

namespace ego_planner {

enum class PrimitiveType {
  kHold = 0,
  kTakeoff = 1,
  kBSplineTrajectory = 2,
  kGatePass = 3,
  kPrecisionLand = 4,
};

class MotionPrimitive {
 public:
  virtual ~MotionPrimitive() = default;
  virtual StateSetpoint evaluate(double t_rel_s, const VehicleState& current_state) const = 0;
  virtual bool isCompleted(double t_rel_s, const VehicleState& current_state) const = 0;
  virtual PrimitiveType type() const = 0;
  virtual const char* name() const = 0;
  virtual double duration() const = 0;
};

// 1. HoldPrimitive: active zero-velocity / position-hold hover primitive
class HoldPrimitive : public MotionPrimitive {
 public:
  HoldPrimitive(const Vec3& hold_pos_ned, double hold_yaw_rad, double duration_s = 1e6);

  StateSetpoint evaluate(double t_rel_s, const VehicleState& current_state) const override;
  bool isCompleted(double t_rel_s, const VehicleState& current_state) const override;
  PrimitiveType type() const override { return PrimitiveType::kHold; }
  const char* name() const override { return "hold"; }
  double duration() const override { return duration_s_; }

 private:
  Vec3 hold_pos_ned_;
  double hold_yaw_rad_;
  double duration_s_;
};

// 2. TakeoffPrimitive: smooth minimum-jerk vertical climb + heading alignment
class TakeoffPrimitive : public MotionPrimitive {
 public:
  TakeoffPrimitive(const Vec3& start_pos_ned, double target_alt_m,
                   double start_yaw_rad, double target_yaw_rad,
                   double duration_s = 1.0);

  StateSetpoint evaluate(double t_rel_s, const VehicleState& current_state) const override;
  bool isCompleted(double t_rel_s, const VehicleState& current_state) const override;
  PrimitiveType type() const override { return PrimitiveType::kTakeoff; }
  const char* name() const override { return "takeoff"; }
  double duration() const override { return duration_s_; }

 private:
  Vec3 start_pos_ned_;
  double target_alt_m_;
  double start_yaw_rad_;
  double target_yaw_rad_;
  double duration_s_;
};

// 3. UniformBSplinePrimitive: EGO-Planner-V2 Cubic Uniform B-Spline in R^3 with lookahead yaw
class UniformBSplinePrimitive : public MotionPrimitive {
 public:
  UniformBSplinePrimitive() = default;
  UniformBSplinePrimitive(const std::vector<Vec3>& control_points, double knot_dt,
                          double yaw_lookahead_s = 0.35);

  void setControlPoints(const std::vector<Vec3>& control_points, double knot_dt);

  // Evaluate position, velocity, acceleration at time t_rel_s in [0, duration()]
  void evaluatePVA(double t_rel_s, Vec3& pos, Vec3& vel, Vec3& acc) const;

  StateSetpoint evaluate(double t_rel_s, const VehicleState& current_state) const override;
  bool isCompleted(double t_rel_s, const VehicleState& current_state) const override;
  PrimitiveType type() const override { return PrimitiveType::kBSplineTrajectory; }
  const char* name() const override { return "bspline"; }
  double duration() const override { return total_duration_s_; }

  const std::vector<Vec3>& controlPoints() const { return ctrl_pts_; }
  std::vector<Vec3>& controlPointsMutable() { return ctrl_pts_; }
  double knotDt() const { return knot_dt_; }
  int numSegments() const { return std::max(0, static_cast<int>(ctrl_pts_.size()) - 3); }

  // Construct a clamped cubic B-spline interpolating/approximating a polyline of waypoints
  // with given initial position/velocity and terminal position/velocity.
  static UniformBSplinePrimitive fromWaypoints(const std::vector<Vec3>& waypoints,
                                               const Vec3& init_vel_ned,
                                               const Vec3& end_vel_ned,
                                               double desired_speed_ms,
                                               double knot_dt = 0.12);

 private:
  std::vector<Vec3> ctrl_pts_;
  double knot_dt_{0.12};
  double total_duration_s_{0.0};
  double yaw_lookahead_s_{0.35};
};

// 4. GatePassPrimitive: explicit aperture-constrained gate crossing primitive
class GatePassPrimitive : public MotionPrimitive {
 public:
  GatePassPrimitive(const GateSpec& gate, const Vec3& refined_center_ned,
                    double pass_speed_ms = 2.6, double post_margin_m = 0.26);

  StateSetpoint evaluate(double t_rel_s, const VehicleState& current_state) const override;
  bool isCompleted(double t_rel_s, const VehicleState& current_state) const override;
  PrimitiveType type() const override { return PrimitiveType::kGatePass; }
  const char* name() const override { return "gate_pass"; }
  double duration() const override { return 2.5; }

 private:
  GateSpec gate_;
  Vec3 center_ned_;
  double pass_speed_ms_;
  double post_margin_m_;
};

// 5. PrecisionLandPrimitive: horizontal capture over pad/mark + controlled touchdown descent
class PrecisionLandPrimitive : public MotionPrimitive {
 public:
  PrecisionLandPrimitive(const Vec2& target_xy_ned, double capture_radius_m = 0.48,
                         double approach_alt_m = 0.38, double descent_rate_ms = 0.68,
                         double touchdown_alt_m = 0.052);

  StateSetpoint evaluate(double t_rel_s, const VehicleState& current_state) const override;
  bool isCompleted(double t_rel_s, const VehicleState& current_state) const override;
  PrimitiveType type() const override { return PrimitiveType::kPrecisionLand; }
  const char* name() const override { return "precision_land"; }
  double duration() const override { return 10.0; }

  Vec2 targetXY() const { return target_xy_ned_; }

 private:
  Vec2 target_xy_ned_;
  double capture_radius_m_;
  double approach_alt_m_;
  double descent_rate_ms_;
  double touchdown_alt_m_;
};

}  // namespace ego_planner
