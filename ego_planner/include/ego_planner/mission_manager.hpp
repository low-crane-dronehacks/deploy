#pragma once

#include <array>
#include <memory>
#include <string>
#include <vector>

#include "ego_planner/bspline_optimizer.hpp"
#include "ego_planner/grid_map.hpp"
#include "ego_planner/math_types.hpp"
#include "ego_planner/motion_primitives.hpp"

namespace ego_planner {

struct MissionConfig {
  bool arm_authorized{false};
  bool enable_return_leg{false};
  bool prefer_window{true};
  double control_dt_s{0.10};
  double stream_lead_s{1.0};
  double arm_wait_s{15.0};
  double depth_stale_s{0.50};
  double depth_lost_land_s{3.0};
  double health_bad_land_s{1.0};
  double max_flight_s{180.0};
  double geofence_m{8.0};
  double airborne_alt_m{0.30};
  double cruise_alt_m{0.78};
  double desired_speed_ms{2.25};
  double replan_period_s{0.20};
  std::string planner_type{"bspline"};   // "bspline" | "rrt_star"
  std::string controller_type{"ego"};    // "ego" (primitive tracker) | "mppi"
  std::array<double, 3> v_max{{5.0, 3.0, 2.0}};
  double yaw_rate_max{1.0471975511965976};
  Vec2 launch_mark_ned{0.9144, -2.5146};
  Vec2 pad_center_ned{-0.7925, -2.0422};
  double pad_radius_m{0.9906};
  double return_radius_m{0.6096};
  double return_hold_s{3.0};
  double cp_margin_m{0.15};
  std::string prior_bin_path{};
};

// Abstract base class for hierarchical mission legs (RunCourseLeg, ReturnHomeLeg, etc.)
class MissionLeg {
 public:
  virtual ~MissionLeg() = default;
  virtual void reset(const VehicleState& state, const GridMap& grid) = 0;
  virtual StateSetpoint step(double t_s, double dt_s, const VehicleState& state,
                             const GridMap& grid, const BSplineOptimizer& opt) = 0;
  virtual bool isFinished(const VehicleState& state) const = 0;
  virtual const char* legName() const = 0;
  virtual const char* activePrimitiveName() const = 0;
  virtual int nextCheckpointIndex() const = 0;
};

// Leg 1: Main DroneHacks 2026 Maze Run (Launch -> CP1..CP4 -> Pad Landing)
class RunCourseLeg : public MissionLeg {
 public:
  explicit RunCourseLeg(const MissionConfig& cfg, const std::vector<GateSpec>& gates);

  void reset(const VehicleState& state, const GridMap& grid) override;
  StateSetpoint step(double t_s, double dt_s, const VehicleState& state,
                     const GridMap& grid, const BSplineOptimizer& opt) override;
  bool isFinished(const VehicleState& /*state*/) const override { return leg_done_; }
  const char* legName() const override { return "run_course"; }
  const char* activePrimitiveName() const override {
    return active_primitive_ ? active_primitive_->name() : "none";
  }
  int nextCheckpointIndex() const override { return next_cp_; }

  static int detectSectorCheckpoint(const Vec2& xy_ned);

 private:
  enum class SubPhase {
    kTakeoff = 0,
    kNavigateGates = 1,
    kLandOnPad = 2,
    kFinished = 3,
  };

  void checkGateCrossings(const Vec2& prev_xy, const Vec2& cur_xy);
  void replanBSpline(double t_s, const VehicleState& state, const GridMap& grid,
                     const BSplineOptimizer& opt);
  std::vector<Vec3> buildRemainingAnchors(const VehicleState& state, const GridMap& grid) const;

  MissionConfig cfg_;
  std::vector<GateSpec> gates_;
  SubPhase sub_phase_{SubPhase::kTakeoff};
  int next_cp_{0};
  int waypoint_stage_{0};
  bool has_prev_xy_{false};
  Vec2 prev_xy_{0.0, 0.0};
  double prim_start_t_{0.0};
  double last_replan_t_{-1e9};
  int last_grid_updates_{-1};
  bool leg_done_{false};
  std::shared_ptr<MotionPrimitive> active_primitive_;
  UniformBSplinePrimitive active_spline_;
};

// Leg 2: Optional Return-Home Leg (Pad Wait -> Climb -> Cross Barrier -> Land at Launch Mark)
class ReturnHomeLeg : public MissionLeg {
 public:
  explicit ReturnHomeLeg(const MissionConfig& cfg);

  void reset(const VehicleState& state, const GridMap& grid) override;
  StateSetpoint step(double t_s, double dt_s, const VehicleState& state,
                     const GridMap& grid, const BSplineOptimizer& opt) override;
  bool isFinished(const VehicleState& /*state*/) const override { return leg_done_; }
  const char* legName() const override { return "return_home"; }
  const char* activePrimitiveName() const override {
    return active_primitive_ ? active_primitive_->name() : "none";
  }
  int nextCheckpointIndex() const override { return 5; }

 private:
  enum class SubPhase {
    kPadHold = 0,
    kClimbAndReturn = 1,
    kLandAtMark = 2,
    kHoldAtMark = 3,
    kFinished = 4,
  };

  MissionConfig cfg_;
  SubPhase sub_phase_{SubPhase::kPadHold};
  double phase_start_t_{0.0};
  double last_replan_t_{-1e9};
  bool leg_done_{false};
  std::shared_ptr<MotionPrimitive> active_primitive_;
  UniformBSplinePrimitive active_spline_;
};

// Top-Level Hierarchical Mission Manager + Safety Monitor + SE(3)/Velocity Tracker
class MissionManager {
 public:
  explicit MissionManager(const MissionConfig& cfg = MissionConfig{});

  void reset();
  void onDepthTimestamp(double t_s);
  void onDepthFrame(double t_s, const float* depth_norm, int width, int height,
                    double hfov_deg, double vfov_deg, double far_clip,
                    const Vec3& pos_ned, const Quat& q_wxyz, int stride = 4);
  void onEstimatorHealth(double t_s, bool healthy);

  // Step the mission state machine, active leg, and trajectory controller
  ControlOutput step(double t_s, const VehicleState& state);

  // Convert a NED velocity + yawspeed setpoint into normalized yaw-aligned body action [-1, 1]^4
  std::array<double, 4> velocityToBodyAction(const Vec3& vel_ned, double yawspeed_rad_s,
                                             const Quat& q_wxyz) const;

  const GridMap& gridMap() const { return grid_; }
  GridMap& gridMapMutable() { return grid_; }
  const BSplineOptimizer& optimizer() const { return optimizer_; }
  const MissionConfig& config() const { return cfg_; }
  const std::vector<std::string>& consumeEvents() { return events_; }
  void clearEvents() { events_.clear(); }

  static std::vector<GateSpec> defaultGates();

 private:
  enum class Phase {
    kWait = 0,
    kArm = 1,
    kFly = 2,
    kLand = 3,
    kDone = 4,
  };

  void setPhase(double t_s, Phase new_phase, const std::string& why);

  MissionConfig cfg_;
  GridMap grid_;
  BSplineOptimizer optimizer_;
  std::vector<GateSpec> gates_;
  std::vector<std::unique_ptr<MissionLeg>> legs_;
  std::size_t active_leg_idx_{0};

  Phase phase_{Phase::kWait};
  std::string why_{"starting"};
  std::string result_{""};
  double t_phase_start_{-1.0};
  double last_step_t_{-1.0};
  double depth_t_{-1e9};
  double health_t_{-1e9};
  bool healthy_{false};
  bool airborne_{false};
  bool leg_initialized_{false};
  Vec3 home_ned_{0.0, 0.0, 0.0};
  Vec3 arena_offset_ned_{0.0, 0.0, 0.0};
  std::vector<std::string> events_;
};

}  // namespace ego_planner
