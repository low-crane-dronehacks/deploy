#include <cassert>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

#include "ego_planner/bspline_optimizer.hpp"
#include "ego_planner/c_api.h"
#include "ego_planner/global_planner_interface.hpp"
#include "ego_planner/grid_map.hpp"
#include "ego_planner/local_controller_interface.hpp"
#include "ego_planner/mission_manager.hpp"
#include "ego_planner/motion_primitives.hpp"
#include "ego_planner/mppi_controller.hpp"
#include "ego_planner/quadrotor_dynamics.hpp"
#include "ego_planner/rrt_star_planner.hpp"

namespace {

void check(bool cond, const char* msg) {
  if (!cond) {
    std::cerr << "FAIL: " << msg << std::endl;
    std::exit(1);
  }
}

void testGridMapAndPrior(const std::string& prior_path) {
  ego_planner::GridMap grid;
  check(grid.loadPriorBinary(prior_path), "loadPriorBinary succeeded");

  ego_planner::Vec3 grad{};
  const double d_launch =
      grid.getDistanceAndGradient(ego_planner::Vec3{0.9144, -2.5146, -0.78}, grad);
  std::cout << "d_launch = " << d_launch << " m\n";
  check(d_launch > 0.35, "launch pose clearance > 0.35 m");
  check(std::isfinite(grad.x) && std::isfinite(grad.y) && std::isfinite(grad.z),
        "ESDF gradient finite");

  std::vector<float> depth(128 * 80, 0.30f);  // 0.30 * 5.0m = 1.50 m
  grid.updateFromDepth(depth.data(), 128, 80, 72.0, 50.0, 5.0,
                       ego_planner::Vec3{0.9144, -2.5146, -0.78},
                       ego_planner::Quat::fromEulerZYX(0.0, 0.0, 0.0), 4);
  check(grid.liveUpdatesCount() == 1, "grid liveUpdatesCount incremented");
  const double d_wall = grid.getDistance(ego_planner::Vec3{0.9144 + 1.50, -2.5146, -0.78});
  check(d_wall < 0.15, "raycasted wall voxel has near-zero ESDF distance");
  std::cout << "[PASS] testGridMapAndPrior (d_launch=" << d_launch << "m, d_wall=" << d_wall
            << "m)\n";
}

void testMotionPrimitives() {
  ego_planner::VehicleState st{};
  st.pos_ned = ego_planner::Vec3{0.9144, -2.5146, -0.08};

  ego_planner::TakeoffPrimitive takeoff(st.pos_ned, 0.78, 0.0, 0.0, 0.90);
  const auto sp0 = takeoff.evaluate(0.45, st);
  check(sp0.pos_ned.z < -0.20 && sp0.vel_ned.z < 0.0, "TakeoffPrimitive climbs upward (vz < 0)");

  ego_planner::PrecisionLandPrimitive land(ego_planner::Vec2{-0.7925, -2.0422});
  st.pos_ned = ego_planner::Vec3{-0.7925, -2.0422, -0.30};
  const auto sp_land = land.evaluate(0.50, st);
  check(sp_land.vel_ned.z > 0.0, "PrecisionLandPrimitive descends (vz > 0)");
  std::cout << "[PASS] testMotionPrimitives\n";
}

void testBSplineOptimizer(const std::string& prior_path) {
  ego_planner::GridMap grid;
  check(grid.loadPriorBinary(prior_path), "loadPriorBinary for optimizer");
  ego_planner::BSplineOptimizer opt;

  const ego_planner::Vec3 start{0.9144, -2.5146, -0.78};
  const std::vector<ego_planner::Vec3> anchors = {
      ego_planner::Vec3{2.35, -2.48, -0.78},
      ego_planner::Vec3{2.80, -2.30, -0.78},
      ego_planner::Vec3{2.55, -1.45, -0.78},
  };
  auto spline = opt.planAndOptimizeTrajectory(
      grid, start, ego_planner::Vec3{0.0, 0.0, 0.0}, anchors,
      ego_planner::Vec3{0.0, 0.0, 0.0}, 3.2);
  check(spline.duration() > 0.4, "B-spline duration > 0.4 s");

  ego_planner::VehicleState dummy{};
  double min_d = 10.0;
  for (double t = 0.0; t <= spline.duration(); t += 0.05) {
    const auto sp = spline.evaluate(t, dummy);
    min_d = std::min(min_d, grid.getDistance(sp.pos_ned));
  }
  check(min_d > 0.25, "Optimized B-spline maintains > 0.25 m obstacle clearance");
  std::cout << "[PASS] testBSplineOptimizer (duration=" << spline.duration()
            << "s, min_clearance=" << min_d << "m)\n";
}

void testMissionManagerAndCAPI(const std::string& prior_path) {
  EgoPlannerCConfig cfg{};
  cfg.arm_authorized = 1;
  cfg.enable_return_leg = 0;
  cfg.prefer_window = 1;
  cfg.cruise_alt_m = 0.78;
  cfg.desired_speed_ms = 3.35;
  cfg.replan_period_s = 0.20;
  cfg.prior_bin_path = prior_path.c_str();

  EgoPlannerHandle* h = ego_planner_create(&cfg);
  check(h != nullptr, "ego_planner_create returned handle");

  EgoPlannerCState st{};
  st.pos_ned[0] = 0.9144;
  st.pos_ned[1] = -2.5146;
  st.pos_ned[2] = -0.08;
  st.quat_wxyz[0] = 1.0;
  st.rangefinder_m = 0.08;
  st.armed = 0;
  st.offboard = 0;
  st.landed = 1;
  st.pose_valid = 1;

  EgoPlannerCOutput out{};
  ego_planner_on_estimator_health(h, 0.1, 1);
  ego_planner_on_depth_timestamp(h, 0.1);
  ego_planner_step(h, 0.1, &st, &out);
  check(std::string(out.phase) == "WAIT", "Initial phase is WAIT while streaming lead setpoints");

  ego_planner_on_estimator_health(h, 1.5, 1);
  ego_planner_on_depth_timestamp(h, 1.5);
  ego_planner_step(h, 1.5, &st, &out);
  check(std::string(out.phase) == "ARM", "Transitions to ARM after stream lead");

  st.armed = 1;
  st.offboard = 1;
  st.landed = 0;
  ego_planner_on_estimator_health(h, 1.7, 1);
  ego_planner_on_depth_timestamp(h, 1.7);
  ego_planner_step(h, 1.7, &st, &out);
  ego_planner_step(h, 1.8, &st, &out);
  check(std::string(out.phase) == "FLY", "Transitions to FLY when armed & offboard");
  check(std::string(out.active_leg) == "run_course", "Active leg is run_course");
  check(out.body_action[2] < 0.0, "Commands upward climb at launch (body_action[2] < 0)");

  ego_planner_destroy(h);
  std::cout << "[PASS] testMissionManagerAndCAPI\n";
}

void testQuadrotorDynamics() {
  ego_planner::QuadrotorDynamics plant;
  ego_planner::QuadrotorState st{};
  st.pos_ned = ego_planner::Vec3{0.0, 0.0, -1.0};
  st.vel_ned = ego_planner::Vec3{0.0, 0.0, 0.0};
  st.q_wxyz = ego_planner::Quat{1.0, 0.0, 0.0, 0.0};

  ego_planner::QuadrotorAction act{};
  act.vel_cmd_yaw_aligned = ego_planner::Vec3{1.5, 0.0, 0.0}; // 1.5 m/s forward command
  act.yaw_rate_cmd_rad_s = 0.0;

  for (int i = 0; i < 5; ++i) {
    st = plant.step(st, act, 0.10);
  }
  check(st.vel_ned.x > 0.0, "Quadrotor accelerates forward under forward command");
  check(std::isfinite(st.pos_ned.x) && std::isfinite(st.pos_ned.z),
        "Quadrotor state remains finite");
  std::cout << "[PASS] testQuadrotorDynamics (vx_next=" << st.vel_ned.x << " m/s)\n";
}

void testRRTStarPlanner(const std::string& prior_path) {
  ego_planner::GridMap grid;
  check(grid.loadPriorBinary(prior_path), "loadPriorBinary for RRT*");

  ego_planner::RRTStarPlanner rrt;
  // Test steering
  ego_planner::Vec3 a{0.0, 0.0, -0.8};
  ego_planner::Vec3 b{1.0, 0.0, -0.8};
  ego_planner::Vec3 steered = rrt.steer(a, b, 0.25);
  check((steered - a).norm() <= 0.2501, "RRT* steer clamps step size");

  // Test line-of-sight planning in launch corridor
  ego_planner::Vec3 start{0.9144, -2.5146, -0.78};
  ego_planner::Vec3 goal{2.00, -2.5146, -0.78};
  std::vector<ego_planner::Vec3> path;
  bool ok = rrt.plan(start, goal, grid, path);
  check(ok && path.size() >= 2, "RRT* successfully plans in corridor");

  ego_planner::GlobalTrajectory traj;
  check(rrt.generateTrajectory(path, 2.25, traj), "RRT* generates trajectory");
  check(traj.total_duration_s > 0.1, "RRT* trajectory duration > 0.1s");
  std::cout << "[PASS] testRRTStarPlanner (pts=" << path.size() << ", duration=" << traj.total_duration_s << "s)\n";
}

void testMPPIController(const std::string& prior_path) {
  ego_planner::GridMap grid;
  check(grid.loadPriorBinary(prior_path), "loadPriorBinary for MPPI");

  ego_planner::MPPIConfig mppi_cfg;
  mppi_cfg.num_samples = 32; // small sample count for fast CPU unit test
  mppi_cfg.horizon_steps = 10;
  ego_planner::MPPIController mppi(mppi_cfg);

  ego_planner::GlobalTrajectory traj;
  traj.points.push_back(ego_planner::GlobalTrajectoryPoint{
      ego_planner::Vec3{0.9144, -2.5146, -0.78}, ego_planner::Vec3{2.0, 0.0, 0.0},
      ego_planner::Vec3{0.0, 0.0, 0.0}, 0.0, 0.0, 0.0, 0.0, 0.0});
  traj.points.push_back(ego_planner::GlobalTrajectoryPoint{
      ego_planner::Vec3{2.0, -2.5146, -0.78}, ego_planner::Vec3{2.0, 0.0, 0.0},
      ego_planner::Vec3{0.0, 0.0, 0.0}, 0.0, 0.0, 0.5, 1.08, 0.0});
  traj.total_duration_s = 0.5;
  traj.total_length_m = 1.08;

  ego_planner::VehicleState st{};
  st.pos_ned = ego_planner::Vec3{0.9144, -2.5146, -0.78};
  st.vel_ned = ego_planner::Vec3{0.0, 0.0, 0.0};
  st.q_wxyz = ego_planner::Quat{1.0, 0.0, 0.0, 0.0};

  ego_planner::ControlOutput cmd = mppi.computeCommand(st, traj, grid, 0.10);
  check(cmd.stream_velocity, "MPPI outputs stream_velocity=true");
  check(std::isfinite(cmd.vel_ned.x) && std::isfinite(cmd.vel_ned.y) && std::isfinite(cmd.vel_ned.z),
        "MPPI velocity setpoints are finite");
  std::cout << "[PASS] testMPPIController (vel_sp=(" << cmd.vel_ned.x << ", "
            << cmd.vel_ned.y << ", " << cmd.vel_ned.z << "))\n";
}

}  // namespace

int main(int argc, char** argv) {
  std::string prior_path = "config/arena_prior.bin";
  if (argc > 1) {
    prior_path = argv[1];
  }
  testGridMapAndPrior(prior_path);
  testMotionPrimitives();
  testBSplineOptimizer(prior_path);
  testMissionManagerAndCAPI(prior_path);
  testQuadrotorDynamics();
  testRRTStarPlanner(prior_path);
  testMPPIController(prior_path);
  std::cout << "ALL C++ EGO-PLANNER-V2 & AUTONOMY TESTS PASSED\n";
  return 0;
}
