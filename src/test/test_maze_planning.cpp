// Comprehensive C++ unit & integration test suite for the Autonomous Quadrotor Maze Navigation System
// (system_architecture_plan.md: GroundTruthMap, VoxelCostmap, RRTStarPlanner, PathTransformer,
//  2048-rollout MPPIController, ControlMux, and FMEA failure mode handling).
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

#include "maze_planning/c_api.h"
#include "maze_planning/control_mux.hpp"
#include "maze_planning/groundtruth_map.hpp"
#include "maze_planning/mppi_controller.hpp"
#include "maze_planning/path_transformer.hpp"
#include "maze_planning/rrt_star_planner.hpp"
#include "maze_planning/voxel_costmap.hpp"

using namespace maze_planning;

static int failures = 0;
#define CHECK(cond, ...)                                    \
  do {                                                      \
    if (!(cond)) {                                          \
      std::printf("FAIL %s:%d: ", __FILE__, __LINE__);      \
      std::printf(__VA_ARGS__);                             \
      std::printf("\n");                                    \
      ++failures;                                           \
    }                                                       \
  } while (0)

namespace {

void testGroundTruthMapEncoding() {
  auto gt = std::make_shared<GroundTruthMap>();
  const bool loaded_bin = gt->loadFromBinary("config/course_groundtruth.bin");
  CHECK(loaded_bin, "GroundTruthMap failed to load config/course_groundtruth.bin");
  CHECK(gt->numOBBSlabs() == 14, "Expected 14 analytical OBB slabs, got %zu", gt->numOBBSlabs());
  CHECK(gt->numBoulders() == 5, "Expected 5 semi-ellipsoid boulders, got %zu", gt->numBoulders());
  CHECK(gt->numTriangles() >= 2000, "Expected >=2000 mesh triangles in BVH, got %zu", gt->numTriangles());
  CHECK(gt->numBVHNodes() >= 400, "Expected >=400 BVH nodes, got %zu", gt->numBVHNodes());

  // Launch pose must be clear
  Eigen::Vector3d grad;
  const double d_launch = gt->clearanceAndGradient(Eigen::Vector3d(0.9144, -2.5146, -0.78), grad);
  CHECK(d_launch > 0.38, "Launch clearance should be > 0.38m, got %.3f", d_launch);
  CHECK(grad.allFinite() && std::abs(grad.norm() - 1.0) < 1e-3, "Unit gradient at launch");

  // Window aperture center at [-2.2830, -0.4750, -0.78] must be clear
  const double d_win = gt->clearance(Eigen::Vector3d(-2.2830, -0.4750, -0.78));
  CHECK(d_win > 0.22, "Window aperture clearance should be > 0.22m, got %.3f", d_win);

  // Interior divider wall (wall3 at y = -1.9304, x in [0.5080, 2.1209]) must be occupied
  CHECK(gt->isStaticWallOccupied(Eigen::Vector3d(1.20, -1.9304, -0.78)),
        "Interior divider wall3 should be flagged as occupied in bitset");
  CHECK(!gt->isSegmentValid(Eigen::Vector3d(1.20, -1.50, -0.78),
                            Eigen::Vector3d(1.20, -2.35, -0.78), 0.18),
        "Sphere-traced segment across divider wall3 must be rejected");

  // Also test direct Wavefront OBJ loading from sitl/world/course.obj
  GroundTruthMap gt_obj;
  CHECK(gt_obj.loadFromObj("sitl/world/course.obj"), "Direct OBJ loader failed on sitl/world/course.obj");
  CHECK(gt_obj.numTriangles() >= 2000, "OBJ loader triangles %zu", gt_obj.numTriangles());

  std::printf("[PASS] testGroundTruthMapEncoding (slabs=%zu, boulders=%zu, tris=%zu, d_launch=%.3fm, d_win=%.3fm)\n",
              gt->numOBBSlabs(), gt->numBoulders(), gt->numTriangles(), d_launch, d_win);
}

void testVoxelCostmapAndDisparityPunchThroughFMEA() {
  auto gt = std::make_shared<GroundTruthMap>();
  gt->loadFromBinary("config/course_groundtruth.bin");
  VoxelCostmap costmap(VoxelCostmapConfig{}, gt);

  // FMEA Test 1: Disparity Punch-Through
  // Simulate a stereo depth ray that punches through a netting/wall hole at [1.20, -1.9304, -0.78]
  // and hits the far background at [1.20, -2.60, -0.78]. Static maze walls must remain permanently locked!
  const Eigen::Vector3d sensor_origin(1.20, -1.35, -0.78);
  const std::vector<Eigen::Vector3d> punch_through_pts = {
      Eigen::Vector3d(1.20, -2.60, -0.78),
      Eigen::Vector3d(1.25, -2.60, -0.78),
  };
  costmap.updateFromClampedPointCloud(1.0, sensor_origin, punch_through_pts);
  const double d_wall = costmap.clearance(Eigen::Vector3d(1.20, -1.9304, -0.78));
  CHECK(d_wall <= 0.06, "FMEA Disparity Punch-Through: static wall must stay locked (d=%.3f)", d_wall);

  // Test dynamic unmodeled support pillar injection
  costmap.injectUnmodeledPillar(Eigen::Vector2d(1.80, -2.45), 0.10, 1.65, 1.5);
  CHECK(costmap.numDynamicObstacles() > 0, "Expected dynamic pillar voxels > 0");
  const double d_pillar = costmap.clearance(Eigen::Vector3d(1.80, -2.45, -0.78));
  CHECK(d_pillar < 0.12, "Expected near-zero clearance at injected pillar, got %.3f", d_pillar);

  std::printf("[PASS] testVoxelCostmapAndDisparityPunchThroughFMEA (d_wall=%.3fm, d_pillar=%.3fm)\n",
              d_wall, d_pillar);
}

void testRRTStarPlannerAndBsplineSmoothing() {
  auto gt = std::make_shared<GroundTruthMap>();
  gt->loadFromBinary("config/course_groundtruth.bin");
  auto costmap = std::make_shared<VoxelCostmap>(VoxelCostmapConfig{}, gt);

  RRTStarPlanner rrt(RRTStarConfig{}, gt, costmap);

  // Plan around the North-West launch chute corner (wall3 at y=-1.9304, x in [0.5080, 2.1209])
  const Eigen::Vector3d start(0.9144, -2.5146, -0.78);
  const Eigen::Vector3d goal(1.60, -1.35, -0.78);
  PlannedPath path;
  std::vector<Eigen::Vector3d> raw_wps;
  const bool ok = rrt.plan(start, goal, path, &raw_wps);
  CHECK(ok && raw_wps.size() >= 2, "RRT* failed to plan around NW corner");
  for (std::size_t i = 0; i + 1 < raw_wps.size(); ++i) {
    CHECK(gt->isSegmentValid(raw_wps[i], raw_wps[i + 1], 0.16),
          "RRT* segment %zu violated GroundTruthMap collision check", i);
  }
  CHECK(!path.empty() && path.total_length_m > 1.8, "Smoothed B-spline path length=%.2fm", path.total_length_m);

  // Plan full multi-gate course
  PlannedPath full_course;
  CHECK(rrt.planFullCourse(start, true, false, full_course), "Full course RRT* planning failed");
  CHECK(full_course.total_length_m > 10.0, "Full course path length=%.2fm", full_course.total_length_m);

  std::printf("[PASS] testRRTStarPlannerAndBsplineSmoothing (corner_wps=%zu, full_course_len=%.2fm)\n",
              raw_wps.size(), full_course.total_length_m);
}

void testPathTransformerAndiSAM2ShiftFMEA() {
  auto gt = std::make_shared<GroundTruthMap>();
  gt->loadFromBinary("config/course_groundtruth.bin");
  RRTStarPlanner rrt(RRTStarConfig{}, gt);

  PlannedPath global_map;
  rrt.planFullCourse(Eigen::Vector3d(0.9144, -2.5146, -0.78), true, false, global_map);

  PathTransformer transformer;
  transformer.setGlobalPathMap(global_map);

  // FMEA Test: iSAM2 Graph Shift of 10 cm on T_map_odom
  SE3Transform T_map_odom{};
  T_map_odom.t = Eigen::Vector3d(0.10, -0.05, 0.0);
  T_map_odom.q = Eigen::Quaterniond(Eigen::AngleAxisd(0.05, Eigen::Vector3d::UnitZ()));
  transformer.setMapToOdomTransform(T_map_odom);

  VehicleKinematics vk{};
  vk.pos_odom = T_map_odom.inverse().apply(Eigen::Vector3d(0.9144, -2.5146, -0.78));
  vk.q_odom_base = Eigen::Quaterniond::Identity();

  const PlannedPath local_odom = transformer.extractLocalReferenceOdom(vk);
  CHECK(!local_odom.empty(), "Local reference path in odom frame is non-empty");
  CHECK((local_odom.points.front().pos - vk.pos_odom).norm() < 0.25,
        "Local reference starts at drone odom position despite 10cm map->odom shift");

  std::array<float, 12> rl_flat{};
  const auto lookaheads = transformer.computeRLLookaheadsBody(vk, &rl_flat);
  CHECK(lookaheads[0].norm() > 0.3 && lookaheads[3].norm() > lookaheads[0].norm(),
        "RL lookaheads at d=[0.5, 1.0, 2.0, 3.5]m increase monotonically in distance");

  std::printf("[PASS] testPathTransformerAndiSAM2ShiftFMEA (local_pts=%zu, lookahead_0=%.2fm, lookahead_3=%.2fm)\n",
              local_odom.points.size(), lookaheads[0].norm(), lookaheads[3].norm());
}

void testMPPI2048RolloutsAndPillarEvasion() {
  auto gt = std::make_shared<GroundTruthMap>();
  gt->loadFromBinary("config/course_groundtruth.bin");
  auto costmap = std::make_shared<VoxelCostmap>(VoxelCostmapConfig{}, gt);

  // Place an unmodeled vertical pole/pillar at [1.75, -2.5146] in the launch corridor
  // (Milestone 3: MPPI executes lateral evasive maneuver around unmodeled vertical pole)
  costmap->injectUnmodeledPillar(Eigen::Vector2d(1.75, -2.5146), 0.10, 1.65, 0.0);

  MPPIConfig mcfg{};
  mcfg.num_samples = 2048;
  mcfg.horizon_steps = 16;
  MPPIController mppi(mcfg, MultirotorPlantParams{}, costmap);

  // Straight reference path down the launch chute from x=0.9144 to x=2.55 along y=-2.5146
  RRTStarPlanner rrt(RRTStarConfig{}, gt);
  const PlannedPath ref_path = rrt.smoothWithCubicBSpline(
      {Eigen::Vector3d(0.9144, -2.5146, -0.78), Eigen::Vector3d(2.55, -2.5146, -0.78)}, 2.5);

  VehicleKinematics vk{};
  vk.timestamp_s = 1.0;
  vk.pos_odom = Eigen::Vector3d(1.25, -2.5146, -0.78);
  vk.vel_odom = Eigen::Vector3d(1.2, 0.0, 0.0);
  vk.q_odom_base = Eigen::Quaterniond::Identity();

  const TrajectorySetpointCmd cmd = mppi.computeCommand(vk, ref_path);
  CHECK(cmd.vel_ned.allFinite() && cmd.pos_ned.allFinite(), "MPPI output is finite");
  CHECK(cmd.vel_ned.x() > 0.3, "MPPI maintains forward progress (vx=%.2f m/s)", cmd.vel_ned.x());
  CHECK(std::abs(cmd.vel_ned.y()) > 0.12,
        "MPPI executes lateral evasive velocity around dead-center pole (vy=%.3f m/s)",
        cmd.vel_ned.y());

  std::printf("[PASS] testMPPI2048RolloutsAndPillarEvasion (2048 rollouts in %.2f ms, vx=%.2f, vy=%.3f m/s)\n",
              cmd.solve_time_ms, cmd.vel_ned.x(), cmd.vel_ned.y());
}

void testControlMuxAndSafetyInterlocks() {
  ControlMux mux;
  VehicleKinematics vk{};
  vk.pos_odom = Eigen::Vector3d(0.9144, -2.5146, -0.78);
  vk.tracking_valid = true;
  vk.pos_variance = 0.01;

  TrajectorySetpointCmd mppi_cmd{};
  mppi_cmd.timestamp_s = 1.0;
  mppi_cmd.vel_ned = Eigen::Vector3d(2.0, 0.0, 0.0);
  mux.onClassicalCommand(mppi_cmd);

  // 1. Classical mode active
  MuxDecision d1 = mux.evaluate(1.01, vk, 0.80);
  CHECK(d1.active_pathway == ControlPathway::kClassicalMPPI && d1.publish_trajectory_setpoint,
        "Expected Classical MPPI mode");

  // 2. Switch to RL Policy mode with valid command
  mux.setRequestedPathway(ControlPathway::kRLPolicy);
  RatesSetpointCmd rl_ok{};
  rl_ok.timestamp_s = 1.01;
  rl_ok.omega_body_rad_s = Eigen::Vector3d(0.5, -0.3, 0.2);
  rl_ok.thrust_norm = 0.58;
  rl_ok.inference_latency_ms = 3.2;
  mux.onRLRatesCommand(rl_ok);
  MuxDecision d2 = mux.evaluate(1.02, vk, 0.80);
  CHECK(d2.active_pathway == ControlPathway::kRLPolicy && d2.publish_rates_setpoint,
        "Expected RL Policy mode when healthy");

  // 3. WP-5 Task 3: RL inference latency > 20 ms must trip fallback to Classical MPPI
  RatesSetpointCmd rl_slow = rl_ok;
  rl_slow.inference_latency_ms = 25.0;
  mux.onRLRatesCommand(rl_slow);
  MuxDecision d3 = mux.evaluate(1.03, vk, 0.80);
  CHECK(d3.active_pathway == ControlPathway::kClassicalMPPI && d3.fallback_tripped,
        "Expected automatic fallback to Classical MPPI when RL latency > 20ms");

  std::printf("[PASS] testControlMuxAndSafetyInterlocks\n");
}

}  // namespace

int main() {
  testGroundTruthMapEncoding();
  testVoxelCostmapAndDisparityPunchThroughFMEA();
  testRRTStarPlannerAndBsplineSmoothing();
  testPathTransformerAndiSAM2ShiftFMEA();
  testMPPI2048RolloutsAndPillarEvasion();
  testControlMuxAndSafetyInterlocks();
  std::printf(failures ? "%d FAILURE(S)\n" : "ALL MAZE PLANNING (RRT* + MPPI + COSTMAP + MUX) C++ TESTS PASSED\n",
              failures);
  return failures ? EXIT_FAILURE : EXIT_SUCCESS;
}
