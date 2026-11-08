// Comprehensive C++ unit & integration test suite for the Autonomous Quadrotor Maze Navigation System
// (system_architecture_plan.md: GroundTruthMap, VoxelCostmap, RRTStarPlanner, PathTransformer,
//  MPPIController on the quad.py plant (QuadPlant), ControlMux, and FMEA failure mode handling).
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

#include "maze_planning/astar_planner.hpp"
#include "maze_planning/c_api.h"
#include "maze_planning/control_mux.hpp"
#include "maze_planning/course_progress.hpp"
#include "maze_planning/groundtruth_map.hpp"
#include "maze_planning/mppi_controller.hpp"
#include "maze_planning/path_transformer.hpp"
#include "maze_planning/quad_plant.hpp"
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

void testMPPIRolloutsAndPillarEvasion() {
  auto gt = std::make_shared<GroundTruthMap>();
  gt->loadFromBinary("config/course_groundtruth.bin");
  auto costmap = std::make_shared<VoxelCostmap>(VoxelCostmapConfig{}, gt);

  // Place an unmodeled vertical pole/pillar at [1.75, -2.5146] in the launch corridor
  // (Milestone 3: MPPI executes lateral evasive maneuver around unmodeled vertical pole)
  costmap->injectUnmodeledPillar(Eigen::Vector2d(1.75, -2.5146), 0.10, 1.65, 0.0);

  MPPIConfig mcfg{};   // defaults: 1024 rollouts x 16 steps through the quad.py plant
  MPPIController mppi(mcfg, QuadPlantParams{}, costmap);

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

  std::printf("[PASS] testMPPIRolloutsAndPillarEvasion (%d rollouts in %.2f ms, vx=%.2f, vy=%.3f m/s)\n",
              mcfg.num_samples, cmd.solve_time_ms, cmd.vel_ned.x(), cmd.vel_ned.y());
}

#include "quad_reference.inc"   // generated by scripts/gen_quad_reference.py from dronehacks-sim quad.py

// The C++ plant must reproduce quad.py (float32 torch vs double here: allow a few mm / mm/s).
void testQuadPlantMatchesQuadPy() {
  QuadPlantParams pp = QuadPlantParams::quadPy();   // quad.py's own velocity loop, as in the generator
  pp.batt_v = 16.8;   // the generator's battery: full, no internal resistance
  QuadPlant plant(pp);
  const Eigen::Quaterniond q0(Eigen::AngleAxisd(kQuadRefYaw0, Eigen::Vector3d::UnitZ()));
  QuadState s = plant.fromNed(Eigen::Vector3d(0.9, -2.5, -0.8), Eigen::Vector3d::Zero(), q0);
  double max_dp = 0.0, max_dv = 0.0, max_dyaw = 0.0;
  for (const auto& r : kQuadRef) {
    plant.stepYawAligned(s, Eigen::Vector3d(r[0], r[1], r[2]), r[3], kQuadRefDt);
    max_dp = std::max(max_dp, (s.posNed() - Eigen::Vector3d(r[4], r[5], r[6])).norm());
    max_dv = std::max(max_dv, (s.velNed() - Eigen::Vector3d(r[7], r[8], r[9])).norm());
    max_dyaw = std::max(max_dyaw, std::abs(wrapPi(s.yawNed() - r[10])));
  }
  CHECK(max_dp < 5e-3 && max_dv < 5e-3 && max_dyaw < 2e-3,
        "QuadPlant diverges from quad.py: max |dp| %.4f m, |dv| %.4f m/s, |dyaw| %.4f rad", max_dp, max_dv,
        max_dyaw);
  std::printf("[PASS] testQuadPlantMatchesQuadPy (%zu steps, max |dp| %.1e m, |dv| %.1e m/s, |dyaw| %.1e rad)\n",
              sizeof(kQuadRef) / sizeof(kQuadRef[0]), max_dp, max_dv, max_dyaw);
}

// World-frame commands: hover holds, and a NED velocity is tracked whatever the heading (the Lee controller
// gets it as a yaw-aligned command re-expressed with the current yaw).
void testQuadPlantWorldVelocity() {
  QuadPlant plant;
  QuadState s = plant.fromNed(Eigen::Vector3d(0, 0, -1), Eigen::Vector3d::Zero(), Eigen::Quaterniond::Identity());
  plant.stepWorld(s, Eigen::Vector3d::Zero(), 0.0, 2.0);
  CHECK((s.posNed() - Eigen::Vector3d(0, 0, -1)).norm() < 1e-3, "hover drifts %.4f m",
        (s.posNed() - Eigen::Vector3d(0, 0, -1)).norm());

  const Eigen::Quaterniond q(Eigen::AngleAxisd(1.2, Eigen::Vector3d::UnitZ()));
  QuadState w = plant.fromNed(Eigen::Vector3d(0, 0, -1), Eigen::Vector3d::Zero(), q);
  plant.stepWorld(w, Eigen::Vector3d(1.0, 0.0, 0.0), 0.0, 6.0);
  // default loop = PX4's PID: its integrator removes the drag-induced steady-state error
  const double v_ss = 1.0;
  CHECK(std::abs(w.velNed().x() - v_ss) < 0.03 && std::abs(w.velNed().y()) < 0.03,
        "world velocity not tracked at yaw 1.2: v=(%.3f, %.3f)", w.velNed().x(), w.velNed().y());
  // quad.py's Lee controller commands yaw RATE only (desired heading = current heading, no heading hold), and the
  // deadcat's asymmetric rotors leak a little yaw torque while accelerating: quad.py itself ends this run at
  // 1.2292 rad. So: small drift, not none.
  CHECK(std::abs(wrapPi(w.yawNed() - 1.2)) < 0.05, "heading drifted %.3f rad without a yaw command",
        wrapPi(w.yawNed() - 1.2));
  std::printf("[PASS] testQuadPlantWorldVelocity (v=(%.3f, %.3f) m/s at yaw 1.2, steady state %.3f)\n",
              w.velNed().x(), w.velNed().y(), v_ss);
}

// The default velocity loop is PX4's (fleet MPC_* gains): a 1 m/s step must rise like PX4's PID (P 1.8 + I + D on
// acceleration; 0.74 s to 63% here, attitude and motor lag included), slower than quad.py's Lee loop (K_vel 3:
// 0.49 s).
void testQuadPlantPx4VelocityStep() {
  auto rise63 = [](const QuadPlantParams& pp) {
    QuadPlant plant(pp);
    QuadState s = plant.fromNed(Eigen::Vector3d(0, 0, -1), Eigen::Vector3d::Zero(), Eigen::Quaterniond::Identity());
    for (int i = 1; i <= 400; ++i) {
      plant.stepWorld(s, Eigen::Vector3d(1.0, 0.0, 0.0), 0.0, 0.01);
      if (s.velNed().x() >= 0.632) return 0.01 * i;
    }
    return 99.0;
  };
  const double t_px4 = rise63(QuadPlantParams{});
  const double t_lee = rise63(QuadPlantParams::quadPy());
  CHECK(t_px4 > 0.4 && t_px4 < 0.8, "PX4 loop 63%% rise %.2f s (expected ~0.6 s for MPC_XY_VEL_P_ACC 1.8)", t_px4);
  CHECK(t_lee < t_px4, "quad.py Lee loop (%.2f s) should be stiffer than PX4's (%.2f s)", t_lee, t_px4);
  std::printf("[PASS] testQuadPlantPx4VelocityStep (63%% rise: PX4 PID %.2f s, quad.py Lee %.2f s)\n", t_px4, t_lee);
}

// Heading regression (RViz heading vs rollout directions): with the vehicle yawed 1 rad off a straight path,
// MPPI must still command motion ALONG the path (world-frame controls, path-frame exploration), turn the
// heading onto it (orientation + FOV critics), and keep doing so as the vehicle yaws between cycles (the warm
// start is world-frame). Closed loop against the same plant at 20 Hz.
void testMPPIHeadingOffsetClosedLoop() {
  auto gt = std::make_shared<GroundTruthMap>();
  gt->loadFromBinary("config/course_groundtruth.bin");
  auto costmap = std::make_shared<VoxelCostmap>(VoxelCostmapConfig{}, gt);
  RRTStarPlanner rrt(RRTStarConfig{}, gt);
  const PlannedPath ref = rrt.smoothWithCubicBSpline(
      {Eigen::Vector3d(0.9144, -2.5146, -0.78), Eigen::Vector3d(2.55, -2.5146, -0.78)}, 1.2);

  MPPIConfig mcfg{};
  mcfg.num_samples = 512;
  MPPIController mppi(mcfg, QuadPlantParams{}, costmap);
  QuadPlant truth;
  const double yaw0 = 1.0;
  QuadState s = truth.fromNed(Eigen::Vector3d(1.05, -2.5146, -0.78), Eigen::Vector3d(0.6, 0.0, 0.0),
                              Eigen::Quaterniond(Eigen::AngleAxisd(yaw0, Eigen::Vector3d::UnitZ())));

  // Commands are judged only in the cruise phase (x < 1.8): over the last ~0.7 m MPPI brakes to stop at the end
  // of this short path, so it commands backwards on purpose.
  double first_dir_err = 0.0, max_dir_err_after = 0.0, max_lat = 0.0, min_clr = 1e9;
  for (int i = 0; i < 24; ++i) {   // 1.2 s
    VehicleKinematics vk{};
    vk.timestamp_s = 1.0 + 0.05 * i;
    vk.pos_odom = s.posNed();
    vk.vel_odom = s.velNed();
    vk.q_odom_base = Eigen::Quaterniond(s.rotNed());
    const TrajectorySetpointCmd cmd = mppi.computeCommand(vk, ref);
    const double dir_err = std::abs(std::atan2(cmd.vel_ned.y(), cmd.vel_ned.x()));   // path heading is 0
    if (i == 0) first_dir_err = dir_err;
    if (i >= 2 && s.posNed().x() < 1.8) max_dir_err_after = std::max(max_dir_err_after, dir_err);
    truth.stepWorld(s, cmd.vel_ned, cmd.yawspeed_rad_s, 0.05);
    max_lat = std::max(max_lat, std::abs(s.posNed().y() + 2.5146));
    min_clr = std::min(min_clr, costmap->clearance(s.posNed()));
  }
  CHECK(first_dir_err < 0.35, "first command leaves the path direction by %.2f rad at a 1 rad heading offset",
        first_dir_err);
  CHECK(max_dir_err_after < 0.5, "cruise command direction drifts %.2f rad off the path while the vehicle yaws",
        max_dir_err_after);
  // quad.py caps yaw rate at 60 deg/s and its yaw-rate loop lags, so 1 rad of offset takes > 1 s to remove
  CHECK(std::abs(s.yawNed()) < 0.4, "heading not brought onto the path: %.2f rad after 1.2 s", s.yawNed());
  CHECK(max_lat < 0.15, "lateral deviation %.3f m", max_lat);
  CHECK(min_clr > 0.21, "came within %.3f m of a wall", min_clr);
  std::printf("[PASS] testMPPIHeadingOffsetClosedLoop (dir err %.2f / %.2f rad, yaw %.2f -> %.2f rad, lat %.3f m)\n",
              first_dir_err, max_dir_err_after, yaw0, s.yawNed(), max_lat);
}

// Pillar Dead-Center split must be left/right of the PATH, not of the body: same scene as the pillar test, but
// the vehicle is yawed 1.2 rad off the corridor.
void testMPPIPillarSplitIsPathRelative() {
  auto gt = std::make_shared<GroundTruthMap>();
  gt->loadFromBinary("config/course_groundtruth.bin");
  auto costmap = std::make_shared<VoxelCostmap>(VoxelCostmapConfig{}, gt);
  costmap->injectUnmodeledPillar(Eigen::Vector2d(1.75, -2.5146), 0.10, 1.65, 0.0);
  MPPIController mppi(MPPIConfig{}, QuadPlantParams{}, costmap);
  RRTStarPlanner rrt(RRTStarConfig{}, gt);
  const PlannedPath ref_path = rrt.smoothWithCubicBSpline(
      {Eigen::Vector3d(0.9144, -2.5146, -0.78), Eigen::Vector3d(2.55, -2.5146, -0.78)}, 2.5);
  VehicleKinematics vk{};
  vk.timestamp_s = 1.0;
  vk.pos_odom = Eigen::Vector3d(1.25, -2.5146, -0.78);
  vk.vel_odom = Eigen::Vector3d(1.2, 0.0, 0.0);
  vk.q_odom_base = Eigen::Quaterniond(Eigen::AngleAxisd(1.2, Eigen::Vector3d::UnitZ()));
  const TrajectorySetpointCmd cmd = mppi.computeCommand(vk, ref_path);
  CHECK(cmd.vel_ned.x() > 0.3, "no forward progress at a 1.2 rad heading offset (vx=%.2f)", cmd.vel_ned.x());
  CHECK(std::abs(cmd.vel_ned.y()) > 0.12, "no lateral evasion at a 1.2 rad heading offset (vy=%.3f)",
        cmd.vel_ned.y());
  std::printf("[PASS] testMPPIPillarSplitIsPathRelative (yaw 1.2: vx=%.2f, vy=%.3f m/s)\n", cmd.vel_ned.x(),
              cmd.vel_ned.y());
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

// Regression: a replan from mid-course must go start -> remaining anchors -> pad, not loop back through the
// whole course from the first anchor; and re-delivering the same global path must not reset progress.
void testRRTStarReplanContinuesFromProgress() {
  auto gt = std::make_shared<GroundTruthMap>();
  gt->loadFromBinary("config/course_groundtruth.bin");
  RRTStarPlanner rrt(RRTStarConfig{}, gt);
  const Eigen::Vector3d launch(0.9144, -2.5146, -0.78);

  PlannedPath full;
  std::vector<double> anchor_s;
  CHECK(rrt.planFullCourse(launch, true, false, full, 0, &anchor_s), "full course plan failed");
  CHECK(anchor_s.size() == rrt.courseAnchors(true, false).size(), "anchor_s size %zu", anchor_s.size());
  for (std::size_t i = 1; i < anchor_s.size(); ++i) {
    CHECK(anchor_s[i] >= anchor_s[i - 1], "anchor arc lengths not monotonic at %zu", i);
  }
  const Eigen::Vector3d pad(gt->padCenterNed().x(), gt->padCenterNed().y(), -0.78);
  CHECK((full.points.back().pos - pad).norm() < 0.05, "course does not end at the pad");

  CourseProgress progress;
  progress.reset(full, anchor_s, 0);
  const double s_mid = 0.6 * full.total_length_m;
  for (double s = 0.0; s <= s_mid; s += 0.1) progress.update(full.sampleByArcLength(s).pos);
  const Eigen::Vector3d mid = full.sampleByArcLength(s_mid).pos;
  const std::size_t next = progress.update(mid);
  CHECK(next > 0 && next < anchor_s.size(), "next anchor %zu after 60%% of the course", next);

  PlannedPath replanned;
  CHECK(rrt.planFullCourse(mid, true, false, replanned, next), "mid-course replan failed");
  const double remaining = full.total_length_m - s_mid;
  CHECK(replanned.total_length_m < remaining + 2.0,
        "replan len %.2fm vs %.2fm remaining: it loops back through passed anchors",
        replanned.total_length_m, remaining);
  CHECK((replanned.points.front().pos - mid).norm() < 0.05, "replan does not start at the vehicle");
  CHECK((replanned.points.back().pos - pad).norm() < 0.05, "replan does not end at the pad");

  PathTransformer tf;
  tf.setGlobalPathMap(full);
  VehicleKinematics st{};
  st.pos_odom = full.sampleByArcLength(4.0).pos;
  for (int i = 0; i < 40; ++i) tf.extractLocalReferenceOdom(st);
  const double s_before = tf.currentArcLengthM();
  tf.setGlobalPathMap(full);   // same path again
  CHECK(std::abs(tf.currentArcLengthM() - s_before) < 1e-9, "same path reset progress to %.2f",
        tf.currentArcLengthM());

  std::printf("[PASS] testRRTStarReplanContinuesFromProgress (next=%zu, replan=%.2fm, remaining=%.2fm)\n",
              next, replanned.total_length_m, remaining);
}

void testMPPICandidateRollouts() {
  auto gt = std::make_shared<GroundTruthMap>();
  gt->loadFromBinary("config/course_groundtruth.bin");
  RRTStarPlanner rrt(RRTStarConfig{}, gt);
  PlannedPath path;
  rrt.planFullCourse(Eigen::Vector3d(0.9144, -2.5146, -0.78), true, false, path);

  MPPIConfig cfg{};
  cfg.viz_num_candidates = 64;
  MPPIController mppi(cfg, QuadPlantParams{}, std::make_shared<VoxelCostmap>(VoxelCostmapConfig{}, gt));
  VehicleKinematics st{};
  st.pos_odom = path.points.front().pos;
  mppi.computeCommand(st, path);

  const MPPICandidateSet& c = mppi.lastCandidates();
  CHECK(c.num_candidates == 64 && c.points_per_candidate == cfg.horizon_steps + 1,
        "candidates %d x %d", c.num_candidates, c.points_per_candidate);
  CHECK(c.positions.size() == static_cast<std::size_t>(64 * (cfg.horizon_steps + 1)), "positions size");
  CHECK((c.positions[0] - st.pos_odom).norm() < 1e-9, "candidate does not start at x0");
  CHECK(c.max_cost >= c.min_cost && c.min_cost >= mppi.lastMinCost() - 1e-9, "candidate cost range");
  int distinct = 0;
  for (int k = 1; k < c.num_candidates; ++k) {
    if ((c.positions[(k + 1) * c.points_per_candidate - 1] - c.positions[c.points_per_candidate - 1]).norm() > 1e-3) {
      ++distinct;
    }
  }
  CHECK(distinct > c.num_candidates / 2, "only %d candidates differ from the mean rollout", distinct);

  MPPIController off(MPPIConfig{}, QuadPlantParams{});
  off.computeCommand(st, path);
  CHECK(off.lastCandidates().num_candidates == 0 && off.lastCandidates().positions.empty(),
        "viz_num_candidates=0 still stores candidates");
  std::printf("[PASS] testMPPICandidateRollouts (%d candidates, cost %.1f..%.1f)\n", c.num_candidates,
              c.min_cost, c.max_cost);
}

// Checkpoints from config/arena_prior.json (one per checkpoint, the window for checkpoint 3), then the pad.
std::vector<PlanGoal> courseGoalsFromArenaPrior() {
  auto g = [](const char* name, double cx, double cy, double cz, double ax, double ay, double bx, double by,
              double nx, double ny) {
    PlanGoal q;
    q.normal = Eigen::Vector2d(nx, ny);
    q.name = name;
    q.center = Eigen::Vector3d(cx, cy, cz);
    q.has_segment = true;
    q.seg_a = Eigen::Vector2d(ax, ay);
    q.seg_b = Eigen::Vector2d(bx, by);
    return q;
  };
  std::vector<PlanGoal> goals = {
      g("corridor_end", 2.5844, -2.5739, -0.8, 2.5844, -3.048, 2.5844, -1.9304, 1.0, 0.0),
      g("gate_a", 2.5601, 0.3048, -0.8, 2.1209, 0.3048, 3.048, 0.3048, 0.0, 1.0),
      g("tunnel_exit", -0.3048, 2.3114, -0.8, -0.3048, 1.5748, -0.3048, 3.048, -1.0, 0.0),
      g("window", -2.283, -0.475, -0.78, -2.7015, -0.6273, -1.8423, -0.3146, 0.34202, -0.939693),
  };
  PlanGoal pad;
  pad.name = "pad";
  pad.center = Eigen::Vector3d(-0.7925, -2.0422, -0.8);
  goals.push_back(pad);
  return goals;
}

void testAStarPlannerCourse() {
  auto gt = std::make_shared<GroundTruthMap>();
  gt->loadFromBinary("config/course_groundtruth.bin");
  AStarConfig cfg{};
  AStarPlanner astar(cfg, gt);
  const auto goals = courseGoalsFromArenaPrior();
  const Eigen::Vector3d launch(0.9144, -2.5146, -0.8);

  const AStarPlanResult r = astar.plan(launch, goals);
  CHECK(r.ok, "A* course plan failed");
  CHECK(r.skipped.empty(), "A* skipped %zu goals", r.skipped.size());
  CHECK(r.not_crossed.empty(), "A* path misses %zu gate segments", r.not_crossed.size());
  CHECK(r.min_clearance_m >= cfg.clearance_m * cfg.clearance_fallback_ratio,
        "A* path clearance %.3f m below %.3f m", r.min_clearance_m, cfg.clearance_m * cfg.clearance_fallback_ratio);
  const PlannedPath& p = r.path;
  CHECK((p.points.front().pos.head<2>() - launch.head<2>()).norm() < 1e-9, "path does not start at the vehicle");
  CHECK((p.points.back().pos.head<2>() - goals.back().center.head<2>()).norm() < 0.08, "path does not end at the pad");
  double vmax = 0.0;
  for (std::size_t i = 1; i < p.points.size(); ++i) {
    CHECK(p.points[i].t_s > p.points[i - 1].t_s, "time not increasing at %zu", i);
    vmax = std::max(vmax, p.points[i].vel.norm());
  }
  CHECK(p.points.front().vel.norm() < 1e-9 && p.points.back().vel.norm() < 1e-9, "path must start and end at rest");
  CHECK(vmax <= cfg.cruise_speed_m_s + 1e-9 && vmax > 1.0, "peak speed %.2f", vmax);
  for (std::size_t i = 1; i < r.goal_s_m.size(); ++i) {
    CHECK(r.goal_s_m[i] >= r.goal_s_m[i - 1], "goal arc lengths not monotonic at %zu", i);
  }

  const AStarPlanResult r2 = astar.plan(launch, goals);
  CHECK(r2.path.points.size() == p.points.size() &&
            (r2.path.points.back().pos - p.points.back().pos).norm() < 1e-12 &&
            std::abs(r2.path.total_duration_s - p.total_duration_s) < 1e-12,
        "A* is not deterministic");

  // Replan from mid-course through the remaining goals, at the current speed.
  CourseProgress progress;
  progress.reset(p, r.goal_s_m, 0);
  const double s_mid = 0.5 * p.total_length_m;
  for (double s = 0.0; s <= s_mid; s += 0.1) progress.update(p.sampleByArcLength(s).pos);
  const PathPoint mid = p.sampleByArcLength(s_mid);
  const std::size_t next = progress.update(mid.pos);
  const std::vector<PlanGoal> rest(goals.begin() + static_cast<std::ptrdiff_t>(next), goals.end());
  const AStarPlanResult rr = astar.plan(mid.pos, rest, mid.vel.norm());
  CHECK(next > 0 && next < goals.size() && rr.ok, "mid-course replan (next=%zu) failed", next);
  CHECK(rr.path.total_length_m < p.total_length_m - s_mid + 1.5, "replan %.2fm vs %.2fm remaining",
        rr.path.total_length_m, p.total_length_m - s_mid);
  CHECK(std::abs(rr.path.points.front().vel.norm() - mid.vel.norm()) < 0.05, "replan does not start at speed");

  // A blocked corridor moves the path.
  auto gt_blocked = std::make_shared<GroundTruthMap>(*gt);
  gt_blocked->markCorridorBlocked(p.sampleByArcLength(0.3 * p.total_length_m).pos, 0.15);
  AStarPlanner astar_b(cfg, gt_blocked);
  const AStarPlanResult rb = astar_b.plan(launch, goals);
  CHECK(rb.ok && std::abs(rb.path.total_length_m - p.total_length_m) > 1e-3, "blocked corridor did not change the path");

  std::printf("[PASS] testAStarPlannerCourse (len=%.2fm, %.2fs, min_clr=%.3fm, %zu expansions, replan from goal "
              "#%zu: %.2fm)\n", p.total_length_m, p.total_duration_s, r.min_clearance_m, r.expanded, next,
              rr.path.total_length_m);
}

// SITL / the drone: odom's origin is the launch mark, not the arena origin. Closed loop from where every
// 2026-11-08 SITL flight stalled (just short of gate A, facing east), with odom offset from map as there: MPPI must
// take the vehicle through gate A. A map-frame cost evaluated in odom (the outer-net barrier did) puts a phantom
// wall at arena E = 0.135 here; sim_loopback (map == odom) cannot show it.
void testMPPIGateAWithOdomOffset() {
  auto gt = std::make_shared<GroundTruthMap>();
  gt->loadFromBinary("config/course_groundtruth.bin");
  auto costmap = std::make_shared<VoxelCostmap>(VoxelCostmapConfig{}, gt);
  AStarConfig acfg{};
  acfg.cruise_speed_m_s = 0.5;
  AStarPlanner astar(acfg, gt);
  const auto all = courseGoalsFromArenaPrior();
  const std::vector<PlanGoal> goals(all.begin() + 1, all.begin() + 3);   // gate_a, tunnel_exit
  const Eigen::Vector3d start_map(2.54, 0.13, -0.8);
  const AStarPlanResult r = astar.plan(start_map, goals);
  CHECK(r.ok && r.skipped.empty(), "A* from the gate A stall point failed");
  const Eigen::Vector3d lm(0.9144, -2.5146, 0.0);                       // launch mark = odom origin (map frame)
  PlannedPath ref_odom = r.path;
  for (auto& pt : ref_odom.points) pt.pos -= lm;
  SE3Transform T_map_odom{};
  T_map_odom.t = lm;

  MPPIConfig mcfg{};
  mcfg.num_samples = 512;
  MPPIController mppi(mcfg, QuadPlantParams{}, costmap);
  mppi.setOdomToMapTransform(T_map_odom);
  QuadPlant truth;
  QuadState s = truth.fromNed(start_map - lm, Eigen::Vector3d::Zero(),
                              Eigen::Quaterniond(Eigen::AngleAxisd(1.48, Eigen::Vector3d::UnitZ())));
  double min_clr = 1e9, max_e = -1e9;
  for (int i = 0; i < 120; ++i) {   // 6 s
    VehicleKinematics vk{};
    vk.timestamp_s = 1.0 + 0.05 * i;
    vk.pos_odom = s.posNed();
    vk.vel_odom = s.velNed();
    vk.q_odom_base = Eigen::Quaterniond(s.rotNed());
    const TrajectorySetpointCmd cmd = mppi.computeCommand(vk, ref_odom);
    truth.stepWorld(s, cmd.vel_ned, cmd.yawspeed_rad_s, 0.05);
    const Eigen::Vector3d p_map = s.posNed() + lm;
    max_e = std::max(max_e, p_map.y());
    min_clr = std::min(min_clr, costmap->clearance(p_map));
  }
  CHECK(max_e > 0.80, "did not get through gate A in 6 s: max arena E %.3f m (gate line 0.305; path heads on to E > 1)", max_e);
  CHECK(min_clr > 0.21, "came within %.3f m of a wall or the net", min_clr);
  std::printf("[PASS] testMPPIGateAWithOdomOffset (max arena E %.2f m, min clearance %.3f m)\n", max_e, min_clr);
}

// The window (arena N -2.28) lies south of the old odom-frame CBF limit (odom N -2.85 = arena N -1.94), which
// stopped southward motion there. Same odom offset as SITL; MPPI must reach the window line.
void testMPPIWindowWithOdomOffset() {
  auto gt = std::make_shared<GroundTruthMap>();
  gt->loadFromBinary("config/course_groundtruth.bin");
  auto costmap = std::make_shared<VoxelCostmap>(VoxelCostmapConfig{}, gt);
  AStarConfig acfg{};
  acfg.cruise_speed_m_s = 0.5;
  AStarPlanner astar(acfg, gt);
  const auto all = courseGoalsFromArenaPrior();
  const std::vector<PlanGoal> goals(all.begin() + 3, all.end());   // window, pad
  const PlanGoal& win = all[3];
  const Eigen::Vector3d start_map(-1.70, 0.20, -0.8);
  const AStarPlanResult r = astar.plan(start_map, goals);
  CHECK(r.ok && r.skipped.empty(), "A* to the window failed");
  const Eigen::Vector3d lm(0.9144, -2.5146, 0.0);
  PlannedPath ref_odom = r.path;
  for (auto& pt : ref_odom.points) pt.pos -= lm;
  SE3Transform T_map_odom{};
  T_map_odom.t = lm;
  MPPIConfig mcfg{};
  mcfg.num_samples = 512;
  MPPIController mppi(mcfg, QuadPlantParams{}, costmap);
  mppi.setOdomToMapTransform(T_map_odom);
  QuadPlant truth;
  const Eigen::Vector2d d0 = r.path.points[std::min<std::size_t>(5, r.path.points.size() - 1)].pos.head<2>() -
                             start_map.head<2>();
  QuadState s = truth.fromNed(start_map - lm, Eigen::Vector3d::Zero(),
                              Eigen::Quaterniond(Eigen::AngleAxisd(std::atan2(d0.y(), d0.x()), Eigen::Vector3d::UnitZ())));
  const Eigen::Vector2d n = win.normal.normalized();
  double best = -1e9, min_clr = 1e9;
  for (int i = 0; i < 240; ++i) {   // 12 s
    VehicleKinematics vk{};
    vk.timestamp_s = 1.0 + 0.05 * i;
    vk.pos_odom = s.posNed();
    vk.vel_odom = s.velNed();
    vk.q_odom_base = Eigen::Quaterniond(s.rotNed());
    const TrajectorySetpointCmd cmd = mppi.computeCommand(vk, ref_odom);
    truth.stepWorld(s, cmd.vel_ned, cmd.yawspeed_rad_s, 0.05);
    const Eigen::Vector3d p_map = s.posNed() + lm;
    best = std::max(best, (p_map.head<2>() - win.seg_a).dot(n));
    min_clr = std::min(min_clr, costmap->clearance(p_map));
  }
  CHECK(best > 0.15, "did not cross the window line by the scoring margin in 12 s: %.3f m past it", best);
  CHECK(min_clr > 0.21, "came within %.3f m of a wall or the net", min_clr);
  std::printf("[PASS] testMPPIWindowWithOdomOffset (%.2f m past the window line, min clearance %.3f m)\n", best, min_clr);
}

int main() {
  testGroundTruthMapEncoding();
  testVoxelCostmapAndDisparityPunchThroughFMEA();
  testRRTStarPlannerAndBsplineSmoothing();
  testRRTStarReplanContinuesFromProgress();
  testAStarPlannerCourse();
  testPathTransformerAndiSAM2ShiftFMEA();
  testQuadPlantMatchesQuadPy();
  testQuadPlantWorldVelocity();
  testQuadPlantPx4VelocityStep();
  testMPPIRolloutsAndPillarEvasion();
  testMPPIPillarSplitIsPathRelative();
  testMPPIHeadingOffsetClosedLoop();
  testMPPIGateAWithOdomOffset();
  testMPPIWindowWithOdomOffset();
  testMPPICandidateRollouts();
  testControlMuxAndSafetyInterlocks();
  std::printf(failures ? "%d FAILURE(S)\n" : "ALL MAZE PLANNING (RRT* + MPPI + COSTMAP + MUX) C++ TESTS PASSED\n",
              failures);
  return failures ? EXIT_FAILURE : EXIT_SUCCESS;
}
