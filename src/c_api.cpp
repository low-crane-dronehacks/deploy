#include "maze_planning/c_api.h"

#include <algorithm>
#include <cstring>
#include <memory>
#include <vector>

#include "maze_planning/control_mux.hpp"
#include "maze_planning/course_progress.hpp"
#include "maze_planning/groundtruth_map.hpp"
#include "maze_planning/mppi_controller.hpp"
#include "maze_planning/path_transformer.hpp"
#include "maze_planning/rrt_star_planner.hpp"
#include "maze_planning/voxel_costmap.hpp"

struct MazePipelineHandle {
  std::shared_ptr<maze_planning::GroundTruthMap> gt_map;
  std::shared_ptr<maze_planning::VoxelCostmap> costmap;
  std::unique_ptr<maze_planning::RRTStarPlanner> rrt_planner;
  std::unique_ptr<maze_planning::PathTransformer> path_transformer;
  std::unique_ptr<maze_planning::MPPIController> mppi;
  std::unique_ptr<maze_planning::ControlMux> mux;

  bool prefer_window{true};
  bool include_return_leg{false};
  maze_planning::PlannedPath global_path_map;
  maze_planning::CourseProgress progress;

  // Plan from start through the anchors still ahead (all of them for a fresh run) and track progress on it.
  bool planCourse(const Eigen::Vector3d& start_map, std::size_t first_anchor_idx) {
    std::vector<double> anchor_s;
    if (!rrt_planner->planFullCourse(start_map, prefer_window, include_return_leg, global_path_map,
                                     first_anchor_idx, &anchor_s)) {
      return false;
    }
    progress.reset(global_path_map, std::move(anchor_s), first_anchor_idx);
    path_transformer->setGlobalPathMap(global_path_map);
    return true;
  }
};

extern "C" {

MazePipelineHandle* maze_pipeline_create(const MazePipelineCConfig* cfg) {
  auto* h = new MazePipelineHandle();
  h->gt_map = std::make_shared<maze_planning::GroundTruthMap>();

  const char* gt_path = (cfg && cfg->groundtruth_map_path && cfg->groundtruth_map_path[0] != '\0')
                            ? cfg->groundtruth_map_path
                            : "config/course_groundtruth.bin";
  if (!h->gt_map->loadFromBinary(gt_path)) {
    if (!h->gt_map->loadFromObj("sitl/world/course.obj")) {
      h->gt_map->initDefaultCourse();
    }
  }

  maze_planning::VoxelCostmapConfig vcfg{};
  h->costmap = std::make_shared<maze_planning::VoxelCostmap>(vcfg, h->gt_map);

  maze_planning::RRTStarConfig rcfg{};
  if (cfg && cfg->cruise_speed_m_s > 0.2) {
    rcfg.cruise_speed_m_s = cfg->cruise_speed_m_s;
  }
  h->rrt_planner = std::make_unique<maze_planning::RRTStarPlanner>(rcfg, h->gt_map, h->costmap);

  maze_planning::PathTransformerConfig ptcfg{};
  h->path_transformer = std::make_unique<maze_planning::PathTransformer>(ptcfg);

  maze_planning::MPPIConfig mcfg{};
  if (cfg && cfg->mppi_samples > 16) mcfg.num_samples = cfg->mppi_samples;
  if (cfg && cfg->mppi_horizon_steps > 4) mcfg.horizon_steps = cfg->mppi_horizon_steps;
  maze_planning::MultirotorPlantParams plant{};
  h->mppi = std::make_unique<maze_planning::MPPIController>(mcfg, plant, h->costmap);

  h->mux = std::make_unique<maze_planning::ControlMux>();
  if (cfg) {
    h->prefer_window = (cfg->prefer_window != 0);
    h->include_return_leg = (cfg->include_return_leg != 0);
  }

  const Eigen::Vector3d start_default(0.9144, -2.5146, -0.78);
  h->planCourse(start_default, 0);
  return h;
}

void maze_pipeline_destroy(MazePipelineHandle* h) { delete h; }

void maze_pipeline_reset(MazePipelineHandle* h, const double start_pos_map[3]) {
  if (!h) return;
  h->gt_map->clearBlockedCorridors();
  h->costmap->clearDynamicObstacles();
  h->mppi->reset();
  Eigen::Vector3d start(0.9144, -2.5146, -0.78);
  if (start_pos_map) {
    start = Eigen::Vector3d(start_pos_map[0], start_pos_map[1], start_pos_map[2]);
  }
  h->planCourse(start, 0);
}

double maze_pipeline_query_groundtruth(const MazePipelineHandle* h,
                                       const double pos_map[3],
                                       double grad_out[3]) {
  if (!h || !pos_map) return 0.0;
  Eigen::Vector3d g(0.0, 0.0, 0.0);
  const double d = h->gt_map->clearanceAndGradient(
      Eigen::Vector3d(pos_map[0], pos_map[1], pos_map[2]), g);
  if (grad_out) {
    grad_out[0] = g.x();
    grad_out[1] = g.y();
    grad_out[2] = g.z();
  }
  return d;
}

double maze_pipeline_query_costmap(const MazePipelineHandle* h,
                                   const double pos_map[3]) {
  if (!h || !pos_map) return 0.0;
  return h->costmap->clearance(Eigen::Vector3d(pos_map[0], pos_map[1], pos_map[2]));
}

void maze_pipeline_inject_pillar(MazePipelineHandle* h,
                                 double x_map,
                                 double y_map,
                                 double radius_m,
                                 double height_m) {
  if (!h) return;
  h->costmap->injectUnmodeledPillar(Eigen::Vector2d(x_map, y_map), radius_m, height_m, 0.0);
}

void maze_pipeline_clear_dynamic_obstacles(MazePipelineHandle* h) {
  if (!h) return;
  h->costmap->clearDynamicObstacles();
}

void maze_pipeline_update_point_cloud(MazePipelineHandle* h,
                                      double stamp_s,
                                      const double sensor_origin_map[3],
                                      const double* points_xyz,
                                      int num_points) {
  if (!h || !sensor_origin_map || !points_xyz || num_points <= 0) return;
  std::vector<Eigen::Vector3d> pts(static_cast<std::size_t>(num_points));
  for (int i = 0; i < num_points; ++i) {
    pts[i] = Eigen::Vector3d(points_xyz[3 * i + 0], points_xyz[3 * i + 1], points_xyz[3 * i + 2]);
  }
  h->costmap->updateFromClampedPointCloud(
      stamp_s,
      Eigen::Vector3d(sensor_origin_map[0], sensor_origin_map[1], sensor_origin_map[2]),
      pts);
}

void maze_pipeline_set_map_to_odom(MazePipelineHandle* h,
                                   const double t_xyz[3],
                                   const double q_wxyz[4]) {
  if (!h || !t_xyz || !q_wxyz) return;
  maze_planning::SE3Transform T_map_odom{};
  T_map_odom.t = Eigen::Vector3d(t_xyz[0], t_xyz[1], t_xyz[2]);
  T_map_odom.q = Eigen::Quaterniond(q_wxyz[0], q_wxyz[1], q_wxyz[2], q_wxyz[3]).normalized();
  h->path_transformer->setMapToOdomTransform(T_map_odom);
  h->mppi->setOdomToMapTransform(T_map_odom);
}

int maze_pipeline_plan_point_to_point(MazePipelineHandle* h,
                                      const double start_map[3],
                                      const double goal_map[3],
                                      double* out_waypoints_xyz,
                                      int max_waypoints) {
  if (!h || !start_map || !goal_map) return 0;
  maze_planning::PlannedPath path;
  std::vector<Eigen::Vector3d> wps;
  const bool ok = h->rrt_planner->plan(
      Eigen::Vector3d(start_map[0], start_map[1], start_map[2]),
      Eigen::Vector3d(goal_map[0], goal_map[1], goal_map[2]),
      path, &wps);
  if (!ok) return 0;
  h->global_path_map = path;
  h->path_transformer->setGlobalPathMap(h->global_path_map);

  const int n = std::min(static_cast<int>(wps.size()), max_waypoints);
  if (out_waypoints_xyz) {
    for (int i = 0; i < n; ++i) {
      out_waypoints_xyz[3 * i + 0] = wps[i].x();
      out_waypoints_xyz[3 * i + 1] = wps[i].y();
      out_waypoints_xyz[3 * i + 2] = wps[i].z();
    }
  }
  return static_cast<int>(wps.size());
}

void maze_pipeline_step(MazePipelineHandle* h,
                        const MazeVehicleCState* state,
                        MazeStepCOutput* out) {
  if (!h || !state || !out) return;
  std::memset(out, 0, sizeof(MazeStepCOutput));

  maze_planning::VehicleKinematics vk{};
  vk.timestamp_s = state->timestamp_s;
  vk.pos_odom = Eigen::Vector3d(state->pos_odom[0], state->pos_odom[1], state->pos_odom[2]);
  vk.vel_odom = Eigen::Vector3d(state->vel_odom[0], state->vel_odom[1], state->vel_odom[2]);
  vk.q_odom_base = Eigen::Quaterniond(
      state->quat_wxyz[0], state->quat_wxyz[1], state->quat_wxyz[2], state->quat_wxyz[3]).normalized();

  // 1. Extract local reference window in odom frame (/plan/local_reference) & RL lookaheads (/plan/rl_lookaheads)
  const maze_planning::PlannedPath local_ref =
      h->path_transformer->extractLocalReferenceOdom(vk);
  std::array<float, 12> rl_flat{};
  h->path_transformer->computeRLLookaheadsBody(vk, &rl_flat);
  for (int i = 0; i < 12; ++i) out->rl_lookaheads_body[i] = rl_flat[i];

  // 2. Run MPPI controller (2048 rollouts)
  maze_planning::TrajectorySetpointCmd cmd = h->mppi->computeCommand(vk, local_ref);

  // 3. FMEA Pillar Dead-Center: if MPPI requests a global replan, mark corridor blocked in RRT* and replan
  //    from the current pose through the anchors still ahead (not from the first anchor again).
  int replanned = 0;
  const Eigen::Vector3d cur_map = h->path_transformer->mapToOdomTransform().apply(vk.pos_odom);
  const std::size_t next_anchor = h->progress.update(cur_map);
  if (cmd.request_global_replan) {
    h->gt_map->markCorridorBlocked(cmd.blocked_obstacle_pos, 0.36);
    if (h->planCourse(cur_map, next_anchor)) {
      h->mppi->reset();
      replanned = 1;
    }
  }

  for (int i = 0; i < 3; ++i) {
    out->pos_ned[i] = cmd.pos_ned[i];
    out->vel_ned[i] = cmd.vel_ned[i];
    out->acc_ned[i] = cmd.acc_ned[i];
  }
  out->yaw_rad = cmd.yaw_rad;
  out->yawspeed_rad_s = cmd.yawspeed_rad_s;
  for (int i = 0; i < 4; ++i) out->body_action[i] = cmd.body_action[i];
  out->obstacle_hold = cmd.obstacle_hold ? 1 : 0;
  out->replanned = replanned;
  out->solve_time_ms = cmd.solve_time_ms;
  out->min_rollout_cost = cmd.min_rollout_cost;
  out->remaining_dist_m = h->path_transformer->remainingDistanceM();
}

}  // extern "C"
