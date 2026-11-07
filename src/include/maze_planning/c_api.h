#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct MazePipelineHandle MazePipelineHandle;

typedef struct {
  const char* groundtruth_map_path;
  int prefer_window;
  int include_return_leg;
  int mppi_samples;
  int mppi_horizon_steps;
  double cruise_speed_m_s;
} MazePipelineCConfig;

typedef struct {
  double timestamp_s;
  double pos_odom[3];
  double vel_odom[3];
  double quat_wxyz[4];
} MazeVehicleCState;

typedef struct {
  double pos_ned[3];
  double vel_ned[3];
  double acc_ned[3];
  double yaw_rad;
  double yawspeed_rad_s;
  double body_action[4];
  float rl_lookaheads_body[12];
  int obstacle_hold;
  int replanned;
  double solve_time_ms;
  double min_rollout_cost;
  double remaining_dist_m;
} MazeStepCOutput;

MazePipelineHandle* maze_pipeline_create(const MazePipelineCConfig* cfg);
void maze_pipeline_destroy(MazePipelineHandle* h);
void maze_pipeline_reset(MazePipelineHandle* h, const double start_pos_map[3]);

double maze_pipeline_query_groundtruth(const MazePipelineHandle* h,
                                       const double pos_map[3],
                                       double grad_out[3]);

double maze_pipeline_query_costmap(const MazePipelineHandle* h,
                                   const double pos_map[3]);

void maze_pipeline_inject_pillar(MazePipelineHandle* h,
                                 double x_map,
                                 double y_map,
                                 double radius_m,
                                 double height_m);

void maze_pipeline_clear_dynamic_obstacles(MazePipelineHandle* h);

void maze_pipeline_update_point_cloud(MazePipelineHandle* h,
                                      double stamp_s,
                                      const double sensor_origin_map[3],
                                      const double* points_xyz,
                                      int num_points);

void maze_pipeline_set_map_to_odom(MazePipelineHandle* h,
                                   const double t_xyz[3],
                                   const double q_wxyz[4]);

int maze_pipeline_plan_point_to_point(MazePipelineHandle* h,
                                      const double start_map[3],
                                      const double goal_map[3],
                                      double* out_waypoints_xyz,
                                      int max_waypoints);

void maze_pipeline_step(MazePipelineHandle* h,
                        const MazeVehicleCState* state,
                        MazeStepCOutput* out);

#ifdef __cplusplus
}
#endif
