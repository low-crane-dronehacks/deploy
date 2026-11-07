#pragma once

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct EgoPlannerHandle EgoPlannerHandle;

typedef struct {
  int arm_authorized;
  int enable_return_leg;
  int prefer_window;
  double cruise_alt_m;
  double desired_speed_ms;
  double replan_period_s;
  const char* prior_bin_path;
} EgoPlannerCConfig;

typedef struct {
  double pos_ned[3];
  double vel_ned[3];
  double quat_wxyz[4];
  double rangefinder_m;
  int armed;
  int offboard;
  int landed;
  int pose_valid;
} EgoPlannerCState;

typedef struct {
  int publish_offboard;
  int publish_arm_cmd;
  int publish_offboard_switch;
  int publish_land_cmd;
  double vel_ned[3];
  double yawspeed_rad_s;
  double body_action[4];
  int next_checkpoint;
  char phase[32];
  char why[64];
  char result[64];
  char active_leg[32];
  char active_primitive[32];
} EgoPlannerCOutput;

EgoPlannerHandle* ego_planner_create(const EgoPlannerCConfig* cfg);
void ego_planner_destroy(EgoPlannerHandle* h);
void ego_planner_reset(EgoPlannerHandle* h);

void ego_planner_on_estimator_health(EgoPlannerHandle* h, double t_s, int healthy);
void ego_planner_on_depth_timestamp(EgoPlannerHandle* h, double t_s);
void ego_planner_on_depth_frame(EgoPlannerHandle* h, double t_s, const float* depth_norm,
                                int width, int height, double hfov_deg, double vfov_deg,
                                double far_clip, const double pos_ned[3],
                                const double quat_wxyz[4], int stride);

void ego_planner_step(EgoPlannerHandle* h, double t_s, const EgoPlannerCState* state,
                      EgoPlannerCOutput* out);

double ego_planner_query_esdf(const EgoPlannerHandle* h, const double pos_ned[3],
                              double grad_out[3]);

#ifdef __cplusplus
}
#endif
