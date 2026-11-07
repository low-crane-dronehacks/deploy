#include "ego_planner/c_api.h"

#include <cstring>

#include "ego_planner/mission_manager.hpp"

struct EgoPlannerHandle {
  ego_planner::MissionManager manager;
  explicit EgoPlannerHandle(const ego_planner::MissionConfig& cfg) : manager(cfg) {}
};

namespace {

void copyStr(char* dst, std::size_t cap, const std::string& src) {
  if (cap == 0) return;
  std::strncpy(dst, src.c_str(), cap - 1);
  dst[cap - 1] = '\0';
}

}  // namespace

extern "C" {

EgoPlannerHandle* ego_planner_create(const EgoPlannerCConfig* cfg) {
  ego_planner::MissionConfig mcfg{};
  if (cfg != nullptr) {
    mcfg.arm_authorized = (cfg->arm_authorized != 0);
    mcfg.enable_return_leg = (cfg->enable_return_leg != 0);
    mcfg.prefer_window = (cfg->prefer_window != 0);
    if (cfg->cruise_alt_m > 0.1) mcfg.cruise_alt_m = cfg->cruise_alt_m;
    if (cfg->desired_speed_ms > 0.1) mcfg.desired_speed_ms = cfg->desired_speed_ms;
    if (cfg->replan_period_s > 0.01) mcfg.replan_period_s = cfg->replan_period_s;
    if (cfg->prior_bin_path != nullptr) {
      mcfg.prior_bin_path = cfg->prior_bin_path;
    }
  }
  return new EgoPlannerHandle(mcfg);
}

void ego_planner_destroy(EgoPlannerHandle* h) { delete h; }

void ego_planner_reset(EgoPlannerHandle* h) {
  if (h != nullptr) h->manager.reset();
}

void ego_planner_on_estimator_health(EgoPlannerHandle* h, double t_s, int healthy) {
  if (h != nullptr) h->manager.onEstimatorHealth(t_s, healthy != 0);
}

void ego_planner_on_depth_timestamp(EgoPlannerHandle* h, double t_s) {
  if (h != nullptr) h->manager.onDepthTimestamp(t_s);
}

void ego_planner_on_depth_frame(EgoPlannerHandle* h, double t_s, const float* depth_norm,
                                int width, int height, double hfov_deg, double vfov_deg,
                                double far_clip, const double pos_ned[3],
                                const double quat_wxyz[4], int stride) {
  if (h == nullptr || depth_norm == nullptr) return;
  const ego_planner::Vec3 p{pos_ned[0], pos_ned[1], pos_ned[2]};
  const ego_planner::Quat q{quat_wxyz[0], quat_wxyz[1], quat_wxyz[2], quat_wxyz[3]};
  h->manager.onDepthFrame(t_s, depth_norm, width, height, hfov_deg, vfov_deg, far_clip, p, q,
                          stride);
}

void ego_planner_step(EgoPlannerHandle* h, double t_s, const EgoPlannerCState* state,
                      EgoPlannerCOutput* out) {
  if (h == nullptr || state == nullptr || out == nullptr) return;
  ego_planner::VehicleState vs{};
  vs.timestamp_s = t_s;
  vs.pos_ned = ego_planner::Vec3{state->pos_ned[0], state->pos_ned[1], state->pos_ned[2]};
  vs.vel_ned = ego_planner::Vec3{state->vel_ned[0], state->vel_ned[1], state->vel_ned[2]};
  vs.q_wxyz = ego_planner::Quat{state->quat_wxyz[0], state->quat_wxyz[1], state->quat_wxyz[2],
                                state->quat_wxyz[3]};
  vs.range_m = state->rangefinder_m;
  vs.armed = (state->armed != 0);
  vs.nav_state = (state->offboard != 0) ? 14 : 0;
  vs.landed = (state->landed != 0);
  vs.pos_valid = (state->pose_valid != 0);
  vs.att_valid = true;

  const ego_planner::ControlOutput co = h->manager.step(t_s, vs);

  std::memset(out, 0, sizeof(EgoPlannerCOutput));
  out->publish_offboard = co.stream_velocity ? 1 : 0;
  out->publish_arm_cmd = (co.command_flags & ego_planner::kCmdArm) ? 1 : 0;
  out->publish_offboard_switch = (co.command_flags & ego_planner::kCmdOffboard) ? 1 : 0;
  out->publish_land_cmd = (co.command_flags & ego_planner::kCmdLand) ? 1 : 0;
  out->vel_ned[0] = co.vel_ned.x;
  out->vel_ned[1] = co.vel_ned.y;
  out->vel_ned[2] = co.vel_ned.z;
  out->yawspeed_rad_s = co.yawspeed_rad_s;
  for (int i = 0; i < 4; ++i) out->body_action[i] = co.body_action[i];
  out->next_checkpoint = co.next_checkpoint;
  copyStr(out->phase, sizeof(out->phase), co.mission_phase);
  copyStr(out->why, sizeof(out->why), co.why);
  copyStr(out->result, sizeof(out->result), co.result);
  copyStr(out->active_leg, sizeof(out->active_leg), co.active_leg);
  copyStr(out->active_primitive, sizeof(out->active_primitive), co.active_primitive);
}

double ego_planner_query_esdf(const EgoPlannerHandle* h, const double pos_ned[3],
                              double grad_out[3]) {
  if (h == nullptr || pos_ned == nullptr) return 0.0;
  ego_planner::Vec3 g{};
  const double d = h->manager.gridMap().getDistanceAndGradient(
      ego_planner::Vec3{pos_ned[0], pos_ned[1], pos_ned[2]}, g);
  if (grad_out != nullptr) {
    grad_out[0] = g.x;
    grad_out[1] = g.y;
    grad_out[2] = g.z;
  }
  return d;
}

}  // extern "C"
