#include "ego_planner/local_controller_interface.hpp"

#include <algorithm>
#include <cmath>

namespace ego_planner {

ControlOutput PrimitiveTracker::computeCommand(const VehicleState& state,
                                               const GlobalTrajectory& ref_traj,
                                               const GridMap& grid_map,
                                               double dt) {
  ControlOutput out;
  out.stream_velocity = true;
  out.mission_phase = "fly";
  out.active_primitive = "primitive_tracker";

  if (ref_traj.empty()) {
    out.vel_ned = Vec3{0.0, 0.0, 0.0};
    return out;
  }

  // Find vehicle projection along the trajectory
  double current_arc_s = 0.0;
  GlobalTrajectoryPoint current_target = ref_traj.projectPoint(state.pos_ned, current_arc_s);

  // Advance cursor forward by lookahead distance: lookahead = max(0.4m, speed * 0.4s)
  double lookahead_m = std::max(0.40, desired_speed_ * 0.40);
  cursor_arc_m_ = std::max(cursor_arc_m_, current_arc_s);
  double target_arc_m = std::min(cursor_arc_m_ + lookahead_m, ref_traj.total_length_m);

  // Evaluate target point
  GlobalTrajectoryPoint target_pt = ref_traj.evaluateAtTime(target_arc_m / std::max(desired_speed_, 0.5));

  // PD setpoint tracking
  Vec3 pos_err = target_pt.pos_ned - state.pos_ned;
  double kp_pos = 1.6;
  Vec3 feedforward_vel = target_pt.vel_ned;
  if (feedforward_vel.norm() < 1e-4) {
    Vec3 dir = target_pt.pos_ned - state.pos_ned;
    double d = dir.norm();
    feedforward_vel = d > 1e-4 ? (dir / d) * desired_speed_ : Vec3{0.0, 0.0, 0.0};
  }

  out.vel_ned = feedforward_vel + pos_err * kp_pos;

  // Clamp vertical velocity
  out.vel_ned.z = clamp(out.vel_ned.z, -1.8, 1.8);

  // Clamp horizontal velocity
  double v_horiz = std::hypot(out.vel_ned.x, out.vel_ned.y);
  double max_horiz = desired_speed_ * 1.25;
  if (v_horiz > max_horiz) {
    out.vel_ned.x *= (max_horiz / v_horiz);
    out.vel_ned.y *= (max_horiz / v_horiz);
  }

  // Reactive push if approaching obstacle
  Vec3 grad;
  double dist = grid_map.getDistanceAndGradient(state.pos_ned, grad);
  if (dist < 0.28) {
    double push_scale = (0.28 - dist) * 2.5;
    out.vel_ned += grad * push_scale;
  }

  // Heading setpoint
  out.yaw_rad = target_pt.yaw_rad;
  Vec3 rpy = state.q_wxyz.toEulerZYX();
  double yaw_err = wrapPi(target_pt.yaw_rad - rpy.z);
  out.yawspeed_rad_s = clamp(yaw_err * 2.0, -kPi / 3.0, kPi / 3.0);

  // Body-frame normalized action
  double cy = std::cos(rpy.z);
  double sy = std::sin(rpy.z);
  double v_fwd = cy * out.vel_ned.x + sy * out.vel_ned.y;
  double v_right = -sy * out.vel_ned.x + cy * out.vel_ned.y;
  double v_down = out.vel_ned.z;

  out.body_action[0] = clamp(v_fwd / 5.0, -1.0, 1.0);
  out.body_action[1] = clamp(v_right / 3.0, -1.0, 1.0);
  out.body_action[2] = clamp(v_down / 2.0, -1.0, 1.0);
  out.body_action[3] = clamp(out.yawspeed_rad_s / (kPi / 3.0), -1.0, 1.0);

  return out;
}

}  // namespace ego_planner
