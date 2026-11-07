// Decision trigger for the policy: when to run a decision, at training's 10 Hz.
//
// Arrival-driven modes ("phase_locked" = default, "min_interval"): run the policy as soon as a depth frame arrives
// instead of on a free-running 10 Hz timer (which adds ~33-50 ms average wait between a depth frame and the
// decision that uses it):
//   DecisionGate gate("phase_locked");             // period 0.1 s
//   on each depth frame (camera stamp t):          if (gate.on_depth(t)) run_policy_on_this_frame();
//   on the control/publish timer:                  if (gate.is_stale(now, 0.3)) hold_or_zero();
//
// Clock-driven mode ("fixed_clock", today's Python node behaviour): a 100 ms timer decides on the NEWEST depth
// frame received so far. on_depth() only records the frame (always returns false); the node's timer calls
// on_tick(now). on_tick returns true whenever any depth has been received, even if no new frame arrived since the
// last tick (the newest frame is then used again, exactly like the Python timer), unless that frame is older than
// max_age (pass the stale threshold; default: never stale).
//   on each depth frame (camera stamp t):          gate.on_depth(t); newest = frame;
//   on a period timer:                             if (gate.on_tick(now, 0.3)) run_policy_on(newest);
//
// All times are seconds on ONE clock: camera header stamps (on_depth) and the same clock for `now`.
//
// Modes
//   phase_locked (default): a decision is due every `period` on a schedule anchored at the first frame. A frame is
//     accepted when t >= due - (1 - accept_frac) * period (default: up to 20 ms early) AND at least
//     min_gap_frac * period (default 50 ms) after the last decision. On accept the schedule advances by one period;
//     if the next due time is then still <= t (a whole period behind: dropped frames, a stall), the schedule
//     restarts at t + period (no catch-up burst). The AVERAGE rate stays at 1/period whenever frames arrive faster
//     than that; individual intervals are whole camera periods around it (15 Hz camera -> 133/67 ms = 10.0 Hz).
//   min_interval: accept the first frame with t - t_last_decision >= accept_frac * period (every interval >= 80 ms,
//     but a 15 Hz camera then gives 133 ms intervals = 7.5 Hz).
//   fixed_clock: see above; the timer (period) sets the rate, the gate only tracks the newest frame.
#pragma once

#include <limits>
#include <string>

namespace policynav {

class DecisionGate {
 public:
  enum class Mode { kPhaseLocked, kMinInterval, kFixedClock };

  // "phase_locked" | "min_interval" | "fixed_clock" -> Mode; anything else throws std::invalid_argument.
  static Mode parse_mode(const std::string& name);
  static const char* mode_name(Mode m);

  struct Params {
    double period = 0.1;         // s; training's 10 Hz decision period (fixed_clock: the node's timer period)
    double accept_frac = 0.8;    // a frame up to (1 - accept_frac) * period early counts as on time
    double min_gap_frac = 0.5;   // phase_locked: never two decisions closer than this * period
    Mode mode = Mode::kPhaseLocked;
  };

  DecisionGate() : DecisionGate(Params{}) {}
  explicit DecisionGate(const Params& p) : p_(p) {}
  explicit DecisionGate(const std::string& mode, double period = 0.1);   // e.g. from a ROS parameter

  // A new depth frame with camera stamp t (s). Arrival modes: true = run the policy on this frame now.
  // fixed_clock: records the frame as the newest and returns false (decisions come from on_tick).
  bool on_depth(double t);
  // fixed_clock only (throws std::logic_error in the arrival modes): a timer tick at `now`. true = run the policy on
  // the newest frame (stamp: last_depth_stamp()); false if no depth yet or the newest is older than max_age.
  bool on_tick(double now, double max_age = std::numeric_limits<double>::infinity());
  // No depth within max_age (s) of `now` (same clock as the stamps), or none ever: the caller holds / zeroes.
  bool is_stale(double now, double max_age) const { return !have_depth_ || now - last_depth_ > max_age; }
  // Forget the schedule (e.g. with policy.reset()); arrival modes accept the next frame immediately.
  void reset();

  const Params& params() const { return p_; }
  Mode mode() const { return p_.mode; }
  bool has_decided() const { return have_decision_; }
  double last_decision_stamp() const { return last_decision_; }   // stamp of the frame the last decision used
  double last_depth_stamp() const { return last_depth_; }

 private:
  Params p_;
  bool have_depth_ = false, have_decision_ = false;
  double last_depth_ = 0.0, last_decision_ = 0.0, due_ = 0.0;
};

}  // namespace policynav
