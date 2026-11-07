// Event-driven decision trigger: run the policy as soon as a depth frame arrives, at ~10 Hz on average (training's
// decision rate), instead of on a free-running 10 Hz timer (which adds ~50 ms average wait between a depth frame
// and the decision that uses it).
//
//   DecisionGate gate;                       // period 0.1 s
//   on each depth frame (camera stamp t):    if (gate.on_depth(t)) run_policy_on_this_frame();
//   on the control/publish timer:            if (gate.is_stale(now, 0.3)) hold_or_zero();
//
// All times are seconds on ONE clock: camera header stamps (on_depth) and the same clock for `now` (is_stale).
//
// Modes
//   kPhaseLocked (default): a decision is due every `period` on a schedule anchored at the first frame. A frame is
//     accepted when t >= due - (1 - accept_frac) * period (default: up to 20 ms early) AND at least
//     min_gap_frac * period (default 50 ms) after the last decision. On accept the schedule advances by one period;
//     if the next due time is then still <= t (a whole period behind: dropped frames, a stall), the schedule
//     restarts at t + period (no catch-up burst). The AVERAGE rate stays at 1/period whenever frames arrive faster than that;
//     individual intervals are whole camera periods around it (15 Hz camera -> 133/67 ms alternating = 10.0 Hz).
//   kMinInterval: accept the first frame with t - t_last_decision >= accept_frac * period (literal "at most 10 Hz":
//     every interval >= 80 ms, but a 15 Hz camera then gives 133 ms intervals = 7.5 Hz).
#pragma once

namespace policynav {

class DecisionGate {
 public:
  enum class Mode { kPhaseLocked, kMinInterval };

  struct Params {
    double period = 0.1;         // s; training's 10 Hz decision period
    double accept_frac = 0.8;    // a frame up to (1 - accept_frac) * period early counts as on time
    double min_gap_frac = 0.5;   // kPhaseLocked: never two decisions closer than this * period
    Mode mode = Mode::kPhaseLocked;
  };

  DecisionGate() : DecisionGate(Params{}) {}
  explicit DecisionGate(const Params& p) : p_(p) {}

  // A new depth frame with camera stamp t (s). true = run the policy on this frame now.
  bool on_depth(double t);
  // No depth within max_age (s) of `now` (same clock as the stamps), or none ever: the caller holds / zeroes.
  bool is_stale(double now, double max_age) const { return !have_depth_ || now - last_depth_ > max_age; }
  // Forget the schedule (e.g. with policy.reset()); the next frame is accepted immediately.
  void reset();

  const Params& params() const { return p_; }
  bool has_decided() const { return have_decision_; }
  double last_decision_stamp() const { return last_decision_; }
  double last_depth_stamp() const { return last_depth_; }

 private:
  Params p_;
  bool have_depth_ = false, have_decision_ = false;
  double last_depth_ = 0.0, last_decision_ = 0.0, due_ = 0.0;
};

}  // namespace policynav
