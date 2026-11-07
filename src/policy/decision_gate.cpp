#include "policynav/decision_gate.hpp"

#include <stdexcept>

namespace policynav {

DecisionGate::Mode DecisionGate::parse_mode(const std::string& name) {
  if (name == "phase_locked") return Mode::kPhaseLocked;
  if (name == "min_interval") return Mode::kMinInterval;
  if (name == "fixed_clock") return Mode::kFixedClock;
  throw std::invalid_argument("DecisionGate: unknown mode '" + name +
                              "' (expected 'phase_locked', 'min_interval' or 'fixed_clock')");
}

const char* DecisionGate::mode_name(Mode m) {
  switch (m) {
    case Mode::kPhaseLocked: return "phase_locked";
    case Mode::kMinInterval: return "min_interval";
    case Mode::kFixedClock: return "fixed_clock";
  }
  return "?";
}

DecisionGate::DecisionGate(const std::string& mode, double period) {
  if (!(period > 0.0)) throw std::invalid_argument("DecisionGate: period must be > 0");
  p_.mode = parse_mode(mode);
  p_.period = period;
}

void DecisionGate::reset() {
  have_decision_ = false;
  last_decision_ = due_ = 0.0;
}

bool DecisionGate::on_depth(double t) {
  // Stale-depth bookkeeping uses the newest stamp seen; an out-of-order (older) frame never triggers a decision.
  if (!have_depth_ || t > last_depth_) last_depth_ = t;
  have_depth_ = true;
  if (p_.mode == Mode::kFixedClock) return false;

  if (!have_decision_) {
    have_decision_ = true;
    last_decision_ = t;
    due_ = t + p_.period;
    return true;
  }
  if (t <= last_decision_) return false;
  const double since = t - last_decision_;

  if (p_.mode == Mode::kMinInterval) {
    if (since < p_.accept_frac * p_.period) return false;
    last_decision_ = t;
    return true;
  }

  const double min_gap = p_.min_gap_frac * p_.period;
  if (t < due_ - (1.0 - p_.accept_frac) * p_.period || since < min_gap) return false;
  last_decision_ = t;
  due_ += p_.period;
  if (due_ <= t) due_ = t + p_.period;   // a whole period behind (dropped frames / stall): restart the schedule
  return true;
}

bool DecisionGate::on_tick(double now, double max_age) {
  if (p_.mode != Mode::kFixedClock)
    throw std::logic_error(std::string("DecisionGate::on_tick is for 'fixed_clock'; mode is '") + mode_name(p_.mode) +
                           "' (use on_depth)");
  if (is_stale(now, max_age)) return false;
  have_decision_ = true;
  last_decision_ = last_depth_;
  return true;
}

}  // namespace policynav
