#include "policynav/decision_gate.hpp"

namespace policynav {

void DecisionGate::reset() {
  have_decision_ = false;
  last_decision_ = due_ = 0.0;
}

bool DecisionGate::on_depth(double t) {
  // Stale-depth bookkeeping uses the newest stamp seen; an out-of-order (older) frame never triggers a decision.
  if (!have_depth_ || t > last_depth_) last_depth_ = t;
  have_depth_ = true;

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

}  // namespace policynav
