// DecisionGate on synthetic camera-stamp sequences: decision rate / intervals per camera rate and mode, dropouts,
// out-of-order stamps, reset() and is_stale(). Exit code 0 = all checks passed.
//   policy_core_decision_gate_test
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "policynav/decision_gate.hpp"

using policynav::DecisionGate;

static int g_fail = 0;
#define CHECK(cond)                                                     \
  do {                                                                  \
    if (!(cond)) {                                                      \
      std::printf("  FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);     \
      ++g_fail;                                                         \
    }                                                                   \
  } while (0)

struct Stats {
  double rate_hz, min_ms, max_ms, mean_ms, timer_wait_ms;
  int n;
};

// Frames at `hz` with uniform +-jitter_s on each stamp, for `dur` s.
static std::vector<double> stamps(double hz, double jitter_s, double dur, unsigned seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<double> J(-jitter_s, jitter_s);
  std::vector<double> t;
  for (int i = 0; i / hz < dur; ++i) t.push_back(5.0 + i / hz + J(rng));
  return t;
}

static Stats run(DecisionGate gate, const std::vector<double>& ts) {
  std::vector<double> acc;
  for (double t : ts)
    if (gate.on_depth(t)) acc.push_back(t);
  Stats s{0, 1e9, 0, 0, 0, int(acc.size())};
  for (std::size_t i = 1; i < acc.size(); ++i) {
    const double d = 1e3 * (acc[i] - acc[i - 1]);
    s.min_ms = std::min(s.min_ms, d);
    s.max_ms = std::max(s.max_ms, d);
  }
  s.rate_hz = (acc.size() - 1) / (acc.back() - acc.front());
  s.mean_ms = 1e3 / s.rate_hz;
  // For comparison: a free-running 10 Hz timer (arbitrary phase) uses the newest frame at each tick; mean wait from
  // that frame's arrival to the tick (frame arrival ~ stamp here; the gate's own wait is 0 by construction).
  // Averaged over 20 timer phases.
  double w = 0;
  int nw = 0;
  for (int ph = 0; ph < 20; ++ph) {
    std::size_t j = 0;
    for (double tick = ts.front() + 0.005 * ph; tick < ts.back(); tick += 0.1) {
      while (j + 1 < ts.size() && ts[j + 1] <= tick) ++j;
      w += tick - ts[j];
      ++nw;
    }
  }
  s.timer_wait_ms = 1e3 * w / nw;
  return s;
}

int main() {
  const double dur = 60.0;
  struct Case {
    const char* name;
    double hz, jitter;
  } cases[] = {{"15 Hz +-2 ms", 15.0, 0.002}, {"13.5 Hz +-2 ms", 13.5, 0.002}, {"10 Hz +-2 ms", 10.0, 0.002},
               {"20 Hz +-2 ms", 20.0, 0.002}, {"30 Hz +-2 ms", 30.0, 0.002},   {"7 Hz +-2 ms", 7.0, 0.002}};

  std::printf("%-15s %-12s %8s %8s %8s %8s %16s\n", "camera", "mode", "rate Hz", "min ms", "max ms", "mean ms",
              "10Hz-timer wait");
  for (const auto& c : cases) {
    const auto ts = stamps(c.hz, c.jitter, dur, 1234u);
    for (auto mode : {DecisionGate::Mode::kPhaseLocked, DecisionGate::Mode::kMinInterval}) {
      DecisionGate::Params p;
      p.mode = mode;
      const Stats s = run(DecisionGate(p), ts);
      const bool pl = mode == DecisionGate::Mode::kPhaseLocked;
      std::printf("%-15s %-12s %8.3f %8.1f %8.1f %8.1f %13.1f ms\n", c.name, pl ? "phase-locked" : "min-interval",
                  s.rate_hz, s.min_ms, s.max_ms, s.mean_ms, s.timer_wait_ms);
      const double expect = std::min(10.0, c.hz);
      if (pl) {
        CHECK(std::fabs(s.rate_hz - expect) < 0.02 * expect);   // averages 10 Hz (or every frame when slower)
        CHECK(s.min_ms >= 50.0 - 1e-9);
      } else {
        CHECK(s.min_ms >= 80.0 - 1e-9);
        CHECK(s.rate_hz <= 10.0 * 1.25 + 1e-9);
      }
    }
  }

  {  // the 15 Hz pattern explicitly: phase-locked alternates 2 and 1 camera periods (133 / 67 ms)
    DecisionGate g;
    std::vector<int> picked;
    for (int i = 0; i < 12; ++i)
      if (g.on_depth(i / 15.0)) picked.push_back(i);
    const std::vector<int> want = {0, 2, 3, 5, 6, 8, 9, 11};
    CHECK(picked == want);
  }
  {  // dropout: 15 Hz, 1 s gap, then 15 Hz again: accept the first frame after the gap, no burst afterwards
    DecisionGate g;
    std::vector<double> acc;
    for (int i = 0; i < 15; ++i)
      if (g.on_depth(i / 15.0)) acc.push_back(i / 15.0);
    for (int i = 30; i < 60; ++i)
      if (g.on_depth(i / 15.0)) acc.push_back(i / 15.0);
    auto it = std::find_if(acc.begin(), acc.end(), [](double t) { return t >= 1.9; });
    CHECK(it != acc.end() && std::fabs(*it - 2.0) < 1e-9);
    for (std::size_t i = 1; i < acc.size(); ++i) CHECK(acc[i] - acc[i - 1] >= 0.05 - 1e-9);
    int after = int(acc.end() - it);   // 2.0 .. 3.93 s: ~20 decisions at 10 Hz
    CHECK(after >= 19 && after <= 21);
  }
  {  // out-of-order / duplicate stamps never trigger; reset() accepts the next frame immediately
    DecisionGate g;
    CHECK(g.on_depth(1.0));
    CHECK(!g.on_depth(1.0));
    CHECK(!g.on_depth(0.5));
    CHECK(g.last_depth_stamp() == 1.0);
    CHECK(!g.on_depth(1.03));
    g.reset();
    CHECK(g.on_depth(1.04));
    CHECK(!g.on_depth(1.06));
  }
  {  // stale
    DecisionGate g;
    CHECK(g.is_stale(0.0, 0.3));   // no depth yet
    g.on_depth(10.0);
    CHECK(!g.is_stale(10.2, 0.3));
    CHECK(g.is_stale(10.31, 0.3));
    g.on_depth(10.05);   // not accepted, but fresh depth
    CHECK(!g.is_stale(10.31, 0.3));
  }

  std::printf("%s (%d failures)\n", g_fail ? "FAILED" : "PASSED", g_fail);
  return g_fail ? 1 : 0;
}
