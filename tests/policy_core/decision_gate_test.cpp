// DecisionGate on synthetic camera-stamp sequences: decision rate / intervals per camera rate and mode, dropouts,
// out-of-order stamps, reset() and is_stale(). Exit code 0 = all checks passed.
//   policy_core_decision_gate_test
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <stdexcept>
#include <string>
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
  double rate_hz, min_ms, max_ms, age_mean_ms, age_p95_ms, reused_pct;
};

// Frames at `hz` with uniform +-jitter_s on each stamp, for `dur` s.
static std::vector<double> stamps(double hz, double jitter_s, double dur, unsigned seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<double> J(-jitter_s, jitter_s);
  std::vector<double> t;
  for (int i = 0; i / hz < dur; ++i) t.push_back(5.0 + i / hz + J(rng));
  return t;
}

// Decisions as (decision time, stamp of the frame used). Frame arrival is taken as its stamp (the upstream camera ->
// node latency is the same for every mode), so "age" = decision time - stamp = the wait the trigger adds.
static Stats summarize(std::vector<std::pair<double, double>> dec, double span, int n_runs) {
  Stats s{0, 1e9, 0, 0, 0, 0};
  std::vector<double> age;
  int reused = 0;
  for (std::size_t i = 0; i < dec.size(); ++i) {
    age.push_back(1e3 * (dec[i].first - dec[i].second));
    if (i && dec[i].first > dec[i - 1].first) {   // consecutive within one run
      const double d = 1e3 * (dec[i].first - dec[i - 1].first);
      s.min_ms = std::min(s.min_ms, d);
      s.max_ms = std::max(s.max_ms, d);
      reused += dec[i].second == dec[i - 1].second;
    }
  }
  s.rate_hz = dec.size() / (span * n_runs);
  for (double a : age) s.age_mean_ms += a / age.size();
  std::sort(age.begin(), age.end());
  s.age_p95_ms = age[std::size_t(0.95 * (age.size() - 1))];
  s.reused_pct = 100.0 * reused / dec.size();
  return s;
}

static Stats run_arrival(DecisionGate gate, const std::vector<double>& ts) {
  std::vector<std::pair<double, double>> dec;
  for (double t : ts)
    if (gate.on_depth(t)) dec.push_back({t, t});   // decides on arrival: age 0 (+ compute, not modelled)
  return summarize(dec, ts.back() - ts.front(), 1);
}

// fixed_clock: a period timer at 20 different phases; each tick decides on the newest frame received so far.
static Stats run_fixed_clock(const std::vector<double>& ts) {
  std::vector<std::pair<double, double>> dec;
  for (int ph = 0; ph < 20; ++ph) {
    DecisionGate g("fixed_clock");
    std::size_t j = 0;
    for (double tick = ts.front() + 0.005 * ph; tick < ts.back(); tick += g.params().period) {
      while (j < ts.size() && ts[j] <= tick) g.on_depth(ts[j++]);
      if (g.on_tick(tick, 0.3)) dec.push_back({tick, g.last_decision_stamp()});
    }
  }
  return summarize(dec, ts.back() - ts.front(), 20);
}

int main() {
  const double dur = 60.0;
  struct Case {
    const char* name;
    double hz, jitter;
  } cases[] = {{"15 Hz +-2 ms", 15.0, 0.002}, {"13.5 Hz +-2 ms", 13.5, 0.002}, {"10 Hz +-2 ms", 10.0, 0.002},
               {"20 Hz +-2 ms", 20.0, 0.002}, {"30 Hz +-2 ms", 30.0, 0.002},   {"7 Hz +-2 ms", 7.0, 0.002}};

  std::printf("%-15s %-13s %8s %7s %7s %13s %12s %9s\n", "camera", "mode", "rate Hz", "min ms", "max ms",
              "age mean ms", "age p95 ms", "reused %");
  for (const auto& c : cases) {
    const auto ts = stamps(c.hz, c.jitter, dur, 1234u);
    for (const char* mode : {"phase_locked", "min_interval", "fixed_clock"}) {
      const auto m = DecisionGate::parse_mode(mode);
      const Stats s = m == DecisionGate::Mode::kFixedClock ? run_fixed_clock(ts) : run_arrival(DecisionGate(mode), ts);
      std::printf("%-15s %-13s %8.3f %7.1f %7.1f %13.1f %12.1f %9.1f\n", c.name, mode, s.rate_hz, s.min_ms, s.max_ms,
                  s.age_mean_ms, s.age_p95_ms, s.reused_pct);
      const double expect = std::min(10.0, c.hz);
      if (m == DecisionGate::Mode::kPhaseLocked) {
        CHECK(std::fabs(s.rate_hz - expect) < 0.02 * expect);   // averages 10 Hz (or every frame when slower)
        CHECK(s.min_ms >= 50.0 - 1e-9);
        CHECK(s.age_p95_ms == 0.0);
      } else if (m == DecisionGate::Mode::kMinInterval) {
        CHECK(s.min_ms >= 80.0 - 1e-9);
        CHECK(s.rate_hz <= 10.0 * 1.25 + 1e-9);
      } else {
        CHECK(std::fabs(s.rate_hz - 10.0) < 0.02 * 10.0);         // the clock rate, whatever the camera does
        CHECK(std::fabs(s.min_ms - 100.0) < 1e-6 && std::fabs(s.max_ms - 100.0) < 1e-6);
        CHECK(s.age_mean_ms > 0.0 && s.age_p95_ms <= 1e3 / c.hz + 2.0 + 1e-6);   // < one camera period (+jitter)
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

  {  // mode strings
    using M = DecisionGate::Mode;
    CHECK(DecisionGate::parse_mode("phase_locked") == M::kPhaseLocked);
    CHECK(DecisionGate::parse_mode("min_interval") == M::kMinInterval);
    CHECK(DecisionGate::parse_mode("fixed_clock") == M::kFixedClock);
    for (M m : {M::kPhaseLocked, M::kMinInterval, M::kFixedClock})
      CHECK(DecisionGate::parse_mode(DecisionGate::mode_name(m)) == m);
    CHECK(DecisionGate().mode() == M::kPhaseLocked);   // default
    for (const char* bad : {"phase-locked", "", "Fixed_Clock", "timer"}) {
      bool threw = false;
      try {
        DecisionGate g(bad);
      } catch (const std::invalid_argument& e) {
        threw = std::string(e.what()).find("expected 'phase_locked', 'min_interval' or 'fixed_clock'") != std::string::npos;
        if (std::string(bad) == "timer") std::printf("unknown mode -> %s\n", e.what());
      }
      CHECK(threw);
    }
    bool threw = false;
    try {
      DecisionGate g("fixed_clock", 0.0);
    } catch (const std::invalid_argument&) {
      threw = true;
    }
    CHECK(threw);
    threw = false;
    try {
      DecisionGate("phase_locked").on_tick(1.0);
    } catch (const std::logic_error&) {
      threw = true;
    }
    CHECK(threw);   // on_tick only in fixed_clock
  }
  {  // fixed_clock: on_depth never triggers; ticks reuse the newest frame until it is stale
    DecisionGate g("fixed_clock");
    CHECK(!g.on_tick(1.0, 0.3));   // no depth yet
    CHECK(!g.on_depth(1.00));
    CHECK(!g.on_depth(1.07));
    CHECK(!g.on_depth(1.03));      // out of order: newest stays 1.07
    CHECK(g.on_tick(1.10, 0.3) && g.last_decision_stamp() == 1.07);
    CHECK(g.on_tick(1.20, 0.3) && g.last_decision_stamp() == 1.07);   // no new frame: same frame again
    CHECK(g.on_tick(1.30));                                            // default max_age: never stale
    CHECK(!g.on_tick(1.40, 0.3));                                      // 0.33 s old: stale
    g.on_depth(1.45);
    CHECK(g.on_tick(1.50, 0.3) && g.last_decision_stamp() == 1.45);
  }

  std::printf("%s (%d failures)\n", g_fail ? "FAILED" : "PASSED", g_fail);
  return g_fail ? 1 : 0;
}
