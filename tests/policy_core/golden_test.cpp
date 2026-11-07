// Golden test: replays tests/policy_core/make_golden.py's inputs through the C++ policy core and compares outputs
// (max abs error per output, NaN == NaN), then times the C++ path (median of 50).
//   policy_core_golden_test <golden.npz> <policy.npz>      exit code 0 = within tolerance
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <functional>
#include <string>
#include <vector>

#include "policynav/npz.hpp"
#include "policynav/obs.hpp"
#include "policynav/policy.hpp"

using namespace policynav;

static int g_fail = 0;

static void report(const char* what, double err, double tol, const std::string& extra = "") {
  const bool ok = err <= tol;
  std::printf("  %-44s max abs err %.3e  (tol %.0e) %s %s\n", what, err, tol, ok ? "ok" : "FAIL", extra.c_str());
  if (!ok) ++g_fail;
}

template <class A, class B>
static double maxerr(const A* a, const B* b, std::size_t n, std::size_t* n_diff = nullptr) {
  double m = 0;
  std::size_t nd = 0;
  for (std::size_t i = 0; i < n; ++i) {
    const double x = double(a[i]), y = double(b[i]);
    if (std::isnan(x) && std::isnan(y)) continue;
    const double e = (std::isnan(x) || std::isnan(y)) ? INFINITY : std::fabs(x - y);
    if (e > 0) ++nd;
    m = std::max(m, e);
  }
  if (n_diff) *n_diff = nd;
  return m;
}

static double median_ms(const std::function<void()>& fn, int n = 50) {
  fn();
  std::vector<double> t;
  for (int i = 0; i < n; ++i) {
    auto t0 = std::chrono::steady_clock::now();
    fn();
    t.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
  }
  std::sort(t.begin(), t.end());
  return n % 2 ? t[n / 2] : 0.5 * (t[n / 2 - 1] + t[n / 2]);
}

static Mat3 mat3(const std::vector<double>& v, std::size_t off) {
  Mat3 R;
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) R[i][j] = v[off + 3 * i + j];
  return R;
}

int main(int argc, char** argv) {
  if (argc < 3) {
    std::fprintf(stderr, "usage: %s <golden.npz> <policy.npz>\n", argv[0]);
    return 2;
  }
  const NpzFile G(argv[1]);
  PolicyStepper stepper(argv[2]);
  Policy& policy = stepper.policy();
  const PolicyMeta& meta = policy.meta();
  std::printf("policy %s: source %s, %d actions, proprio %d, features %d, hidden %d, depth %dx%d x%d\n", argv[2],
              meta.source.c_str(), meta.num_actions, meta.proprio_dim, policy.feature_dim(), policy.hidden_size(),
              meta.depth_h, meta.depth_w, meta.frame_stack);

  // ── reprojection ──
  const NpyArray& raw = G["depth_raw"];
  const std::size_t F = raw.shape[0], Hs = raw.shape[1], Ws = raw.shape[2], src_n = Hs * Ws;
  const float* rawp = raw.f32();
  const auto K = G["reproj_K"].to_double(), R = G["reproj_R"].to_double(), T = G["reproj_t"].to_double();
  const auto S = G["reproj_stride"].to_int64();
  const auto axis = G["reproj_axis_err"].to_double();
  const NpyArray& rout = G["reproj_out"];
  const std::size_t n_cfg = rout.shape[0], out_n = rout.shape[2] * rout.shape[3];
  std::printf("reprojection: %zu real frames %zux%zu, %zu calibrations\n", F, Hs, Ws, n_cfg);
  std::vector<DepthReprojector> reps;
  for (std::size_t c = 0; c < n_cfg; ++c) {
    reps.push_back(make_reprojector(meta, {K[4 * c], K[4 * c + 1], K[4 * c + 2], K[4 * c + 3]}, int(Hs), int(Ws),
                                    mat3(R, 9 * c), {T[3 * c], T[3 * c + 1], T[3 * c + 2]}, int(S[c])));
    double e = 0;
    std::size_t nd_tot = 0;
    std::vector<float> out(out_n);
    for (std::size_t f = 0; f < F; ++f) {
      reps[c](rawp + f * src_n, out.data());
      std::size_t nd = 0;
      e = std::max(e, maxerr(out.data(), rout.f32() + (c * F + f) * out_n, out_n, &nd));
      nd_tot += nd;
    }
    char name[64], extra[96];
    std::snprintf(name, sizeof name, "reproject cfg %zu (stride %lld)", c, (long long)S[c]);
    std::snprintf(extra, sizeof extra, "[%zu of %zu px differ]", nd_tot, F * out_n);
    report(name, e, 1e-6, extra);
    report("  axis_err_deg", std::fabs(reps[c].axis_err_deg() - axis[c]), 1e-9);
  }

  // ── decode_depth ──
  for (const char* enc : {"16UC1", "32FC1"}) {
    const std::string p = std::string(enc) == "16UC1" ? "dec16_" : "dec32_";
    const NpyArray& buf = G[p + "buf"];
    const NpyArray& want = G[p + "out"];
    const DepthImage d = decode_depth(buf.data.data(), buf.data.size(), enc, std::uint32_t(want.shape[0]),
                                      std::uint32_t(want.shape[1]), std::uint32_t(G[p + "step"].to_int64()[0]));
    report((std::string("decode_depth ") + enc).c_str(), maxerr(d.data.data(), want.f32(), want.size()), 0.0);
  }

  // ── proprio / action_to_setpoint ──
  {
    const auto q = G["pp_q"].to_double(), vel = G["pp_vel"].to_double(), rng = G["pp_range"].to_double();
    const NpyArray& prev = G["pp_prev"];
    const NpyArray& act = G["pp_act"];
    const NpyArray& want = G["pp_proprio"];
    const auto vsp = G["pp_vel_sp"].to_double(), ysp = G["pp_yawspeed"].to_double();
    const std::size_t N = want.shape[0], P = want.shape[1], A = prev.shape[1];
    double ep = 0, ev = 0, ey = 0;
    for (std::size_t i = 0; i < N; ++i) {
      const QuatWXYZ qi{q[4 * i], q[4 * i + 1], q[4 * i + 2], q[4 * i + 3]};
      std::vector<float> pa(prev.f32() + i * A, prev.f32() + (i + 1) * A);
      const auto pro = proprio({vel[3 * i], vel[3 * i + 1], vel[3 * i + 2]}, qi, rng[i], pa, meta.v_max);
      ep = std::max(ep, maxerr(pro.data(), want.f32() + i * P, P));
      const Setpoint sp = action_to_setpoint(act.f32() + i * A, qi, meta.v_max, meta.yaw_rate_max);
      ev = std::max(ev, maxerr(sp.vel_ned.data(), &vsp[3 * i], 3));
      ey = std::max(ey, std::fabs(sp.yawspeed - ysp[i]));
    }
    report("proprio (256 synthetic states)", ep, 1e-6);
    report("action_to_setpoint vel_ned", ev, 1e-9);
    report("action_to_setpoint yawspeed", ey, 1e-12);
  }

  // ── single policy steps from a fresh GRU state ──
  const NpyArray& sd = G["single_depth"];
  const NpyArray& sp = G["single_proprio"];
  const NpyArray& so = G["single_out"];
  {
    const std::size_t N = sd.shape[0], dn = sd.size() / N, P = sp.shape[1], A = so.shape[1];
    double e = 0;
    for (std::size_t i = 0; i < N; ++i) {
      policy.reset();
      const auto out = policy(sd.f32() + i * dn, sp.f32() + i * P);
      e = std::max(e, maxerr(out.data(), so.f32() + i * A, A));
    }
    report("policy forward, single steps (raw actions)", e, 1e-4, "[" + std::to_string(N) + " cases]");
  }

  // ── multi-step sequence (GRU state carried, reset() mid-sequence), nodes' per-decision step ──
  {
    const auto q = G["seq_q"].to_double(), vel = G["seq_vel"].to_double(), rng = G["seq_range"].to_double();
    const auto fr = G["seq_frame"].to_int64(), rs = G["seq_reset"].to_int64();
    const NpyArray& wa = G["seq_action"];
    const NpyArray& wp = G["seq_proprio"];
    const NpyArray& wh = G["seq_h"];
    const auto wv = G["seq_vel_sp"].to_double(), wy = G["seq_yawspeed"].to_double();
    const std::size_t Tn = wa.shape[0], A = wa.shape[1], P = wp.shape[1], Hn = wh.shape[1];
    for (int e2e = 0; e2e < 2; ++e2e) {   // 0: Python's reprojected frames; 1: C++ reprojection of the raw frames
      PolicyStepper st(argv[2]);
      double ea = 0, ep = 0, eh = 0, ev = 0, ey = 0;
      std::vector<float> img(out_n);
      for (std::size_t k = 0; k < Tn; ++k) {
        if (std::find(rs.begin(), rs.end(), std::int64_t(k)) != rs.end()) {
          st.reset();
          bool zero = std::all_of(st.policy().hidden().begin(), st.policy().hidden().end(), [](float v) { return v == 0; }) &&
                      std::all_of(st.prev_action().begin(), st.prev_action().end(), [](float v) { return v == 0; });
          if (!zero) {
            std::printf("  FAIL reset() left state non-zero\n");
            ++g_fail;
          }
        }
        if (e2e) reps[0](rawp + std::size_t(fr[k]) * src_n, img.data());
        else img.assign(rout.f32() + std::size_t(fr[k]) * out_n, rout.f32() + std::size_t(fr[k] + 1) * out_n);
        const QuatWXYZ qk{q[4 * k], q[4 * k + 1], q[4 * k + 2], q[4 * k + 3]};
        const auto r = st.step(img, {vel[3 * k], vel[3 * k + 1], vel[3 * k + 2]}, qk, rng[k]);
        ea = std::max(ea, maxerr(r.action.data(), wa.f32() + k * A, A));
        ep = std::max(ep, maxerr(r.proprio.data(), wp.f32() + k * P, P));
        eh = std::max(eh, maxerr(st.policy().hidden().data(), wh.f32() + k * Hn, Hn));
        ev = std::max(ev, maxerr(r.setpoint.vel_ned.data(), &wv[3 * k], 3));
        ey = std::max(ey, std::fabs(r.setpoint.yawspeed - wy[k]));
      }
      const std::string tag = e2e ? "seq e2e (C++ reproj) " : "seq " + std::to_string(Tn) + " steps ";
      report((tag + "action (clipped)").c_str(), ea, 1e-4);
      report((tag + "proprio (prev action fed back)").c_str(), ep, 1e-4);
      report((tag + "GRU hidden state").c_str(), eh, 1e-4);
      report((tag + "setpoint vel_ned [m/s]").c_str(), ev, 5e-4);
      report((tag + "setpoint yawspeed [rad/s]").c_str(), ey, 2e-4);
    }
  }

  // ── timings (median of 50) ──
  std::vector<float> out(out_n);
  const double t_rep = median_ms([&] { reps[0](rawp, out.data()); });
  const std::size_t dn = sd.size() / sd.shape[0];
  const double t_pol = median_ms([&] { policy(sd.f32(), sp.f32()); });
  std::vector<float> img(rout.f32(), rout.f32() + out_n);
  const double t_step = median_ms([&] { stepper.step(img, {1.0, 0.2, -0.1}, {0.98, 0.0, 0.05, 0.19}, 1.0); });
  (void)dn;
  std::printf("timing (median of 50): C++ reproject %.3f ms, policy forward %.3f ms, full step %.3f ms\n", t_rep, t_pol,
              t_step);
  if (G.has("py_ms_reproj"))
    std::printf("timing (median of 50): Python reproject %.3f ms, policy forward %.3f ms, full step %.3f ms\n",
                G["py_ms_reproj"].to_double()[0], G["py_ms_policy"].to_double()[0], G["py_ms_step"].to_double()[0]);

  std::printf("%s (%d failures)\n", g_fail ? "FAILED" : "PASSED", g_fail);
  return g_fail ? 1 : 0;
}
