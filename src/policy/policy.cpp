// Port of policynav/policy_np.py: conv (im2col + GEMM) -> ELU [-> 3x3 s2 p1 maxpool] x n_conv, GRU cell, MLP.
#include "policynav/policy.hpp"

#include <Eigen/Core>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

#include "mini_json.hpp"
#include "policynav/npz.hpp"

namespace policynav {
namespace {

using MatR = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using VecF = Eigen::VectorXf;

inline float elu(float x) { return x > 0.0f ? x : std::expm1(std::min(x, 0.0f)); }
inline float sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }

MatR to_matrix(const NpyArray& a, std::size_t rows, std::size_t cols) {
  if (a.size() != rows * cols) throw std::runtime_error("policy: weight size mismatch");
  const std::vector<float> v = a.to_float();
  return Eigen::Map<const MatR>(v.data(), Eigen::Index(rows), Eigen::Index(cols));
}

VecF to_vector(const NpyArray& a) {
  const std::vector<float> v = a.to_float();
  return Eigen::Map<const VecF>(v.data(), Eigen::Index(v.size()));
}

struct ConvLayer {
  int out_c, in_c, kh, kw, sh, sw, ph, pw;
  bool pool;
  MatR w;   // (O, C*kh*kw)
  VecF b;
  // shapes for the configured input
  int H, W, Ho, Wo, Hp, Wp;   // input, conv output, after pool
};

}  // namespace

struct Policy::Impl {
  PolicyMeta meta;
  std::vector<ConvLayer> conv;
  MatR w_ih, w_hh;
  VecF b_ih, b_hh;
  std::vector<std::pair<MatR, VecF>> mlp;
  int hidden = 0, feat_dim = 0;
  std::vector<float> h;
  // scratch
  std::vector<float> padded, act, pooled, layer_in;
  MatR cols, conv_out;
  VecF feat, gi, gh, y;

  void forward_conv(const float* depth);
};

namespace {

PolicyMeta parse_meta(const std::string& s) {
  const json::Value j = json::parse(s);
  PolicyMeta m;
  m.json = s;
  auto num = [&](const char* k, double dflt) { return j.has(k) ? j.at(k).number() : dflt; };
  auto str = [&](const char* k) { return j.has(k) && j.at(k).type == json::Value::kString ? j.at(k).string() : std::string(); };
  m.mode = str("mode");
  m.source = str("source");
  m.num_actions = int(j.at("num_actions").number());
  m.proprio_dim = int(j.at("proprio_dim").number());
  m.frame_stack = int(num("frame_stack", 2));
  m.n_conv = int(j.at("n_conv").number());
  m.n_mlp = int(j.at("n_mlp").number());
  if (j.has("depth_hw")) {
    const auto& hw = j.at("depth_hw").arr;
    if (hw.size() != 2) throw std::runtime_error("policy meta: depth_hw must have 2 entries");
    m.depth_h = int(hw[0].number());
    m.depth_w = int(hw[1].number());
  }
  if (j.has("v_max")) {
    const auto& v = j.at("v_max").arr;
    if (v.size() != 3) throw std::runtime_error("policy meta: v_max must have 3 entries");
    for (int i = 0; i < 3; ++i) m.v_max[i] = v[i].number();
  }
  m.yaw_rate_max = num("yaw_rate_max", m.yaw_rate_max);
  if (j.has("v_obs_scale")) {
    const auto& v = j.at("v_obs_scale").arr;
    if (v.size() != 3) throw std::runtime_error("policy meta: v_obs_scale must have 3 entries");
    for (int i = 0; i < 3; ++i) m.v_obs_scale[i] = v[i].number();
  }
  for (double v : m.v_obs_scale)
    if (!std::isfinite(v) || v <= 0)
      throw std::runtime_error("policy meta: v_obs_scale must be finite and positive");
  m.far_clip = num("far_clip", m.far_clip);
  m.hfov_deg = num("hfov_deg", m.hfov_deg);
  m.vfov_deg = num("vfov_deg", m.vfov_deg);
  m.rf_max = num("rf_max", m.rf_max);
  return m;
}

}  // namespace

Policy::Policy(const std::string& npz_path) : impl_(std::make_unique<Impl>()) {
  const NpzFile z(npz_path);
  Impl& P = *impl_;
  P.meta = parse_meta(z["meta"].to_string());
  const PolicyMeta& m = P.meta;

  int C = m.frame_stack, H = m.depth_h, W = m.depth_w;
  for (int i = 0; i < m.n_conv; ++i) {
    const std::string p = "conv" + std::to_string(i) + "_";
    const NpyArray& wa = z[p + "w"];
    if (wa.ndim() != 4) throw std::runtime_error("policy: " + p + "w must be 4-D");
    ConvLayer L;
    L.out_c = int(wa.shape[0]);
    L.in_c = int(wa.shape[1]);
    L.kh = int(wa.shape[2]);
    L.kw = int(wa.shape[3]);
    if (L.in_c != C) throw std::runtime_error("policy: " + p + "w expects " + std::to_string(L.in_c) + " channels, got " + std::to_string(C));
    const auto st = z[p + "stride"].to_int64(), pd = z[p + "pad"].to_int64();
    if (st.size() != 2 || pd.size() != 2) throw std::runtime_error("policy: " + p + "stride/pad must have 2 entries");
    L.sh = int(st[0]);
    L.sw = int(st[1]);
    L.ph = int(pd[0]);
    L.pw = int(pd[1]);
    L.pool = z[p + "pool"].to_bool();
    L.w = to_matrix(wa, L.out_c, std::size_t(L.in_c) * L.kh * L.kw);
    L.b = to_vector(z[p + "b"]);
    if (L.b.size() != L.out_c) throw std::runtime_error("policy: " + p + "b size mismatch");
    L.H = H;
    L.W = W;
    L.Ho = (H + 2 * L.ph - L.kh) / L.sh + 1;
    L.Wo = (W + 2 * L.pw - L.kw) / L.sw + 1;
    L.Hp = L.pool ? (L.Ho + 2 - 3) / 2 + 1 : L.Ho;
    L.Wp = L.pool ? (L.Wo + 2 - 3) / 2 + 1 : L.Wo;
    if (L.Ho < 1 || L.Wo < 1 || L.Hp < 1 || L.Wp < 1) throw std::runtime_error("policy: " + p + " output is empty");
    C = L.out_c;
    H = L.Hp;
    W = L.Wp;
    P.conv.push_back(std::move(L));
  }
  const int cnn_dim = C * H * W;
  P.feat_dim = m.proprio_dim + cnn_dim;

  const NpyArray& wih = z["gru_w_ih"];
  const NpyArray& whh = z["gru_w_hh"];
  if (wih.ndim() != 2 || whh.ndim() != 2) throw std::runtime_error("policy: GRU weights must be 2-D");
  P.hidden = int(whh.shape[1]);
  if (int(wih.shape[0]) != 3 * P.hidden || int(whh.shape[0]) != 3 * P.hidden)
    throw std::runtime_error("policy: GRU weight rows must be 3 * hidden");
  if (int(wih.shape[1]) != P.feat_dim)
    throw std::runtime_error("policy: GRU input is " + std::to_string(wih.shape[1]) + ", features are " +
                             std::to_string(P.feat_dim) + " (proprio " + std::to_string(m.proprio_dim) + " + CNN " +
                             std::to_string(cnn_dim) + ")");
  P.w_ih = to_matrix(wih, wih.shape[0], wih.shape[1]);
  P.w_hh = to_matrix(whh, whh.shape[0], whh.shape[1]);
  P.b_ih = to_vector(z["gru_b_ih"]);
  P.b_hh = to_vector(z["gru_b_hh"]);
  if (P.b_ih.size() != 3 * P.hidden || P.b_hh.size() != 3 * P.hidden) throw std::runtime_error("policy: GRU bias size");

  int in = P.hidden;
  for (int i = 0; i < m.n_mlp; ++i) {
    const std::string p = "mlp" + std::to_string(i) + "_";
    const NpyArray& wa = z[p + "w"];
    if (wa.ndim() != 2 || int(wa.shape[1]) != in) throw std::runtime_error("policy: " + p + "w shape mismatch");
    VecF b = to_vector(z[p + "b"]);
    if (b.size() != Eigen::Index(wa.shape[0])) throw std::runtime_error("policy: " + p + "b size mismatch");
    P.mlp.emplace_back(to_matrix(wa, wa.shape[0], wa.shape[1]), std::move(b));
    in = int(wa.shape[0]);
  }
  if (in != m.num_actions) throw std::runtime_error("policy: MLP output " + std::to_string(in) + " != num_actions");
  P.h.assign(P.hidden, 0.0f);
  P.feat.resize(P.feat_dim);
}

Policy::~Policy() = default;
Policy::Policy(Policy&&) noexcept = default;
Policy& Policy::operator=(Policy&&) noexcept = default;

const PolicyMeta& Policy::meta() const { return impl_->meta; }
int Policy::hidden_size() const { return impl_->hidden; }
int Policy::feature_dim() const { return impl_->feat_dim; }
void Policy::reset() { std::fill(impl_->h.begin(), impl_->h.end(), 0.0f); }
const std::vector<float>& Policy::hidden() const { return impl_->h; }
void Policy::set_hidden(const std::vector<float>& h) {
  if (int(h.size()) != impl_->hidden) throw std::invalid_argument("Policy::set_hidden: wrong size");
  impl_->h = h;
}

void Policy::Impl::forward_conv(const float* depth) {
  const float* x = depth;
  for (const ConvLayer& L : conv) {
    // zero pad
    const int Hpd = L.H + 2 * L.ph, Wpd = L.W + 2 * L.pw;
    padded.assign(std::size_t(L.in_c) * Hpd * Wpd, 0.0f);
    for (int c = 0; c < L.in_c; ++c)
      for (int r = 0; r < L.H; ++r)
        std::copy_n(x + (std::size_t(c) * L.H + r) * L.W, L.W,
                    &padded[(std::size_t(c) * Hpd + r + L.ph) * Wpd + L.pw]);
    // im2col: rows (c, i, j), cols (oh, ow)
    const int K = L.in_c * L.kh * L.kw, N = L.Ho * L.Wo;
    cols.resize(K, N);
    for (int c = 0; c < L.in_c; ++c)
      for (int i = 0; i < L.kh; ++i)
        for (int j = 0; j < L.kw; ++j) {
          float* dst = cols.data() + std::size_t((c * L.kh + i) * L.kw + j) * N;
          for (int oh = 0; oh < L.Ho; ++oh) {
            const float* srow = &padded[(std::size_t(c) * Hpd + oh * L.sh + i) * Wpd + j];
            for (int ow = 0; ow < L.Wo; ++ow) *dst++ = srow[ow * L.sw];
          }
        }
    conv_out.noalias() = L.w * cols;
    conv_out.colwise() += L.b;
    act.resize(std::size_t(L.out_c) * N);
    const float* co = conv_out.data();
    for (std::size_t k = 0; k < act.size(); ++k) act[k] = elu(co[k]);
    if (L.pool) {   // 3x3, stride 2, pad 1 with -inf
      pooled.resize(std::size_t(L.out_c) * L.Hp * L.Wp);
      for (int c = 0; c < L.out_c; ++c)
        for (int ph = 0; ph < L.Hp; ++ph)
          for (int pw = 0; pw < L.Wp; ++pw) {
            float mx = -std::numeric_limits<float>::infinity();
            for (int i = 0; i < 3; ++i) {
              const int r = 2 * ph + i - 1;
              if (r < 0 || r >= L.Ho) continue;
              for (int j = 0; j < 3; ++j) {
                const int q = 2 * pw + j - 1;
                if (q < 0 || q >= L.Wo) continue;
                mx = std::max(mx, act[(std::size_t(c) * L.Ho + r) * L.Wo + q]);
              }
            }
            pooled[(std::size_t(c) * L.Hp + ph) * L.Wp + pw] = mx;
          }
      std::swap(act, pooled);
    }
    std::swap(act, layer_in);   // this layer's output is the next layer's input (no aliasing with act / pooled)
    x = layer_in.data();
  }
  std::copy(x, x + (feat_dim - meta.proprio_dim), feat.data() + meta.proprio_dim);
}

std::vector<float> Policy::operator()(const float* depth, const float* proprio) {
  Impl& P = *impl_;
  std::copy(proprio, proprio + P.meta.proprio_dim, P.feat.data());   // 1-D groups first, then CNN
  P.forward_conv(depth);

  const int H = P.hidden;
  Eigen::Map<VecF> h(P.h.data(), H);
  P.gi.noalias() = P.w_ih * P.feat;
  P.gi += P.b_ih;
  P.gh.noalias() = P.w_hh * h;
  P.gh += P.b_hh;
  for (int k = 0; k < H; ++k) {
    const float r = sigmoid(P.gi[k] + P.gh[k]);
    const float u = sigmoid(P.gi[H + k] + P.gh[H + k]);
    const float n = std::tanh(P.gi[2 * H + k] + r * P.gh[2 * H + k]);
    P.h[k] = (1.0f - u) * n + u * P.h[k];
  }
  P.y = h;
  for (std::size_t i = 0; i < P.mlp.size(); ++i) {
    VecF z = P.mlp[i].first * P.y;
    z += P.mlp[i].second;
    if (i + 1 < P.mlp.size())
      for (Eigen::Index k = 0; k < z.size(); ++k) z[k] = elu(z[k]);
    P.y = std::move(z);
  }
  return std::vector<float>(P.y.data(), P.y.data() + P.y.size());
}

std::vector<float> Policy::operator()(const std::vector<float>& depth, const std::vector<float>& proprio) {
  const PolicyMeta& m = impl_->meta;
  if (depth.size() != std::size_t(m.frame_stack) * m.depth_h * m.depth_w)
    throw std::invalid_argument("Policy: depth stack has " + std::to_string(depth.size()) + " values, expected " +
                                std::to_string(m.frame_stack * m.depth_h * m.depth_w));
  if (int(proprio.size()) != m.proprio_dim)
    throw std::invalid_argument("Policy: proprio has " + std::to_string(proprio.size()) + " values, expected " +
                                std::to_string(m.proprio_dim));
  return (*this)(depth.data(), proprio.data());
}

DepthReprojector make_reprojector(const PolicyMeta& meta, const std::array<double, 4>& K, int src_h, int src_w,
                                  const Mat3& R_body_cam, const Vec3& t_body_cam, int stride) {
  DepthReprojector::Params p;
  p.out_h = meta.depth_h;
  p.out_w = meta.depth_w;
  p.hfov_deg = meta.hfov_deg;
  p.vfov_deg = meta.vfov_deg;
  p.far = meta.far_clip;
  p.stride = stride;
  return DepthReprojector(K, src_h, src_w, R_body_cam, t_body_cam, p);
}

PolicyStepper::PolicyStepper(const std::string& npz_path)
    : policy_(npz_path), stack_(policy_.meta().frame_stack), prev_action_(policy_.meta().num_actions, 0.0f) {}

void PolicyStepper::reset() {
  policy_.reset();
  stack_.reset();
  std::fill(prev_action_.begin(), prev_action_.end(), 0.0f);
}

PolicyStepper::Result PolicyStepper::step(const std::vector<float>& depth_img, const Vec3& vel_ned, const QuatWXYZ& q,
                                          double range_m) {
  const PolicyMeta& m = policy_.meta();
  if (depth_img.size() != std::size_t(m.depth_h) * m.depth_w)
    throw std::invalid_argument("PolicyStepper: depth frame size mismatch");
  if (m.num_actions < 4) throw std::runtime_error("PolicyStepper: policy has fewer than 4 actions");
  Result res;
  const std::vector<float>& img = stack_.push(depth_img);
  res.proprio = proprio(vel_ned, q, range_m, prev_action_, m.v_obs_scale);
  res.action = policy_(img, res.proprio);
  for (float& a : res.action) a = std::max(-1.0f, std::min(1.0f, a));
  prev_action_ = res.action;
  res.setpoint = action_to_setpoint(res.action.data(), q, m.v_max, m.yaw_rate_max);
  return res;
}

}  // namespace policynav
