// The trained CNN+GRU actor in C++: a port of policynav/policy_np.py (NumpyPolicy), float32 like NumPy.
// Weights: the same .npz files (policies/*.npz), read directly (no Python at runtime).
//
//   policynav::Policy policy("policies/cur4_clean50_800.npz");
//   std::vector<float> a = policy(depth_stack /* (frame_stack, H, W) */, proprio /* (proprio_dim) */);
//   policy.reset();                                      // zero the GRU state
//
// PolicyStepper bundles what skybeta_node.py / handheld_node.py do per decision (frame stack push, proprio from
// velocity / attitude / range / previous action, forward, clip to [-1, 1], previous-action feedback, setpoint).
#pragma once

#include <memory>
#include <string>
#include <vector>

#include "policynav/obs.hpp"

namespace policynav {

struct PolicyMeta {
  std::string mode, source;
  int num_actions = 4, proprio_dim = 10, frame_stack = 2, n_conv = 0, n_mlp = 0;
  int depth_h = 80, depth_w = 128;
  Vec3 v_max{5.0, 3.0, 2.0};
  Vec3 v_obs_scale{5.0, 3.0, 2.0};  // fixed training scale, including legacy exports
  double yaw_rate_max = 1.0, far_clip = 5.0, hfov_deg = 72.0, vfov_deg = 50.0, rf_max = 4.0;
  std::string json;   // the raw 'meta' entry
};

class Policy {
 public:
  explicit Policy(const std::string& npz_path);
  ~Policy();
  Policy(Policy&&) noexcept;
  Policy& operator=(Policy&&) noexcept;

  const PolicyMeta& meta() const;
  int hidden_size() const;
  int feature_dim() const;   // proprio_dim + flattened CNN features (the GRU input)

  void reset();   // GRU state <- 0
  // depth: (frame_stack, depth_h, depth_w) C-order; proprio: proprio_dim values. -> num_actions raw outputs
  // (the Gaussian mean; NOT clipped, exactly as NumpyPolicy.__call__). Advances the GRU state.
  std::vector<float> operator()(const float* depth, const float* proprio);
  std::vector<float> operator()(const std::vector<float>& depth, const std::vector<float>& proprio);

  const std::vector<float>& hidden() const;
  void set_hidden(const std::vector<float>& h);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

// A DepthReprojector configured from the policy's meta (training camera, far clip) with the nodes' stride 2.
DepthReprojector make_reprojector(const PolicyMeta& meta, const std::array<double, 4>& K, int src_h, int src_w,
                                  const Mat3& R_body_cam, const Vec3& t_body_cam, int stride = 2);

class PolicyStepper {
 public:
  explicit PolicyStepper(const std::string& npz_path);

  struct Result {
    std::vector<float> action;    // clipped to [-1, 1], float32 (num_actions)
    std::vector<float> proprio;   // what the policy saw
    Setpoint setpoint;            // from action[0..3]
  };
  // One decision on the newest reprojected depth frame (depth_h * depth_w, from DepthReprojector).
  Result step(const std::vector<float>& depth_img, const Vec3& vel_ned, const QuatWXYZ& q, double range_m);
  // policy.reset(); stack.reset(); prev_action = 0 (the nodes do this when a flight starts / memory 'fresh').
  void reset();

  Policy& policy() { return policy_; }
  const PolicyMeta& meta() const { return policy_.meta(); }
  const std::vector<float>& prev_action() const { return prev_action_; }

 private:
  Policy policy_;
  FrameStack stack_;
  std::vector<float> prev_action_;
};

}  // namespace policynav
