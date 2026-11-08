// Observation / action plumbing between ROS-PX4 data and the trained policy: a C++ port of policynav/obs.py and
// policynav/node.py:decode_depth. PURE: no ROS. Conventions (see obs.py's docstring):
//   depth    frame stack, newest first, of z-depth / far_clip on the 128x80, 72x50 deg training camera looking
//            along body +x; a policy pixel with too few valid source pixels is INVALID and reads as near (0.0).
//   proprio  [body-frame velocity / v_max (3), downward range / 1.5 (1), roll, pitch (2), previous action (A)]
//   action   [forward, right, down, yaw_rate] in [-1, 1], yaw-aligned body frame -> NED velocity + yawspeed.
// Frames: PX4 NED world, FRD body; quaternions are (w, x, y, z), FRD body -> NED.
// Numerics follow NumPy: the reprojection runs in float64 (as obs.py does) with the same operation order and no
// fused multiply-adds, so its float32 output matches bit-for-bit in practice.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <string>
#include <vector>

namespace policynav {

using Vec3 = std::array<double, 3>;
using Mat3 = std::array<std::array<double, 3>, 3>;   // row-major, R[row][col]
using QuatWXYZ = std::array<double, 4>;

Mat3 quat_to_mat(double w, double x, double y, double z);
inline Mat3 quat_to_mat(const QuatWXYZ& q) { return quat_to_mat(q[0], q[1], q[2], q[3]); }

struct RPY {
  double roll, pitch, yaw;
};
// NED roll, pitch, yaw (ZYX) of an FRD->NED rotation.
RPY rpy_from_R(const Mat3& R);

// Metric depth image, row-major float32, NaN = invalid (no stereo match).
struct DepthImage {
  int height = 0, width = 0;
  std::vector<float> data;
};

// sensor_msgs/Image payload -> metric depth, exactly as node.py:decode_depth:
//   16UC1 / mono16 (mm, 0 = invalid) -> m;  32FC1 (m): +inf -> 1000 m (far, NOT invalid);  NaN / <= 0 -> NaN.
// `step` is the row stride in bytes. Throws std::invalid_argument on another encoding or a short buffer.
DepthImage decode_depth(const std::uint8_t* data, std::size_t size, const std::string& encoding, std::uint32_t height,
                        std::uint32_t width, std::uint32_t step);

// Maps the real depth image onto the training camera's grid. Built once per (CameraInfo, TF).
class DepthReprojector {
 public:
  struct Params {
    int out_h = 80, out_w = 128;
    double hfov_deg = 72.0, vfov_deg = 50.0;
    double far = 5.0, min_valid = 0.5;
    int stride = 1;   // the nodes use 2
    int edge_fill_cols = 0;  // opt-in training preprocessing, bounded to half the image
  };
  // K = (fx, fy, cx, cy) of the source image; R_body_cam / t_body_cam: optical frame (x right, y down, z forward)
  // -> body FLU, as TF base_link <- camera gives them.
  DepthReprojector(const std::array<double, 4>& K, int src_h, int src_w, const Mat3& R_body_cam,
                   const Vec3& t_body_cam, const Params& params);

  // (src_h, src_w) metric depth (NaN / <= 0 / inf = no return) -> (out_h, out_w) normalised depth in [0, 1].
  void operator()(const float* depth_m, float* out) const;
  std::vector<float> operator()(const DepthImage& depth_m) const;

  int src_h() const { return src_h_; }
  int src_w() const { return src_w_; }
  int out_h() const { return p_.out_h; }
  int out_w() const { return p_.out_w; }
  double axis_err_deg() const { return axis_err_deg_; }   // optical axis vs body x, degrees
  const Mat3& R_frd() const { return R_; }                  // optical -> body FRD
  const Vec3& t_frd() const { return t_; }

 private:
  Params p_;
  int src_h_, src_w_;
  Mat3 R_;
  Vec3 t_;
  std::vector<double> rays_;   // (N, 3) body-FRD rays of the strided source pixels
  double fxp_, fyp_, axis_err_deg_;
};

// Depth frame stack, newest first. A fresh stack is filled with its first frame (as in training).
class FrameStack {
 public:
  explicit FrameStack(int n = 2) : n_(n) {}
  void reset() { frames_.clear(); }
  // img: one (H, W) frame. Returns the (n, H, W) stack, valid until the next push / reset.
  const std::vector<float>& push(const std::vector<float>& img);
  int size() const { return n_; }

 private:
  int n_;
  std::deque<std::vector<float>> frames_;
  std::vector<float> stacked_;
};

// [v_body / v_max (3), range / 1.5, roll, pitch, prev_action...] as float32 (computed in float64, like obs.py).
std::vector<float> proprio(const Vec3& vel_ned, const QuatWXYZ& q, double range_m, const std::vector<float>& prev_action,
                           const Vec3& v_max);

struct Setpoint {
  Vec3 vel_ned;      // m/s, NED
  double yawspeed;   // rad/s
};
// action[0..3] (clipped to [-1, 1]) -> PX4 TrajectorySetpoint velocity in the yaw-aligned frame + yawspeed.
Setpoint action_to_setpoint(const float* action4, const QuatWXYZ& q, const Vec3& v_max, double yaw_rate_max);

}  // namespace policynav
