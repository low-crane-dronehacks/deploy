// C++ port of policynav/obs.py + node.py:decode_depth. Compiled with -ffp-contract=off (see CMakeLists.txt): the
// operation order below mirrors NumPy's element-wise evaluation so floor() bin decisions match exactly.
#include "policynav/obs.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace policynav {
namespace {
constexpr double kPi = 3.14159265358979323846;
double radians(double deg) { return deg * (kPi / 180.0); }   // == Python math.radians
double degrees(double rad) { return rad * (180.0 / kPi); }   // == Python math.degrees
double clip1(double x) { return std::max(-1.0, std::min(1.0, x)); }
}  // namespace

Mat3 quat_to_mat(double w, double x, double y, double z) {
  Mat3 R;
  R[0] = {1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)};
  R[1] = {2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)};
  R[2] = {2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)};
  return R;
}

RPY rpy_from_R(const Mat3& R) {
  const double pitch = std::asin(std::max(-1.0, std::min(1.0, -R[2][0])));
  return {std::atan2(R[2][1], R[2][2]), pitch, std::atan2(R[1][0], R[0][0])};
}

DepthImage decode_depth(const std::uint8_t* data, std::size_t size, const std::string& encoding, std::uint32_t height,
                        std::uint32_t width, std::uint32_t step) {
  std::string enc(encoding);
  std::transform(enc.begin(), enc.end(), enc.begin(), [](unsigned char c) { return char(std::tolower(c)); });
  DepthImage img;
  img.height = int(height);
  img.width = int(width);
  img.data.resize(std::size_t(height) * width);
  const float nan = std::numeric_limits<float>::quiet_NaN();
  if (enc == "16uc1" || enc == "mono16") {
    if (step < 2 * width || std::size_t(step) * height > size) throw std::invalid_argument("depth image buffer too small");
    for (std::uint32_t r = 0; r < height; ++r) {
      const std::uint8_t* row = data + std::size_t(r) * step;
      for (std::uint32_t c = 0; c < width; ++c) {
        std::uint16_t v;
        std::memcpy(&v, row + 2 * c, 2);
        const float m = float(v) * 1e-3f;   // NumPy: float32 array * Python float -> float32 multiply
        img.data[std::size_t(r) * width + c] = (m <= 0.0f) ? nan : m;
      }
    }
  } else if (enc == "32fc1") {
    if (step < 4 * width || std::size_t(step) * height > size) throw std::invalid_argument("depth image buffer too small");
    for (std::uint32_t r = 0; r < height; ++r) {
      const std::uint8_t* row = data + std::size_t(r) * step;
      for (std::uint32_t c = 0; c < width; ++c) {
        float m;
        std::memcpy(&m, row + 4 * c, 4);
        if (std::isinf(m) && m > 0) m = 1e3f;          // +inf = nothing within range: far, NOT invalid
        if (std::isnan(m) || m <= 0.0f) m = nan;       // 0 / NaN / -inf = no stereo match: invalid
        img.data[std::size_t(r) * width + c] = m;
      }
    }
  } else {
    throw std::invalid_argument("unsupported depth encoding '" + encoding + "'");
  }
  return img;
}

DepthReprojector::DepthReprojector(const std::array<double, 4>& K, int src_h, int src_w, const Mat3& R_body_cam,
                                   const Vec3& t_body_cam, const Params& params)
    : p_(params), src_h_(src_h), src_w_(src_w) {
  if (p_.stride < 1 || src_h < 1 || src_w < 1 || p_.out_h < 1 || p_.out_w < 1)
    throw std::invalid_argument("DepthReprojector: bad sizes");
  const double fx = K[0], fy = K[1], cx = K[2], cy = K[3];
  const double flu_to_frd[3] = {1.0, -1.0, -1.0};
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) R_[i][j] = flu_to_frd[i] * R_body_cam[i][j];
    t_[i] = flu_to_frd[i] * t_body_cam[i];
  }
  const int hs = (src_h + p_.stride - 1) / p_.stride, ws = (src_w + p_.stride - 1) / p_.stride;
  rays_.resize(std::size_t(hs) * ws * 3);
  std::size_t k = 0;
  for (int v = 0; v < src_h; v += p_.stride) {
    for (int u = 0; u < src_w; u += p_.stride, k += 3) {
      const double rc0 = (double(u) + 0.5 - cx) / fx, rc1 = (double(v) + 0.5 - cy) / fy, rc2 = 1.0;
      for (int j = 0; j < 3; ++j) rays_[k + j] = (rc0 * R_[j][0] + rc1 * R_[j][1]) + rc2 * R_[j][2];
    }
  }
  fxp_ = 0.5 * p_.out_w / std::tan(radians(p_.hfov_deg) / 2.0);
  fyp_ = 0.5 * p_.out_h / std::tan(radians(p_.vfov_deg) / 2.0);
  axis_err_deg_ = degrees(std::acos(clip1(R_[0][2])));
}

void DepthReprojector::operator()(const float* depth_m, float* out) const {
  const int H = p_.out_h, W = p_.out_w, s = p_.stride;
  const std::size_t n_out = std::size_t(H) * W;
  std::vector<std::int64_t> n_all(n_out, 0), n_ok(n_out, 0);
  std::vector<double> zsum(n_out, 0.0);
  const double halfW = 0.5 * W, halfH = 0.5 * H;
  std::size_t k = 0;
  for (int r = 0; r < src_h_; r += s) {
    const float* row = depth_m + std::size_t(r) * src_w_;
    for (int c = 0; c < src_w_; c += s, k += 3) {
      const float zf = row[c];
      const bool ok = std::isfinite(zf) && zf > 0.0f;
      const double* ray = &rays_[k];
      double d0, d1, d2;   // ok: the body FRD point t + z * ray; invalid: the ray itself
      if (ok) {
        const double z = double(zf);
        d0 = t_[0] + ray[0] * z;
        d1 = t_[1] + ray[1] * z;
        d2 = t_[2] + ray[2] * z;
      } else {
        d0 = ray[0];
        d1 = ray[1];
        d2 = ray[2];
      }
      if (!(d0 > 1e-6)) continue;   // not in front of the virtual camera
      const double uf = std::floor(d1 / d0 * fxp_ + halfW);
      const double vf = std::floor(d2 / d0 * fyp_ + halfH);
      if (!(uf >= 0.0 && uf < W && vf >= 0.0 && vf < H)) continue;
      const std::size_t idx = std::size_t(vf) * W + std::size_t(uf);
      ++n_all[idx];
      if (ok) {
        ++n_ok[idx];
        zsum[idx] += std::min(d0, p_.far);
      }
    }
  }
  std::vector<unsigned char> vmask(n_out);
  for (std::size_t i = 0; i < n_out; ++i) {
    const double mean = zsum[i] / double(std::max<std::int64_t>(n_ok[i], 1));
    const bool valid = (double(n_ok[i]) >= p_.min_valid * double(std::max<std::int64_t>(n_all[i], 1))) && n_ok[i] > 0;
    out[i] = valid ? float(std::max(0.0, std::min(1.0, mean / p_.far))) : 0.0f;   // invalid -> near
    vmask[i] = valid;
  }
  const int n = std::max(0, std::min(p_.edge_fill_cols, W / 2));
  for (int r = 0; r < H; ++r) {
    const std::size_t row = std::size_t(r) * W;
    auto fill = [&](int c, int src) {
      if (!vmask[row + c] && vmask[row + src]) {
        out[row + c] = out[row + src];
        vmask[row + c] = 1;
      }
    };
    for (int c = W - n; c < W; ++c) fill(c, c - 1);
    for (int c = n - 1; c >= 0; --c) fill(c, c + 1);
  }
}

std::vector<float> DepthReprojector::operator()(const DepthImage& depth_m) const {
  if (depth_m.height != src_h_ || depth_m.width != src_w_ ||
      depth_m.data.size() != std::size_t(src_h_) * std::size_t(src_w_))
    throw std::invalid_argument("DepthReprojector: depth image is " + std::to_string(depth_m.height) + "x" +
                                std::to_string(depth_m.width) + ", calibrated for " + std::to_string(src_h_) + "x" +
                                std::to_string(src_w_));
  std::vector<float> out(std::size_t(p_.out_h) * p_.out_w);
  (*this)(depth_m.data.data(), out.data());
  return out;
}

const std::vector<float>& FrameStack::push(const std::vector<float>& img) {
  if (frames_.empty()) {
    frames_.assign(n_, img);
  } else {
    frames_.push_front(img);
    frames_.pop_back();
  }
  const std::size_t sz = img.size();
  stacked_.resize(sz * n_);
  for (int i = 0; i < n_; ++i) {
    if (frames_[i].size() != sz) throw std::invalid_argument("FrameStack: frame size changed");
    std::copy(frames_[i].begin(), frames_[i].end(), stacked_.begin() + i * sz);
  }
  return stacked_;
}

std::vector<float> proprio(const Vec3& vel_ned, const QuatWXYZ& q, double range_m, const std::vector<float>& prev_action,
                           const Vec3& v_max) {
  const Mat3 R = quat_to_mat(q);
  const RPY rpy = rpy_from_R(R);
  std::vector<float> out;
  out.reserve(6 + prev_action.size());
  for (int i = 0; i < 3; ++i) {   // v_body = R.T @ vel_ned
    const double vb = (R[0][i] * vel_ned[0] + R[1][i] * vel_ned[1]) + R[2][i] * vel_ned[2];
    out.push_back(float(vb / v_max[i]));
  }
  out.push_back(float(range_m / 1.5));
  out.push_back(float(rpy.roll));
  out.push_back(float(rpy.pitch));
  for (float a : prev_action) out.push_back(a);
  return out;
}

Setpoint action_to_setpoint(const float* action4, const QuatWXYZ& q, const Vec3& v_max, double yaw_rate_max) {
  double a[4];
  for (int i = 0; i < 4; ++i) a[i] = clip1(double(action4[i]));
  const double yaw = rpy_from_R(quat_to_mat(q)).yaw;
  const double f = a[0] * v_max[0], r = a[1] * v_max[1], d = a[2] * v_max[2];
  const double c = std::cos(yaw), s = std::sin(yaw);
  return {{c * f - s * r, s * f + c * r, d}, a[3] * yaw_rate_max};
}

}  // namespace policynav
