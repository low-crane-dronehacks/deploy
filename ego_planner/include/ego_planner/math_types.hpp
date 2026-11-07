#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

namespace ego_planner {

constexpr double kPi = 3.14159265358979323846;

inline double clamp(double v, double lo, double hi) {
  return std::max(lo, std::min(hi, v));
}

inline double wrapPi(double angle) {
  while (angle > kPi) angle -= 2.0 * kPi;
  while (angle < -kPi) angle += 2.0 * kPi;
  return angle;
}

struct Vec2 {
  double x{0.0};
  double y{0.0};

  Vec2() = default;
  constexpr Vec2(double x_, double y_) : x(x_), y(y_) {}

  Vec2 operator+(const Vec2& o) const { return {x + o.x, y + o.y}; }
  Vec2 operator-(const Vec2& o) const { return {x - o.x, y - o.y}; }
  Vec2 operator*(double s) const { return {x * s, y * s}; }
  Vec2 operator/(double s) const { return {x / s, y / s}; }
  double dot(const Vec2& o) const { return x * o.x + y * o.y; }
  double normSq() const { return x * x + y * y; }
  double norm() const { return std::sqrt(normSq()); }
  Vec2 normalized(double eps = 1e-9) const {
    double n = norm();
    return n > eps ? (*this / n) : Vec2{0.0, 0.0};
  }
};

inline Vec2 operator*(double s, const Vec2& v) { return v * s; }

struct Vec3 {
  double x{0.0};
  double y{0.0};
  double z{0.0};

  Vec3() = default;
  constexpr Vec3(double x_, double y_, double z_) : x(x_), y(y_), z(z_) {}

  double operator[](int i) const { return i == 0 ? x : (i == 1 ? y : z); }
  double& operator[](int i) { return i == 0 ? x : (i == 1 ? y : z); }

  Vec3 operator+(const Vec3& o) const { return {x + o.x, y + o.y, z + o.z}; }
  Vec3 operator-(const Vec3& o) const { return {x - o.x, y - o.y, z - o.z}; }
  Vec3 operator-() const { return {-x, -y, -z}; }
  Vec3 operator*(double s) const { return {x * s, y * s, z * s}; }
  Vec3 operator/(double s) const { return {x / s, y / s, z / s}; }
  Vec3& operator+=(const Vec3& o) {
    x += o.x;
    y += o.y;
    z += o.z;
    return *this;
  }
  Vec3& operator-=(const Vec3& o) {
    x -= o.x;
    y -= o.y;
    z -= o.z;
    return *this;
  }
  Vec3& operator*=(double s) {
    x *= s;
    y *= s;
    z *= s;
    return *this;
  }

  double dot(const Vec3& o) const { return x * o.x + y * o.y + z * o.z; }
  Vec3 cross(const Vec3& o) const {
    return {y * o.z - z * o.y, z * o.x - x * o.z, x * o.y - y * o.x};
  }
  Vec3 cwiseProduct(const Vec3& o) const { return {x * o.x, y * o.y, z * o.z}; }
  Vec3 cwiseQuotient(const Vec3& o) const { return {x / o.x, y / o.y, z / o.z}; }
  double normSq() const { return x * x + y * y + z * z; }
  double norm() const { return std::sqrt(normSq()); }
  Vec2 xy() const { return {x, y}; }
  Vec3 normalized(double eps = 1e-9) const {
    double n = norm();
    return n > eps ? (*this / n) : Vec3{0.0, 0.0, 0.0};
  }
};

inline Vec3 operator*(double s, const Vec3& v) { return v * s; }

struct Mat3 {
  double m[3][3]{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};

  static Mat3 identity() { return Mat3{}; }

  Vec3 operator*(const Vec3& v) const {
    return {
        m[0][0] * v.x + m[0][1] * v.y + m[0][2] * v.z,
        m[1][0] * v.x + m[1][1] * v.y + m[1][2] * v.z,
        m[2][0] * v.x + m[2][1] * v.y + m[2][2] * v.z,
    };
  }

  Vec3 transposeMul(const Vec3& v) const {
    return {
        m[0][0] * v.x + m[1][0] * v.y + m[2][0] * v.z,
        m[0][1] * v.x + m[1][1] * v.y + m[2][1] * v.z,
        m[0][2] * v.x + m[1][2] * v.y + m[2][2] * v.z,
    };
  }

  Mat3 operator*(const Mat3& o) const {
    Mat3 r;
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        r.m[i][j] = m[i][0] * o.m[0][j] + m[i][1] * o.m[1][j] + m[i][2] * o.m[2][j];
      }
    }
    return r;
  }

  Mat3 transposed() const {
    Mat3 r;
    for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j) r.m[i][j] = m[j][i];
    return r;
  }
};

// Quaternion (w, x, y, z) representing FRD body -> NED world rotation (matching PX4 VehicleAttitude)
struct Quat {
  double w{1.0};
  double x{0.0};
  double y{0.0};
  double z{0.0};

  Quat() = default;
  constexpr Quat(double w_, double x_, double y_, double z_) : w(w_), x(x_), y(y_), z(z_) {}

  Mat3 toRotationMatrix() const {
    Mat3 R;
    R.m[0][0] = 1.0 - 2.0 * (y * y + z * z);
    R.m[0][1] = 2.0 * (x * y - z * w);
    R.m[0][2] = 2.0 * (x * z + y * w);
    R.m[1][0] = 2.0 * (x * y + z * w);
    R.m[1][1] = 1.0 - 2.0 * (x * x + z * z);
    R.m[1][2] = 2.0 * (y * z - x * w);
    R.m[2][0] = 2.0 * (x * z - y * w);
    R.m[2][1] = 2.0 * (y * z + x * w);
    R.m[2][2] = 1.0 - 2.0 * (x * x + y * y);
    return R;
  }

  // ZYX Euler angles (roll, pitch, yaw) in NED
  Vec3 toEulerZYX() const {
    Mat3 R = toRotationMatrix();
    double pitch = std::asin(clamp(-R.m[2][0], -1.0, 1.0));
    double roll = std::atan2(R.m[2][1], R.m[2][2]);
    double yaw = std::atan2(R.m[1][0], R.m[0][0]);
    return {roll, pitch, yaw};
  }

  static Quat fromEulerZYX(double roll, double pitch, double yaw) {
    double cr = std::cos(roll * 0.5), sr = std::sin(roll * 0.5);
    double cp = std::cos(pitch * 0.5), sp = std::sin(pitch * 0.5);
    double cy = std::cos(yaw * 0.5), sy = std::sin(yaw * 0.5);
    return {
        cy * cr * cp + sy * sr * sp,
        cy * sr * cp - sy * cr * sp,
        cy * cr * sp + sy * sr * cp,
        sy * cr * cp - cy * sr * sp,
    };
  }
};

struct GateSpec {
  int index{0};
  std::string name;
  int checkpoint{0};
  bool is_window{false};
  Vec2 A_ned{0.0, 0.0};
  Vec2 B_ned{0.0, 0.0};
  Vec2 tangent_ned{1.0, 0.0};
  Vec2 normal_ned{0.0, 1.0};
  double length_m{1.0};
  Vec3 center_ned{0.0, 0.0, -0.8};
  double alt_min_m{0.45};
  double alt_max_m{1.35};
  double alt_nominal_m{0.80};
};

struct StateSetpoint {
  Vec3 pos_ned{0.0, 0.0, -0.8};
  Vec3 vel_ned{0.0, 0.0, 0.0};
  Vec3 acc_ned{0.0, 0.0, 0.0};
  double yaw_rad{0.0};
  double yawspeed_rad_s{0.0};
};

struct VehicleState {
  double timestamp_s{0.0};
  Vec3 pos_ned{0.0, 0.0, 0.0};
  Vec3 vel_ned{0.0, 0.0, 0.0};
  Quat q_wxyz{1.0, 0.0, 0.0, 0.0};
  double range_m{0.08};
  bool pos_valid{false};
  bool att_valid{false};
  bool estimator_healthy{false};
  bool armed{false};
  bool landed{true};
  int nav_state{-1};
};

enum CommandFlag : uint32_t {
  kCmdNone = 0,
  kCmdOffboard = 1 << 0,
  kCmdArm = 1 << 1,
  kCmdLand = 1 << 2,
  kCmdDisarm = 1 << 3,
};

struct ControlOutput {
  bool stream_velocity{false};
  Vec3 pos_ned{0.0, 0.0, 0.0};
  Vec3 vel_ned{0.0, 0.0, 0.0};
  Vec3 acc_ned{0.0, 0.0, 0.0};
  double yaw_rad{0.0};
  double yawspeed_rad_s{0.0};
  // Normalized action [v_fwd, v_right, v_down, yaw_rate] in [-1, 1] in yaw-aligned body frame
  std::array<double, 4> body_action{{0.0, 0.0, 0.0, 0.0}};
  uint32_t command_flags{kCmdNone};
  int next_checkpoint{0};
  std::string mission_phase{"wait"};
  std::string active_leg{"run"};
  std::string active_primitive{"hold"};
  std::string why{"starting"};
  std::string result{""};
};

}  // namespace ego_planner
