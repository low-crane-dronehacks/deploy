// vpi_stereo: rectify the OAK's raw mono pair (OpenCV remap, maps built once from camera_info + the left->right TF)
// and run VPI census + SGM stereo disparity (CUDA backend). No Python in the path.
//
//   in   /oak/{left,right}/image_mono     mono8 raw (1024x640); frame ids come from /oak/{left,right}/camera_info
//        /oak/{left,right}/camera_info    raw K, D (skybeta: D=0, R=I, no common P), then TF left_optical -> right_optical
//   out  /policynav/depth/image           32FC1 metres, NaN = invalid (no match / low confidence / out of range):
//                                         the same contract as the Python matchers
//        /policynav/depth/camera_info     K of the rectified left image (P[3] = 0), frame <left optical>_rect
//        TF <left optical> -> <left optical>_rect (static; rotation of the rectified frame)
//
// Census matching is insensitive to brightness differences between the eyes (glare, exposure). Low-confidence
// pixels become holes, which the policy reads as near: holes are safe, confidently wrong depth is not.
#include <vpi/Image.h>
#include <vpi/Status.h>
#include <vpi/Stream.h>
#include <vpi/algo/ConvertImageFormat.h>
#include <vpi/algo/StereoDisparity.h>

#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cmath>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "geometry_msgs/msg/transform_stamped.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/camera_info.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "tf2/LinearMath/Matrix3x3.h"
#include "tf2/LinearMath/Quaternion.h"
#include "tf2_ros/buffer.h"
#include "tf2_ros/static_transform_broadcaster.h"
#include "tf2_ros/transform_listener.h"

#define VPI_CHECK(expr)                                                                        \
  do {                                                                                         \
    VPIStatus s_ = (expr);                                                                     \
    if (s_ != VPI_SUCCESS) {                                                                   \
      char buf[VPI_MAX_STATUS_MESSAGE_LENGTH];                                                 \
      vpiGetLastStatusMessage(buf, sizeof(buf));                                               \
      throw std::runtime_error(std::string(#expr) + ": " + vpiStatusGetName(s_) + ": " + buf); \
    }                                                                                          \
  } while (0)

using sensor_msgs::msg::CameraInfo;
using sensor_msgs::msg::Image;

namespace {
double stamp_s(const builtin_interfaces::msg::Time& t) { return t.sec + 1e-9 * t.nanosec; }

double ms_since(std::chrono::steady_clock::time_point t0) {
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

// Per-stage timing, reported as p50 / p95 / max every report period and then reset.
struct StageTimes {
  std::vector<std::pair<std::string, std::vector<double>>> stages;
  void add(const std::string& name, double ms) {
    for (auto& s : stages)
      if (s.first == name) { s.second.push_back(ms); return; }
    stages.push_back({name, {ms}});
  }
  std::string report() {
    std::string out;
    char buf[160];
    for (auto& s : stages) {
      auto& v = s.second;
      if (v.empty()) continue;
      std::sort(v.begin(), v.end());
      auto pct = [&v](double q) { return v[std::min(v.size() - 1, static_cast<size_t>(q * (v.size() - 1) + 0.5))]; };
      std::snprintf(buf, sizeof(buf), "\n  %-26s p50 %6.1f  p95 %6.1f  max %6.1f ms  (n=%zu)", s.first.c_str(), pct(0.5),
                    pct(0.95), v.back(), v.size());
      out += buf;
      v.clear();
    }
    return out;
  }
};
}  // namespace

class VpiStereo : public rclcpp::Node {
 public:
  VpiStereo()
      : Node("policynav_vpi_stereo"), tf_buffer_(get_clock()), tf_listener_(tf_buffer_), tf_pub_(this) {
    max_disp_ = declare_parameter<int>("max_disparity", 96);
    window_ = declare_parameter<int>("window", 5);
    conf_thr_ = declare_parameter<int>("conf_threshold", 32767);
    min_m_ = declare_parameter<double>("min_depth_m", 0.2);
    max_m_ = declare_parameter<double>("max_depth_m", 20.0);
    integer_ = declare_parameter<bool>("integer_disparity", false);
    out_w_ = declare_parameter<int>("out_w", 640);
    out_h_ = declare_parameter<int>("out_h", 400);
    rate_hz_ = declare_parameter<double>("rate_hz", 15.0);
    slop_s_ = declare_parameter<double>("sync_slop_s", 0.005);
    const std::string ns_l = declare_parameter<std::string>("left_ns", "/oak/left");
    const std::string ns_r = declare_parameter<std::string>("right_ns", "/oak/right");
    const std::string img = declare_parameter<std::string>("image_topic", "image_mono");
    auto qos = rclcpp::SensorDataQoS();
    sub_info_l_ = create_subscription<CameraInfo>(ns_l + "/camera_info", qos, [this](CameraInfo::ConstSharedPtr m) {
      if (!info_l_ && m->k[0] > 0) info_l_ = m;
    });
    sub_info_r_ = create_subscription<CameraInfo>(ns_r + "/camera_info", qos, [this](CameraInfo::ConstSharedPtr m) {
      if (!info_r_ && m->k[0] > 0) info_r_ = m;
    });
    sub_l_ = create_subscription<Image>(ns_l + "/" + img, qos, [this](Image::ConstSharedPtr m) {
      times_.add("0 camera->rx left (age)", 1e3 * (now().seconds() - stamp_s(m->header.stamp)));
      left_ = m;
      try_pair();
    });
    sub_r_ = create_subscription<Image>(ns_r + "/" + img, qos, [this](Image::ConstSharedPtr m) {
      times_.add("0 camera->rx right (age)", 1e3 * (now().seconds() - stamp_s(m->header.stamp)));
      right_ = m;
      try_pair();
    });
    const double report_s = declare_parameter<double>("timing_report_s", 5.0);
    if (report_s > 0)
      report_timer_ = create_wall_timer(std::chrono::duration<double>(report_s), [this]() {
        const std::string r = times_.report();
        if (!r.empty())
          RCLCPP_INFO(get_logger(), "timing (pairs %zu, unpaired %zu, decimated %zu):%s", n_pairs_, unpaired_,
                      n_decimated_, r.c_str());
      });
    pub_ = create_publisher<Image>("/policynav/depth/image", qos);
    pub_info_ = create_publisher<CameraInfo>("/policynav/depth/camera_info", qos);
    VPI_CHECK(vpiStreamCreate(VPI_BACKEND_CUDA, &stream_));
    RCLCPP_INFO(get_logger(), "VPI stereo (CUDA): max disparity %d, window %d, confidence >= %d, <= %.1f Hz, %dx%d",
                max_disp_, window_, conf_thr_, rate_hz_, out_w_, out_h_);
  }

  ~VpiStereo() override {
    if (stream_) vpiStreamSync(stream_);
    if (payload_) vpiPayloadDestroy(payload_);
    for (VPIImage im : {in8_l_, in8_r_, in16_l_, in16_r_, disp_, conf_})
      if (im) vpiImageDestroy(im);
    if (stream_) vpiStreamDestroy(stream_);
  }

 private:
  // Rectification from the raw intrinsics and the right camera's pose in the left optical frame, exactly as
  // policynav/stereo.py stereo_rect_from_extrinsics(): output grid out_w x out_h, alpha = 0, zero disparity.
  bool setup_rectification() {
    const CameraInfo& L = *info_l_;
    const CameraInfo& R = *info_r_;
    geometry_msgs::msg::TransformStamped tf;
    try {
      tf = tf_buffer_.lookupTransform(L.header.frame_id, R.header.frame_id, tf2::TimePointZero);
    } catch (const std::exception& e) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000, "waiting for TF %s -> %s: %s", L.header.frame_id.c_str(),
                           R.header.frame_id.c_str(), e.what());
      return false;
    }
    const auto& q = tf.transform.rotation;
    const auto& t = tf.transform.translation;
    tf2::Matrix3x3 m(tf2::Quaternion(q.x, q.y, q.z, q.w));
    cv::Matx33d R_lr;                                       // p_left = R_lr p_right + t_lr
    for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j) R_lr(i, j) = m[i][j];
    const cv::Vec3d t_lr(t.x, t.y, t.z);
    if (!std::isfinite(t.x) || t.x <= 0.0) {
      RCLCPP_ERROR_THROTTLE(get_logger(), *get_clock(), 5000,
          "Invalid stereo baseline: right eye must be at +x in left optical frame. Check calibration/eye ordering; no depth published.");
      return false;
    }
    const cv::Matx33d R_rl = R_lr.t();                      // stereoRectify wants x_right = R x_left + T
    const cv::Vec3d T = -(R_rl * t_lr);
    const double sx = static_cast<double>(out_w_) / L.width, sy = static_cast<double>(out_h_) / L.height;
    const cv::Matx33d S(sx, 0, 0.5 * sx - 0.5, 0, sy, 0.5 * sy - 0.5, 0, 0, 1);
    const cv::Matx33d K1(L.k[0], L.k[1], L.k[2], L.k[3], L.k[4], L.k[5], L.k[6], L.k[7], L.k[8]);
    const cv::Matx33d K2(R.k[0], R.k[1], R.k[2], R.k[3], R.k[4], R.k[5], R.k[6], R.k[7], R.k[8]);
    auto dist = [](const CameraInfo& c) {                  // skybeta: all zeros; honour a real model if one appears
      std::vector<double> d;
      if (c.distortion_model == "plumb_bob" || c.distortion_model == "rational_polynomial")
        d.assign(c.d.begin(), c.d.begin() + std::min<size_t>(c.d.size(), 8));
      if (d.empty()) d.assign(5, 0.0);
      return d;
    };
    const std::vector<double> D1 = dist(L), D2 = dist(R);
    cv::Mat R1, R2, P1, P2, Q;
    const cv::Size out(out_w_, out_h_);
    const cv::Matx33d K1s = S * K1, K2s = S * K2;           // intrinsics re-expressed on the output grid
    cv::stereoRectify(cv::Mat(K1s), D1, cv::Mat(K2s), D2, out, cv::Mat(R_rl), cv::Mat(T), R1, R2, P1, P2, Q,
                      cv::CALIB_ZERO_DISPARITY, 0.0);
    // remap: output pixel -> SOURCE pixel, so the source-resolution K with the output-grid P
    cv::initUndistortRectifyMap(cv::Mat(K1), D1, R1, P1(cv::Rect(0, 0, 3, 3)), out, CV_32FC1, m1x_, m1y_);
    cv::initUndistortRectifyMap(cv::Mat(K2), D2, R2, P2(cv::Rect(0, 0, 3, 3)), out, CV_32FC1, m2x_, m2y_);
    fx_ = P1.at<double>(0, 0);
    cx_ = P1.at<double>(0, 2);
    cy_ = P1.at<double>(1, 2);
    baseline_ = -P2.at<double>(0, 3) / P2.at<double>(0, 0);
    if (!std::isfinite(baseline_) || baseline_ <= 0.0) {
      RCLCPP_ERROR_THROTTLE(get_logger(), *get_clock(), 5000,
          "Invalid rectified stereo baseline: check calibration/eye ordering; no depth published.");
      return false;
    }
    src_w_ = L.width;
    src_h_ = L.height;
    frame_ = L.header.frame_id + "_rect";

    // static TF: the rectified frame's axes expressed in the left camera frame (rotation R1^T)
    cv::Mat R1t = R1.t();
    tf2::Matrix3x3 rm(R1t.at<double>(0, 0), R1t.at<double>(0, 1), R1t.at<double>(0, 2), R1t.at<double>(1, 0),
                      R1t.at<double>(1, 1), R1t.at<double>(1, 2), R1t.at<double>(2, 0), R1t.at<double>(2, 1),
                      R1t.at<double>(2, 2));
    tf2::Quaternion rq;
    rm.getRotation(rq);
    geometry_msgs::msg::TransformStamped ts;
    ts.header.stamp = now();
    ts.header.frame_id = L.header.frame_id;
    ts.child_frame_id = frame_;
    ts.transform.rotation.x = rq.x();
    ts.transform.rotation.y = rq.y();
    ts.transform.rotation.z = rq.z();
    ts.transform.rotation.w = rq.w();
    tf_pub_.sendTransform(ts);
    if (baseline_ < 0.02 || baseline_ > 0.2)
      RCLCPP_ERROR(get_logger(), "implausible baseline %.4f m from the TF: check camera_info / calibration", baseline_);
    RCLCPP_INFO(get_logger(), "rectified from TF extrinsics: fx %.1f at %d px, baseline %.2f cm, min range %.2f m", fx_,
                out_w_, baseline_ * 100.0, fx_ * baseline_ / max_disp_);
    rect_ready_ = true;
    return true;
  }

  void setup_vpi(int w, int h) {
    w_ = w;
    h_ = h;
    const uint64_t flags = VPI_BACKEND_CUDA | VPI_BACKEND_CPU;
    VPI_CHECK(vpiImageCreate(w, h, VPI_IMAGE_FORMAT_U8, flags, &in8_l_));
    VPI_CHECK(vpiImageCreate(w, h, VPI_IMAGE_FORMAT_U8, flags, &in8_r_));
    VPI_CHECK(vpiImageCreate(w, h, VPI_IMAGE_FORMAT_Y16_ER, flags, &in16_l_));
    VPI_CHECK(vpiImageCreate(w, h, VPI_IMAGE_FORMAT_Y16_ER, flags, &in16_r_));
    VPI_CHECK(vpiImageCreate(w, h, VPI_IMAGE_FORMAT_S16, flags, &disp_));
    VPI_CHECK(vpiImageCreate(w, h, VPI_IMAGE_FORMAT_U16, flags, &conf_));
    VPIStereoDisparityEstimatorCreationParams cp;
    VPI_CHECK(vpiInitStereoDisparityEstimatorCreationParams(&cp));
    cp.maxDisparity = max_disp_;
    VPI_CHECK(vpiCreateStereoDisparityEstimator(VPI_BACKEND_CUDA, w, h, VPI_IMAGE_FORMAT_Y16_ER, &cp, &payload_));
    RCLCPP_INFO(get_logger(), "VPI images %dx%d ready", w, h);
  }

  // rectify a raw mono8 ROS image straight into the VPI U8 image's host memory (one pass, no extra copy)
  void rectify_into(const Image& m, const cv::Mat& mx, const cv::Mat& my, VPIImage dst) {
    VPIImageData d;
    VPI_CHECK(vpiImageLockData(dst, VPI_LOCK_WRITE, VPI_IMAGE_BUFFER_HOST_PITCH_LINEAR, &d));
    const auto& pl = d.buffer.pitch.planes[0];
    cv::Mat src(m.height, m.width, CV_8UC1, const_cast<uint8_t*>(m.data.data()), m.step);
    cv::Mat out(h_, w_, CV_8UC1, pl.data, pl.pitchBytes);
    cv::remap(src, out, mx, my, cv::INTER_LINEAR);
    VPI_CHECK(vpiImageUnlock(dst));
  }

  void try_pair() {
    if (!left_ || !right_) return;
    const double tl = stamp_s(left_->header.stamp), tr = stamp_s(right_->header.stamp);
    if (std::abs(tl - tr) > slop_s_) {                      // drop the older eye and wait for its partner
      if (++unpaired_ % 150 == 0)
        RCLCPP_WARN(get_logger(), "%zu images without a partner within %.1f ms", unpaired_, slop_s_ * 1e3);
      (tl < tr ? left_ : right_).reset();
      return;
    }
    auto l = left_, r = right_;
    left_.reset();
    right_.reset();
    // Match at most rate_hz (15 Hz, as trained), decimating on camera stamps. The 0.8 tolerance matters: with 30 Hz
    // input, a strict 1/rate_hz gap rejects every other pair about half the time (stamp jitter puts it a hair under
    // 66.7 ms), which halves the rate to ~10 Hz.
    const double now_s = tl;
    if (now_s - t_last_ < 0.8 / rate_hz_) { ++n_decimated_; return; }
    if (!info_l_ || !info_r_) {
      RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 5000, "waiting for /oak/{left,right}/camera_info");
      return;
    }
    if (!rect_ready_ && !setup_rectification()) return;
    if (l->encoding != "mono8" || r->encoding != "mono8" || l->width != src_w_ || l->height != src_h_ ||
        r->width != src_w_ || r->height != src_h_) {
      RCLCPP_ERROR_THROTTLE(get_logger(), *get_clock(), 5000, "expected %ux%u mono8 raw images from both eyes", src_w_,
                            src_h_);
      return;
    }
    t_last_ = now_s;
    try {
      if (w_ != out_w_ || h_ != out_h_) setup_vpi(out_w_, out_h_);
    } catch (const std::exception& e) {
      RCLCPP_ERROR_THROTTLE(get_logger(), *get_clock(), 5000, "VPI setup: %s", e.what());
      return;
    }
    match(*l, *r);
  }

  void match(const Image& l, const Image& r) {
    const double fB = fx_ * baseline_;
    const auto t0 = std::chrono::steady_clock::now();
    times_.add("1 camera->match start (age)", 1e3 * (now().seconds() - stamp_s(l.header.stamp)));
    try {
      auto ts = std::chrono::steady_clock::now();
      rectify_into(l, m1x_, m1y_, in8_l_);
      rectify_into(r, m2x_, m2y_, in8_r_);
      times_.add("2 rectify x2 (CPU)", ms_since(ts));
      ts = std::chrono::steady_clock::now();
      VPIConvertImageFormatParams cvp;
      VPI_CHECK(vpiInitConvertImageFormatParams(&cvp));
      VPI_CHECK(vpiSubmitConvertImageFormat(stream_, VPI_BACKEND_CUDA, in8_l_, in16_l_, &cvp));
      VPI_CHECK(vpiSubmitConvertImageFormat(stream_, VPI_BACKEND_CUDA, in8_r_, in16_r_, &cvp));
      VPIStereoDisparityEstimatorParams sp;
      VPI_CHECK(vpiInitStereoDisparityEstimatorParams(&sp));
      sp.windowSize = window_;
      sp.maxDisparity = max_disp_;
      sp.confidenceThreshold = conf_thr_;
      VPI_CHECK(vpiSubmitStereoDisparityEstimator(stream_, VPI_BACKEND_CUDA, payload_, in16_l_, in16_r_, disp_, conf_,
                                                  &sp));
      VPI_CHECK(vpiStreamSync(stream_));
      times_.add("3 VPI convert+SGM (GPU)", ms_since(ts));
    } catch (const std::exception& e) {
      RCLCPP_ERROR_THROTTLE(get_logger(), *get_clock(), 5000, "VPI: %s", e.what());
      return;
    }
    const auto t_depth = std::chrono::steady_clock::now();
    auto out = std::make_unique<Image>();
    out->header.stamp = l.header.stamp;
    out->header.frame_id = frame_;
    out->width = w_;
    out->height = h_;
    out->encoding = "32FC1";
    out->is_bigendian = false;
    out->step = 4 * w_;
    out->data.resize(static_cast<size_t>(out->step) * h_);
    auto* z = reinterpret_cast<float*>(out->data.data());
    const float nan = std::numeric_limits<float>::quiet_NaN();
    size_t valid = 0;
    try {
      VPIImageData dd, cd;
      VPI_CHECK(vpiImageLockData(disp_, VPI_LOCK_READ, VPI_IMAGE_BUFFER_HOST_PITCH_LINEAR, &dd));
      VPI_CHECK(vpiImageLockData(conf_, VPI_LOCK_READ, VPI_IMAGE_BUFFER_HOST_PITCH_LINEAR, &cd));
      const auto& dp = dd.buffer.pitch.planes[0];
      const auto& cpl = cd.buffer.pitch.planes[0];
      for (int y = 0; y < h_; ++y) {
        const auto* drow = reinterpret_cast<const int16_t*>(static_cast<const uint8_t*>(dp.data) + y * dp.pitchBytes);
        const auto* crow = reinterpret_cast<const uint16_t*>(static_cast<const uint8_t*>(cpl.data) + y * cpl.pitchBytes);
        for (int x = 0; x < w_; ++x) {
          float d = drow[x] / 32.0f;                              // S16 Q10.5
          if (integer_) d = std::floor(d + 0.5f);
          float v = nan;
          if (d > 0.f && crow[x] >= conf_thr_) {
            const float m = static_cast<float>(fB / d);
            if (m >= min_m_ && m <= max_m_) { v = m; ++valid; }
          }
          z[y * w_ + x] = v;
        }
      }
      vpiImageUnlock(disp_);
      vpiImageUnlock(conf_);
    } catch (const std::exception& e) {
      RCLCPP_ERROR_THROTTLE(get_logger(), *get_clock(), 5000, "VPI: %s", e.what());
      return;
    }
    times_.add("4 disparity->metres (CPU)", ms_since(t_depth));
    auto ci = std::make_unique<CameraInfo>();
    ci->header = out->header;
    ci->width = w_;
    ci->height = h_;
    ci->k = {fx_, 0.0, cx_, 0.0, fx_, cy_, 0.0, 0.0, 1.0};
    ci->p = {fx_, 0.0, cx_, 0.0, 0.0, fx_, cy_, 0.0, 0.0, 0.0, 1.0, 0.0};
    ci->r = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
    ci->distortion_model = "plumb_bob";
    ci->d = {0.0, 0.0, 0.0, 0.0, 0.0};
    const auto t_pub = std::chrono::steady_clock::now();
    pub_->publish(std::move(out));
    pub_info_->publish(std::move(ci));
    times_.add("5 publish depth", ms_since(t_pub));
    times_.add("6 match total", ms_since(t0));
    times_.add("7 camera->depth out (age)", 1e3 * (now().seconds() - stamp_s(l.header.stamp)));
    ++n_pairs_;
    const double ms = ms_since(t0);
    RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 5000, "VPI stereo (rectify + match) %.1f ms, valid %.0f%%", ms,
                         100.0 * valid / (static_cast<double>(w_) * h_));
  }

  int max_disp_, window_, conf_thr_, out_w_, out_h_;
  double min_m_, max_m_, rate_hz_, slop_s_;
  bool integer_;
  int w_ = 0, h_ = 0;
  uint32_t src_w_ = 0, src_h_ = 0;
  size_t unpaired_ = 0, n_pairs_ = 0, n_decimated_ = 0;
  StageTimes times_;
  rclcpp::TimerBase::SharedPtr report_timer_;
  double t_last_ = 0.0;
  bool rect_ready_ = false;
  double fx_ = 0, cx_ = 0, cy_ = 0, baseline_ = 0;
  std::string frame_;
  cv::Mat m1x_, m1y_, m2x_, m2y_;
  VPIStream stream_ = nullptr;
  VPIPayload payload_ = nullptr;
  VPIImage in8_l_ = nullptr, in8_r_ = nullptr, in16_l_ = nullptr, in16_r_ = nullptr, disp_ = nullptr, conf_ = nullptr;
  Image::ConstSharedPtr left_, right_;
  CameraInfo::ConstSharedPtr info_l_, info_r_;
  tf2_ros::Buffer tf_buffer_;
  tf2_ros::TransformListener tf_listener_;
  tf2_ros::StaticTransformBroadcaster tf_pub_;
  rclcpp::Subscription<CameraInfo>::SharedPtr sub_info_l_, sub_info_r_;
  rclcpp::Subscription<Image>::SharedPtr sub_l_, sub_r_;
  rclcpp::Publisher<Image>::SharedPtr pub_;
  rclcpp::Publisher<CameraInfo>::SharedPtr pub_info_;
};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<VpiStereo>());
  rclcpp::shutdown();
  return 0;
}
