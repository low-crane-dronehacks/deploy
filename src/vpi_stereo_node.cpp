// vpi_stereo: VPI census + SGM stereo disparity (CUDA backend) on the rectified pair from policynav's depth node.
//
//   in   /policynav/stereo/{left,right}/image_rect  mono8, rectified (policynav/depth_node.py backend:=external)
//        /policynav/stereo/camera_info              rectified geometry: K, and P[3] = -fx * baseline
//   out  /policynav/depth/image                     32FC1 metres, NaN = invalid (no match / low confidence /
//                                                   out of range): the same contract as the Python matchers
//        /policynav/depth/camera_info
//
// Census matching is insensitive to brightness differences between the eyes (glare, exposure). Low-confidence
// pixels become holes, which the policy reads as near: holes are safe, confidently wrong depth is not.
#include <vpi/Image.h>
#include <vpi/Status.h>
#include <vpi/Stream.h>
#include <vpi/algo/ConvertImageFormat.h>
#include <vpi/algo/StereoDisparity.h>

#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/camera_info.hpp"
#include "sensor_msgs/msg/image.hpp"

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

class VpiStereo : public rclcpp::Node {
 public:
  VpiStereo() : Node("policynav_vpi_stereo") {
    max_disp_ = declare_parameter<int>("max_disparity", 96);
    window_ = declare_parameter<int>("window", 5);
    conf_thr_ = declare_parameter<int>("conf_threshold", 32767);
    min_m_ = declare_parameter<double>("min_depth_m", 0.2);
    max_m_ = declare_parameter<double>("max_depth_m", 20.0);
    integer_ = declare_parameter<bool>("integer_disparity", false);
    auto qos = rclcpp::SensorDataQoS();
    sub_info_ = create_subscription<CameraInfo>("/policynav/stereo/camera_info", qos,
                                                [this](CameraInfo::ConstSharedPtr m) { info_ = m; });
    sub_l_ = create_subscription<Image>("/policynav/stereo/left/image_rect", qos,
                                        [this](Image::ConstSharedPtr m) { left_ = m; try_match(); });
    sub_r_ = create_subscription<Image>("/policynav/stereo/right/image_rect", qos,
                                        [this](Image::ConstSharedPtr m) { right_ = m; try_match(); });
    pub_ = create_publisher<Image>("/policynav/depth/image", qos);
    pub_info_ = create_publisher<CameraInfo>("/policynav/depth/camera_info", qos);
    VPI_CHECK(vpiStreamCreate(VPI_BACKEND_CUDA, &stream_));
    RCLCPP_INFO(get_logger(), "VPI stereo (CUDA): max disparity %d, window %d, confidence >= %d", max_disp_, window_,
                conf_thr_);
  }

  ~VpiStereo() override {
    vpiStreamSync(stream_);
    vpiPayloadDestroy(payload_);
    for (VPIImage im : {in8_l_, in8_r_, in16_l_, in16_r_, disp_, conf_}) vpiImageDestroy(im);
    vpiStreamDestroy(stream_);
  }

 private:
  void setup(int w, int h) {
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

  // copy a mono8 ROS image into a VPI U8 image (host pitch-linear lock)
  void upload(const Image& m, VPIImage dst) {
    VPIImageData d;
    VPI_CHECK(vpiImageLockData(dst, VPI_LOCK_WRITE, VPI_IMAGE_BUFFER_HOST_PITCH_LINEAR, &d));
    const auto& pl = d.buffer.pitch.planes[0];
    auto* base = static_cast<uint8_t*>(pl.data);
    for (int y = 0; y < h_; ++y) std::memcpy(base + y * pl.pitchBytes, m.data.data() + y * m.step, w_);
    VPI_CHECK(vpiImageUnlock(dst));
  }

  void try_match() {
    if (!left_ || !right_ || !info_) return;
    if (left_->header.stamp != right_->header.stamp) return;   // wait for the matching pair
    auto l = left_, r = right_;
    left_.reset();
    right_.reset();
    if (l->encoding != "mono8" || r->encoding != "mono8" || l->width != r->width || l->height != r->height) {
      RCLCPP_ERROR_THROTTLE(get_logger(), *get_clock(), 5000, "expected matching mono8 rectified images");
      return;
    }
    if (w_ != static_cast<int>(l->width) || h_ != static_cast<int>(l->height)) setup(l->width, l->height);
    const double fx = info_->p[0], fB = -info_->p[3];         // P[3] = -fx * baseline
    if (fx <= 0 || fB <= 0) {
      RCLCPP_ERROR_THROTTLE(get_logger(), *get_clock(), 5000, "camera_info P has no baseline (P[3] >= 0)");
      return;
    }
    const auto t0 = std::chrono::steady_clock::now();
    try {
      upload(*l, in8_l_);
      upload(*r, in8_r_);
      VPIConvertImageFormatParams cv;
      VPI_CHECK(vpiInitConvertImageFormatParams(&cv));
      VPI_CHECK(vpiSubmitConvertImageFormat(stream_, VPI_BACKEND_CUDA, in8_l_, in16_l_, &cv));
      VPI_CHECK(vpiSubmitConvertImageFormat(stream_, VPI_BACKEND_CUDA, in8_r_, in16_r_, &cv));
      VPIStereoDisparityEstimatorParams sp;
      VPI_CHECK(vpiInitStereoDisparityEstimatorParams(&sp));
      sp.windowSize = window_;
      sp.maxDisparity = max_disp_;
      sp.confidenceThreshold = conf_thr_;
      VPI_CHECK(vpiSubmitStereoDisparityEstimator(stream_, VPI_BACKEND_CUDA, payload_, in16_l_, in16_r_, disp_, conf_,
                                                  &sp));
      VPI_CHECK(vpiStreamSync(stream_));
    } catch (const std::exception& e) {
      RCLCPP_ERROR_THROTTLE(get_logger(), *get_clock(), 5000, "VPI: %s", e.what());
      return;
    }
    auto out = std::make_unique<Image>();
    out->header = l->header;
    out->width = w_;
    out->height = h_;
    out->encoding = "32FC1";
    out->is_bigendian = false;
    out->step = 4 * w_;
    out->data.resize(static_cast<size_t>(out->step) * h_);
    auto* z = reinterpret_cast<float*>(out->data.data());
    const float nan = std::numeric_limits<float>::quiet_NaN();
    VPIImageData dd, cd;
    VPI_CHECK(vpiImageLockData(disp_, VPI_LOCK_READ, VPI_IMAGE_BUFFER_HOST_PITCH_LINEAR, &dd));
    VPI_CHECK(vpiImageLockData(conf_, VPI_LOCK_READ, VPI_IMAGE_BUFFER_HOST_PITCH_LINEAR, &cd));
    const auto& dp = dd.buffer.pitch.planes[0];
    const auto& cpl = cd.buffer.pitch.planes[0];
    size_t valid = 0;
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
    auto ci = std::make_unique<CameraInfo>(*info_);
    ci->header = l->header;
    pub_->publish(std::move(out));
    pub_info_->publish(std::move(ci));
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 5000, "VPI stereo %.1f ms, valid %.0f%%", ms,
                         100.0 * valid / (static_cast<double>(w_) * h_));
  }

  int max_disp_, window_, conf_thr_;
  double min_m_, max_m_;
  bool integer_;
  int w_ = 0, h_ = 0;
  VPIStream stream_ = nullptr;
  VPIPayload payload_ = nullptr;
  VPIImage in8_l_ = nullptr, in8_r_ = nullptr, in16_l_ = nullptr, in16_r_ = nullptr, disp_ = nullptr, conf_ = nullptr;
  Image::ConstSharedPtr left_, right_;
  CameraInfo::ConstSharedPtr info_;
  rclcpp::Subscription<CameraInfo>::SharedPtr sub_info_;
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
