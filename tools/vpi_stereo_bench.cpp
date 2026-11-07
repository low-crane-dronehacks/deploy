// Offline benchmark for vpi_stereo_node.cpp stages on real OAK pairs (no ROS, not built by CMake). Compares the
// original pipeline with candidate optimizations, stage by stage, and checks output equivalence.
//
//   data dir: left.u8 / right.u8 (N x 640 x 1024 raw mono8) + calib.txt (lines "<key> <n> v0 v1 ...": left_K right_K
//   t_lr q_lr_xyzw), exported from scratch/oak_pairs.npz with:
//     python3 -c "import numpy as np; z=np.load('oak_pairs.npz'); z['left'].tofile('left.u8');
//       z['right'].tofile('right.u8'); open('calib.txt','w').writelines(k+' '+str(z[k].size)+' '+
//       ' '.join(repr(float(x)) for x in z[k].ravel())+'\n' for k in ['left_K','right_K','t_lr','q_lr_xyzw'])"
//
//   inside skybeta:0.3.2 (docker run --runtime nvidia --network none ...):
//   g++ -O2 -std=c++17 tools/vpi_stereo_bench.cpp -I/opt/nvidia/vpi3/include -I/usr/include/opencv4 \
//       -L/opt/nvidia/vpi3/lib64 -lnvvpi -lopencv_core -lopencv_imgproc -lopencv_calib3d -o /tmp/bench
//   LD_LIBRARY_PATH=/opt/nvidia/vpi3/lib64 /tmp/bench <data dir> [iterations]
#include <vpi/Image.h>
#include <vpi/Status.h>
#include <vpi/Stream.h>
#include <vpi/WarpMap.h>
#include <vpi/algo/ConvertImageFormat.h>
#include <vpi/algo/Remap.h>
#include <vpi/algo/StereoDisparity.h>

#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <functional>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#define VPI_CHECK(expr)                                                                        \
  do {                                                                                         \
    VPIStatus s_ = (expr);                                                                     \
    if (s_ != VPI_SUCCESS) {                                                                   \
      char buf[VPI_MAX_STATUS_MESSAGE_LENGTH];                                                 \
      vpiGetLastStatusMessage(buf, sizeof(buf));                                               \
      throw std::runtime_error(std::string(#expr) + ": " + vpiStatusGetName(s_) + ": " + buf); \
    }                                                                                          \
  } while (0)

using Clock = std::chrono::steady_clock;
static double ms_since(Clock::time_point t0) {
  return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}
#include <ctime>
static double cpu_ms() {  // process CPU time (all threads)
  timespec t;
  clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &t);
  return t.tv_sec * 1e3 + t.tv_nsec * 1e-6;
}
static double median(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}

static const int W = 640, H = 400, SW = 1024, SH = 640, MAX_DISP = 96, WINDOW = 5, CONF = 32767;
static const double MIN_M = 0.2, MAX_M = 20.0;

struct Rect {
  cv::Mat m1x, m1y, m2x, m2y;
  double fx, baseline;
};

// Same math as VpiStereo::setup_rectification (D = 0 for this camera).
static Rect build_rect(std::map<std::string, std::vector<double>>& c) {
  auto& kl = c["left_K"];
  auto& kr = c["right_K"];
  auto& t = c["t_lr"];
  auto& q = c["q_lr_xyzw"];
  // quaternion -> matrix (tf2 convention)
  const double x = q[0], y = q[1], z = q[2], w = q[3];
  const double s = 2.0 / (x * x + y * y + z * z + w * w);
  cv::Matx33d R_lr(1 - s * (y * y + z * z), s * (x * y - w * z), s * (x * z + w * y), s * (x * y + w * z),
                   1 - s * (x * x + z * z), s * (y * z - w * x), s * (x * z - w * y), s * (y * z + w * x),
                   1 - s * (x * x + y * y));
  const cv::Vec3d t_lr(t[0], t[1], t[2]);
  const cv::Matx33d R_rl = R_lr.t();
  const cv::Vec3d T = -(R_rl * t_lr);
  const double sx = double(W) / SW, sy = double(H) / SH;
  const cv::Matx33d S(sx, 0, 0.5 * sx - 0.5, 0, sy, 0.5 * sy - 0.5, 0, 0, 1);
  const cv::Matx33d K1(kl.data()), K2(kr.data());
  const std::vector<double> D(5, 0.0);
  cv::Mat R1, R2, P1, P2, Q;
  const cv::Size out(W, H);
  const cv::Matx33d K1s = S * K1, K2s = S * K2;
  cv::stereoRectify(cv::Mat(K1s), D, cv::Mat(K2s), D, out, cv::Mat(R_rl), cv::Mat(T), R1, R2, P1, P2, Q,
                    cv::CALIB_ZERO_DISPARITY, 0.0);
  Rect r;
  cv::initUndistortRectifyMap(cv::Mat(K1), D, R1, P1(cv::Rect(0, 0, 3, 3)), out, CV_32FC1, r.m1x, r.m1y);
  cv::initUndistortRectifyMap(cv::Mat(K2), D, R2, P2(cv::Rect(0, 0, 3, 3)), out, CV_32FC1, r.m2x, r.m2y);
  r.fx = P1.at<double>(0, 0);
  r.baseline = std::abs(-P2.at<double>(0, 3) / P2.at<double>(0, 0));
  return r;
}

// ---- disparity -> metres: original loop (verbatim) ----
static size_t depth_old(const cv::Mat& disp, const cv::Mat& conf, double fB, bool integer, float* z) {
  const float nan = std::numeric_limits<float>::quiet_NaN();
  size_t valid = 0;
  for (int y = 0; y < H; ++y) {
    const auto* drow = disp.ptr<int16_t>(y);
    const auto* crow = conf.ptr<uint16_t>(y);
    for (int x = 0; x < W; ++x) {
      float d = drow[x] / 32.0f;
      if (integer) d = std::floor(d + 0.5f);
      float v = nan;
      if (d > 0.f && crow[x] >= CONF) {
        const float m = static_cast<float>(fB / d);
        if (m >= MIN_M && m <= MAX_M) { v = m; ++valid; }
      }
      z[y * W + x] = v;
    }
  }
  return valid;
}

// ---- disparity -> metres: 64K lookup table indexed by the raw S16 disparity ----
static std::vector<float> make_lut(double fB, bool integer) {
  const float nan = std::numeric_limits<float>::quiet_NaN();
  std::vector<float> lut(65536, nan);
  for (int i = 0; i < 65536; ++i) {
    const int16_t raw = static_cast<int16_t>(static_cast<uint16_t>(i));
    float d = raw / 32.0f;
    if (integer) d = std::floor(d + 0.5f);
    if (d > 0.f) {
      const float m = static_cast<float>(fB / d);
      if (m >= MIN_M && m <= MAX_M) lut[i] = m;
    }
  }
  return lut;
}
static size_t depth_lut(const cv::Mat& disp, const cv::Mat& conf, const float* lut, float* z) {
  const float nan = std::numeric_limits<float>::quiet_NaN();
  size_t valid = 0;
  for (int y = 0; y < H; ++y) {
    const auto* drow = disp.ptr<uint16_t>(y);
    const auto* crow = conf.ptr<uint16_t>(y);
    float* zr = z + y * W;
    for (int x = 0; x < W; ++x) {
      const float v = crow[x] >= CONF ? lut[drow[x]] : nan;
      zr[x] = v;
      valid += (v == v);
    }
  }
  return valid;
}

int main(int argc, char** argv) {
  if (argc < 2) { std::fprintf(stderr, "usage: %s <data dir> [iters]\n", argv[0]); return 2; }
  const std::string dir = argv[1];
  const int iters = argc > 2 ? std::atoi(argv[2]) : 24;
  std::map<std::string, std::vector<double>> calib;
  {
    std::ifstream f(dir + "/calib.txt");
    std::string line;
    while (std::getline(f, line)) {
      std::istringstream ss(line);
      std::string k; size_t n; ss >> k >> n;
      std::vector<double> v(n);
      for (auto& e : v) ss >> e;
      calib[k] = v;
    }
  }
  auto read_all = [&](const std::string& p) {
    std::ifstream f(dir + "/" + p, std::ios::binary);
    return std::vector<uint8_t>((std::istreambuf_iterator<char>(f)), {});
  };
  const auto L = read_all("left.u8"), R = read_all("right.u8");
  const int N = L.size() / (SW * SH);
  std::printf("%d pairs, %d iterations, OpenCV threads %d\n", N, iters, cv::getNumThreads());
  Rect rc = build_rect(calib);
  const double fB = rc.fx * rc.baseline;
  std::printf("fx %.3f baseline %.5f\n", rc.fx, rc.baseline);
  auto srcL = [&](int i) { return cv::Mat(SH, SW, CV_8UC1, const_cast<uint8_t*>(L.data()) + size_t(i) * SW * SH); };
  auto srcR = [&](int i) { return cv::Mat(SH, SW, CV_8UC1, const_cast<uint8_t*>(R.data()) + size_t(i) * SW * SH); };

  // ---------- 1. remap: float maps vs CV_16SC2 fixed-point ----------
  cv::Mat f1a, f1b, f2a, f2b;
  auto t0 = Clock::now();
  cv::convertMaps(rc.m1x, rc.m1y, f1a, f1b, CV_16SC2);
  cv::convertMaps(rc.m2x, rc.m2y, f2a, f2b, CV_16SC2);
  std::printf("convertMaps once: %.2f ms\n", ms_since(t0));
  {
    std::vector<double> tf, tx, cf, cx;
    cv::Mat o1(H, W, CV_8UC1), o2(H, W, CV_8UC1);
    double maxdiff = 0;
    for (int it = 0; it < iters; ++it) {
      const int i = it % N;
      t0 = Clock::now();
      double c0 = cpu_ms();
      cv::remap(srcL(i), o1, rc.m1x, rc.m1y, cv::INTER_LINEAR);
      tf.push_back(ms_since(t0));
      cf.push_back(cpu_ms() - c0);
      t0 = Clock::now();
      c0 = cpu_ms();
      cv::remap(srcL(i), o2, f1a, f1b, cv::INTER_LINEAR);
      tx.push_back(ms_since(t0));
      cx.push_back(cpu_ms() - c0);
      maxdiff = std::max(maxdiff, cv::norm(o1, o2, cv::NORM_INF));
      cv::remap(srcR(i), o1, rc.m2x, rc.m2y, cv::INTER_LINEAR);
      cv::remap(srcR(i), o2, f2a, f2b, cv::INTER_LINEAR);
      maxdiff = std::max(maxdiff, cv::norm(o1, o2, cv::NORM_INF));
    }
    std::printf("[remap one eye] float32 maps p50 %.3f ms (CPU %.3f) | CV_16SC2 maps p50 %.3f ms (CPU %.3f) | max |diff| "
                "%g\n", median(tf), median(cf), median(tx), median(cx), maxdiff);
    const int nt = cv::getNumThreads();
    cv::setNumThreads(1);
    tf.clear(); tx.clear(); cf.clear(); cx.clear();
    for (int it = 0; it < iters; ++it) {
      const int i = it % N;
      t0 = Clock::now();
      double c0 = cpu_ms();
      cv::remap(srcL(i), o1, rc.m1x, rc.m1y, cv::INTER_LINEAR);
      tf.push_back(ms_since(t0));
      cf.push_back(cpu_ms() - c0);
      t0 = Clock::now();
      c0 = cpu_ms();
      cv::remap(srcL(i), o2, f1a, f1b, cv::INTER_LINEAR);
      tx.push_back(ms_since(t0));
      cx.push_back(cpu_ms() - c0);
    }
    cv::setNumThreads(nt);
    std::printf("[remap one eye, 1 thread] float32 maps p50 %.3f ms (CPU %.3f) | CV_16SC2 maps p50 %.3f ms (CPU %.3f)\n",
                median(tf), median(cf), median(tx), median(cx));
  }

  // ---------- VPI setup ----------
  VPIStream stream;
  VPI_CHECK(vpiStreamCreate(VPI_BACKEND_CUDA, &stream));
  const uint64_t flags = VPI_BACKEND_CUDA | VPI_BACKEND_CPU;
  VPIImage in8_l, in8_r, in16_l, in16_r, disp, conf, disp2, conf2, raw_l, rem_l;
  VPI_CHECK(vpiImageCreate(W, H, VPI_IMAGE_FORMAT_U8, flags, &in8_l));
  VPI_CHECK(vpiImageCreate(W, H, VPI_IMAGE_FORMAT_U8, flags, &in8_r));
  VPI_CHECK(vpiImageCreate(W, H, VPI_IMAGE_FORMAT_Y16_ER, flags, &in16_l));
  VPI_CHECK(vpiImageCreate(W, H, VPI_IMAGE_FORMAT_Y16_ER, flags, &in16_r));
  VPI_CHECK(vpiImageCreate(W, H, VPI_IMAGE_FORMAT_S16, flags, &disp));
  VPI_CHECK(vpiImageCreate(W, H, VPI_IMAGE_FORMAT_U16, flags, &conf));
  VPI_CHECK(vpiImageCreate(W, H, VPI_IMAGE_FORMAT_S16, flags, &disp2));
  VPI_CHECK(vpiImageCreate(W, H, VPI_IMAGE_FORMAT_U16, flags, &conf2));
  VPI_CHECK(vpiImageCreate(SW, SH, VPI_IMAGE_FORMAT_U8, flags, &raw_l));
  VPI_CHECK(vpiImageCreate(W, H, VPI_IMAGE_FORMAT_U8, flags, &rem_l));
  VPIStereoDisparityEstimatorCreationParams cp;
  VPI_CHECK(vpiInitStereoDisparityEstimatorCreationParams(&cp));
  cp.maxDisparity = MAX_DISP;
  VPIPayload p16, p8;
  VPI_CHECK(vpiCreateStereoDisparityEstimator(VPI_BACKEND_CUDA, W, H, VPI_IMAGE_FORMAT_Y16_ER, &cp, &p16));
  VPI_CHECK(vpiCreateStereoDisparityEstimator(VPI_BACKEND_CUDA, W, H, VPI_IMAGE_FORMAT_U8, &cp, &p8));
  VPIStereoDisparityEstimatorParams sp;
  VPI_CHECK(vpiInitStereoDisparityEstimatorParams(&sp));
  sp.windowSize = WINDOW;
  sp.maxDisparity = MAX_DISP;
  sp.confidenceThreshold = CONF;
  VPIConvertImageFormatParams cvp;
  VPI_CHECK(vpiInitConvertImageFormatParams(&cvp));

  auto lock_mat = [](VPIImage im, VPILockMode mode, int type, int w, int h) {
    VPIImageData d;
    VPI_CHECK(vpiImageLockData(im, mode, VPI_IMAGE_BUFFER_HOST_PITCH_LINEAR, &d));
    const auto& pl = d.buffer.pitch.planes[0];
    return cv::Mat(h, w, type, pl.data, pl.pitchBytes);
  };
  auto rectify_into = [&](const cv::Mat& src, const cv::Mat& a, const cv::Mat& b, VPIImage dst, double* lock_ms) {
    auto tl = Clock::now();
    cv::Mat out = lock_mat(dst, VPI_LOCK_WRITE, CV_8UC1, W, H);
    if (lock_ms) *lock_ms += ms_since(tl);
    cv::remap(src, out, a, b, cv::INTER_LINEAR);
    tl = Clock::now();
    VPI_CHECK(vpiImageUnlock(dst));
    if (lock_ms) *lock_ms += ms_since(tl);
  };

  // ---------- 2. VPI: U8 -> Y16_ER convert + SGM vs SGM directly on U8 ----------
  {
    std::vector<double> t16, t8, tlock;
    size_t ndiff_d = 0, ndiff_c = 0;
    for (int it = 0; it < iters; ++it) {
      const int i = it % N;
      double lk = 0;
      rectify_into(srcL(i), f1a, f1b, in8_l, &lk);
      rectify_into(srcR(i), f2a, f2b, in8_r, &lk);
      tlock.push_back(lk);
      t0 = Clock::now();
      VPI_CHECK(vpiSubmitConvertImageFormat(stream, VPI_BACKEND_CUDA, in8_l, in16_l, &cvp));
      VPI_CHECK(vpiSubmitConvertImageFormat(stream, VPI_BACKEND_CUDA, in8_r, in16_r, &cvp));
      VPI_CHECK(vpiSubmitStereoDisparityEstimator(stream, VPI_BACKEND_CUDA, p16, in16_l, in16_r, disp, conf, &sp));
      VPI_CHECK(vpiStreamSync(stream));
      t16.push_back(ms_since(t0));
      t0 = Clock::now();
      VPI_CHECK(vpiSubmitStereoDisparityEstimator(stream, VPI_BACKEND_CUDA, p8, in8_l, in8_r, disp2, conf2, &sp));
      VPI_CHECK(vpiStreamSync(stream));
      t8.push_back(ms_since(t0));
      cv::Mat a = lock_mat(disp, VPI_LOCK_READ, CV_16SC1, W, H), b = lock_mat(disp2, VPI_LOCK_READ, CV_16SC1, W, H);
      cv::Mat c = lock_mat(conf, VPI_LOCK_READ, CV_16UC1, W, H), d = lock_mat(conf2, VPI_LOCK_READ, CV_16UC1, W, H);
      ndiff_d += cv::countNonZero(a != b);
      ndiff_c += cv::countNonZero(c != d);
      for (VPIImage im : {disp, disp2, conf, conf2}) vpiImageUnlock(im);
    }
    std::printf("[VPI] convert+SGM(Y16_ER) p50 %.3f ms | SGM(U8) p50 %.3f ms | differing px: disp %zu conf %zu\n",
                median(t16), median(t8), ndiff_d, ndiff_c);
    std::printf("[VPI] lock/unlock of the two U8 inputs for CPU write: p50 %.3f ms\n", median(tlock));
  }

  // ---------- 3. VPI Remap on GPU (dense warp map from the OpenCV maps) ----------
  {
    VPIWarpMap wm;
    std::memset(&wm, 0, sizeof(wm));
    wm.grid.numHorizRegions = 1;
    wm.grid.numVertRegions = 1;
    wm.grid.regionWidth[0] = W;
    wm.grid.regionHeight[0] = H;
    wm.grid.horizInterval[0] = 1;
    wm.grid.vertInterval[0] = 1;
    VPI_CHECK(vpiWarpMapAllocData(&wm));
    for (int y = 0; y < H; ++y) {
      auto* row = reinterpret_cast<VPIKeypointF32*>(reinterpret_cast<uint8_t*>(wm.keypoints) + y * wm.pitchBytes);
      for (int x = 0; x < W; ++x) { row[x].x = rc.m1x.at<float>(y, x); row[x].y = rc.m1y.at<float>(y, x); }
    }
    VPIPayload prem;
    VPI_CHECK(vpiCreateRemap(VPI_BACKEND_CUDA, &wm, &prem));
    std::vector<double> tup, tgpu;
    double maxdiff = 0;
    double meandiff = 0;
    cv::Mat ref(H, W, CV_8UC1);
    for (int it = 0; it < iters; ++it) {
      const int i = it % N;
      t0 = Clock::now();
      cv::Mat dst = lock_mat(raw_l, VPI_LOCK_WRITE, CV_8UC1, SW, SH);
      srcL(i).copyTo(dst);
      VPI_CHECK(vpiImageUnlock(raw_l));
      tup.push_back(ms_since(t0));
      t0 = Clock::now();
      VPI_CHECK(vpiSubmitRemap(stream, VPI_BACKEND_CUDA, prem, raw_l, rem_l, VPI_INTERP_LINEAR, VPI_BORDER_ZERO, 0));
      VPI_CHECK(vpiStreamSync(stream));
      tgpu.push_back(ms_since(t0));
      cv::remap(srcL(i), ref, f1a, f1b, cv::INTER_LINEAR);
      cv::Mat g = lock_mat(rem_l, VPI_LOCK_READ, CV_8UC1, W, H);
      cv::Mat df;
      cv::absdiff(g, ref, df);
      maxdiff = std::max(maxdiff, cv::norm(df, cv::NORM_INF));
      meandiff += cv::mean(df)[0] / iters;
      vpiImageUnlock(rem_l);
    }
    std::printf("[VPI remap one eye] host copy into VPI image p50 %.3f ms | GPU remap+sync p50 %.3f ms | vs OpenCV: "
                "max |diff| %g, mean %.3f\n",
                median(tup), median(tgpu), maxdiff, meandiff);
    vpiPayloadDestroy(prem);
    vpiWarpMapFreeData(&wm);
  }

  // ---------- 4. disparity -> metres ----------
  for (bool integer : {false, true}) {
    std::vector<double> told, tlut, tlutlock;
    t0 = Clock::now();
    const auto lut = make_lut(fB, integer);
    const double t_make = ms_since(t0);
    std::vector<float> z1(W * H), z2(W * H);
    size_t mism = 0, v1 = 0, v2 = 0;
    for (int it = 0; it < iters; ++it) {
      const int i = it % N;
      rectify_into(srcL(i), f1a, f1b, in8_l, nullptr);
      rectify_into(srcR(i), f2a, f2b, in8_r, nullptr);
      VPI_CHECK(vpiSubmitStereoDisparityEstimator(stream, VPI_BACKEND_CUDA, p8, in8_l, in8_r, disp, conf, &sp));
      VPI_CHECK(vpiStreamSync(stream));
      t0 = Clock::now();
      cv::Mat a = lock_mat(disp, VPI_LOCK_READ, CV_16SC1, W, H), c = lock_mat(conf, VPI_LOCK_READ, CV_16UC1, W, H);
      tlutlock.push_back(ms_since(t0));
      t0 = Clock::now();
      v1 = depth_old(a, c, fB, integer, z1.data());
      told.push_back(ms_since(t0));
      t0 = Clock::now();
      v2 = depth_lut(a, c, lut.data(), z2.data());
      tlut.push_back(ms_since(t0));
      vpiImageUnlock(disp);
      vpiImageUnlock(conf);
      if (v1 != v2) ++mism;
      if (std::memcmp(z1.data(), z2.data(), z1.size() * 4) != 0) ++mism;  // NaN payloads identical (quiet_NaN)
    }
    std::printf("[depth integer=%d] old loop p50 %.3f ms | LUT p50 %.3f ms (LUT build %.2f ms once) | lock p50 %.3f "
                "| bitwise mismatching frames %zu | valid %.1f%%\n",
                integer, median(told), median(tlut), t_make, median(tlutlock), mism, 100.0 * v2 / (W * H));
  }

  // ---------- 5. end-to-end (both eyes ready; CPU time of match()) old vs new ----------
  {
    const auto lut = make_lut(fB, false);
    std::vector<double> told, tnew, tnew_crit;
    std::vector<float> z1(W * H), z2(W * H);
    size_t mism = 0;
    for (int it = 0; it < iters; ++it) {
      const int i = it % N;
      t0 = Clock::now();
      rectify_into(srcL(i), rc.m1x, rc.m1y, in8_l, nullptr);
      rectify_into(srcR(i), rc.m2x, rc.m2y, in8_r, nullptr);
      VPI_CHECK(vpiSubmitConvertImageFormat(stream, VPI_BACKEND_CUDA, in8_l, in16_l, &cvp));
      VPI_CHECK(vpiSubmitConvertImageFormat(stream, VPI_BACKEND_CUDA, in8_r, in16_r, &cvp));
      VPI_CHECK(vpiSubmitStereoDisparityEstimator(stream, VPI_BACKEND_CUDA, p16, in16_l, in16_r, disp, conf, &sp));
      VPI_CHECK(vpiStreamSync(stream));
      {
        cv::Mat a = lock_mat(disp, VPI_LOCK_READ, CV_16SC1, W, H), c = lock_mat(conf, VPI_LOCK_READ, CV_16UC1, W, H);
        depth_old(a, c, fB, false, z1.data());
        vpiImageUnlock(disp);
        vpiImageUnlock(conf);
      }
      told.push_back(ms_since(t0));

      t0 = Clock::now();
      rectify_into(srcL(i), f1a, f1b, in8_l, nullptr);  // in the node this happens while waiting for the right eye
      const double t_left = ms_since(t0);
      auto t1 = Clock::now();
      rectify_into(srcR(i), f2a, f2b, in8_r, nullptr);
      VPI_CHECK(vpiSubmitStereoDisparityEstimator(stream, VPI_BACKEND_CUDA, p8, in8_l, in8_r, disp, conf, &sp));
      VPI_CHECK(vpiStreamSync(stream));
      {
        cv::Mat a = lock_mat(disp, VPI_LOCK_READ, CV_16SC1, W, H), c = lock_mat(conf, VPI_LOCK_READ, CV_16UC1, W, H);
        depth_lut(a, c, lut.data(), z2.data());
        vpiImageUnlock(disp);
        vpiImageUnlock(conf);
      }
      tnew_crit.push_back(ms_since(t1));
      tnew.push_back(t_left + ms_since(t1));
      if (std::memcmp(z1.data(), z2.data(), z1.size() * 4) != 0) ++mism;
    }
    std::printf("[end-to-end] old p50 %.3f ms | new p50 %.3f ms (after right eye: %.3f ms) | frames not bit-identical "
                "%zu/%d\n",
                median(told), median(tnew), median(tnew_crit), mism, iters);
  }
  // ---------- 7. GPU rectification (VPI Remap) for both eyes: time, CPU, and effect on the depth output ----------
  {
    auto make_remap = [&](const cv::Mat& mx, const cv::Mat& my) {
      VPIWarpMap wm;
      std::memset(&wm, 0, sizeof(wm));
      wm.grid.numHorizRegions = 1;
      wm.grid.numVertRegions = 1;
      wm.grid.regionWidth[0] = W;
      wm.grid.regionHeight[0] = H;
      wm.grid.horizInterval[0] = 1;
      wm.grid.vertInterval[0] = 1;
      VPI_CHECK(vpiWarpMapAllocData(&wm));
      for (int y = 0; y < H; ++y) {
        auto* row = reinterpret_cast<VPIKeypointF32*>(reinterpret_cast<uint8_t*>(wm.keypoints) + y * wm.pitchBytes);
        for (int x = 0; x < W; ++x) { row[x].x = mx.at<float>(y, x); row[x].y = my.at<float>(y, x); }
      }
      VPIPayload p;
      VPI_CHECK(vpiCreateRemap(VPI_BACKEND_CUDA, &wm, &p));
      vpiWarpMapFreeData(&wm);
      return p;
    };
    VPIPayload pr1 = make_remap(rc.m1x, rc.m1y), pr2 = make_remap(rc.m2x, rc.m2y);
    VPIImage raw_r;
    VPI_CHECK(vpiImageCreate(SW, SH, VPI_IMAGE_FORMAT_U8, flags, &raw_r));
    const auto lut = make_lut(fB, false);
    std::vector<float> zc(W * H), zg(W * H);
    std::vector<double> tcpu_w, tcpu_c, tgpu_w, tgpu_c, relerr;
    size_t flips = 0, both = 0;
    for (int it = 0; it < iters; ++it) {
      const int i = it % N;
      // CPU rectification path (new node): wall + process CPU of rectify both eyes
      auto t = Clock::now();
      double c0 = cpu_ms();
      rectify_into(srcL(i), f1a, f1b, in8_l, nullptr);
      rectify_into(srcR(i), f2a, f2b, in8_r, nullptr);
      tcpu_w.push_back(ms_since(t));
      tcpu_c.push_back(cpu_ms() - c0);
      VPI_CHECK(vpiSubmitStereoDisparityEstimator(stream, VPI_BACKEND_CUDA, p8, in8_l, in8_r, disp, conf, &sp));
      VPI_CHECK(vpiStreamSync(stream));
      {
        cv::Mat a = lock_mat(disp, VPI_LOCK_READ, CV_16SC1, W, H), c = lock_mat(conf, VPI_LOCK_READ, CV_16UC1, W, H);
        depth_lut(a, c, lut.data(), zc.data());
        vpiImageUnlock(disp); vpiImageUnlock(conf);
      }
      // GPU path: copy raw into VPI, remap on CUDA (timed incl. sync)
      t = Clock::now();
      c0 = cpu_ms();
      for (auto pr : {std::make_pair(raw_l, srcL(i)), std::make_pair(raw_r, srcR(i))}) {
        cv::Mat dst = lock_mat(pr.first, VPI_LOCK_WRITE, CV_8UC1, SW, SH);
        pr.second.copyTo(dst);
        VPI_CHECK(vpiImageUnlock(pr.first));
      }
      VPI_CHECK(vpiSubmitRemap(stream, VPI_BACKEND_CUDA, pr1, raw_l, in8_l, VPI_INTERP_LINEAR, VPI_BORDER_ZERO, 0));
      VPI_CHECK(vpiSubmitRemap(stream, VPI_BACKEND_CUDA, pr2, raw_r, in8_r, VPI_INTERP_LINEAR, VPI_BORDER_ZERO, 0));
      VPI_CHECK(vpiStreamSync(stream));
      tgpu_w.push_back(ms_since(t));
      tgpu_c.push_back(cpu_ms() - c0);
      VPI_CHECK(vpiSubmitStereoDisparityEstimator(stream, VPI_BACKEND_CUDA, p8, in8_l, in8_r, disp, conf, &sp));
      VPI_CHECK(vpiStreamSync(stream));
      {
        cv::Mat a = lock_mat(disp, VPI_LOCK_READ, CV_16SC1, W, H), c = lock_mat(conf, VPI_LOCK_READ, CV_16UC1, W, H);
        depth_lut(a, c, lut.data(), zg.data());
        vpiImageUnlock(disp); vpiImageUnlock(conf);
      }
      for (int k = 0; k < W * H; ++k) {
        const bool vc = zc[k] == zc[k], vg = zg[k] == zg[k];
        if (vc != vg) ++flips;
        else if (vc) { ++both; relerr.push_back(std::abs(zg[k] - zc[k]) / zc[k]); }
      }
    }
    std::sort(relerr.begin(), relerr.end());
    std::printf("[rectify both eyes] CPU 16SC2 p50 %.3f ms (CPU %.3f) | VPI CUDA remap incl. upload p50 %.3f ms (CPU %.3f)\n",
                median(tcpu_w), median(tcpu_c), median(tgpu_w), median(tgpu_c));
    std::printf("[GPU remap effect on depth] valid/invalid flips %.2f%% of px | both valid: rel err p50 %.4f p95 %.4f "
                "p99 %.4f, exact %.1f%%\n",
                100.0 * flips / (double(W) * H * iters), relerr[relerr.size() / 2], relerr[relerr.size() * 95 / 100],
                relerr[relerr.size() * 99 / 100],
                100.0 * (std::upper_bound(relerr.begin(), relerr.end(), 0.0) - relerr.begin()) / relerr.size());
    vpiPayloadDestroy(pr1);
    vpiPayloadDestroy(pr2);
    vpiImageDestroy(raw_r);
  }

  // ---------- 6. (info) SGM alone, inputs already resident; and without diagonal paths (changes the output) ----------
  {
    std::vector<double> tres, tnodiag;
    VPIStereoDisparityEstimatorCreationParams cp2 = cp;
    cp2.includeDiagonals = 0;
    VPIPayload pnd;
    VPI_CHECK(vpiCreateStereoDisparityEstimator(VPI_BACKEND_CUDA, W, H, VPI_IMAGE_FORMAT_U8, &cp2, &pnd));
    for (int it = 0; it < iters; ++it) {
      t0 = Clock::now();
      VPI_CHECK(vpiSubmitStereoDisparityEstimator(stream, VPI_BACKEND_CUDA, p8, in8_l, in8_r, disp, conf, &sp));
      VPI_CHECK(vpiStreamSync(stream));
      tres.push_back(ms_since(t0));
      t0 = Clock::now();
      VPI_CHECK(vpiSubmitStereoDisparityEstimator(stream, VPI_BACKEND_CUDA, pnd, in8_l, in8_r, disp2, conf2, &sp));
      VPI_CHECK(vpiStreamSync(stream));
      tnodiag.push_back(ms_since(t0));
    }
    std::printf("[info] SGM(U8) back-to-back, no CPU touch p50 %.3f ms | includeDiagonals=0 p50 %.3f ms\n", median(tres),
                median(tnodiag));
    vpiPayloadDestroy(pnd);
  }
  vpiStreamDestroy(stream);
  return 0;
}
