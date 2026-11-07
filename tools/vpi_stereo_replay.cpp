// Offline replay harness for the REAL vpi_stereo node code (not a copy): compiles src/vpi_stereo_node.cpp (or any
// other revision of it, e.g. `git show <rev>:src/vpi_stereo_node.cpp`) into this program, feeds it the OAK pairs
// the way the camera delivers them (left, then the right ~gap_ms later), and records
//   - latency from the RIGHT image's arrival to the depth image being published (the pair's critical path),
//   - process CPU time per pair (all threads, both callbacks),
//   - the published depth images (raw float32, for a bitwise comparison between revisions).
// Run ONLY inside a container started with --network none (ROS_LOCALHOST_ONLY=1, its own domain id): it publishes.
//
//   data dir as for tools/vpi_stereo_bench.cpp (left.u8, right.u8, calib.txt)
//   build (inside skybeta:0.3.2, ROS sourced), NODE_CPP = path of the node source to test:
//     cmake -S <dir with the CMakeLists below> -B /tmp/b -DNODE_CPP=... && cmake --build /tmp/b
//       cmake_minimum_required(VERSION 3.8) project(replay CXX)
//       find_package(ament_cmake REQUIRED) find_package(rclcpp REQUIRED) find_package(sensor_msgs REQUIRED)
//       find_package(geometry_msgs REQUIRED) find_package(tf2 REQUIRED) find_package(tf2_ros REQUIRED)
//       find_package(OpenCV REQUIRED COMPONENTS core imgproc calib3d)
//       find_package(vpi 3 REQUIRED PATHS /opt/nvidia/vpi3/lib64/cmake)
//       add_executable(replay vpi_stereo_replay.cpp) target_compile_features(replay PUBLIC cxx_std_17)
//       target_compile_definitions(replay PRIVATE NODE_CPP="${NODE_CPP}")
//       ament_target_dependencies(replay rclcpp sensor_msgs geometry_msgs tf2 tf2_ros)
//       target_include_directories(replay PRIVATE ${OpenCV_INCLUDE_DIRS})
//       target_link_libraries(replay vpi ${OpenCV_LIBS})
//     (add target_compile_definitions(replay PRIVATE OLD_NODE) for revisions before on_image(), e.g. dac0e0c)
//   replay <data dir> <out.f32> [pairs=40] [gap_ms=17] [--ros-args -p gpu_rectify:=true ...]
#include <ctime>
#include <fstream>
#include <map>
#include <sstream>
#include <thread>

// everything the node includes, first, so the access hack below only touches the node's own class
#include <vpi/Image.h>
#include <vpi/Status.h>
#include <vpi/Stream.h>
#include <vpi/algo/StereoDisparity.h>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include "geometry_msgs/msg/transform_stamped.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/camera_info.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "tf2/LinearMath/Matrix3x3.h"
#include "tf2/LinearMath/Quaternion.h"
#include "tf2_ros/buffer.h"
#include "tf2_ros/static_transform_broadcaster.h"
#include "tf2_ros/transform_listener.h"

#define private public                                      // reach the node's callbacks and state
#define main vpi_stereo_node_main
#include NODE_CPP
#undef main
#undef private

static double cpu_ms() {
  timespec t;
  clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &t);
  return t.tv_sec * 1e3 + t.tv_nsec * 1e-6;
}

int main(int argc, char** argv) {
  if (argc < 3) { std::fprintf(stderr, "usage: %s <data dir> <out.f32> [pairs] [gap_ms]\n", argv[0]); return 2; }
  const std::string dir = argv[1];
  const int pairs = argc > 3 ? std::atoi(argv[3]) : 40;
  const double gap_ms = argc > 4 ? std::atof(argv[4]) : 17.0;
  rclcpp::init(argc, argv);
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
  const int SW = 1024, SH = 640, N = L.size() / (SW * SH);

  auto node = std::make_shared<VpiStereo>();
  auto info = [&](const std::string& frame, const std::vector<double>& k) {
    auto ci = std::make_shared<CameraInfo>();
    ci->header.frame_id = frame;
    ci->width = SW;
    ci->height = SH;
    ci->distortion_model = "plumb_bob";
    ci->d = {0, 0, 0, 0, 0};
    for (int i = 0; i < 9; ++i) ci->k[i] = k[i];
    return ci;
  };
  node->info_l_ = info("oak_left_optical", calib["left_K"]);
  node->info_r_ = info("oak_right_optical", calib["right_K"]);
  geometry_msgs::msg::TransformStamped tf;                  // right optical frame's pose in the left optical frame
  tf.header.frame_id = "oak_left_optical";
  tf.child_frame_id = "oak_right_optical";
  tf.transform.translation.x = calib["t_lr"][0];
  tf.transform.translation.y = calib["t_lr"][1];
  tf.transform.translation.z = calib["t_lr"][2];
  tf.transform.rotation.x = calib["q_lr_xyzw"][0];
  tf.transform.rotation.y = calib["q_lr_xyzw"][1];
  tf.transform.rotation.z = calib["q_lr_xyzw"][2];
  tf.transform.rotation.w = calib["q_lr_xyzw"][3];
  node->tf_buffer_.setTransform(tf, "replay", true);

  auto sink = std::make_shared<rclcpp::Node>("replay_sink");
  std::vector<Image::ConstSharedPtr> got;
  auto sub = sink->create_subscription<Image>("/policynav/depth/image", rclcpp::SensorDataQoS().keep_last(50),
                                              [&](Image::ConstSharedPtr m) { got.push_back(m); });
  rclcpp::executors::SingleThreadedExecutor ex;
  ex.add_node(sink);
  for (int i = 0; i < 20; ++i) { ex.spin_some(); std::this_thread::sleep_for(std::chrono::milliseconds(50)); }

  auto make = [&](const std::vector<uint8_t>& buf, int i, int64_t st_ns, const char* frame) {
    auto m = std::make_shared<Image>();
    m->header.stamp.sec = st_ns / 1000000000LL;
    m->header.stamp.nanosec = st_ns % 1000000000LL;
    m->header.frame_id = frame;
    m->width = SW;
    m->height = SH;
    m->encoding = "mono8";
    m->step = SW;
    m->data.assign(buf.begin() + size_t(i) * SW * SH, buf.begin() + size_t(i + 1) * SW * SH);
    return Image::ConstSharedPtr(m);
  };
  int64_t st = node->now().nanoseconds();
  std::vector<double> crit, cpu;
  std::vector<std::vector<float>> outs;
  for (int k = 0; k < pairs + 1; ++k) {                     // pair 0 sets up rectification + VPI (not timed)
    const int i = k % N;
    st += 66666667;                                         // 15 Hz pairs: none decimated
    auto l = make(L, i, st, "oak_left_optical"), r = make(R, i, st, "oak_right_optical");
    const double c0 = cpu_ms();
    node->left_ = l;                                        // what the subscription callbacks do (minus the age stat)
#ifdef OLD_NODE
    node->try_pair();
#else
    node->on_image(0);
#endif
    const double c1 = cpu_ms();
    std::this_thread::sleep_until(std::chrono::steady_clock::now() + std::chrono::duration<double, std::milli>(gap_ms));
    const auto t0 = std::chrono::steady_clock::now();
    const double c2 = cpu_ms();
    node->right_ = r;
#ifdef OLD_NODE
    node->try_pair();
#else
    node->on_image(1);
#endif
    const double c3 = cpu_ms();
    if (k > 0) {
      crit.push_back(ms_since(t0));
      cpu.push_back((c1 - c0) + (c3 - c2));
    }
    for (int w = 0; w < 100 && got.size() < size_t(k + 1); ++w) {
      ex.spin_some();
      if (got.size() < size_t(k + 1)) std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(20));  // the camera's 30 Hz cadence
  }
  std::sort(crit.begin(), crit.end());
  std::sort(cpu.begin(), cpu.end());
  std::printf("pairs %zu, depth images received %zu\n", crit.size(), got.size());
  std::printf("right arrival -> depth published: p50 %.3f ms  p90 %.3f ms | CPU per pair p50 %.3f ms\n",
              crit[crit.size() / 2], crit[crit.size() * 9 / 10], cpu[cpu.size() / 2]);
  std::ofstream f(argv[2], std::ios::binary);
  for (auto& m : got) f.write(reinterpret_cast<const char*>(m->data.data()), m->data.size());
  std::printf("node timing report:%s\n", node->times_.report().c_str());
  node.reset();
  rclcpp::shutdown();
  return 0;
}
