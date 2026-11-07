cmake_minimum_required(VERSION 3.8)
project(policynav)
# One package for skybeta's team_ws: the Python nodes (policy, stereo rectification/SGBM, handheld) and, when VPI
# is installed (libnvvpi3 + vpi3-dev, as in skybeta:0.3.2), the C++ VPI stereo matcher.

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)

find_package(vpi 3 QUIET PATHS /opt/nvidia/vpi3/lib/cmake /opt/nvidia/vpi3/lib64/cmake)
if(vpi_FOUND)
  find_package(rclcpp REQUIRED)
  find_package(sensor_msgs REQUIRED)
  find_package(geometry_msgs REQUIRED)
  find_package(tf2 REQUIRED)
  find_package(tf2_ros REQUIRED)
  find_package(OpenCV REQUIRED COMPONENTS core imgproc calib3d)
  add_executable(vpi_stereo src/vpi_stereo_node.cpp)
  target_compile_features(vpi_stereo PUBLIC cxx_std_17)
  ament_target_dependencies(vpi_stereo rclcpp sensor_msgs geometry_msgs tf2 tf2_ros)
  target_include_directories(vpi_stereo PRIVATE ${OpenCV_INCLUDE_DIRS})
  target_link_libraries(vpi_stereo vpi ${OpenCV_LIBS})
  find_package(nav_msgs REQUIRED)
  add_executable(latency_probe tools/latency_probe.cpp)
  target_compile_features(latency_probe PUBLIC cxx_std_17)
  ament_target_dependencies(latency_probe rclcpp sensor_msgs nav_msgs geometry_msgs)
  install(TARGETS vpi_stereo latency_probe DESTINATION lib/${PROJECT_NAME})
  message(STATUS "policynav: building vpi_stereo (VPI ${vpi_VERSION})")
else()
  message(WARNING "policynav: VPI not found, skipping vpi_stereo (stereo_backend:=sgbm still works)")
endif()

# C++ policy core (no ROS, no Python): .npz loader, observation plumbing (policynav/obs.py) and the NumPy policy
# forward (policynav/policy_np.py). Built -O3 regardless of CMAKE_BUILD_TYPE (colcon's default is unoptimised).
find_package(ZLIB REQUIRED)
find_package(Eigen3 REQUIRED NO_MODULE)
add_library(policynav_core STATIC
  src/policy/npz.cpp
  src/policy/obs.cpp
  src/policy/decision_gate.cpp
  src/policy/policy.cpp)
# obs.cpp mirrors NumPy's float64 operation order (floor() bin decisions): no fused multiply-adds there.
set_source_files_properties(src/policy/obs.cpp PROPERTIES COMPILE_OPTIONS -ffp-contract=off)
set_target_properties(policynav_core PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_compile_features(policynav_core PUBLIC cxx_std_17)
target_compile_options(policynav_core PRIVATE -O3 -Wall -Wextra)
target_include_directories(policynav_core PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
target_link_libraries(policynav_core PRIVATE ZLIB::ZLIB Eigen3::Eigen)
add_executable(policy_core_decision_gate_test tests/policy_core/decision_gate_test.cpp)
target_link_libraries(policy_core_decision_gate_test policynav_core)
# Golden test against the Python implementation: inputs/outputs from tests/policy_core/make_golden.py (offline).
add_executable(policy_core_golden_test tests/policy_core/golden_test.cpp)
target_compile_options(policy_core_golden_test PRIVATE -O2)
target_link_libraries(policy_core_golden_test policynav_core)
install(TARGETS policy_core_decision_gate_test policy_core_golden_test DESTINATION lib/${PROJECT_NAME})
if(BUILD_TESTING)
  enable_testing()
  add_test(NAME policy_core_decision_gate COMMAND policy_core_decision_gate_test)
endif()

ament_python_install_package(${PROJECT_NAME})
install(PROGRAMS scripts/skybeta_policy scripts/stereo_depth scripts/handheld DESTINATION lib/${PROJECT_NAME})
install(FILES launch/skybeta.launch.py launch/skybeta_climb_forward.launch.py DESTINATION share/${PROJECT_NAME}/launch)
install(PROGRAMS onboard/examples/climb_forward_hold_skybeta.py DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY policies/ DESTINATION share/${PROJECT_NAME}/policies FILES_MATCHING PATTERN "*.npz")

ament_package()
