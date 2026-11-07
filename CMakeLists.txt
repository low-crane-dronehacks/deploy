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
  add_executable(vpi_stereo src/vpi_stereo_node.cpp)
  target_compile_features(vpi_stereo PUBLIC cxx_std_17)
  ament_target_dependencies(vpi_stereo rclcpp sensor_msgs)
  target_link_libraries(vpi_stereo vpi)
  install(TARGETS vpi_stereo DESTINATION lib/${PROJECT_NAME})
  message(STATUS "policynav: building vpi_stereo (VPI ${vpi_VERSION})")
else()
  message(WARNING "policynav: VPI not found, skipping vpi_stereo (stereo_backend:=sgbm still works)")
endif()

ament_python_install_package(${PROJECT_NAME})
install(PROGRAMS scripts/skybeta_policy scripts/stereo_depth scripts/handheld DESTINATION lib/${PROJECT_NAME})
install(FILES launch/skybeta.launch.py launch/skybeta_climb_forward.launch.py DESTINATION share/${PROJECT_NAME}/launch)
install(PROGRAMS onboard/examples/climb_forward_hold_skybeta.py DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY policies/ DESTINATION share/${PROJECT_NAME}/policies FILES_MATCHING PATTERN "*.npz")

ament_package()
