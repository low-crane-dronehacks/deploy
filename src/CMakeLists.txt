cmake_minimum_required(VERSION 3.16)
project(isam2_depth CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
add_compile_options(-Wall -Wextra -Wno-unused-parameter)

find_package(GTSAM 4.2 REQUIRED)        # config/versions.yaml: gtsam 4.2 (ros-jazzy-gtsam)
find_package(Eigen3 REQUIRED)

# Core: no ROS, so it can be tested anywhere and called in-process by the MPPI.
add_library(isam2_depth_core STATIC depth_icp.cpp distance_grid.cpp depth_isam2.cpp)
target_include_directories(isam2_depth_core PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include> $<INSTALL_INTERFACE:include>)
target_link_libraries(isam2_depth_core PUBLIC gtsam Eigen3::Eigen)
set_target_properties(isam2_depth_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

include(CTest)
if(BUILD_TESTING)
  add_executable(test_depth_isam2 test/test_depth_isam2.cpp)
  target_link_libraries(test_depth_isam2 isam2_depth_core)
  add_test(NAME test_depth_isam2 COMMAND test_depth_isam2)
endif()

# ROS 2 node: only when built in a ROS environment that has px4_msgs (colcon build --base-paths src).
find_package(ament_cmake QUIET)
find_package(px4_msgs QUIET)
if(ament_cmake_FOUND AND px4_msgs_FOUND)
  find_package(rclcpp REQUIRED)
  find_package(sensor_msgs REQUIRED)
  find_package(nav_msgs REQUIRED)
  find_package(tf2_ros REQUIRED)
  add_executable(isam2_depth_node ros/isam2_depth_node.cpp)
  target_link_libraries(isam2_depth_node isam2_depth_core)
  target_link_libraries(isam2_depth_node rclcpp::rclcpp tf2_ros::tf2_ros
    ${sensor_msgs_TARGETS} ${nav_msgs_TARGETS} ${px4_msgs_TARGETS})
  install(TARGETS isam2_depth_node DESTINATION lib/${PROJECT_NAME})
  ament_package()
else()
  message(STATUS "isam2_depth: ament_cmake / px4_msgs not found, building the core library and test only")
endif()
