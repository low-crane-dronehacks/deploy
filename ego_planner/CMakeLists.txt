cmake_minimum_required(VERSION 3.16)
project(ego_planner VERSION 2.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

add_compile_options(-Wall -Wextra -O3)

# Core C++17 EGO-Planner-V2 & Autonomy library (zero ROS dependencies, usable in ROS 2, C++, and Python ctypes)
add_library(ego_planner_core SHARED
  src/grid_map.cpp
  src/motion_primitives.cpp
  src/bspline_optimizer.cpp
  src/quadrotor_dynamics.cpp
  src/global_planner_interface.cpp
  src/rrt_star_planner.cpp
  src/local_controller_interface.cpp
  src/mppi_controller.cpp
  src/mission_manager.cpp
  src/c_api.cpp
)
target_include_directories(ego_planner_core PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

# Standalone C++ test executable
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_ego_planner_cpp.cpp")
  add_executable(test_ego_planner_cpp tests/test_ego_planner_cpp.cpp)
  target_link_libraries(test_ego_planner_cpp PRIVATE ego_planner_core)
endif()

# Optional ROS 2 node build when ament_cmake + rclcpp + px4_msgs are available
find_package(ament_cmake QUIET)
find_package(rclcpp QUIET)
find_package(px4_msgs QUIET)

if(ament_cmake_FOUND AND rclcpp_FOUND AND px4_msgs_FOUND)
  find_package(std_msgs REQUIRED)
  find_package(sensor_msgs REQUIRED)
  find_package(geometry_msgs REQUIRED)
  find_package(tf2 REQUIRED)
  find_package(tf2_ros REQUIRED)

  add_executable(ego_planner_node src/ego_planner_node.cpp)
  target_link_libraries(ego_planner_node ego_planner_core)
  ament_target_dependencies(ego_planner_node
    rclcpp
    std_msgs
    sensor_msgs
    geometry_msgs
    tf2
    tf2_ros
    px4_msgs
  )

  install(TARGETS ego_planner_core ego_planner_node
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION lib/${PROJECT_NAME}
  )
  install(DIRECTORY include/
    DESTINATION include
  )
  ament_package()
else()
  install(TARGETS ego_planner_core
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
  )
  install(DIRECTORY include/
    DESTINATION include
  )
endif()
