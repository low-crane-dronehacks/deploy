cmake_minimum_required(VERSION 3.16)
project(pkg_maze_control CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
add_compile_options(-Wall -Wextra -Wno-unused-parameter -O3)

find_package(Eigen3 REQUIRED)
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(pkg_maze_planning REQUIRED)
find_package(px4_msgs QUIET)

set(SRC_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../src")

add_executable(control_mux_node ${SRC_ROOT}/ros/control_mux_node.cpp)
target_include_directories(control_mux_node PRIVATE ${SRC_ROOT}/include)
ament_target_dependencies(control_mux_node
  rclcpp nav_msgs std_msgs geometry_msgs pkg_maze_planning
)
if(px4_msgs_FOUND)
  target_compile_definitions(control_mux_node PRIVATE HAVE_PX4_MSGS=1)
  ament_target_dependencies(control_mux_node px4_msgs)
endif()

install(TARGETS control_mux_node
  DESTINATION lib/${PROJECT_NAME}
)
ament_package()
