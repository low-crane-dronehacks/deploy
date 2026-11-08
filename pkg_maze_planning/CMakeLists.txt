cmake_minimum_required(VERSION 3.16)
project(pkg_maze_planning CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
add_compile_options(-Wall -Wextra -Wno-unused-parameter -O3)

find_package(Eigen3 REQUIRED)
find_package(OpenMP QUIET)
find_package(GTSAM 4.2 QUIET)

set(SRC_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../src")

add_library(maze_planning_core SHARED
  ${SRC_ROOT}/distance_grid.cpp
  ${SRC_ROOT}/groundtruth_map.cpp
  ${SRC_ROOT}/voxel_costmap.cpp
  ${SRC_ROOT}/rrt_star_planner.cpp
  ${SRC_ROOT}/path_transformer.cpp
  ${SRC_ROOT}/mppi_controller.cpp
  ${SRC_ROOT}/control_mux.cpp
  ${SRC_ROOT}/c_api.cpp
)
target_include_directories(maze_planning_core PUBLIC
  $<BUILD_INTERFACE:${SRC_ROOT}/include>
  $<INSTALL_INTERFACE:include>
)
target_link_libraries(maze_planning_core PUBLIC Eigen3::Eigen)
if(OpenMP_CXX_FOUND)
  target_link_libraries(maze_planning_core PUBLIC OpenMP::OpenMP_CXX)
endif()
if(GTSAM_FOUND)
  target_compile_definitions(maze_planning_core PUBLIC HAVE_GTSAM=1)
  target_link_libraries(maze_planning_core PUBLIC gtsam)
endif()

include(CTest)
if(BUILD_TESTING)
  add_executable(test_maze_planning ${SRC_ROOT}/test/test_maze_planning.cpp)
  target_link_libraries(test_maze_planning PRIVATE maze_planning_core)
  add_test(NAME test_maze_planning COMMAND test_maze_planning)
endif()

find_package(ament_cmake QUIET)
find_package(px4_msgs QUIET)
if(ament_cmake_FOUND)
  find_package(rclcpp REQUIRED)
  find_package(sensor_msgs REQUIRED)
  find_package(nav_msgs REQUIRED)
  find_package(std_msgs REQUIRED)
  find_package(geometry_msgs REQUIRED)
  find_package(tf2_ros REQUIRED)

  add_executable(rrt_star_planner_node ${SRC_ROOT}/ros/rrt_star_planner_node.cpp)
  target_link_libraries(rrt_star_planner_node maze_planning_core)
  ament_target_dependencies(rrt_star_planner_node
    rclcpp tf2_ros sensor_msgs nav_msgs geometry_msgs
  )

  add_executable(path_transformer_node ${SRC_ROOT}/ros/path_transformer_node.cpp)
  target_link_libraries(path_transformer_node maze_planning_core)
  ament_target_dependencies(path_transformer_node
    rclcpp tf2_ros nav_msgs std_msgs geometry_msgs
  )

  add_executable(mppi_controller_node ${SRC_ROOT}/ros/mppi_controller_node.cpp)
  target_link_libraries(mppi_controller_node maze_planning_core)
  ament_target_dependencies(mppi_controller_node
    rclcpp tf2_ros sensor_msgs nav_msgs std_msgs geometry_msgs
  )
  if(px4_msgs_FOUND)
    target_compile_definitions(mppi_controller_node PRIVATE HAVE_PX4_MSGS=1)
    ament_target_dependencies(mppi_controller_node px4_msgs)
  endif()

  add_executable(maze_planning_node ${SRC_ROOT}/ros/maze_planning_node.cpp)
  target_link_libraries(maze_planning_node maze_planning_core)
  ament_target_dependencies(maze_planning_node
    rclcpp tf2_ros sensor_msgs nav_msgs std_msgs geometry_msgs
  )
  if(px4_msgs_FOUND)
    target_compile_definitions(maze_planning_node PRIVATE HAVE_PX4_MSGS=1)
    ament_target_dependencies(maze_planning_node px4_msgs)
  endif()

  ament_export_include_directories(include ${EIGEN3_INCLUDE_DIRS})
  ament_export_libraries(maze_planning_core)

  install(TARGETS maze_planning_core
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
  )
  install(TARGETS
    rrt_star_planner_node
    path_transformer_node
    mppi_controller_node
    maze_planning_node
    DESTINATION lib/${PROJECT_NAME}
  )
  install(DIRECTORY ${SRC_ROOT}/include/ DESTINATION include)
  ament_package()
else()
  install(TARGETS maze_planning_core
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
  )
  install(DIRECTORY ${SRC_ROOT}/include/ DESTINATION include)
endif()
