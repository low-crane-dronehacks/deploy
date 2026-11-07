cmake_minimum_required(VERSION 3.16)
project(isam2_depth CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
add_compile_options(-Wall -Wextra -Wno-unused-parameter -O3)

find_package(Eigen3 REQUIRED)
find_package(OpenMP QUIET)
find_package(GTSAM 4.2 QUIET)        # config/versions.yaml: gtsam 4.2 (ros-jazzy-gtsam)

# 1. Core Maze Planning Library (GroundTruthMap, VoxelCostmap, RRT*, PathTransformer, 2048-rollout MPPI, ControlMux)
add_library(maze_planning_core SHARED
  distance_grid.cpp
  groundtruth_map.cpp
  voxel_costmap.cpp
  rrt_star_planner.cpp
  path_transformer.cpp
  mppi_controller.cpp
  control_mux.cpp
  c_api.cpp
)
target_include_directories(maze_planning_core PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
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

# 2. Core iSAM2 Depth Factor Graph Library (when GTSAM 4.2 is installed)
if(GTSAM_FOUND)
  add_library(isam2_depth_core STATIC depth_icp.cpp distance_grid.cpp depth_isam2.cpp)
  target_include_directories(isam2_depth_core PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include> $<INSTALL_INTERFACE:include>)
  target_link_libraries(isam2_depth_core PUBLIC gtsam Eigen3::Eigen)
  set_target_properties(isam2_depth_core PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()

include(CTest)
if(BUILD_TESTING)
  add_executable(test_maze_planning test/test_maze_planning.cpp)
  target_link_libraries(test_maze_planning PRIVATE maze_planning_core)
  add_test(NAME test_maze_planning COMMAND test_maze_planning)

  if(GTSAM_FOUND)
    add_executable(test_depth_isam2 test/test_depth_isam2.cpp)
    target_link_libraries(test_depth_isam2 isam2_depth_core)
    add_test(NAME test_depth_isam2 COMMAND test_depth_isam2)
  endif()
endif()

# 3. ROS 2 Nodes (colcon build --base-paths src)
find_package(ament_cmake QUIET)
find_package(px4_msgs QUIET)
if(ament_cmake_FOUND AND px4_msgs_FOUND)
  find_package(rclcpp REQUIRED)
  find_package(sensor_msgs REQUIRED)
  find_package(nav_msgs REQUIRED)
  find_package(std_msgs REQUIRED)
  find_package(geometry_msgs REQUIRED)
  find_package(tf2_ros REQUIRED)

  add_executable(maze_planning_node ros/maze_planning_node.cpp)
  target_link_libraries(maze_planning_node maze_planning_core rclcpp::rclcpp tf2_ros::tf2_ros
    ${sensor_msgs_TARGETS} ${nav_msgs_TARGETS} ${std_msgs_TARGETS} ${geometry_msgs_TARGETS} ${px4_msgs_TARGETS})

  add_executable(path_transformer_node ros/path_transformer_node.cpp)
  target_link_libraries(path_transformer_node maze_planning_core rclcpp::rclcpp tf2_ros::tf2_ros
    ${nav_msgs_TARGETS} ${std_msgs_TARGETS} ${geometry_msgs_TARGETS})

  add_executable(control_mux_node ros/control_mux_node.cpp)
  target_link_libraries(control_mux_node maze_planning_core rclcpp::rclcpp
    ${nav_msgs_TARGETS} ${std_msgs_TARGETS} ${px4_msgs_TARGETS})

  install(TARGETS maze_planning_core
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
  )
  install(TARGETS maze_planning_node path_transformer_node control_mux_node
    DESTINATION lib/${PROJECT_NAME}
  )

  if(GTSAM_FOUND)
    add_executable(isam2_depth_node ros/isam2_depth_node.cpp)
    target_link_libraries(isam2_depth_node isam2_depth_core maze_planning_core)
    target_link_libraries(isam2_depth_node rclcpp::rclcpp tf2_ros::tf2_ros
      ${sensor_msgs_TARGETS} ${nav_msgs_TARGETS} ${px4_msgs_TARGETS})
    install(TARGETS isam2_depth_node DESTINATION lib/${PROJECT_NAME})
  endif()

  install(DIRECTORY include/ DESTINATION include)
  ament_package()
else()
  message(STATUS "isam2_depth: ament_cmake / px4_msgs not found, building core libraries and tests only")
endif()
