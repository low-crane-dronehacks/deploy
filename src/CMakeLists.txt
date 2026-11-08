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

# 3. ROS 2 Nodes (colcon build --base-paths src) — supports both ROS 2 Humble (Ubuntu 22.04) and ROS 2 Jazzy (Ubuntu 24.04)
find_package(ament_cmake QUIET)
find_package(px4_msgs QUIET)
if(ament_cmake_FOUND)
  find_package(rclcpp REQUIRED)
  find_package(sensor_msgs REQUIRED)
  find_package(nav_msgs REQUIRED)
  find_package(std_msgs REQUIRED)
  find_package(geometry_msgs REQUIRED)
  find_package(tf2_ros REQUIRED)
  find_package(visualization_msgs QUIET)
  find_package(std_srvs QUIET)

  # Dedicated modular ROS 2 nodes matching Section 2.2, 2.3, WP-3, and WP-4:
  #   1. rrt_star_planner_node -> /plan/global_rrt
  #   2. path_transformer_node -> /plan/local_reference & /plan/rl_lookaheads
  #   3. mppi_controller_node  -> /costmap/local_voxels & /control/trajectory_setpoint
  #   4. control_mux_node      -> /fmu/in/trajectory_setpoint OR /fmu/in/vehicle_rates_setpoint
  add_executable(rrt_star_planner_node ros/rrt_star_planner_node.cpp)
  target_link_libraries(rrt_star_planner_node maze_planning_core)
  ament_target_dependencies(rrt_star_planner_node
    rclcpp tf2_ros sensor_msgs nav_msgs geometry_msgs
  )

  add_executable(path_transformer_node ros/path_transformer_node.cpp)
  target_link_libraries(path_transformer_node maze_planning_core)
  ament_target_dependencies(path_transformer_node
    rclcpp tf2_ros nav_msgs std_msgs geometry_msgs
  )

  add_executable(mppi_controller_node ros/mppi_controller_node.cpp)
  target_link_libraries(mppi_controller_node maze_planning_core)
  ament_target_dependencies(mppi_controller_node
    rclcpp tf2_ros sensor_msgs nav_msgs std_msgs geometry_msgs
  )
  if(px4_msgs_FOUND)
    target_compile_definitions(mppi_controller_node PRIVATE HAVE_PX4_MSGS=1)
    ament_target_dependencies(mppi_controller_node px4_msgs)
  endif()

  add_executable(control_mux_node ros/control_mux_node.cpp)
  target_link_libraries(control_mux_node maze_planning_core)
  ament_target_dependencies(control_mux_node
    rclcpp nav_msgs std_msgs geometry_msgs
  )
  if(px4_msgs_FOUND)
    target_compile_definitions(control_mux_node PRIVATE HAVE_PX4_MSGS=1)
    ament_target_dependencies(control_mux_node px4_msgs)
  endif()

  # Optional combined single-process node for zero-copy benchmarking
  add_executable(maze_planning_node ros/maze_planning_node.cpp)
  target_link_libraries(maze_planning_node maze_planning_core)
  ament_target_dependencies(maze_planning_node
    rclcpp tf2_ros sensor_msgs nav_msgs std_msgs geometry_msgs
  )
  if(px4_msgs_FOUND)
    target_compile_definitions(maze_planning_node PRIVATE HAVE_PX4_MSGS=1)
    ament_target_dependencies(maze_planning_node px4_msgs)
  endif()

  install(TARGETS maze_planning_core
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
  )
  install(TARGETS
    rrt_star_planner_node
    path_transformer_node
    mppi_controller_node
    control_mux_node
    maze_planning_node
    DESTINATION lib/${PROJECT_NAME}
  )

  if(GTSAM_FOUND AND px4_msgs_FOUND AND visualization_msgs_FOUND AND std_srvs_FOUND)
    add_executable(isam2_depth_node ros/isam2_depth_node.cpp)
    target_link_libraries(isam2_depth_node isam2_depth_core maze_planning_core)
    ament_target_dependencies(isam2_depth_node
      rclcpp tf2_ros sensor_msgs nav_msgs px4_msgs visualization_msgs std_srvs
    )
    install(TARGETS isam2_depth_node DESTINATION lib/${PROJECT_NAME})
  else()
    message(WARNING "isam2_depth_node NOT built: needs GTSAM 4.2 (found: ${GTSAM_FOUND}; set GTSAM_DIR, e.g. "
                    "/opt/ros/jazzy/lib/cmake/GTSAM), px4_msgs (${px4_msgs_FOUND}, build it first: "
                    "colcon build --packages-up-to isam2_depth) visualization_msgs (${visualization_msgs_FOUND}) and std_srvs (${std_srvs_FOUND})")
  endif()

  install(DIRECTORY include/ DESTINATION include)
  ament_package()
else()
  install(TARGETS maze_planning_core
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
  )
  install(DIRECTORY include/ DESTINATION include)
  message(STATUS "isam2_depth: ament_cmake not found, building core libraries and tests only")
endif()
