# Autonomy Stack Architecture & Developer Guide

This document describes the modular architecture of the DroneHacks 2026 autonomy stack and provides actionable instructions for colleagues implementing **MPPI**, **RRT\***, and **iSAM2**.

---

## 1. System Overview & Architecture

The codebase has been refactored to decouple state estimation, global planning, local control, and mission orchestration while preserving the production-tested ROS 2 / PX4 flight harness.

```mermaid
flowchart TD
    subgraph Estimation["State Estimation (VIO -> PX4; iSAM2 for planning only)"]
        CuVSLAM["cuVSLAM (onboard/slam)"] --> EVBridge["onboard/vio/ev_bridge.py"]
        EVBridge --> PX4Odom["/fmu/in/vehicle_visual_odometry -> EKF2"]
        PX4Odom --> PX4Est["PX4 vehicle_local_position + attitude"]
        PX4Est --> iSAM2["C++ iSAM2 depth graph (src/, docs/ISAM2.md)"]
        DepthCamE["/camera/depth/image_raw"] --> iSAM2
        iSAM2 -.->|"pose + distance grid (not wired yet; never back to PX4)"| GlobalPlanner
    end

    subgraph Priors["Arena Priors & Clearance Field"]
        PriorBin["config/arena_prior.bin"] --> GridMap["GridMap 3D ESDF (grid_map.hpp)"]
        DepthCam["/camera/depth/image_raw"] -->|Live Updates| GridMap
    end

    subgraph GlobalPlan["Global Trajectory Planning (RRT* / B-Spline)"]
        GridMap --> GlobalPlanner["GlobalPlannerBase (global_planner_interface.hpp)"]
        Gates["Gate Sequence (Gate 0 -> 1 -> 2 -> 3 -> 4 Window -> Pad)"] --> GlobalPlanner
        GlobalPlanner -->|BSplineGlobalPlanner (Default)| RefTraj["GlobalTrajectory (p, v, a, yaw)"]
        GlobalPlanner -.->|RRTStarPlanner (In Progress)| RefTraj
    end

    subgraph LocalControl["Local Control & Tracking (MPPI / Primitive)"]
        RefTraj --> Controller["LocalControllerBase (local_controller_interface.hpp)"]
        GridMap -->|O(1) Obstacle Cost| Controller
        QuadDyn["QuadrotorDynamics (quadrotor_dynamics.hpp)"] -->|Rollouts| Controller
        Controller -->|PrimitiveTracker (Default)| CmdOut["ControlOutput (vel_sp_ned, yawspeed)"]
        Controller -.->|MPPIController (In Progress)| CmdOut
    end

    subgraph Execution["Mission Orchestration & PX4 Bridge"]
        MissionSM["MissionManager (WAIT -> ARM -> FLY -> LAND -> DONE)"] --> Controller
        CBF["Outer-Net Control Barrier Function"] --> CmdOut
        CmdOut --> ROS2Node["ego_planner_node.cpp"]
        ROS2Node -->|TrajectorySetpoint| PX4["PX4 Autopilot (SITL or Hardware)"]
    end
```

---

## 2. Directory Structure & Key Files

| Component | Path | Description |
| :--- | :--- | :--- |
| **Arena Prior Assets** | [`config/arena_prior.json`](file:///home/ejkaba/prj/deploy/config/arena_prior.json), [`config/arena_prior.bin`](file:///home/ejkaba/prj/deploy/config/arena_prior.bin) | Surveyed 3D locations of gates, window (+5 pt bonus), pad, and $123\times123\times62$ 5cm voxel ESDF. |
| **Continuous ESDF** | [`ego_planner/include/ego_planner/grid_map.hpp`](file:///home/ejkaba/prj/deploy/ego_planner/include/ego_planner/grid_map.hpp), `src/grid_map.cpp` | Trilinear continuous obstacle clearance `getDistance()`, analytical gradient $\nabla d(p)$, and `getObstacleCost()`. |
| **Global Planner Interface** | [`ego_planner/include/ego_planner/global_planner_interface.hpp`](file:///home/ejkaba/prj/deploy/ego_planner/include/ego_planner/global_planner_interface.hpp) | Abstract `GlobalPlannerBase` and working `BSplineGlobalPlanner`. |
| **RRT\* Global Planner** | [`ego_planner/include/ego_planner/rrt_star_planner.hpp`](file:///home/ejkaba/prj/deploy/ego_planner/include/ego_planner/rrt_star_planner.hpp), `src/rrt_star_planner.cpp` | Complete RRT* scaffold with steering, neighborhood query, parent choice, and asymptotic rewiring. |
| **Local Controller Interface** | [`ego_planner/include/ego_planner/local_controller_interface.hpp`](file:///home/ejkaba/prj/deploy/ego_planner/include/ego_planner/local_controller_interface.hpp) | Abstract `LocalControllerBase` and working `PrimitiveTracker`. |
| **Quadrotor Dynamics** | [`ego_planner/include/ego_planner/quadrotor_dynamics.hpp`](file:///home/ejkaba/prj/deploy/ego_planner/include/ego_planner/quadrotor_dynamics.hpp), `src/quadrotor_dynamics.cpp` | Full 6-DoF multirotor plant ported from `dronehacks/quad.py` for MPPI forward rollouts. |
| **MPPI Controller** | [`ego_planner/include/ego_planner/mppi_controller.hpp`](file:///home/ejkaba/prj/deploy/ego_planner/include/ego_planner/mppi_controller.hpp), `src/mppi_controller.cpp` | MPPI local controller with stochastic rollout sampling, ESDF obstacle costs, and softmax expectation. |
| **Mission Manager** | [`ego_planner/include/ego_planner/mission_manager.hpp`](file:///home/ejkaba/prj/deploy/ego_planner/include/ego_planner/mission_manager.hpp), `src/mission_manager.cpp` | State machine (`WAIT` $\to$ `ARM` $\to$ `FLY` $\to$ `LAND` $\to$ `DONE`), gate sequence tracking, and CBF barriers. |
| **ROS 2 Flight Node** | [`ego_planner/src/ego_planner_node.cpp`](file:///home/ejkaba/prj/deploy/ego_planner/src/ego_planner_node.cpp) | PX4 uORB offboard streaming, camera depth subscription, and coordinate transforms (FLU $\leftrightarrow$ NED). |
| **Launch Infrastructure** | [`launch/ego_planner.launch.py`](file:///home/ejkaba/prj/deploy/launch/ego_planner.launch.py) (aliased to `launch/autonomy.launch.py`) | Modular launch configuration with parameters for hardware, planner, and controller. |
| **iSAM2 depth graph** | [`src/`](../src/), [`docs/ISAM2.md`](ISAM2.md) | C++ keyframe graph (GTSAM iSAM2): depth ICP + PX4 estimate -> corrected pose + distance grid for the planner / MPPI (`~/obstacles` topic, or `MppiControllerSink` in-process). Planning only: never fed back into PX4. |

---

## 3. Developer Instructions for MPPI

The MPPI package is located in [`ego_planner/include/ego_planner/mppi_controller.hpp`](file:///home/ejkaba/prj/deploy/ego_planner/include/ego_planner/mppi_controller.hpp) and [`ego_planner/src/mppi_controller.cpp`](file:///home/ejkaba/prj/deploy/ego_planner/src/mppi_controller.cpp).

### How to complete / tune MPPI:
1. **Forward Dynamics**: Use [`QuadrotorDynamics::step(state, action, dt)`](file:///home/ejkaba/prj/deploy/ego_planner/src/quadrotor_dynamics.cpp). It already implements:
   - Lee geometric velocity controller
   - Motor lag first-order filter ($\tau = 30\text{ ms}$)
   - Rotor allocation matrix with torque reaction
   - Quadratic aerodynamic drag
2. **Obstacle Cost**: In [`MPPIController::evaluateTrajectoryCost`](file:///home/ejkaba/prj/deploy/ego_planner/src/mppi_controller.cpp), obstacle penalties are evaluated via:
   ```cpp
   double obs_cost = grid_map.getObstacleCost(state.pos_ned, cfg_.safe_distance_m);
   total_cost += cfg_.w_obs * obs_cost;
   ```
   This query is $\mathcal{O}(1)$ without any triangle raycasting.
3. **Parallelization**:
   - For CPU execution: add `#pragma omp parallel for` across the `k = 0 ... cfg_.num_samples` loop.
   - For GPU execution: port the rollout loop to a CUDA kernel or LibTorch tensor operation (sample $K=2048$ trajectories in parallel).
4. **Launching**:
   ```bash
   ros2 launch launch/autonomy.launch.py controller:=mppi arm:=true
   ```

---

## 4. Developer Instructions for RRT\*

The RRT\* planner is located in [`ego_planner/include/ego_planner/rrt_star_planner.hpp`](file:///home/ejkaba/prj/deploy/ego_planner/include/ego_planner/rrt_star_planner.hpp) and [`ego_planner/src/rrt_star_planner.cpp`](file:///home/ejkaba/prj/deploy/ego_planner/src/rrt_star_planner.cpp).

### How to complete / tune RRT\*:
1. **Collision Oracle**: Use `grid_map.isCollisionFree(p, radius)` for point samples, and `grid_map.isSegmentClear(from, to, radius, step)` for edge validation.
2. **Hierarchical / Segment Planning**:
   Instead of running one giant 3D RRT* across the whole maze, plan segment-by-segment through the gate sequence defined in [`config/arena_prior.json`](file:///home/ejkaba/prj/deploy/config/arena_prior.json):
   - Leg 0: Launch $\to$ Gate 0 entry
   - Leg 1: Gate 0 $\to$ Gate 1
   - Leg 2: Gate 1 $\to$ Gate 2
   - Leg 3: Gate 2 $\to$ Gate 3
   - Leg 4: Gate 3 $\to$ Gate 4 (Window aperture at $[-2.283, -0.475, -0.78]$)
   - Leg 5: Gate 4 $\to$ Pad touchdown $[-0.7925, -2.0422, 0.08]$
3. **Trajectory Generation**: The method `generateTrajectory()` interpolates waypoints with continuous arc-length $s$, velocity, and curvature.
4. **Launching**:
   ```bash
   ros2 launch launch/autonomy.launch.py planner:=rrt_star arm:=true
   ```

---

## 5. Developer Instructions for iSAM2

The estimator is the C++ graph in [`src/`](../src/); see [`docs/ISAM2.md`](ISAM2.md) for factors, build, topics and
test results. It is **planning-only**: PX4 keeps flying on cuVSLAM through `ev_bridge`, and the graph corrects
PX4's estimate with depth for the planner and MPPI. Do not publish its output to `/visual_slam/tracking/odometry` or
anything else PX4 fuses: it consumes PX4's estimate, so that would be a feedback loop.

Its distance field reaches the maze MPPI today through `~/obstacles` (section 7). Next steps: publish a
`map_ned -> odom_ned` correction (section 7, "Map alignment"), register depth against `config/arena_prior.bin` (prior-map factors, robust: the as-built arena
differs), and connect its `MppiInput` / `DistanceGrid` to `MPPIController` / `GridMap`.

---

## 6. Building and Running Verification Tests

### 1. Build C++ Core Library and Run Tests
```bash
cd /home/ejkaba/prj/deploy/ego_planner
g++ -std=c++17 -O3 -fPIC -shared -Iinclude \
  src/grid_map.cpp \
  src/motion_primitives.cpp \
  src/bspline_optimizer.cpp \
  src/quadrotor_dynamics.cpp \
  src/global_planner_interface.cpp \
  src/rrt_star_planner.cpp \
  src/local_controller_interface.cpp \
  src/mppi_controller.cpp \
  src/mission_manager.cpp \
  src/c_api.cpp \
  -o build/libego_planner_core.so

g++ -std=c++17 -O3 -Iinclude tests/test_ego_planner_cpp.cpp -Lbuild -lego_planner_core -Wl,-rpath,'$ORIGIN' -o build/test_ego_planner_cpp

cd /home/ejkaba/prj/deploy
./ego_planner/build/test_ego_planner_cpp
```
Output:
```
[PASS] testGridMapAndPrior
[PASS] testMotionPrimitives
[PASS] testBSplineOptimizer
[PASS] testMissionManagerAndCAPI
[PASS] testQuadrotorDynamics
[PASS] testRRTStarPlanner
[PASS] testMPPIController
ALL C++ EGO-PLANNER-V2 & AUTONOMY TESTS PASSED
```

### 2. Run Pytest Suite
```bash
cd /home/ejkaba/prj/deploy
/home/ejkaba/prj/dronehacks-sim/.venv/bin/pytest -v tests
```
Output: `9 passed in 0.34s`.

### 3. Run Headless PX4 SITL Flight
```bash
cd /home/ejkaba/prj/deploy
sbatch /scratch/ejkaba/sitl/run_sitl_ego_test.sbatch
```

---

## 7. Maze stack contracts (`src/`, `launch/maze_autonomy.launch.py`)

Every node talks to the others only through these topics and TF, so any one of them can be replaced (by the
merged trajectory planner, by an estimator) without touching the others.

### Frames (all NED / FRD, like PX4)

| Frame | Meaning | Published by |
| :--- | :--- | :--- |
| `map_ned` | Arena / prior map (GroundTruthMap, gates, pad). Launch mark at (0.9144, -2.5146). | root |
| `odom_ned` | PX4's local frame (origin where PX4 booted), continuous, never jumps | child of `map_ned` |
| `base_link_frd` | Vehicle body | `px4_local_odom_ned.py` (or `sim_loopback_node`) |
| `isam2_world_ned` | iSAM2's world = PX4's local frame | static identity under `odom_ned` |

These names are deliberately not ROS's ENU `odom` / `base_link`, which `onboard/adapters/px4_odom.py` already
publishes from the same estimate. Two publishers of one TF edge in two conventions breaks both.

**Map alignment (`map_ned -> odom_ned`).** With `map_odom_source:=static` (default) the launch file publishes
the launch-mark offset (and `map_odom_yaw`) as a static TF: correct when the vehicle boots on the launch mark
with a true-north heading (SITL). An estimator that knows better (iSAM2 registered against the prior map, or
the merged stack) publishes this edge itself; launch with `map_odom_source:=external`. Exactly one publisher.
Every node warns (throttled) and falls back to identity when the edge is missing; identity is ~2.7 m wrong on
this course, so treat that warning as a failure.

### Topics

| Topic | Type | Frame | Producer -> consumer |
| :--- | :--- | :--- | :--- |
| `/odometry/local` | `nav_msgs/Odometry` (twist in body FRD) | `odom_ned` | `px4_local_odom_ned.py` / `sim_loopback_node` -> all |
| `/plan/global_rrt` | `nav_msgs/Path`, **latched** (reliable, transient_local) | `map_ned` | global planner -> `path_transformer_node` |
| `/plan/local_reference` | `nav_msgs/Path` (timed, see below) | `odom_ned` | `path_transformer_node` -> MPPI |
| `/plan/replan_blocked_corridor` | `geometry_msgs/PointStamped` | `map_ned` | MPPI -> global planner |
| `/oak/points/clamped` | `sensor_msgs/PointCloud2` | any, via TF | depth filter -> MPPI costmap (ray-cleared from the cloud frame's origin) |
| `/isam2_depth_node/obstacles` | `sensor_msgs/PointCloud2` | `isam2_world_ned` | iSAM2 -> MPPI local distance grid (`isam2:=true`) |
| `/control/trajectory_setpoint` | `px4_msgs/TrajectorySetpoint` | NED | MPPI -> `control_mux_node` |
| `/control/cmd_vel` | `geometry_msgs/TwistStamped` | `odom_ned` | MPPI -> `sim_loopback_node` only (debug); the mux ignores it |
| `/viz/mppi_candidates` | `visualization_msgs/MarkerArray` | `odom_ned` | MPPI -> RViz |

### Global planner contract (how to swap RRT* for the merged planner)

Replace `rrt_star_planner_node` with any node that:
1. publishes `nav_msgs/Path` on `/plan/global_rrt` in `map_ned`, latched, **only when the plan changes**. A new
   message means "new plan, starting at the vehicle": `path_transformer_node` restarts progress at s = 0. The same
   path sent again keeps progress, but do not rely on it.
2. puts the timing in the pose stamps: `poses[i].header.stamp = header.stamp + t_i`. Velocities are derived from
   them, so a trajectory optimizer's speed profile reaches MPPI unchanged. A path with equal stamps is timed at
   `cruise_speed_m_s` instead.
3. goes from the current pose to the goal through the gates **not yet passed** when it replans. RRT* does this
   with `CourseProgress` (`include/maze_planning/course_progress.hpp`), which any planner that knows where its
   anchors sit along its path can reuse.
4. handles `/plan/replan_blocked_corridor` (optional: without it a dead-centre pillar only produces a hold).

Nothing downstream (transformer, MPPI, mux, RL lookaheads) changes.

### MPPI visualization

`/viz/mppi_candidates` holds `viz_num_candidates` (default 64; 0 = off) of the sampled rollouts, evenly strided
over the K samples (namespace `mppi_candidates`, coloured green -> red by log cost), and the optimal rollout
(namespace `mppi_optimal`, blue). It is published every `viz_candidates_every_n` control ticks (default 5 =
10 Hz) and only while something subscribes. Markers expire after 0.5 s.

### Standalone demo

`sim_loopback:=true` runs `sim_loopback_node` (a first-order velocity plant) in place of PX4 and the odometry
adapter. It owns `/odometry/local` and TF; never run it next to SITL or the drone (the launch file refuses
`arm:=true` with it).
