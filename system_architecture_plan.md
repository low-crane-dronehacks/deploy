# Autonomous Quadrotor Maze Navigation System: Architecture & Implementation Plan

> **Design plan for the classical stack (branch `isam2`), written before the learned policy was deployed.**
> The learned-policy path on `main` differs from the "RL Policy Stack" sketched below: it is trained in
> [dronesim](https://github.com/low-crane-dronehacks/dronesim) (Warp ray-traced depth, Lee velocity
> controller, rsl_rl PPO) and commands velocity + yaw rate, not body rates. See README.md and AGENT.md.

This document specifies the software architecture, data pipelines, topic structures, design trade-offs, and modular work packages for an autonomous quadrotor navigating a known maze with unknown interior obstacles (e.g., netting walls, support beams, unexpected obstructions). 

It supports two interchangeable execution pathways:
1. **Classical Baseline Stack:** Prior RRT* global guiding path with real-time GPU/CPU MPPI local reactive rollouts.
2. **Reinforcement Learning (RL) Policy Stack (Future Drop-In):** An observation-to-action neural policy (trained via Isaac Lab / Flightmare) consuming compressed depth embeddings, state odometry, and global path progress, outputting body-rate/thrust commands directly to PX4.

---

## 1. System Overview & Core Philosophy

The autonomy system decouples **high-rate continuous local control** from **low-rate metric global estimation**:

* **Sensor / Perception:** Luxonis OAK-D-S2 running DepthAI. Outputs synchronized stereo monochrome streams, high-rate IMU data, and depth point clouds / low-res depth maps with disparity clamping to handle netting/mesh transparency.
* **State Estimation & Mapping:**
  * Front-end VIO streams smooth, high-rate odometry into the local controller, the RL policy observation buffer, and PX4 EKF2.
  * Back-end **iSAM2** (GTSAM) fuses visual landmarks, IMU pre-integration, and prior maze planar/line wall constraints to solve global drift and broadcast the `map -> odom` transform.
* **Global Planning:** **RRT\*** plans kinodynamically feasible or geometric paths over the prior CAD/mesh maze model to provide a topological corridor to the finish line.
* **Dual Execution Pathway (Switchable via ROS 2 multiplexer):**
  * **Pathway A (Classical):** **MPPI (Model Predictive Path Integral)** samples thousands of rollouts per cycle in real time, tracking the global RRT* path while reactively swerving around unmodeled support beams or dynamic clutter detected in the local voxel costmap.
  * **Pathway B (Learning / RL Policy):** A neural actor (exported to ONNX / TensorRT) running at $50\text{--}100\text{ Hz}$. It accepts history buffers of downsampled depth latents, vehicle body twists, and relative lookahead vectors along the RRT* guide path, outputting normalized collective thrust and body rates $(f_{\text{cmd}}, \boldsymbol{\omega}_{\text{cmd}})$.
* **Flight Control:** **PX4 Autopilot (EKF2)** running in `Offboard` mode, ingesting visual odometry via Micro XRCE-DDS and tracking setpoints from either the MPPI controller or the RL policy.

---

## 2. ROS 2 Data Flow & Topic Structure

### 2.1 Coordinate Frame Hierarchy (TF2)

```
   [map]                <-- Globally consistent frame aligned with prior maze map
     │
     │  (Published by iSAM2 Node as TF map -> odom at ~20 Hz)
     ▼
   [odom]               <-- High-rate, continuous, non-jumping local frame
     │
     │  (Published by VIO / IMU Propagation at ~100 Hz)
     ▼
[base_link]             <-- Quadrotor center of mass / IMU body frame
     │
     ├──> [camera_link] <-- OAK-D-S2 optical center
     └──> [px4_link]    <-- Flight Controller EKF2 reference
```

### 2.2 Dual-Track End-to-End Architecture

```
+--------------------------------------------------------------------------------------------------+
|                                         SENSING LAYER                                            |
|                                                                                                  |
|   +------------------------------------------------------------------------------------------+   |
|   |                              DepthAI Node (Luxonis OAK-D-S2)                             |   |
|   +------------------------------------------------------------------------------------------+   |
|        │ (Stereo IR/Mono + IMU)       │ (Clamped Depth Cloud)           │ (Low-Res Depth Image)  |
+--------┼──────────────────────────────┼─────────────────────────────────┼────────────────--------+
         │                              │                                 │
         ▼                              ▼                                 ▼
+------------------------------+  +---------------------------+  +---------------------------------+
|      ESTIMATION LAYER        |  |    LOCAL MAPPING LAYER    |  |     RL OBSERVATION BUFFER       |
|                              |  |                           |  |                                 |
|  +------------------------+  |  |  +---------------------+  |  |  +---------------------------+  |
|  | VIO Front-End / IMU    |  |  |  | STVL / Voxel Map    |  |  |  | State & Depth Preprocessor|  |
|  +------------------------+  |  |  | - Hard-locked walls |  |  |  | - 84x84 Depth / Latent    |  |
|    │ (odom -> base @ 100Hz)  |  |  | - Sensor clears     |  |  |  | - Body linear/ang vel    |  |
|    ├──────────────┬──────────┤  |  |   dynamic obs only  |  |  |  | - Lookahead error vectors|  |
|    │              ▼          |  |  +---------------------+  |  |  +---------------------------+  |
|    │       +---------------+ |  +-------------┬-------------+  +----------------┬----------------+
|    │       | iSAM2 Backend | |                │ (Costmap)                       │ (Obs Vector)
|    │       +---------------+ |                │                                 │
|    │              │          |                │                                 │
|    │              ▼          |                │                                 │
|    │        (map -> odom)    |                │                                 │
+────┼──────────────┼──────────+                │                                 │
     │              │                           │                                 │
     ▼              ▼                           │                                 │
+-----------------------------------------------+---------------------------------┼----------------+
|                   GLOBAL ROUTING & REFERENCE GENERATION                         │                |
|                                                                                 │                |
|   +--------------------------------------------------------------------------+  │                |
|   | Prior RRT* Planner (Static Maze CAD) -> Path Transformer (map -> odom)   |  │                |
|   +--------------------------------------------------------------------------+  │                |
|           │                                                                     │                |
|           ├──> Local Reference Path (nav_msgs/Path) ────────┐                   │                |
|           └──> Local Target Lookaheads ─────────────────────┼───────────────────┼──────────┐     |
+─────────────────────────────────────────────────────────────┼───────────────────┼──────────┼─────+
                                                              │                   │          │
         ┌────────────────────────────────────────────────────┘                   │          │
         ▼                                                                        ▼          ▼
+─────────────────────────────────────────────────────────────+  +---------------------------------+
|               PATHWAY A: CLASSICAL LOCAL STACK              |  |     PATHWAY B: RL POLICY STACK  |
|                                                             |  |                                 |
|  +-------------------------------------------------------+  |  |  +---------------------------+  |
|  | MPPI Controller Node                                  |  |  |  | ONNX / TensorRT Policy    |  |
|  | - Stochastically samples 2048 rollouts around RRT*    |  |  |  | - Evaluates actor network |  |
|  | - Cost penalized by obstacle voxels in costmap        |  |  |  | - Robust to netting noise |  |
|  +-------------------------------------------------------+  |  |  +---------------------------+  |
|               │                                             |  |                │                |
|               ▼ (TrajectorySetpoint: Pos + Vel + Acc)       |  |                ▼ (BodyRate/Thrust|
+───────────────┼─────────────────────────────────────────────+  +────────────────┼────────────────+
                │                                                                 │
                └───────────────────────────────┬─────────────────────────────────┘
                                                ▼
                               +----------------------------------+
                               |     Control Mode Multiplexer     |
                               | (Switches Classical vs. Policy)  |
                               +----------------------------------+
                                                │
                                                ▼
+--------------------------------------------------------------------------------------------------+
|                                       ACTUATION & PX4 LAYER                                      |
|                                                                                                  |
|   +------------------------------------------------------------------------------------------+   |
|   | PX4 Micro XRCE-DDS Bridge Node                                                           |   |
|   | - Publishes: /fmu/in/vehicle_visual_odometry (100 Hz from VIO)                           |   |
|   | - Publishes: /fmu/in/trajectory_setpoint (MPPI mode) OR /fmu/in/vehicle_rates_setpoint   |   |
|   +------------------------------------------------------------------------------------------+   |
|                                               │                                                  |
|                                               ▼                                                  |
|   +------------------------------------------------------------------------------------------+   |
|   | PX4 Autopilot (EKF2 in OFFBOARD Mode)                                                    |   |
|   +------------------------------------------------------------------------------------------+   |
+--------------------------------------------------------------------------------------------------+
```

### 2.3 Detailed Topic Specification

| Topic Name | Type | Rate | Source Node | Target Node | Purpose |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `/oak/imu/data` | `sensor_msgs/Imu` | 200 Hz | DepthAI Driver | VIO / iSAM2 / RL | High-frequency angular velocity & linear acceleration. |
| `/oak/stereo/left/image_rect` | `sensor_msgs/Image` | 30 Hz | DepthAI Driver | VIO / iSAM2 | Rectified left monochrome stream for feature tracking. |
| `/oak/stereo/right/image_rect` | `sensor_msgs/Image` | 30 Hz | DepthAI Driver | VIO / iSAM2 | Rectified right monochrome stream for stereo matching. |
| `/oak/points/clamped` | `sensor_msgs/PointCloud2` | 30 Hz | Depth Filter | STVL Costmap | Stereo point cloud clamped to $z_{\text{max}} \le 3.5\text{ m}$. |
| `/oak/depth/low_res` | `sensor_msgs/Image` | 30–50 Hz | Depth Filter | RL Obs Buffer | Downsampled ($84 \times 84$ or $160 \times 120$) depth image for policy actor input. |
| `/odometry/local` | `nav_msgs/Odometry` | 100 Hz | VIO Front-End | MPPI, RL, PX4 Bridge | Continuous, jump-free local odometry (`odom -> base_link`). |
| `/tf` (`map -> odom`) | `tf2_msgs/TFMessage` | 20 Hz | iSAM2 Backend | TF Buffer | Global correction transform aligning local frame with maze map. |
| `/fmu/in/vehicle_visual_odometry` | `px4_msgs/VehicleOdometry` | 100 Hz | PX4 Bridge Node | PX4 EKF2 | Feeds local visual odometry into PX4 state estimator. |
| `/plan/global_rrt` | `nav_msgs/Path` | 1 Hz / Event | RRT* Planner | Path Transformer | Global geometric path from current pose to maze goal in `map` frame. |
| `/plan/local_reference` | `nav_msgs/Path` | 20 Hz | Path Transformer | MPPI Controller | Sliding lookahead window of RRT* path transformed into `odom` frame. |
| `/plan/rl_lookaheads` | `std_msgs/Float32MultiArray` | 50 Hz | Path Transformer | RL Obs Buffer | 3–5 relative future waypoints $[\Delta x, \Delta y, \Delta z]$ sampled along RRT*. |
| `/costmap/local_voxels` | `nav_msgs/OccupancyGrid` or STVL | 30 Hz | STVL / Costmap2D | MPPI Controller | Combined map of static maze walls + unmodeled obstacles (beams/pillars). |
| `/control/trajectory_setpoint` | `px4_msgs/TrajectorySetpoint` | 50 Hz | MPPI Controller | Control Mux | Classical position, velocity, and yaw rate setpoints. |
| `/control/rates_setpoint` | `px4_msgs/VehicleRatesSetpoint` | 100 Hz | RL Policy Runner | Control Mux | RL body rates $(\omega_x, \omega_y, \omega_z)$ and normalized thrust. |
| `/fmu/in/trajectory_setpoint` | `px4_msgs/TrajectorySetpoint` | 50 Hz | Control Mux | PX4 Offboard | Routed to PX4 when classical mode is active. |
| `/fmu/in/vehicle_rates_setpoint` | `px4_msgs/VehicleRatesSetpoint` | 100 Hz | Control Mux | PX4 Offboard | Routed to PX4 when RL policy mode is active. |

---

## 3. Technology Evaluation: Pros, Cons, and Selection Rationale

### 3.1 Sensor: Luxonis OAK-D-S2 (DepthAI)
* **Rationale:** Lightweight stereo-depth + IMU module handling on-chip stereo rectification and disparity filtering without consuming companion CPU/GPU resources.
* **Pros:** Hardware-synced monochrome frames + IMU; configurable subpixel disparity and on-chip downsampling.
* **Cons:** Stereo matching penetrates netting when holes reveal high-contrast backgrounds; drywalls lack features.
* **Mitigation:** Disparity clamping ($z_{\text{max}} \le 3.5\text{ m}$), bilateral/speckle filtering, and hard-locking the static maze walls in mapping memory.

### 3.2 Global Estimator: iSAM2 (Incremental Smoothing via GTSAM)
* **Rationale:** Provides mathematically optimal non-linear graph optimization incorporating prior geometric constraints (planar walls) to eliminate long-term drift.
* **Pros:** Bayes tree incremental updates maintain constant-time solving; natively integrates heterogeneous sensor priors.
* **Cons:** Graph re-linearization can induce small spatial coordinate corrections.
* **Mitigation:** Decouple estimation: broadcast corrections strictly as $T_{\text{map}}^{\text{odom}}$, keeping the local control loop on smooth local VIO.

### 3.3 Global Planner: RRT*
* **Rationale:** Asymptotically optimal search over the known 2D/3D maze model. Guarantees finding valid topological routes across dead-ends and forks where reactive planners get trapped.
* **Pros:** Prevents cul-de-sac entrapment; operates directly on the known CAD map.
* **Cons:** Paths can be non-smooth; recomputing from scratch is computationally expensive.
* **Mitigation:** Run at low frequency (1 Hz or event-driven) and smooth outputs with cubic B-splines.

### 3.4 Local Control: Classical MPPI vs. Reinforcement Learning Policy

| Metric | Classical Pathway (MPPI) | RL Policy Pathway (Neural Actor) |
| :--- | :--- | :--- |
| **Input Interface** | 3D Voxel Costmap + Lookahead `nav_msgs/Path` | Downsampled Depth Image/Latents + Vector Observation |
| **Output Interface** | `TrajectorySetpoint` (Pos + Vel + Acc feedforward) | `VehicleRatesSetpoint` (Body rates $\boldsymbol{\omega}$ + Thrust) |
| **Control Latency** | 10–20 ms (samples 2048 trajectories) | 2–5 ms (single forward pass through TensorRT/ONNX) |
| **Unmodeled Obstacles (Beams)** | Avoids by checking collision costs in voxel grid | Avoids implicitly if trained in cluttered randomized simulations |
| **Netting Robustness** | Needs explicit voxel raycast locks to prevent punch-through | Highly robust if domain-randomized with missing depth / dropout |
| **Predictability & Safety** | Deterministic: respects hard bounds and clear costs | Black-box: requires safety envelope / fallback filter |
| **Development Cycle** | Ready immediately via parameter tuning | Requires sim-to-real training pipeline (Isaac Lab / Flightmare) |

**Why Support Both?**  
MPPI provides a robust, predictable, and immediately functional baseline that ensures safe hardware testing today. The RL policy provides an upgrade path for ultra-low latency, highly agile flight where unmodeled obstacles can be evaded at high speeds without the overhead of voxel raycasting.

---

## 4. Reinforcement Learning Architecture Specifications (Future Drop-In)

### 4.1 Policy Observation Space $\mathcal{O}_t \in \mathbb{R}^{d}$

The RL observation vector is constructed at $50\text{--}100\text{ Hz}$ from synchronized sensor and estimator buffers:

1. **Vehicle Kinematics (in body frame $\mathcal{B}$):**
   * Linear velocity: $\mathbf{v}_{\mathcal{B}} = R_{\mathcal{B}}^{\text{odom}} \mathbf{v}_{\text{odom}} \in \mathbb{R}^3$
   * Angular velocity: $\boldsymbol{\omega}_{\mathcal{B}} \in \mathbb{R}^3$
   * Projected gravity vector: $\mathbf{g}_{\mathcal{B}} = R_{\mathcal{B}}^{\text{odom}} \begin{bmatrix} 0 & 0 & -1 \end{bmatrix}^\top \in \mathbb{R}^3$
2. **Global Tracking Guidance (from RRT* path):**
   * Relative positions to $N = 4$ lookahead path coordinates spaced at $d = [0.5, 1.0, 2.0, 3.5]\text{ m}$:
     $$\Delta \mathbf{p}_k = R_{\mathcal{B}}^{\text{odom}} (\mathbf{p}_{\text{path}, k}^{\text{odom}} - \mathbf{p}_{\text{drone}}^{\text{odom}}) \in \mathbb{R}^3 \quad (k \in \{1, \dots, 4\})$$
3. **Action History:**
   * Previous action: $\mathbf{a}_{t-1} \in \mathbb{R}^4$
4. **Exteroception (Depth):**
   * *Option 1 (Direct Depth):* Downsampled depth image ($84 \times 84$ or $64 \times 64$, normalized $[0, 1]$) fed into a lightweight CNN/MobileNet encoder.
   * *Option 2 (Spatial Autoencoder / Ray Latent):* 32–64 virtual 1D ray distances sampled from the depth image along the flight envelope.

### 4.2 Policy Action Space $\mathcal{A}_t \in \mathbb{R}^4$

* Body rate commands and collective thrust:
  $$\mathbf{a}_t = \begin{bmatrix} \omega_x^{\text{cmd}} & \omega_y^{\text{cmd}} & \omega_z^{\text{cmd}} & f_{\text{cmd}} \end{bmatrix}^\top$$
* Scaled to vehicle physical limits: $\omega_{x,y} \in [-180, 180]^\circ/\text{s}$, $\omega_z \in [-90, 90]^\circ/\text{s}$, $f_{\text{cmd}} \in [0.1, 0.9]$ normalized thrust.
* Bypasses high-level trajectory tracking and hooks directly into PX4’s rate-control loop via `/fmu/in/vehicle_rates_setpoint`.

### 4.3 Sim-to-Real Domain Randomization Strategy

To ensure zero-shot transfer into the netted maze:
* **Sensor Noise Modeling:** Randomly inject depth dropouts (simulating stereo blind zones on plain walls), depth punch-through artifacts (simulating netting holes), and Gaussian noise $\sigma_z \propto z^2$.
* **Obstacle Randomization:** Procedurally spawn cylinders/boxes ($0.05\text{--}0.2\text{ m}$ diameter) across corridors in simulation to represent unmodeled support beams.
* **Actuation Latency:** Add random latency buffers ($20\text{--}50\text{ ms}$) between policy inference and motor execution.

---

## 5. Work Packages for Modular Multi-Agent Implementation

```
                    +------------------------------------+
                    |        System Implementation       |
                    +------------------------------------+
                                      │
         ┌────────────────┬───────────┴────────────┬────────────────┬────────────────┐
         ▼                ▼                        ▼                ▼                ▼
   [Package WP-1]   [Package WP-2]           [Package WP-3]   [Package WP-4]   [Package WP-5]
      Sensing &       Estimation &              Global &        PX4 Bridge &       RL Policy
     Perception         Mapping              Local Planning     Supervisor        Deployment
```

### Work Package WP-1: Sensing & Perception Node (`pkg_maze_sensing`)
* **Goal:** Wrap DepthAI to produce calibrated, synchronized stereo streams, IMU data, clamped point clouds, and downsampled depth images for the RL actor.
* **Primary Tasks:**
  1. Construct ROS 2 node using `depthai-core`.
  2. Implement an on-chip pipeline with `MonoCamera`, `StereoDepth`, and `IMU`.
  3. Enable Subpixel disparity, Disparity Shift, and Bilateral filtering.
  4. Write a post-processing filter that publishes:
     * `/oak/points/clamped` ($z \le 3.5\text{ m}$, cropped ceiling/floor) for costmap.
     * `/oak/depth/low_res` ($84 \times 84$ float32 matrix) for the RL policy.
* **Deliverables:** `oak_d_node.cpp`, `depth_filter_node.cpp`, `oak_config.yaml`.

### Work Package WP-2: State Estimation & Map Alignment (`pkg_maze_estimator`)
* **Goal:** Implement high-rate local VIO and low-rate iSAM2 global graph optimizer with prior plane factors.
* **Primary Tasks:**
  1. **Front-End VIO:** Deploy OpenVINS / stereo MSCKF subscribing to `/oak/stereo/*` and `/oak/imu/data`. Publish `/odometry/local` at $\ge 100\text{ Hz}$.
  2. **iSAM2 Back-End:** Ingest prior maze CAD wall equations ($\mathbf{n}^\top \mathbf{x} + d = 0$). Add `BetweenFactor<Pose3>` and `PlaneFactor` residuals. Broadcast TF `map -> odom` at $20\text{ Hz}$.
* **Deliverables:** `isam2_backend_node.cpp`, `maze_prior_loader.cpp`, unit tests for bounded drift.

### Work Package WP-3: Classical Planning & Costmap (`pkg_maze_planning`)
* **Goal:** Implement the RRT* global reference solver, STVL costmap, and MPPI local controller.
* **Primary Tasks:**
  1. **RRT* Global Planner:** Ingest 2.5D/3D prior maze map. Compute clearance-constrained optimal path to finish; publish `/plan/global_rrt`.
  2. **Path Transformer:** Track current pose, extract lookahead path segments, and transform into `odom` frame as `/plan/local_reference` and `/plan/rl_lookaheads`.
  3. **Costmap Layer (STVL):** Lock static maze walls into voxel memory; update unmodeled obstacles using `/oak/points/clamped`.
  4. **MPPI Controller:** Run 2048 rollouts at 50 Hz on GPU/CPU to track path and avoid unmodeled obstacles. Output `/control/trajectory_setpoint`.
* **Deliverables:** `rrt_star_planner.cpp`, `path_transformer_node.cpp`, `mppi_controller.cpp`.

### Work Package WP-4: PX4 Bridge & Autonomy Supervisor (`pkg_maze_control`)
* **Goal:** Implement Micro XRCE-DDS bridge, flight state supervisor, and the Control Mode Multiplexer.
* **Primary Tasks:**
  1. Set up Micro XRCE-DDS bridge translating `/odometry/local` into `px4_msgs::msg::VehicleOdometry`.
  2. Implement **Control Mux**:
     * In `CLASSICAL` mode: Route `/control/trajectory_setpoint` to `/fmu/in/trajectory_setpoint`.
     * In `RL_POLICY` mode: Route `/control/rates_setpoint` to `/fmu/in/vehicle_rates_setpoint`.
  3. Write BehaviorTree state machine (`BOOT -> ARMED -> TAKEOFF -> RUNNING -> RECOVERY -> LAND`).
  4. Implement watchdogs: If tracking covariance drops or costmap collision check predicts impact $<0.3\text{ s}$, drop to PX4 loiter/hold.
* **Deliverables:** `px4_bridge_node.cpp`, `control_mux_node.cpp`, `mission_supervisor_node.cpp`.

### Work Package WP-5: RL Policy Deployment (`pkg_maze_rl`)
* **Goal:** Deploy trained neural network policy on the companion computer using ONNX Runtime / TensorRT.
* **Primary Tasks:**
  1. **Observation Assembler Node:**
     * Subscribe to `/odometry/local`, `/oak/imu/data`, `/oak/depth/low_res`, and `/plan/rl_lookaheads`.
     * Transform velocities and lookaheads into current body frame $\mathcal{B}$.
     * Construct normalized flat tensor and depth image array at 50–100 Hz.
  2. **Policy Inference Runner:**
     * Load trained policy (`policy.onnx` or `policy.engine`).
     * Run forward pass on Jetson GPU/DLA or CPU via TensorRT.
     * Scale outputs to body-rate and thrust targets; publish `/control/rates_setpoint`.
  3. **Safety Fallback Interlock:**
     * Monitor policy output variance and network latency. If inference exceeds $20\text{ ms}$ or commands exceed flight envelope limits, trigger Control Mux fallback to classical MPPI stop-and-hover.
* **Deliverables:** `rl_observation_node.cpp`, `policy_runner_node.cpp`, launch scripts.

---

## 6. Failure Mode & Effects Analysis (FMEA)

| Failure Mode | Root Cause | Detection Mechanism | Recovery Action |
| :--- | :--- | :--- | :--- |
| **Disparity Punch-Through** | Netting holes let stereo see background; wall appears empty. | Costmap ray-clearing violates prior solid map bounds. | **Mitigation:** Static maze walls are permanently flagged as occupied. Depth rays cannot clear voxels inside the prior wall bounding boxes. |
| **Pillar Dead-Center** | An unmodeled support beam is centered in a narrow corridor. | MPPI cost spikes along the centerline; forward progress stalls. | MPPI/RL naturally splits distribution left/right. If clearance on both sides is $< r_{\text{drone}}$, supervisor switches to `OBSTACLE_HOLD`, marks edge blocked in RRT*, and triggers a global replan. |
| **iSAM2 Graph Shift** | Landmark re-triangulation shifts global pose by $10\text{ cm}$. | Step change observed on `map -> odom` transform. | Zero impact on flight stability. Local MPPI and RL policy operate strictly in continuous `odom`/body frame; only the lookahead target shifts smoothly. |
| **RL Policy Out-of-Distribution** | Complex beam shadows or glare cause erratic policy outputs. | Action jerk filter triggers or policy value function estimates high risk. | Control Mux trips watchdog and immediately switches control authority to Classical MPPI / PX4 Position Hold. |
| **VIO Tracking Loss** | Fast yaw or feature-poor drywall causes visual drop. | Odometry covariance exceeds threshold or tracking status flags failure. | Supervisor switches PX4 immediately to altitude hold / position hold using internal EKF2 optical flow / baro fallback. |

---

## 7. Implementation Milestones

1. **Milestone 1 (Sensors & VIO Base):** Stand up OAK-D-S2 with DepthAI. Verify VIO odometry publishes cleanly at $\ge 100\text{ Hz}$ and drives PX4 EKF2 in manual Offboard position-hold flight.
2. **Milestone 2 (iSAM2 + Prior Map):** Ingest static maze geometry into GTSAM. Verify that running along the maze corridor continuously publishes a drift-free `map -> odom` transform without discrete pose jumps.
3. **Milestone 3 (Classical Simulation Flight):** In Gazebo / PX4 SITL, place unmodeled vertical poles inside a netted corridor. Verify that RRT* generates global routes and MPPI executes lateral evasive maneuvers around the poles.
4. **Milestone 4 (RL Sim-to-Real Training):** Export the Isaac Lab / Flightmare training environment mirroring the maze and sensor latency. Train the policy actor on lookahead targets and downsampled depth.
5. **Milestone 5 (Hardware Dual-Track Validation):** Validate the Control Mux on physical hardware: fly the maze using classical MPPI, flip the multiplexer switch mid-flight to hand over authority to the RL policy runner, and verify safe traversal.