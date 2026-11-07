# AGENTS.md: context for coding agents

This repo is the code that flies. Training lives in dronehacks-sim; read its AGENTS.md for the rules, the
task, and how policies are trained. Keep this repo small, fast, and robust.

Docs:
- `docs/AUTONOMY_ARCHITECTURE.md`: Master reference for C++ EGO-Planner-v2, MPPI, RRT*, and iSAM2.
- `docs/SITL.md`: Local PX4 SITL + Gazebo Harmonic rig.
- `docs/FLYX_LOCAL.md`: Organizers' sim on pear-omen.
- `docs/POLICIES.md`: Policy files + observation/action contract.
- `docs/HARDWARE.md`: Kit, training-plant assumptions, open data.
- `docs/ISAM2.md`: src/: C++ iSAM2 depth ICP + PX4 factor graph -> state + distance field for an MPPI.
Update them when you change what they describe.

## Autonomy Stack Architecture (`ego_planner/`)

The C++ ROS 2 autonomy stack is decoupled into clear interfaces:
1. **Clearance & Collision**: `GridMap` (`ego_planner/include/ego_planner/grid_map.hpp`) provides $\mathcal{O}(1)$
   continuous 3D ESDF queries, analytical spatial gradients $\nabla d(p)$, and `getObstacleCost()`.
2. **Global Planning**: `GlobalPlannerBase` (`global_planner_interface.hpp`) with `BSplineGlobalPlanner` (production default)
   and `RRTStarPlanner` (`rrt_star_planner.hpp`) for asymptotic global path search.
3. **Local Control**: `LocalControllerBase` (`local_controller_interface.hpp`) with `PrimitiveTracker` (production default)
   and `MPPIController` (`mppi_controller.hpp`) utilizing `QuadrotorDynamics` for 50-100 Hz forward rollouts.
4. **State Estimation**: `onboard/slam/isam2_node.py` factor graph node linking IMU preintegration, visual odometry,
   rangefinder, and `config/arena_prior.json` surveyed anchors to PX4 `vehicle_visual_odometry`.
5. **Mission State Machine**: `MissionManager` (`mission_manager.hpp`) with hierarchical states
   `WAIT` -> `ARM` -> `FLY` -> `LAND` -> `DONE`, gate sequence tracking, and CBF net barrier constraints.

## Rules
- **Never arm or fly without explicit approval from the team for that specific flight** (SITL included when it
  runs on a shared lab PC). Dry runs (`arm:=false`) are always fine.
- One stack for drone / sitl / flyx: platform differences go in `launch/autonomy.launch.py` (and `policynav.launch.py`)
  PROFILES and `config/`, never as `if sim:` branches inside core planner or onboard code.
- Pin versions in `config/versions.yaml`; a PX4/px4_msgs mismatch silently renames topics (`*_v1`).
- The organizers' simulator is a HELD-OUT test: never tune policies or this stack's parameters to it.
- No credentials, tokens or keys in this repo.
- FlyX (the organizers' sim) is proprietary: local non-distributed use only. Never commit its code or images.

## Conventions (what the policy expects; see policynav/obs.py)
- Depth: OAK depth (16UC1 mm, 0 = invalid) is re-projected into the training camera (128x80, 72x50 deg,
  looking along body x). Invalid -> near; a simulator's +inf (no return in range) -> far.
- State: PX4 vehicle_local_position_v1 velocity (NED) and vehicle_attitude (FRD->NED, w first).
- Action: [fwd, right, down, yaw_rate] in [-1,1] x v_max in the yaw-aligned frame -> TrajectorySetpoint
  velocity (NED) + yawspeed, Offboard velocity mode, 10 Hz decisions, 20 Hz stream.
- A launch must start from a heading the policy was trained for (dronehacks-sim banks v3: any heading).

## Gotchas
- gz-physics segfaults on OBJ collision meshes without vertex normals (sitl/export_world.py writes them).
- snap Docker: use `--runtime nvidia -e NVIDIA_VISIBLE_DEVICES=all`, not `--gpus all`; Gazebo ogre2 needs the
  glvnd EGL vendor file (sitl/rig_entry.sh writes it).
- PX4 1.17 ignores the yaw in PX4_GZ_MODEL_POSE; sitl/rig_entry.sh spawns the model itself.
- Their onboard/ services read /etc/flyx/onboard.yaml in their sim; the drone needs its own deployment file.
- FlyX locally: without `net.core.rmem_max/wmem_max` >= 32 MB on the host, FastDDS drops frames and the sim
  falls to ~0.1x real time after ~2 min (looks like a cuVSLAM/estimator failure).
- `sitl/battery.py` is a copy of dronehacks-sim's; keep the two in sync.
