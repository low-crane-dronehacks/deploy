# deploy

Drone software for DroneHacks 2026: one modular ROS 2 / PX4 stack that runs **unchanged** on the aircraft, on our
PX4 SITL rig, and in the organizers' simulator. Only the hardware layer differs (`hardware:=drone|sitl|flyx`).

Supports both learned RL neural policies (`policynav/`) and classical/optimization-based autonomy stacks (`ego_planner/`),
modularized for **EGO-Planner-v2**, **MPPI**, **RRT\***, and **iSAM2**.

```
ego_planner/   C++17 ROS 2 autonomy stack: GridMap 3D ESDF, Motion Primitives, B-Spline Optimizer,
               Quadrotor Dynamics, Global Planner Interface (RRT*), Local Controller Interface (MPPI),
               and Mission Manager state machine
launch/        autonomy.launch.py (alias of ego_planner.launch.py) & policynav.launch.py
config/        arena_prior.bin, arena_prior.json (surveyed 3D course anchors & 5cm voxel ESDF)
src/           C++ iSAM2 factor graph (depth ICP + PX4) -> state + distance field for an MPPI (docs/ISAM2.md)
onboard/       VIO & SLAM: cuVSLAM wrapper, iSAM2 factor graph node (slam/isam2_node.py), EV bridge
policynav/     policy node: depth + PX4 state -> Offboard velocity setpoints (NumPy export)
sitl/          PX4 SITL rig: Docker/Apptainer, PX4 1.17 + Gazebo Harmonic + uXRCE, course world
docs/          AUTONOMY_ARCHITECTURE.md, SITL.md, FLYX_LOCAL.md, POLICIES.md, HARDWARE.md, ISAM2.md
```

## Architecture & Modular Packages

For complete architectural details, interface contracts, and instructions on plugging in **MPPI**, **RRT\***, and **iSAM2**, see:
- [docs/AUTONOMY_ARCHITECTURE.md](docs/AUTONOMY_ARCHITECTURE.md)

## Quick Start & Flight Execution

```bash
# 1. Run unit tests (C++ standalone and Python pytest)
cd ego_planner/build && ./test_ego_planner_cpp
pytest -v tests

# 2. Launch production EGO-Planner-v2 in SITL (arm:=true to fly)
ros2 launch launch/autonomy.launch.py hardware:=sitl arm:=true desired_speed_ms:=2.25

# 3. Launch with modular options (RRT* global planner or MPPI local controller)
ros2 launch launch/autonomy.launch.py hardware:=sitl arm:=true planner:=rrt_star controller:=mppi

# 4. Dry-run / benchmark in fast simulator (10,000x real-time)
python3 ego_planner/scripts/eval_ego_planner_sim.py --episodes 5 --speed 2.25
```

**Nothing arms unless `arm:=true`.** Stop conditions (stale depth, estimator, geofence, time, pilot takeover)
are in `policynav/mission.py`. `python3 -m pytest -q tests` runs the pure tests.

## Status

- [x] policy node, dry-run verified in the organizers' sim and in SITL (actions match training)
- [x] SITL rig (pear-deepak), truth-based external vision, 4S battery sag (docs/SITL.md)
- [x] organizers' sim installed locally on pear-omen; needs the host UDP-buffer sysctl to stay real-time (docs/FLYX_LOCAL.md)
- [ ] SITL flight; SITL with stereo + cuVSLAM instead of truth
- [ ] Jetson image (ARM ROS 2 Jazzy, depthai_ros_driver, cuVSLAM ARM build, uXRCE serial)
- [ ] aircraft: Pixhawk 6C on PX4 v1.17, OAK calibration + TF, VL53L1X, EKF2 external-vision params, RC kill switch
