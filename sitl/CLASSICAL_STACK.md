# Classical stack in the SITL rig: A* + MPPI + iSAM2 + cuVSLAM

For teammates and their coding agents. It covers how we intend to use the rig (`docs/SITL.md`) with the classical
stack from the `isam2` branch, the interface that stack must meet to plug in, what is still missing, and what data
to record on the real drone so most of this can be tested on saved data.

Read `AGENTS.md` (repo root) first. Its rules apply here unchanged. In particular: **never arm without the team's OK
for that specific flight, SITL included**. Dry runs are always fine.

## Status (2026-11-08)
- **The rig** (`sitl/`, branch `sitl-humble`) is written and has not run yet. The first `run_rig.sh build` failed
  configuring PX4: its Gazebo optical-flow plugin needs `libopencv-dev`. The Dockerfile now installs it, but the
  build now succeeds, but `run_rig.sh up` fails at Gazebo startup (see `sitl/HANDOFF.md`). Treat the rig as
  unproven until `docs/SITL.md`'s Status says it flies.
- **The classical stack** (branch `isam2`, which includes `ego-planner-v2`) has `astar_planner_node`,
  `mppi_controller_node`, `isam2_depth_node`, `path_transformer_node`, `control_mux_node`,
  `launch/maze_autonomy.launch.py` and `docs/ISAM2.md`. It was written against Jazzy, px4_msgs `release/1.17` and an
  older `run_rig.sh`. It has not been built on the Humble rig.
- Nothing in this file has been done yet. It is the plan.

## What we use this stack for
1. **Closed loop in SITL:** A* global plan, MPPI local control, and iSAM2 giving the planner its state and obstacle
   field. Everything runs on the drone's real software path: PX4 v1.16.2, skybeta's `flight_interface` and
   `px4_odom_bridge`, and ROS 2 Humble. The truth-based fake VIO comes first, real cuVSLAM later.
2. **Open-loop replay** of bags recorded on the real drone. This tests cuVSLAM, stereo depth and iSAM2 on real
   sensors, checks A* and MPPI against real state, and tunes the rig to match the real drone (see "Data to record").
3. Later, the real drone, through exactly the same launch file.

The policy path (`skybeta.launch.py`) stays as it is. The classical stack is a second team launch, not a replacement.

## Non-negotiables
- **skybeta is read-only.** We never edit `../skybeta`. Organizer updates refuse a dirty tree.
- **Commands go only through `flight_interface`.** That means `/skybeta/setpoint/twist`: TwistStamped, ENU, m/s,
  expires after 1 s, |v| clamped to `max_speed_mps` (1.5). No node of ours publishes on `/fmu/in/*`, requests
  Offboard, or arms. `flight_interface` owns the Offboard heartbeat and the safety envelope on the drone and in the
  rig. `control_mux_node` on `isam2` publishes `/fmu/in/offboard_control_mode` and `trajectory_setpoint` directly, so
  it must not run under skybeta (see Phase 3).
- **iSAM2 is planning-only.** Its output goes to the planner and MPPI, never to PX4. PX4 fuses only
  `/visual_slam/tracking/odometry`, through skybeta's bridge. iSAM2 uses PX4's estimate as odometry, so feeding it
  back would create a loop.
- **One stack.** Platform differences go in launch arguments and `config/`, never `if sim:` inside nodes
  (`AGENTS.md`).
- **Pin versions.** The rig is PX4 `v1.16.2`, px4_msgs `release/1.16`, Humble and Gazebo Harmonic
  (`config/versions.yaml`, `sitl_*` keys). A px4_msgs mismatch silently renames topics (`*_v1`).

## The interface MPPI and iSAM2 must meet

### What the rig provides (domain 150, skybeta's `fastdds_localhost.xml`)
| topic | type | notes |
|---|---|---|
| `/fmu/out/vehicle_local_position[_v1]`, `/fmu/out/vehicle_attitude`, `/fmu/out/vehicle_status[_v1]` | px4_msgs 1.16 | **check the exact names** with `run_rig.sh sh` then `ros2 topic list \| grep fmu`. Make them node parameters; don't hard-code them |
| `/visual_slam/tracking/odometry` | nav_msgs/Odometry, ENU | from `gz_vio.py` now, from cuVSLAM in Phase 4 |
| `/policynav/depth/image` | sensor_msgs/Image 32FC1, metres | NaN = invalid (includes readings under 0.33 m), +inf = no return. Frame `sim_depth_optical` |
| `/policynav/depth/camera_info` | CameraInfo | 640x400, fx ~420 |
| `/tf_static` | | `base_link -> sim_camera_link -> sim_depth_optical` (from `CAMERA_*`) |
| `/skybeta/state` | std_msgs/String (JSON) | `armed`, `offboard_active` |

### What the classical stack must output
- **One** command stream: `/control/cmd_vel` (TwistStamped, NED, m/s plus yaw rate), at 10 Hz or faster.
  `skybeta_twist_adapter` (to be written, Phase 3) converts it to ENU on `/skybeta/setpoint/twist`.
- **Speed:** plan at 1.45 m/s or less. The `isam2` default `cruise_speed_m_s: 2.60` is above the interface clamp,
  and clamped commands are no longer the trajectory MPPI optimized.
- **Altitude:** stay inside `flight_interface`'s 0.3–3.0 m band above the odometry origin, with margin. A*
  `flight_z_ned` has to respect it.
- **Frames:** keep the `isam2` convention. NED `map_ned -> odom_ned -> base_link_frd`, and
  `odom_ned -> isam2_world_ned` is identity. The rig spawns at ENU (-2.5146, 0.9144), yaw pi/2 (`world/spawn.txt`),
  which is the launch file's `LAUNCH_MARK_NED` (0.9144, -2.5146), yaw 0. Keep these two in sync.
- **Files** (`config/course_groundtruth.bin`, `sitl/world/course.obj`, `config/arena_prior.*`): resolve them through
  `get_package_share_directory`. Relative paths break in the rig, which builds into `/cache/ws`.
- **Depth:** `isam2_depth_node` must treat NaN pixels as invalid. It documents 16UC1 (0 = invalid) and 32FC1 (+inf
  = no return); NaN isn't covered yet. MPPI's `obstacle_cloud_topic` defaults to `/oak/points/clamped`, which nothing
  publishes in the rig or on the drone today. Use `""` (the iSAM2 grid only), or add a cloud publisher.
- **The net:** in the sim it is an invisible collision shell, and stereo may not see it on the drone either. Don't
  rely on depth to avoid the net. Use the prior map and a geofence.

### Build requirements
- The rig image needs GTSAM 4.2 (`find_package(GTSAM 4.2)`) and `visualization_msgs`. Plan:
  `ros-humble-gtsam` and `ros-humble-visualization-msgs` in `sitl/Dockerfile`. Check that the apt GTSAM is 4.2 on
  arm64; if not, build 4.2 from source with the system Eigen (an Eigen mismatch causes alignment crashes).
- `run_rig.sh build_ws` has to build the new packages, as a Release build (MPPI with 2048 samples is unusable in
  Debug). It currently builds only `skybeta_flight_interface oakd_vslam_bringup policynav`.
- Changing the Dockerfile means **every machine rebuilds its own image** (`run_rig.sh build`). Images are built
  locally and are not shared between arm64 and x86. Bundle Dockerfile changes so people rebuild once. After that,
  code changes only need `run_rig.sh build ws`.

## Plan
Phases 1–3 are the shortest path to "the classical stack flies in SITL". Phase 4 doesn't depend on them.

**Phase 0: Base rig.** Finish `build`, then `up`, `team`, `pilot fly` (dry run) with the policy. Record the real
PX4 topic names and the idle CPU, GPU and memory use (`jtop`).
*Done:* the policy flies the course in SITL, armed with team OK.

**Phase 1: Bring `isam2` onto the Humble rig.** Branch `classical-sitl` off `sitl-humble`, merge `isam2` (keep
`sitl-humble`'s `sitl/run_rig.sh`), and make the build changes above. Defer `ego_planner/` unless the team decides
otherwise. Run the branch's unit tests (`test_maze_planning`, `test_depth_isam2`, `tests/test_maze_planning.py`) in
the container.
*Done:* `build ws` is clean, `isam2_depth_node` is built (GTSAM and px4_msgs both found), and the tests pass.

**Phase 2: Open loop in the rig (no commands).** New `launch/skybeta_classical.launch.py`, run with
`TEAM_LAUNCH=skybeta_classical.launch.py sitl/run_rig.sh team`:
`px4_local_odom_ned.py -> /odometry/local`, `isam2_depth_node` (on `/policynav/depth/*`), `astar_planner_node`,
`path_transformer_node`, `mppi_controller_node`, and the static TFs. No `control_mux`, and nothing listens to
`/control/cmd_vel` yet.
*Done:* the A* path follows the course from the spawn point. The iSAM2 obstacles line up with the mesh within about
0.1 m. With the drone moved by hand (`gz service ... set_pose`), MPPI rollouts bend around obstacles and the command
points along the path.

**Phase 3: Close the loop through skybeta.** Write `skybeta_twist_adapter` (Python, in `policynav`, modelled on
`skybeta_node.py`):
- converts NED to ENU;
- publishes at 10 Hz;
- sends zero until `offboard_active`;
- on a stale command (over 0.3 s), stale depth or odometry, or `max_flight_s`: sends zero and calls `/skybeta/hold`.

Unit-test the frame conversion. Team decision: the yaw policy (we recommend facing along the velocity, rate-limited).
Fly order: dry run, then armed runs with team OK, then with `VIO_*` and `DEPTH_*` noise, then with a
`VIO_OUTAGE_AT_S`.
*Done:* 3 armed course completions in a row with no geofence or hold, noisy runs completing or stopping safely, and
bags kept per run.

**Phase 4: Real cuVSLAM in place of `gz_vio.py`.**
1. Stereo pair and IMU in `models/x500_skybeta/model.sdf.in`: mono 640x400 at 30 Hz with the OAK-D S2's baseline
   (~7.5 cm) and intrinsics from the drone's calibration, and a 200 Hz IMU.
2. `sitl/gz_stereo.py`, a Python gz-transport bridge, on the topics skybeta's `oakd_vslam.launch.py` feeds cuVSLAM
   (`/oak/{left,right}/image_raw`, `camera_info`, `/oak/imu/data`). All stamps come from the gz clock.
3. `isaac_ros_visual_slam` runs as a sidecar container built from skybeta's own image:
   `--network container:dronehacks-rig --ipc container:dronehacks-rig`, domain 150. It starts only the rectify and
   cuVSLAM part (a launch file of ours in `sitl/`). This keeps the rig isolated and matches the drone's cuVSLAM
   version.
   - skybeta's image is **arm64-only** (`FROM nvcr.io/nvidia/isaac/ros:aarch64-...`). An x86 machine needs an x86
     Isaac ROS base and an NVIDIA GPU, and won't match the drone's version exactly.
4. `VIO=truth|cuvslam` switch in `rig_entry.sh`. READY waits for `/visual_slam/status` to show tracking.
5. **First, look at the rendered stereo frames.** cuVSLAM needs texture, and a flat-shaded world won't track.
6. Then optionally run `stereo_backend:=vpi` on the sim stereo pair instead of the perfect depth camera.

*Done:* drift under about 1% of distance travelled, no track loss on the course, and Phase 3's criteria passing with
`VIO=cuvslam`.

**Running across two machines (option, not the default):** Gazebo and PX4 on an x86 PC, the stack on the Jetson.
This needs a DDS profile limited to the link between the two machines and a separate domain. It deliberately gives
up the rig's loopback-only isolation, so never use it on a drone whose real stack is up.

## Data to record on the drone (for replay)
Replay is **open loop**: MPPI's commands don't move a recorded drone, so it tests the estimator and perception, the
planners' outputs, and the tuning of the rig. MPPI's closed-loop behaviour is still tested in SITL.

**The base recording** is skybeta's `host/run.sh bag` (`container_scripts/record_bag.sh`): the raw OAK pair, both
`camera_info`s, the IMU, `/visual_slam/tracking/odometry` and `status`, `/fmu/in/vehicle_visual_odometry`, and every
`/fmu/out/*`.

**Add these topics** (`record_bag.sh -- <topics>`, or `--all`), and compress with
`-- --compression-mode file --compression-format zstd`:
- `/tf_static`, `/tf`: the camera and IMU mount, which iSAM2 needs
- `/skybeta/state`, `/skybeta/setpoint/twist`: when Offboard was active and what was commanded
- `/policynav/depth/*` when our stack runs
- `/rosout`
- the OAK's on-device depth, if the driver publishes it

The raw stereo pair is about 15 MB/s, so a 2-minute run is about 2 GB.

**Save outside the bag, per run:**
- the PX4 **ulog** from the SD card (EKF2 innovations, motor outputs, high-rate IMU)
- the FC **parameter dump**, into `sitl/params/` for `PX4_PARAMS_DUMP`
- `/etc/skybeta/drone.env` (`CAMERA_*`) and the OAK's EEPROM calibration
- skybeta `VERSION`, the deploy git sha, and the policy file
- notes: drone id, battery voltage at start, lighting, start pose, anything unusual

**Ground truth.** Without it, you can only compare one estimator against another.
- Start and end every run **on the launch mark**: the closing error is the drift over the run.
- Tape-measure a few posts and the checkpoints, to check `config/course_groundtruth.bin` against the real course.
- Static depth checks: a flat wall at measured 0.5, 1, 2 and 3 m, plus a post and a net panel. These give
  `DEPTH_NOISE_K` and show whether stereo sees the net.
- Motion capture, if the venue has it.

**What to record, in this order:**
1. **Handheld walks, props off** (`skybeta.launch.py mode:=handheld`, or `host/run.sh bag`). No flight approval is
   needed, so get many: every corridor at flight height and about 1–1.5 m/s, revisits and loops (for iSAM2's revisit
   ICP links), fast yaw, close passes by the net and posts, dim areas, each starting and ending on the mark.
2. **Static:** about 60 s on the ground (IMU bias and noise), then the wall-distance checks.
3. **Manual flights, team approval per flight:** 30 s hovers (vibration, VIO drift under props), a pilot-flown lap
   in Position mode, and velocity step inputs (stick steps) for MPPI's dynamics model and the rig's response.
4. **Every policy flight you make anyway.**

After each run, check `ros2 bag info` message counts against the expected rates. Dropped stereo frames make a
replay misleading.

**Replay recipe:**
- **cuVSLAM:** `ros2 bag play --clock` into skybeta's container. Run its rectify and cuVSLAM nodes with
  `use_sim_time`. Exclude or remap the recorded `/visual_slam/*`, so the old and new outputs don't collide.
- **iSAM2 and MPPI:** feed them `/fmu/out/*`, TF, and depth re-computed from the recorded stereo. Run once on the
  recorded cuVSLAM and once on a fresh cuVSLAM run, to separate estimator errors from planner errors.

## Open decisions (team)
- Port `ego_planner/` now, or defer it (recommended)?
- Keep `control_mux` for switching between MPPI and the policy? If so, rewrite its output for skybeta twists.
- The yaw policy for the classical stack.
- Which machine is the main rig host? This depends on Phase 0's resource numbers and on whether the x86 machines
  have NVIDIA GPUs.

## Checklist for agents working here
- Don't touch `../skybeta`. Don't arm. Don't publish on `/fmu/in/*` from our nodes.
- New platform behaviour goes in launch arguments and `config/`. Pin any new dependency in `config/versions.yaml`.
- Check PX4 topic names live before relying on a `_v1` suffix.
- Update this file and `docs/SITL.md` when you change what they describe, including the Status sections, and say
  what has actually been run and what has not.
