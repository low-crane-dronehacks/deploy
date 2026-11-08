# Local PX4 SITL rig (`sitl/`)

Evaluation only: tests the policy node on real PX4 software. Never a training target.

**What runs** (one Docker container, `sitl/run_rig.sh`):
- Gazebo Harmonic, headless with EGL rendering, with our course as the world (`sitl/world/`). The world is
  exported from dronehacks-sim's mesh by `sitl/export_world.py`.
- PX4 SITL v1.17 `gz_x500_depth` (x500 frame, forward depth camera).
- Micro-XRCE-DDS agent and ROS 2 Jazzy bridges for `/clock`, depth and camera_info.
- Truth-based external vision (`sitl/ev_bridge.py`): the Gazebo pose plus noise goes to EKF2 in place of VIO.

**Commands** (a lab PC with Docker and the NVIDIA runtime; built and verified on pear-deepak):
```bash
sitl/run_rig.sh build            # once; JOBS=n caps compile parallelism
sitl/run_rig.sh up               # ~90 s to READY; logs in the container under /tmp/rig/*.log
sitl/run_rig.sh policy           # DRY RUN: prints what the policy would command, never arms
sitl/run_rig.sh policy -p arm:=true      # flies; needs the team's OK for this specific flight
sitl/run_rig.sh logs | down
```
**Options:**
- `SPAWN_POSE=x,y,z,roll,pitch,yaw_enu` moves the start. The default is the launch mark (`sitl/world/spawn.txt`).
- `BATTERY_SOC=0..1` sets the pack's charge at takeoff. The 4S 5000 mAh LiPo model (`sitl/battery.py`,
  `sitl/battery_4s5000.json`, copied from dronehacks-sim) gives the loaded voltage at hover. The x500 motor
  constants are scaled by the thrust that voltage leaves, about (V / 16 V)^2.
  - The scaling is static per flight, so a 3-minute run's further sag (~5%) is not modelled.
  - PX4 is told about the pack (`BAT1_*`, `MC_BAT_SCALE_EN 1`), but SITL's own simulated voltage does not
    follow our model.

**Limits** (it is a stand-in, not the competition):
- The net is an invisible collision shell with nothing behind it. The real net shows its 3.8 cm frame (5 posts
  per side, rails at the floor, middle and top) and the room through it.
- Depth comes from Gazebo's depth camera, not stereo, and state comes from truth plus noise, not cuVSLAM.
- x500 dynamics, not our airframe. Replace them once the motor/prop data is in.

**Gotchas:**
- gz-physics segfaults on OBJ meshes without normals; `add_normals.py` adds them.
- With snap Docker, use `--runtime nvidia`, not `--gpus all`.
- ogre2 needs the glvnd EGL vendor file, which `rig_entry.sh` writes.
- PX4 1.17 ignores the yaw in `PX4_GZ_MODEL_POSE`, so `rig_entry.sh` spawns the model itself.

## Native (no Docker): `sitl/run_native.sh`

The same stack on a desktop with ROS 2 Jazzy + Gazebo Harmonic (`ros-jazzy-ros-gz`), with the Gazebo window.
Verified on automata-5 (RTX 5080, driver 580). Prerequisites, each built once:
- PX4 v1.17.0 SITL in `~/PX4-v1.17` (`make px4_sitl` with the Gazebo vendor packages installed builds `gz_bridge`);
  PX4's Python tools need its `requirements.txt` (a shell with `PYTHONNOUSERSITE` set: `env -u PYTHONNOUSERSITE make ...`)
- Micro-XRCE-DDS agent v3.0.1 built in `~/Micro-XRCE-DDS-Agent/build`
- px4_msgs release/1.17 in `~/isam2_ws` ([ISAM2.md](ISAM2.md))
- for `EV_SOURCE=cuvslam`: `~/cuvslam-venv` ([CUVSLAM_LOCAL.md](CUVSLAM_LOCAL.md))

```bash
export ROS_DOMAIN_ID=87                     # your own; PX4's UXRCE_DDS_DOM_ID is set to it
sitl/run_native.sh up                       # Gazebo window, x500_stereo at the launch mark, PX4 on Gazebo's truth
EV_SOURCE=cuvslam sitl/run_native.sh up     # PX4 on cuVSLAM instead (the drone's chain)
sitl/run_native.sh status                   # processes + PX4 xy/z/heading validity
sitl/run_native.sh logs | down
```
Options: `GUI=0` (headless), `MODEL=x500_depth`, `SPAWN_POSE`, `BATTERY_SOC`, `EV_HEADING=none`, `RUN_DIR` (default
`/tmp/dronehacks-native`, logs there), and the paths (`PX4_DIR`, `PX4_MSGS_WS`, `XRCE_AGENT`, `CUVSLAM_VENV`).
Starting never arms; an armed flight needs the team's approval (AGENTS.md).

**Differences from the rig:** PX4 runs from a private copy of its `etc/` (the indoor params are appended there, so the
PX4 checkout stays untouched) and a fresh rootfs each start; every process runs in its own session and `down` stops
exactly the recorded ones.

**Model `sitl/models/x500_stereo`:** PX4's `x500_depth` (depth camera kept for the policy / iSAM2) plus an OAK-D S2-like
rectified mono pair (640x400 mono8, 74.7 deg, fx 419, 7.5 cm, 30 Hz, both eyes with identical stamps) and a 200 Hz IMU
with `vio.yaml`'s noise, bridged to `/stereo/{left,right}/image`, `.../camera_info`, `/stereo/imu`, with static TF
`base_link -> oak_left / oak_right / oak_imu`.

**Position source (`EV_SOURCE`)**, exactly one publisher of `/fmu/in/vehicle_visual_odometry`:

| | feeds PX4 | PX4 local origin |
|---|---|---|
| `truth` (default) | `sitl/ev_bridge.py`: Gazebo pose + noise | Gazebo world origin (launch mark = (0.91, -2.51)) |
| `cuvslam` | `cuvslam_lib_node` (venv) -> `onboard/vio/ev_bridge.py`, spawn yaw as its heading reference | where cuVSLAM started (as on the drone) |

With `cuvslam`, Gazebo's truth is still published on `/sitl/truth_odometry` (VehicleOdometry, NED world frame) for
scoring, not to PX4. The onboard bridge publishes only after its 25 s sole-publisher discovery window.

Checked (2026-11-08, no arming): `truth` -> PX4 xy/z/heading valid at the launch mark, "Ready for takeoff"; stereo
30 Hz with identical stamps, IMU 200 Hz reading +9.78 m/s^2 up at rest, depth ~21 Hz. `cuvslam` -> cuVSLAM tracking
on the pad (265 features/frame), EKF2 fusing EV position + height + yaw (`cs_ev_pos/hgt/yaw`, `cs_yaw_align`),
heading good for control, "Ready for takeoff". Not checked: anything in flight.

**Open:** the course mesh is untextured (flat grey), so cuVSLAM sees few features once it moves; texturing the world is
the next step before trusting `EV_SOURCE=cuvslam` in flight. Note the origin difference above when comparing positions
between the two sources (e.g. the maze stack's static `map_ned -> odom_ned` launch-mark offset assumes PX4's origin
is at the launch mark, which holds for `cuvslam`, not for `truth`).

### The real drone's start procedure in SITL (`SKYBETA=1`)

On the drone (SKYBETA.md) team code never arms or commands PX4: the pilot arms in Altitude, switches to Position,
skybeta's `flight_interface` requests Offboard once the team node's twists are streaming, the team node takes off,
and the pilot lands. `SKYBETA=1` reproduces that, so skybeta-style nodes fly in SITL unchanged:
```bash
SKYBETA=1 EV_SOURCE=cuvslam sitl/run_native.sh up          # + sitl/flight_interface_sim.py, PX4 takes simulated RC
# team node, e.g. (the policynav package is not built here, so run the file; ROS + ~/isam2_ws sourced):
python3 onboard/examples/climb_forward_hold_skybeta.py --ros-args -p fly:=true -p forward_m:=0.0
python3 sitl/sim_pilot.py --check     # disarmed: RC accepted, Altitude -> Position on the ground (no arming)
python3 sitl/sim_pilot.py --arm       # the pilot: arm in Altitude, Position after 2 s, Ctrl-C = land (ARMS)
```
- `sitl/flight_interface_sim.py`: stand-in for skybeta's flight_interface, written from the documented contract
  (`/skybeta/setpoint/{twist,pose}`, `/skybeta/hold`, `/skybeta/state`; |v| clamped to 1.5 m/s; Offboard requested
  once per arming when armed in Position with a setpoint from the last second). Approximations are marked in the
  file: an expired twist and `/skybeta/hold` hold the current position.
- `sitl/sim_pilot.py`: the safety pilot. Streams sticks as RC input (`/fmu/in/manual_control_input`, source RC;
  `SKYBETA=1` sets `COM_RC_IN_MODE 0`), throttle low to arm, centred after the switch to Position (no stick motion
  afterwards, so no RC override of Offboard). Landing is Land mode (a real pilot lands by hand). No flags = dry run.
- Frames: twists are in cuVSLAM's `odom` frame read as ENU, so `SKYBETA=1` gives the vision bridge heading 0
  (`EV_HEADING=odom`) and PX4's NED world is that frame; the stand-in maps velocity by an axis swap.
- Safety: `run_native.sh up` refuses to start next to another PX4 SITL / agent / Gazebo, and each run has its own
  Gazebo partition (`GZ_PARTITION`, saved in `$RUN_DIR/gz_partition` for `gz` commands from other terminals).
  2026-11-08: a second Gazebo started next to a running one shared its transport for ~2 minutes (same world and
  model names); that is why.

**Options added after the first crash (2026-11-08):**
- `EV_SOURCE=truth_odom`: `sitl/truth_odom.py` publishes Gazebo's truth AS `/visual_slam/tracking/odometry` (cuVSLAM's
  convention: start-pose frame, FLU, body twist, covariance), through the same onboard bridge to PX4. The full
  skybeta chain with perfect odometry: if a flight works here and fails with `cuvslam`, the problem is the vision.
- `SKYBETA=1` runs cuVSLAM stereo-only by default (`VSLAM_IMU=0`), as skybeta does on the drone ("IMU fusion off").
  Inertial mode had no gravity estimate on the pad and initialised it during liftoff.
- `SPAWN=launch` (default) = the launch mark, which has only 0.39 m to the wall behind it. `SPAWN=open`: on the landing
  pad, NED (-1.15, -1.95), facing north, 1.05 m clearance at every height (`SPAWN_POSE` overrides both).

The crash that prompted them (PX4 ULog, t from arming): Offboard at 2.1 s, liftoff ~7.5 s, at 8.0 s cuVSLAM (Inertial,
untextured world, ~100 features) reported ~0.23 m of motion that PX4 and the team node corrected backwards; prop
strike on the wall 0.46 m behind the launch mark at ~9 s (roll/pitch ~77 deg), then cuVSLAM lost (-38, -31, 29).

Checked (no arming): `SKYBETA=1 EV_SOURCE=truth_odom` -> spawn on the pad, PX4 xy/z/heading valid, bridge healthy;
`sim_pilot.py --check` -> Altitude -> Position on the ground ("check passed"); `SKYBETA=1 EV_SOURCE=cuvslam` ->
Multicamera, 283 features/frame on the pad, PX4 valid. Not checked: an armed flight with these changes.
