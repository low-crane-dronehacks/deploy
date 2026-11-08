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

### A* wall clearance (`launch/maze_autonomy.launch.py astar_*`)

The A* path keeps the drone's centre `astar_clearance_m` (default 0.30) from walls, but allows
`astar_clearance_m x astar_clearance_fallback_ratio` (default 0.75, so 0.225 m) within 0.3 m of every goal and gate,
and on a whole leg when a gate is otherwise unreachable. `astar_centre_dist_m` / `astar_centre_weight` (0.60 / 1.5)
pull the path to the middle of corridors narrower than 2x the distance. Swept offline on course_groundtruth.bin
(2026-11-08): **corridor_end needs a minimum (clearance x ratio) <= 0.240 m**; any higher and A* skips it ("unreachable,
skipped: corridor_end") and the path misses that checkpoint. Defaults give min 0.240 m along the path. 0.40 x 0.60
keeps corridor_end with more room elsewhere (15.8 m vs 15.3 m path). Results right at 0.24 m depend on the 5 cm grid,
so check the planner log ("min clearance", "skipped") after any change.

`mppi_safe_distance_m` (default 0.42) is the MPPI's obstacle comfort radius: cost starts there, shrinking per horizon
step to the reference's own clearance (floor lethal 0.21 + 0.02).

**Gate A stall (fixed 2026-11-08):** every SITL maze flight stopped at arena E ~0.13, just short of gate A, dithered,
and drifted into the north net. Cause: MPPI's outer-net cost barrier and its net velocity CBF compared the ODOM
position with arena bounds. Odom's origin is the launch mark on SITL and the drone, so the "net" sat at arena
E = 0.135 (cost) / E = 0.34 (velocity clamp), and at N = -1.94 (would have blocked the window). Both now work in the map
frame. sim_loopback publishes map == odom, so it could not show this; `testMPPIGateAWithOdomOffset` and
`testMPPIWindowWithOdomOffset` run MPPI closed loop with the SITL odom offset. Any map-frame rule inside the stack
must go through T_map_odom.

### Our airframe in Gazebo (`MODEL=dronehacks_stereo`, the default since 2026-11-08)

PX4's x500 reaches ~0.39 m from its centre (rotors 0.246 m out, 0.14 m props); the planner and MPPI plan for our drone
(`collision_radius_m` 0.2; the corridor_end pass has ~0.24 m clearance). The first maze-stack flight on the x500 clipped
the corridor wall's end at the right turn 4.7 s after arming (0.5 m sideways in 0.1 s, roll +40 deg) and tumbled out
of the arena. `sitl/models/make_dronehacks_quad.py` generates our quad from PX4's x500 files (sensors, motor plugins,
meshes) with the training plant's numbers (`dronehacks-sim/dronehacks/quad.py`):

| | dronehacks_quad | x500 |
|---|---|---|
| mass, inertia | 1.5 kg, (0.012, 0.012, 0.020) | 2.0 kg, (0.022, 0.022, 0.040) |
| rotors | X at +-0.13 m (0.184 m from centre) | +-0.174 m (0.246 m) |
| max thrust / rotor | 10 N (2.7:1) | 8.5 N (1.7:1) |
| motor time constant | 0.03 s | 0.0125 up / 0.025 down |
| reach (collision) | body box corners 0.187 m; 5 in props: 0.247 m | 0.39 m |

Kept from the x500 deliberately: moment constant 0.016 (training uses 0.07: with x500 tuning that is ~9x the yaw
response) and the prop size (quad.py has none; 5 in assumed). **The real drone's measured reach decides whether the
corridor_end pass is safe**: 0.247 m (5 in props) is already above the planner's 0.24 m there.

PX4 (appended to the private airframe by `run_native.sh` for `dronehacks_*`): `CA_ROTOR0..3` at (+-0.13, +-0.13) FRD
in the x500's rotor order, KM +-0.05; `MPC_THR_HOVER` 0.50 (10 N rotors, 1.5 kg, ESC linear in speed); rate gains
scaled for ~1.6x roll/pitch and ~2.3x yaw acceleration per command (MC_ROLL/PITCHRATE P 0.095 I 0.127 D 0.0019,
MC_YAWRATE P 0.085 I 0.043). A starting tune: verify with a hover before trusting it.

Camera frames: the OAK sits on the nose (depth optical frame at (0.132, 0, 0.019) in base_link; stereo eyes at
(0.12, +-0.0375, 0.02)). The x500 depth frame was published 0.24 m too high (the rig's (0.132, 0, 0.261) counted the
landing-gear height twice: base_link is already at 0.24 m in the model frame); now (0.132, 0, 0.021).
`sitl/rig_entry.sh` (Docker rig) still has the old value.

Checked (no arming): spawns, PX4 boots with the new mixer / gains / hover thrust (read back with `px4-param show`),
position + heading valid, Gazebo link poses match the design, stereo 29 Hz with identical stamps, IMU 200 Hz, depth
28 Hz, TF as above; maze stack stage 0 ALL PASS (stack vs truth 0.03 m, 0.7 deg). Not checked: any flight.
