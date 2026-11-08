# The real drone: organizers' skybeta stack

Drone 17 ("skybeta-17", Jetson Orin Nano, JetPack 6) runs the organizers' skybeta stack v0.3.2 in Docker
(ROS 2 Humble). The source is in `~/skybeta` on the drone; **never edit it**, because organizer updates refuse
a dirty tree. Team code lives in `~/team_ws/src` (`/root/team_ws` in the containers) and is built with
`container_scripts/team_build.sh`. Their guide: `~/skybeta/docs/competitor_guide.md`.
Command reference for building, bench tests and test flights: [DRONE_TESTING.md](DRONE_TESTING.md).

## Contract (what we may touch)
- **Read:**
  - `/visual_slam/tracking/odometry` (cuVSLAM, ENU/FLU, 30 Hz, pure stereo VO with IMU fusion off);
  - `/skybeta/state` (JSON: `armed`, `offboard_active`, `clamped`, ...);
  - the OAK's raw mono pair `/oak/{left,right}/image_raw` + `camera_info`;
  - `/fmu/out/*`, e.g. `vehicle_local_position.dist_bottom` from the VL53L1X. These are the PX4 1.16 names,
    with no `_v1` suffix.
- **Write:** `/skybeta/setpoint/twist` (TwistStamped, `frame_id: odom`, ENU m/s, expires after 1 s). The
  `/skybeta/hold` service freezes the vehicle. **Never publish `/fmu/in/*`; never arm.**
- **Procedure:**
  1. The pilot arms in Altitude, then switches to Position on the ground.
  2. `flight_interface` then requests Offboard once, but only if our setpoints arrived within the last second.
  3. We take off ourselves (a twist with an upward component). PX4 disarms if the vehicle is not airborne
     10 s after **arming**.
- **Envelope (in `flight_interface`):**
  - the norm of `|v|` is clamped to 1.5 m/s by scaling;
  - the altitude band (0.3–3.0 m) applies to pose setpoints only;
  - there is no horizontal geofence;
  - **yaw rate is not limited.**
- **End of flight:** the pilot lands. There is no land command, and no run-end or scoring rule is
  documented; ask the organizers.

## What we run (`launch/skybeta.launch.py`)
| node | does |
|---|---|
| `flight_interface` | skybeta's own, started exactly as their `competitor_example` does |
| `stereo_depth` (`policynav/depth_node.py`) | rectifies the raw pair from the TF extrinsics, because the driver's camera_info is per-eye raw (R = I, D = 0, no common P). Output is 640x400; fx ~420 at alpha 0, the same as our sim and FlyX, with a 74.7 x 51 deg FOV. With `stereo_backend:=sgbm` it also matches (OpenCV SGBM 3-way: 29 ms median on 6 Orin cores with the stack up, measured). Invalid -> NaN -> read as near |
| `vpi_stereo` (`src/vpi_stereo_node.cpp`, `stereo_backend:=vpi`, the default) | VPI C API, CUDA census + SGM with a confidence threshold on the rectified pair -> `/policynav/depth/image`. Built only if CMake finds VPI (libnvvpi3 + vpi3-dev are in skybeta:0.3.2) |
| `skybeta_policy` (`policynav/skybeta_node.py`) | zero twists at 10 Hz until `offboard_active`, then the policy at 10 Hz. Caps: `|v|` <= 1.45, yaw <= 0.6 rad/s. Holds on stale odometry/depth, at the time limit, or after touchdown |
| `handheld` (with `mode:=handheld`) | camera + policy with odometry state. Publishes no setpoints and does not start the flight interface |

```bash
# on the drone, once:  git clone https://github.com/low-crane-dronehacks/deploy ~/team_ws/src/deploy
#                      then skybeta's container_scripts/team_build.sh
# /etc/skybeta/drone.env:  COMPETITOR_PKG=policynav  COMPETITOR_LAUNCH=skybeta.launch.py
ros2 launch policynav skybeta.launch.py mode:=handheld      # props OFF first
```

## Runtime and transport (`launch/skybeta.launch.py`)
What the launch adds around the nodes, and why. Everything here applies to **our** nodes only (`vpi_stereo`,
`stereo_depth`, `skybeta_policy`, `handheld`). `flight_interface` keeps skybeta's environment, nice 0, no pinning.

| arg | default | effect |
|---|---|---|
| `dds_profile` | `team` | `config/fastdds_team.xml`: skybeta's `host/dds/fastdds_localhost.xml` verbatim + an 8 MiB SHM segment. `stack` = inherit skybeta's profile. The team file is used only if skybeta's profile env is set (local scope) and the file parses, else we inherit (a broken profile would put Fast DDS on every interface). |
| `nice` | `5` | niceness of our processes (0..19; 0 = off) |
| `cpus` | `""` | `taskset` core list for our processes, e.g. `3-5` (`""` = all 6) |

Subscription QoS (`policynav/transport.py`): depth image and `/visual_slam/tracking/odometry` are best effort,
keep_last 1 (`LATEST`), so a stalled Python executor reads the newest sample instead of a queue of stale ones,
and cuVSLAM's odometry writer keeps no reliable state for us. The policy nodes subscribe to `/tf_static` only
(`tf_static_only`, default true): the depth-frame chain is all static, and `TransformListener` would otherwise
take every 30 Hz cuVSLAM transform on `/tf` (reliable, keep_last 100) through the Python executor.

### What the drone looks like (inspected 2026-11-07, stack up, our nodes not running)
- **Cores:** skybeta pins nothing (`Cpus_allowed_list: 0-5` on every stack process). cuVSLAM's ~10 busy threads
  (`vslam_container`, ~4 % each), the depthai `XLinkInHost` readers (~3-8 % each), MicroXRCEAgent and
  `flight_interface` (~21 %, Python) all migrate over cores 0-5; ~1.9 of 6 cores busy, evenly spread. There is
  no "VIO core" to avoid, hence `nice` by default and `cpus` as an option.
- **SHM is the data path.** Every participant has a `/dev/shm/fastrtps_<guid>` segment of 549408 B (the 512 KiB
  default) and `fastrtps_port<N>` queues on the user-unicast ports (domain 17: 11661, 11663, ...). Loopback UDP
  runs at ~750 datagrams/s, far below what 30 Hz images would need (~1900/s per reader); that is discovery and
  the uXRCE agent's PX4 topics (the agent has its own UDP-only participant).
- **Fragmentation:** Fast DDS 2.6 cuts every sample to the smallest transport limit (UDP, ~64 KB), SHM
  included, and keeps a writer's fragments in the writer's segment until readers take them. 1 MB depth
  (32FC1 640x400) does not fit a 512 KiB segment: offline, 1 MB best-effort at 15 Hz lost 1.3-3.8 % of frames with
  skybeta's profile and 0 % with the team profile (3 runs each, p50 ~3 ms either way). The stack's
  `/oak/*/image_mono` (655 KB) is written into the vslam container's 512 KiB segment too; we cannot change that.
- **No zero copy across processes:** rmw_fastrtps on Humble uses data-sharing / loans only for bounded
  (fixed-size) types; `sensor_msgs/Image` is unbounded. Zero copy needs intra-process in one process (below).
- Publication mode is already `SYNCHRONOUS` (the Humble rmw default); `emulate_tty` and `output='screen'` cost
  nothing measurable at our log rates (one timing line per 5 s per node).
- Loopback UDP shows receive-buffer drops on the **metatraffic** ports (11660 + 2*id: up to ~16k since boot,
  `/proc/net/udp`), with `rmem_max` = 208 KB. That is discovery traffic, not data; nothing to do from our side.

### One C++ process (planned)
When `vpi_stereo` and the policy are both C++ components, launch them in one container with intra-process comms;
the depth image then moves as a `unique_ptr` with no serialisation and no DDS hop:
```python
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode
ipc = [{'use_intra_process_comms': True}]
ComposableNodeContainer(
    name='policynav', namespace='', package='rclcpp_components',
    executable='component_container_isolated',          # one SingleThreadedExecutor thread per component
    prefix=prefix, additional_env=ours, output='screen', emulate_tty=True,
    composable_node_descriptions=[
        ComposableNode(package='policynav', plugin='policynav::VpiStereo', name='policynav_vpi_stereo',
                       parameters=[...], extra_arguments=ipc),
        ComposableNode(package='policynav', plugin='policynav::Policy', name='policynav_skybeta',
                       parameters=[...], extra_arguments=ipc),
    ])
```
Requirements: publish with `std::unique_ptr<Image>` (`pub->publish(std::move(msg))`), subscribe with
`ConstSharedPtr` or `UniquePtr`; intra-process needs KEEP_LAST + VOLATILE (best effort keep_last 1 is fine).
Other subscribers (rosbag, `latency_probe`) still get a serialised copy. `component_container_isolated` keeps the
stereo callback (GPU waits) from delaying the policy; prefer it over `component_container_mt`, which lets
callbacks of one node run concurrently. Run the policy on depth arrival, not on a free-running 10 Hz timer (today
~50 ms average wait), with a watchdog timer for the stale-depth hold.

### Bench check (props off, `mode:=handheld`, ~60 s per variant)
```bash
# A: old behaviour of this layer          B: new defaults
ros2 launch policynav skybeta.launch.py mode:=handheld dds_profile:=stack nice:=0     # A
ros2 launch policynav skybeta.launch.py mode:=handheld                                 # B
# (QoS / tf_static_only are code: for A use the commit before ae95c69, or -p tf_static_only:=false for the TF part)

# 1. latency: camera->depth->cmd ages and VIO odometry age (run with skybeta's profile, i.e. a plain shell)
ros2 run policynav latency_probe --ros-args -p seconds:=30 -p label:=A \
  -p image_topics:="['/policynav/depth/image']" -p twist_topics:="['/policynav/handheld/cmd']"
# 2. the nodes' own timing lines: vpi_stereo "pairs/unpaired" + stages, handheld "a camera->depth rx (age)"
# 3. our placement + DDS env actually applied
for p in $(pgrep -f 'policynav/(vpi_stereo|handheld|skybeta_policy)'); do
  echo "$p ni=$(ps -o ni= -p $p) $(grep Cpus_allowed_list /proc/$p/status)"; done
ls -la /dev/shm | grep -E 'fastrtps_[0-9a-f]+$'            # B: two 8413728-byte segments appear (ours)
# 4. no /tf subscriber of ours; data on SHM, not UDP (InDatagrams rate should not jump by ~1000/s)
ros2 topic info -v /tf | grep -B1 -A1 policynav             # B: no output
a=$(awk '/^Udp:/{x=$2}END{print x}' /proc/net/snmp); sleep 10; b=$(awk '/^Udp:/{x=$2}END{print x}' /proc/net/snmp); echo $(((b-a)/10))/s
# 5. CPU: per-thread load of the stack and of us over 10 s
top -H -b -d 10 -n 2 | awk '/^top -/{n++} n==2' | head -40
```
Accept B if VIO odometry age p95 drops back toward ~69 ms and our camera->cmd p50/p95 does not get worse. If our
p95 rises under `nice:=5`, try `nice:=2`, or `cpus:=3-5 nice:=0` (a soft partition: the scheduler then puts
more stack threads on 0-2).

## Scripted test flight (no policy)
`onboard/examples/climb_forward_hold_skybeta.py` (launch: `skybeta_climb_forward.launch.py`) uses **PX4 fused
velocity and heading** from `VehicleLocalPosition`, not raw cuVSLAM feedback. It integrates measured vertical
velocity to climb `height`, then measured body-forward velocity to travel `forward_m`, while correcting
integrated height. Forward follows the current PX4 heading; vertical stays gravity-aligned. The mission
rotates commands into ENU for `/skybeta/setpoint/twist`, never sends pose setpoints, and never writes `/fmu/in/*`.
After travel it sends zero velocity for `hold_s`, then continues zero velocity until the pilot lands; this
is not an absolute-position hold. Distances accumulate velocity-integration error. Invalid/stale feedback,
velocity/heading resets and sample-time discontinuities abort the mission to zero velocity; loss of Offboard
stops publishing permanently. It cannot resume itself.

Dry run is the default (`fly:=false`: wait for valid PX4 feedback, print the plan, publish nothing).
`px4_feedback_topic` defaults to `/fmu/out/vehicle_local_position` as resolved on skybeta-17; override it
for a version-suffixed topic when necessary. `target_system` defaults to **17**, the observed PX4 system ID
on this drone; override on other vehicles. Only the organizer interface sends PX4 commands. Its own
`odometry_topic` and the organizer EKF2 fusion configuration remain unchanged; the mission does not disable
vision fusion in PX4. See [DRONE_TESTING.md](DRONE_TESTING.md) for readiness and build steps.

## Known gaps (2026-11-07)
- **`vpi_stereo` has never been compiled.** There is no VPI or ROS on our dev machines. Expect small fixes at
  the first `team_build.sh`. Until it builds, use `stereo_backend:=sgbm`. VPI has no Python bindings in
  skybeta:0.3.2, hence C++.
- **Lens distortion:** the driver publishes D = 0. If the real lenses have noticeable distortion,
  rectification is off. Check the rows of a rectified pair from the recorded bag
  (`~/team_ws/scratch/bags/oak_sample` on the drone).
- **Policies:** only checkpoints trained with `--v-max 1.2 0.6 0.6 --yaw-rate-max 0.6` (`cur4_c50_slow*`, in
  training) match this envelope. Faster ones are scaled down and fly out of distribution. Train for the 80-100 px
  left band too (`left_band_*`).
- **Repo layout:** `package.xml` + `CMakeLists.txt` (ament_cmake + ament_cmake_python) sit at the repo root. colcon does not look for packages inside a package
  directory, so other ROS packages added to this repo need their own repo, or this one moves to a subfolder.
- Nothing here has run on the Jetson yet. The pure parts are tested (`tests/`).
