# The real drone: organizers' skybeta stack

Drone 17 ("skybeta-17", Jetson Orin Nano, JetPack 6) runs the organizers' skybeta stack v0.3.2 in Docker
(ROS 2 Humble). The source is in `~/skybeta` on the drone; **never edit it**, because organizer updates refuse
a dirty tree. Team code lives in `~/team_ws/src` (`/root/team_ws` in the containers) and is built with
`container_scripts/team_build.sh`. Their guide: `~/skybeta/docs/competitor_guide.md`.

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

## Scripted test flight (no policy)
`onboard/examples/climb_forward_hold_skybeta.py` (launch: `skybeta_climb_forward.launch.py`): climb `height`, fly `forward_m`
along the heading at Offboard start, hold a pose `hold_s`, then hover until the pilot lands. Same plan as
`climb_forward_hold.py` (which commands PX4 directly: SITL / FlyX only), through the skybeta contract. Dry run by
default (`fly:=false`: prints the plan from live odometry, publishes nothing).

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
