# Testing on the drone: command reference

Every command for building, bench-testing and flight-testing our code on the real drone (organizers' skybeta
stack). Background and the topic contract: [SKYBETA.md](SKYBETA.md). Their procedure:
`~/skybeta/docs/competitor_guide.md` on the drone.

**Rules:**
- Never edit `~/skybeta`, PX4 parameters or the stack's containers.
- In `/etc/skybeta/drone.env`, only `COMPETITOR_PKG` / `COMPETITOR_LAUNCH` are ours.
- Nothing of ours publishes `/fmu/in/*`.
- Run new code with the props off first, then low and tethered.
- Never fly without the team's approval for that specific flight.
- **Before flying, confirm with the organizers that the drone's motor configuration is correct.** Some drones,
  17 among them, had it overwritten by a software flash (the "flipping drone" issue, 2026-11-07).

## 0. Build

Builds `~/team_ws` as an overlay on the stack's workspace, in a throwaway container. `--network none` keeps it
off the drone's DDS domain.

```bash
docker run --rm --runtime nvidia --network none -e HOST_UID=$(id -u) -e HOST_GID=$(id -g) \
  -v ~/team_ws:/root/team_ws skybeta:0.3.2 bash -lc team_build.sh
```

Inside a stack shell (`cd ~/skybeta && host/run.sh shell`) it is just `team_build.sh`.

## 1. Pre-flight checks (any shell in the stack)

```bash
cd ~/skybeta && host/run.sh shell              # a shell in the running stack container
ros2 topic echo /skybeta/state --once          # px4_link: true, local_enu not null
ros2 topic hz /visual_slam/tracking/odometry   # ~30 Hz (Python hz under-reads: ~27-30 is normal)
ros2 topic echo /visual_slam/status --once     # vo_state 1 = tracking, 2 = lost
```

- The stack starts at power-up. Keep the drone still for the first minute: the VIO origin and heading are its
  startup pose.
- `docker rm -f skybeta-stack` restarts the stack (back in ~10 s) and re-zeroes that origin.

## 2. Throwaway container on the live domain (bench runs)

The common arguments for running our nodes beside the stack, in their own container:

```bash
DARGS="--rm --runtime nvidia --network host --ipc host -e ROS_DOMAIN_ID=25 -e RMW_IMPLEMENTATION=rmw_fastrtps_cpp \
  -v $HOME/skybeta/host/dds/fastdds_localhost.xml:/etc/skybeta/fastdds.xml:ro \
  -e FASTRTPS_DEFAULT_PROFILES_FILE=/etc/skybeta/fastdds.xml -e FASTDDS_DEFAULT_PROFILES_FILE=/etc/skybeta/fastdds.xml \
  -v $HOME/team_ws:/root/team_ws"
SRC="source /opt/ros/humble/setup.bash; source /root/skybeta_ws/install/setup.bash; source /root/team_ws/install/setup.bash"
```

`ROS_DOMAIN_ID` must be this drone's (see `/etc/skybeta/drone.env`; **25** on skybeta-17).

## 3. Handheld bench (camera + depth + policy; never commands PX4)

Starts no `flight_interface` and publishes only `/policynav/*`. Logs "would command: ..." at 1 Hz and per-stage
timing every 5 s. It saves `~/team_ws/logs/handheld/handheld_<stamp>.npz`.

```bash
docker run $DARGS skybeta:0.3.2 bash -c "$SRC; ros2 launch policynav skybeta.launch.py mode:=handheld stereo_backend:=vpi"
# OpenCV fallback (CPU-heavy: saturates all cores, slows VIO; offline comparison only):
#   ... mode:=handheld stereo_backend:=sgbm
```

Stop it with Ctrl-C (`docker run -it` for an interactive terminal).

## 4. Depth node alone + latency

```bash
# vpi_stereo alone: rectify + match timing per stage every 5 s
docker run $DARGS skybeta:0.3.2 bash -c "$SRC; ros2 run policynav vpi_stereo"

# latency probe (read-only): rate by header stamp, age on arrival, MB/s per topic, one table after 20 s
docker run $DARGS skybeta:0.3.2 bash -c "$SRC; ros2 run policynav latency_probe --ros-args -p seconds:=20.0"
```

Other measurement tools:
- GPU/CPU load: run `tegrastats --interval 1000` on the host.
- Per-process CPU: run `top -bn2 -d3` inside the container.

## 5. Scripted test flight: climb, forward, hold (no policy)

`onboard/examples/climb_forward_hold_skybeta.py`:
- Uses PX4 fused NED velocity and heading from `vehicle_local_position`, not raw VIO odometry.
- Integrates measured vertical velocity to climb `height` metres above the Offboard-start level.
- Integrates measured forward velocity to travel `forward_m` metres along the **current** heading.
  Forward/right are yaw-relative; vertical remains aligned with gravity. Turning changes the forward direction.
- During forward travel, corrects integrated height using vertical velocity commands.
- Streams zero velocity for `hold_s`, then continues zero velocity until the pilot lands.
  This requests stopping, not holding an absolute position. There is no land command.

All mission commands go through `/skybeta/setpoint/twist` in ENU after rotation using PX4 heading.
No pose setpoints or direct PX4 commands are sent by the mission. Travel is estimated by integrating
PX4 velocity; bias accumulates, so the distance is approximate. Stage timeouts switch to zero velocity.
Invalid/stale velocity or heading, estimator resets, or sample-time discontinuities abort to zero velocity;
Offboard loss stops publishing permanently. The mission never restarts itself.

The organizer interface still owns its safety envelope and its `odometry_topic` configuration; this change
neither changes EKF2 fusion sources nor disables the organizer's visual-odometry pipeline.
The altitude envelope applies to pose setpoints, not our velocity commands.

Defaults: `height` 1.0 m, `forward_m` 1.0 m, `climb_speed` / `forward_speed` 0.5 m/s, `hold_s` 5 s.
`fly:=false` (the default) prints the plan once valid PX4 feedback arrives and publishes nothing.
Feedback must have `v_xy_valid`, `v_z_valid`, and `heading_good_for_control` true; this is not a replacement
for the organizer's pre-flight checks. Its default topic is `/fmu/out/vehicle_local_position` as observed on
skybeta-17. If the installed message generation uses a suffix, set
`px4_feedback_topic:=/fmu/out/vehicle_local_position_v1` after checking the live topic type.
Both Skybeta launches default `target_system` to **25**, matching skybeta-17's flight controller
(`MAV_SYS_ID` / `vehicle_status.system_id`); override it on other vehicles. `ROS_DOMAIN_ID` is also **25**,
but is a separate setting. The hostname/drone number remains **17**. `component_id: 1` is another separate field.
Recheck after a firmware/configuration change with
`ros2 topic echo /fmu/out/vehicle_status --once --qos-reliability best_effort` on PX4 1.16
(use the version-suffixed topic only if that is what the live stack publishes).
A wrong ID lets the interface read state while its mode commands address the wrong aircraft.

The stack's competitor window launches with no arguments, which would only ever give the dry run. Run it by hand:

```bash
# 1. stop the competitor window: only ONE flight_interface may run
docker exec -it skybeta-stack tmux attach -t skybeta
#    select the "competitor" window, Ctrl-C, then detach with Ctrl-b d

# 2. a shell in the stack; check section 1
cd ~/skybeta && host/run.sh shell

# 3. dry run (props off): prints "DRY RUN would ..."
ros2 launch policynav skybeta_climb_forward.launch.py

# 4. fly (props on, safety pilot on RC, kill switch confirmed, team approval for this flight)
ros2 launch policynav skybeta_climb_forward.launch.py fly:=true height:=1.0 forward_m:=1.0 target_system:=25
#    hover only: add forward_m:=0.0 to the same launch (same PX4 feedback and velocity-only code).
#    Check it without commanding the drone first:
#    ros2 launch policynav skybeta_climb_forward.launch.py forward_m:=0.0
```

Pilot procedure:
- Arm in **Altitude**, throttle low.
- Switch to **Position** within a few seconds. The drone climbs at once, and PX4 disarms it if it is not
  airborne 10 s after arming.
- Take it back by raising the throttle toward centre or flipping the mode switch.
- Kill switch: channel 5.
- The pilot lands it.
- `forward_m` follows the current heading, so face the drone at open space and avoid turning during the test.

Record every flight in a second terminal before arming; see [Flight recording](#flight-recording) below.
The organizer wrapper is also available: `cd ~/skybeta && host/run.sh bag`.

Afterwards, restore the normal competitor window: `docker rm -f skybeta-stack`. Keep the drone still while it
restarts.

## 6. Policy flight (stereo depth + policy via /skybeta/setpoint/twist)

```bash
# by hand (competitor window stopped as in section 5):
ros2 launch policynav skybeta.launch.py stereo_backend:=vpi policy:=policies/<name>.npz target_system:=25
```

To have it start with the stack, set these in `/etc/skybeta/drone.env` and restart the stack:
`COMPETITOR_PKG=policynav` and `COMPETITOR_LAUNCH=skybeta.launch.py`.

Same pilot procedure as section 5. Handheld bench (section 3) first.
Handheld still uses cuVSLAM state; it does not verify the flying policy's PX4 inputs. Before a
separately approved policy flight, confirm PX4 `vehicle_local_position` velocity/heading validity
and inspect `dist_bottom`, plus fresh `vehicle_attitude`. Invalid range retains the last valid reading;
before the first valid reading it uses 4 m. Check `range_age_s` to distinguish held range from live readings.
The flying policy requires both
streams within `feedback_lost_s` (default 0.5 s), with no altitude fallback. Confirm `/skybeta/state`
publishes JSON booleans `armed` and `offboard_active` more frequently than `state_lost_s` (0.5 s).
`/policynav/status` reports `feedback_error`, `lpos_age_s`, `attitude_age_s`, and `state_age_s`.

For a props-off stereo check, compare the normal handheld launch with `swap_eyes:=true` only if physical
images are mislabelled relative to correctly labelled calibration. This swaps images only; camera-info and TF stay fixed.
Verify a known-distance object on each side of the view; an invalid band on the right alone is not proof
of swapped eyes. Nonpositive/nonfinite baselines now prevent depth output instead of merely logging a
warning. The policy default is `c5_s164_1575.npz`, with 16-column edge filling enabled in its metadata.
V5 is included and can be inspected without commands using
`ros2 launch policynav skybeta.launch.py mode:=handheld policy:=policies/c5_s164_1575.npz`.
Older policies retain their no-fill preprocessing. These bench checks do not validate the reported simulation success rates.

Check the selected policy's exported `v_obs_scale`: this policy family uses `[5, 3, 2]` despite action
limits of `[1.2, 0.6, 0.6]`. Missing scale metadata uses `[5, 3, 2]` with a Python warning; malformed scales
are rejected. Python and C++ now both use the observation scale. Offline verification covers synthetic
Python/C++ parity, not ROS topic delivery or actual VPI output. The stack advances at 10 Hz inference;
the [training review](https://github.com/low-crane-dronehacks/deploy/pull/1#issuecomment-6719561536)
reports that training also takes the latest delayed frame at each policy tick. Camera-rate stacking
is not an established requirement; do not change it without checking the training implementation.

### Flight recording

Start this in a **second terminal on the drone**, before the approved flight. Keep it running through
landing and disarm. It records PX4 feedback, commanded motion, policy status/depth, VIO and transforms.
It does not arm or send flight commands.

```bash
# On the drone host: enter a separate shell in the running stack.
cd ~/skybeta && host/run.sh shell

# Inside that shell:
source /opt/ros/humble/setup.bash
source /root/skybeta_ws/install/setup.bash
source /root/team_ws/install/setup.bash
export ROS_DOMAIN_ID=25
mkdir -p /root/team_ws/logs/bags
FLIGHT_BAG="/root/team_ws/logs/bags/flight_$(date -u +%Y%m%dT%H%M%SZ)"
ros2 bag record -o "$FLIGHT_BAG" \
  /fmu/out/vehicle_local_position \
  /fmu/out/vehicle_attitude \
  /fmu/out/vehicle_status \
  /fmu/out/vehicle_land_detected \
  /skybeta/state /skybeta/setpoint/twist \
  /policynav/status /policynav/depth/image /policynav/depth/camera_info \
  /visual_slam/tracking/odometry /visual_slam/status \
  /tf /tf_static
```

Wait for the recorder to subscribe to the expected live topics before arming. The names above target
PX4 1.16; check `ros2 topic list` if a topic is missing. The scripted forward flight does not itself
start policy/depth publishers, so those topics may be absent for that test.

After landing, press **Ctrl-C once** in the recording terminal and wait for the recorder to exit so it
finishes writing the bag. Then, in the same shell:

```bash
ros2 bag info "$FLIGHT_BAG"
```

Check the listed topics and message counts. The bag directory contains `metadata.yaml` and the storage
file(s). It persists on the drone host at `~/team_ws/logs/bags/flight_<UTC timestamp>/` through the
workspace mount; stop recording before restarting/removing the stack container.

For a stereo-debugging **bench** recording, append `/oak/left/image_mono /oak/right/image_mono
/oak/left/camera_info /oak/right/camera_info` to the record command. Raw stereo adds substantial data;
check recorder load and available storage before including it in a flight recording.

## 7. Watching it live: Foxglove on your laptop

`foxglove_bridge` is already in the skybeta image. Run it on the drone in a throwaway container. It joins the
drone's local DDS domain, so `drone.env` does not change and the stack does not restart.

```bash
docker run --rm --network host --ipc host -e ROS_DOMAIN_ID=25 -e RMW_IMPLEMENTATION=rmw_fastrtps_cpp \
  -v ~/skybeta/host/dds/fastdds_localhost.xml:/etc/skybeta/fastdds.xml:ro \
  -e FASTRTPS_DEFAULT_PROFILES_FILE=/etc/skybeta/fastdds.xml -e FASTDDS_DEFAULT_PROFILES_FILE=/etc/skybeta/fastdds.xml \
  skybeta:0.3.2 bash -c "source /opt/ros/humble/setup.bash; source /root/skybeta_ws/install/setup.bash; \
    ros2 launch foxglove_bridge foxglove_bridge_launch.xml port:=8766 capabilities:='[connectionGraph]'"
```

- **Source `/root/skybeta_ws/install` too.** That is where `px4_msgs` lives; without it the bridge reports
  `px4_msgs` not found and cannot decode the `/fmu/*` topics.
- **Port 8766, not Foxglove's default 8765.** 8765 is taken on the Tailscale address by the organizers' status
  agent (`/opt/skybeta-status`), which we must not disturb.
- **`capabilities:='[connectionGraph]'` makes it view-only.** The laptop cannot publish topics, set parameters or
  call services. With the defaults, a stray click could send a setpoint.

On the laptop:
1. Open the Foxglove app (or app.foxglove.dev).
2. Choose Open connection, then Foxglove WebSocket.
3. Connect to one of:
   - `ws://192.168.55.1:8766` (USB cable)
   - `ws://100.110.105.30:8766` (Tailscale, drone 17)
   - the drone's Wi-Fi IP

Useful panels:

| Panel | Topic | Shows |
|---|---|---|
| Log | `/rosout` | Every node's log, including the dry run's `DRY RUN would ...` plan line |
| Plot | `/visual_slam/tracking/odometry` `.pose.pose.position.{x,y,z}` | VIO tracking as you carry the drone |
| Raw Messages | `/skybeta/state` | `px4_link`, `armed`, `offboard_active`, `clamped` |
| Image | `/oak/left/image_mono`; `/policynav/depth/image` (when our pipeline runs) | Camera / stereo depth |
| 3D | TF, odometry | Pose and frames |

- Images are 650 KB to 2 MB per frame. Open only the panels you need, and close the image panels before flying.
- Offline alternative with no load on the drone: record a bag (`cd ~/skybeta && host/run.sh bag`), copy the `.mcap`
  to the laptop and open it in Foxglove.
- rviz2 on the laptop is possible, but needs ROS 2 Humble there plus `SKYBETA_DDS_SCOPE=network` and a stack
  restart. The restart re-zeroes VIO, and the camera streams then leave the drone. Use Foxglove instead.

## 8. Moving code off the drone

`origin` is `git@github.com:low-crane-dronehacks/deploy.git`. SSH goes through the laptop: `~/.ssh/config` sets a
`ProxyCommand` to `127.0.0.1:8888`. With the laptop connected, plain `git push` / `git pull` work. Opening PRs needs
the GitHub web UI (or a token). Without the laptop, export a branch to push from elsewhere:

```bash
git -C ~/team_ws/src/deploy bundle create ~/team_ws/scratch/<branch>.bundle origin/main..<branch>
git -C ~/team_ws/src/deploy format-patch origin/main..<branch> -o ~/team_ws/scratch/<branch>-patches
```
