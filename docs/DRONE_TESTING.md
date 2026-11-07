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
DARGS="--rm --runtime nvidia --network host --ipc host -e ROS_DOMAIN_ID=17 -e RMW_IMPLEMENTATION=rmw_fastrtps_cpp \
  -v $HOME/skybeta/host/dds/fastdds_localhost.xml:/etc/skybeta/fastdds.xml:ro \
  -e FASTRTPS_DEFAULT_PROFILES_FILE=/etc/skybeta/fastdds.xml -e FASTDDS_DEFAULT_PROFILES_FILE=/etc/skybeta/fastdds.xml \
  -v $HOME/team_ws:/root/team_ws"
SRC="source /opt/ros/humble/setup.bash; source /root/skybeta_ws/install/setup.bash; source /root/team_ws/install/setup.bash"
```

`ROS_DOMAIN_ID` must be this drone's (see `/etc/skybeta/drone.env`; 17 on skybeta-17).

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
- Climbs to `height`.
- Flies `forward_m` along the heading the drone had when Offboard started.
- Holds that pose for `hold_s`.
- Then hovers until the pilot lands. There is no land command.

Defaults: `height` 1.0 m, `forward_m` 1.0 m, `climb_speed` / `forward_speed` 0.5 m/s, `hold_s` 5 s.
`fly:=false` (the default) prints the plan from live odometry and publishes nothing.

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
ros2 launch policynav skybeta_climb_forward.launch.py fly:=true height:=1.0 forward_m:=1.0
#    hover only: forward_m:=0.0 (check the dry-run plan line first)
```

Pilot procedure:
- Arm in **Altitude**, throttle low.
- Switch to **Position** within a few seconds. The drone climbs at once, and PX4 disarms it if it is not
  airborne 10 s after arming.
- Take it back by raising the throttle toward centre or flipping the mode switch.
- Kill switch: channel 5.
- The pilot lands it.
- `forward_m` points along the takeoff heading, so face the drone at open space.

Record a bag of every flight in a second terminal: `cd ~/skybeta && host/run.sh bag`.

Afterwards, restore the normal competitor window: `docker rm -f skybeta-stack`. Keep the drone still while it
restarts.

## 6. Policy flight (stereo depth + policy via /skybeta/setpoint/twist)

```bash
# by hand (competitor window stopped as in section 5):
ros2 launch policynav skybeta.launch.py stereo_backend:=vpi policy:=policies/<name>.npz
```

To have it start with the stack, set these in `/etc/skybeta/drone.env` and restart the stack:
`COMPETITOR_PKG=policynav` and `COMPETITOR_LAUNCH=skybeta.launch.py`.

Same pilot procedure as section 5. Handheld bench (section 3) first.

## 7. Moving code off the drone

The drone has no GitHub credentials. Export a branch for someone to push from another machine:

```bash
git -C ~/team_ws/src/deploy bundle create ~/team_ws/scratch/<branch>.bundle origin/main..<branch>
git -C ~/team_ws/src/deploy format-patch origin/main..<branch> -o ~/team_ws/scratch/<branch>-patches
```
