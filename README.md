# deploy

Drone software for DroneHacks 2026: one ROS 2 / PX4 stack that runs **unchanged** on the aircraft, on our
PX4 SITL rig, and in the organizers' simulator. Only the hardware layer differs (`hardware:=drone|sitl|flyx`).
Policies are trained in [dronesim](https://github.com/low-crane-dronehacks/dronesim) (approach and results:
[docs/APPROACH.md](https://github.com/low-crane-dronehacks/dronesim/blob/main/docs/APPROACH.md)) and exported
here as NumPy `.npz` files (no torch needed on the drone). The policy sees only stereo depth and onboard state
(velocity, attitude, downward range); it gets no position or map input.

**Default policies on this branch:** `launch/skybeta.launch.py` (the real drone) uses
`policies/ftsc_s334_1627.npz`; `launch/policynav.launch.py` uses `policies/cur4_clean50_800.npz`. Any export can
be selected with `policy:=policies/<name>.npz`; [docs/POLICIES.md](docs/POLICIES.md) lists them all and the
observation/action contract. The ftw0 policy supersedes the earlier holes-v5h3-s196 candidate.

```
policynav/   policy node: depth + PX4 state -> Offboard velocity setpoints; mission/safety logic (mission.py)
policies/    exported policies (.npz)   python -m dronehacks.export_policy <ckpt> --out policies/<name>.npz
onboard/     VIO: cuVSLAM wrapper, external-vision bridge to PX4, health (from the organizers' workspace)
launch/      policynav.launch.py: hardware:=drone|sitl|flyx, arm:=false|true, policy:=...
config/      versions.yaml (pinned PX4 / px4_msgs / cuVSLAM), oak.yaml (camera), PX4 params
sitl/        PX4 SITL rig: Docker, PX4 1.17 + Gazebo Harmonic + uXRCE, our course world
docs/        SKYBETA.md (the real drone), DRONE_TESTING.md (every test/flight command on the drone), HANDHELD.md, SITL.md, FLYX_LOCAL.md (organizers' sim on a local workstation), POLICIES.md, HARDWARE.md
```

## Run

```bash
# SITL (a desktop with Docker + NVIDIA runtime)
sitl/run_rig.sh build && sitl/run_rig.sh up          # PX4 + Gazebo + bridges, ~90 s
sitl/run_rig.sh policy                                # dry run: reports what the policy would command
sitl/run_rig.sh policy -p arm:=true                   # fly (SPAWN_POSE=x,y,z,r,p,yaw_enu moves the start, BATTERY_SOC=0..1)

# organizers' simulator: hosted workspace, or locally (docs/FLYX_LOCAL.md)
ros2 launch launch/policynav.launch.py hardware:=flyx

# the real drone (organizers' skybeta stack): flight interface + our stereo depth + the policy (docs/SKYBETA.md)
ros2 launch policynav skybeta.launch.py mode:=handheld     # props off first; mode:=fly to fly

# handheld dry run / data collection: camera + policy, carried, never commands PX4 (docs/HANDHELD.md)
ros2 launch launch/policynav.launch.py hardware:=handheld record:=true

# aircraft (UNTESTED)
ros2 launch launch/policynav.launch.py hardware:=drone
```

**Nothing arms unless `arm:=true`.** Stop conditions (stale depth, estimator, geofence, time, pilot takeover)
are in `policynav/mission.py`. `python3 -m pytest -q tests` runs the pure tests.

## Status

- [x] policy node, dry-run verified in the organizers' sim and in SITL (actions match training)
- [x] SITL rig (Docker), truth-based external vision, 4S battery sag (docs/SITL.md)
- [x] organizers' sim installed locally; needs the host UDP-buffer sysctl to stay real-time (docs/FLYX_LOCAL.md)
- [ ] skybeta mode (docs/SKYBETA.md): stereo depth + policy via /skybeta/setpoint/twist; written + pure tests, not yet run on the Jetson
- [ ] handheld mode (`hardware:=handheld`): written + pure tests; first run on the Jetson/laptop with the OAK
- [ ] SITL flight; SITL with stereo + cuVSLAM instead of truth
- [ ] Jetson image (ARM ROS 2 Jazzy, depthai_ros_driver, cuVSLAM ARM build, uXRCE serial)
- [ ] aircraft: Pixhawk 6C on PX4 v1.17, OAK calibration + TF, VL53L1X, EKF2 external-vision params, RC kill switch
