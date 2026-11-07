# deploy

Drone software for DroneHacks 2026: one ROS 2 / PX4 stack that runs **unchanged** on the aircraft, on our
PX4 SITL rig, and in the organizers' simulator. Only the hardware layer differs (`hardware:=drone|sitl|flyx`).
Policies are trained in [dronehacks-sim](https://github.com/ColinBalfour/dronehacks-sim) and exported here as
NumPy `.npz` files (no torch needed on the drone).

```
policynav/   policy node: depth + PX4 state -> Offboard velocity setpoints; mission/safety logic (mission.py)
policies/    exported policies (.npz)   python -m dronehacks.export_policy <ckpt> --out policies/<name>.npz
onboard/     VIO: cuVSLAM wrapper, external-vision bridge to PX4, health (from the organizers' workspace)
launch/      policynav.launch.py: hardware:=drone|sitl|flyx, arm:=false|true, policy:=...
config/      versions.yaml (pinned PX4 / px4_msgs / cuVSLAM), oak.yaml (camera), PX4 params
sitl/        PX4 SITL rig: Docker, PX4 1.17 + Gazebo Harmonic + uXRCE, our course world
docker/      Jetson image (todo)
docs/        SITL.md, FLYX_LOCAL.md (organizers' sim on pear-omen), POLICIES.md, HARDWARE.md
```

## Run

```bash
# SITL (a lab PC with Docker + NVIDIA runtime)
sitl/run_rig.sh build && sitl/run_rig.sh up          # PX4 + Gazebo + bridges, ~90 s
sitl/run_rig.sh policy                                # dry run: reports what the policy would command
sitl/run_rig.sh policy -p arm:=true                   # fly (SPAWN_POSE=x,y,z,r,p,yaw_enu moves the start, BATTERY_SOC=0..1)

# organizers' simulator: hosted workspace, or locally on pear-omen (docs/FLYX_LOCAL.md)
ros2 launch launch/policynav.launch.py hardware:=flyx

# aircraft (UNTESTED)
ros2 launch launch/policynav.launch.py hardware:=drone
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
