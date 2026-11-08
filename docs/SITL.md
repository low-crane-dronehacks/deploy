# Local PX4 SITL rig (`sitl/`)

Evaluation only: tests the policy node on real PX4 software. Never a training target.

**What runs** (one Docker container, `sitl/run_rig.sh`):
- Gazebo Harmonic, headless with EGL rendering, with our course as the world (`sitl/world/`). The world is
  exported from dronesim's course mesh by `sitl/export_world.py`.
- PX4 SITL v1.17 `gz_x500_depth` (x500 frame, forward depth camera).
- Micro-XRCE-DDS agent and ROS 2 Jazzy bridges for `/clock`, depth and camera_info.
- Truth-based external vision (`sitl/ev_bridge.py`): the Gazebo pose plus noise goes to EKF2 in place of VIO.

**Commands** (a desktop with Docker and the NVIDIA runtime):
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
  `sitl/battery_4s5000.json`, copied from dronesim) gives the loaded voltage at hover. The x500 motor
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
