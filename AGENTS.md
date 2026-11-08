# AGENTS.md: context for coding agents

This repo is the code that flies. Training lives in dronehacks-sim; read its AGENTS.md for the rules, the
task, and how policies are trained. Keep this repo small and boring.

Docs: docs/SITL.md (local PX4 rig), sitl/CLASSICAL_STACK.md (plan + interface for A*/MPPI/iSAM2/cuVSLAM in the rig), docs/FLYX_LOCAL.md (organizers' sim on pear-omen), docs/POLICIES.md
(policy files + observation/action contract), docs/HARDWARE.md (kit, training-plant assumptions, open data).
Update them when you change what they describe.

## Rules
- **Never arm or fly without explicit approval from the team for that specific flight** (SITL included when it
  runs on a shared lab PC). Dry runs (`arm:=false`) are always fine.
- One stack for drone / sitl / flyx: platform differences go in `launch/policynav.launch.py` PROFILES and
  `config/`, never as `if sim:` branches inside policynav/ or onboard/.
- Pin versions in `config/versions.yaml`; a PX4/px4_msgs mismatch silently renames topics (`*_v1`).
- The organizers' simulator is a HELD-OUT test: never tune policies or this stack's parameters to it.
- No credentials, tokens or keys in this repo.
- FlyX (the organizers' sim) is proprietary: local non-distributed use only. Never commit its code or images.

## Conventions (what the policy expects; see policynav/obs.py)
- Depth: OAK depth (16UC1 mm, 0 = invalid) is re-projected into the training camera (128x80, 72x50 deg,
  looking along body x). Invalid -> near; a simulator's +inf (no return in range) -> far.
- State: PX4 vehicle_local_position_v1 velocity (NED) and vehicle_attitude (FRD->NED, w first).
- Action: [fwd, right, down, yaw_rate] in [-1,1] x v_max in the yaw-aligned frame -> TrajectorySetpoint
  velocity (NED) + yawspeed, Offboard velocity mode, 10 Hz decisions, 20 Hz stream.
- A launch must start from a heading the policy was trained for (dronehacks-sim banks v3: any heading).

## Gotchas
- gz-physics segfaults on OBJ collision meshes without vertex normals (sitl/export_world.py writes them).
- snap Docker: use `--runtime nvidia -e NVIDIA_VISIBLE_DEVICES=all`, not `--gpus all`; Gazebo ogre2 needs the
  glvnd EGL vendor file (sitl/rig_entry.sh writes it).
- PX4 1.17 ignores the yaw in PX4_GZ_MODEL_POSE; sitl/rig_entry.sh spawns the model itself.
- Their onboard/ services read /etc/flyx/onboard.yaml in their sim; the drone needs its own deployment file.
- FlyX locally: without `net.core.rmem_max/wmem_max` >= 32 MB on the host, FastDDS drops frames and the sim
  falls to ~0.1x real time after ~2 min (looks like a cuVSLAM/estimator failure).
- `sitl/battery.py` is a copy of dronehacks-sim's; keep the two in sync.
