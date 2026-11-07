# Policies

| file | trained | what it can do |
|---|---|---|
| `policies/cur4_real_s2_700.npz` | realism curriculum: net+frame -> +stereo depth -> all effects at 640x400 stereo (dronehacks-sim-eval `cur4_realfull_s2` @700) | **57% from launch** on the `real` preset (all effects, eval-grade stereo; 128 deterministic launches); only 1-10% in the ideal sim (it expects realistic stereo). Provisional best, 2026-11-07 |
| `policies/comp_v3_s1.npz` | real course, ideal sim, v3 starts (any launch heading) | 92% success from the launch mark in our sim; 1.1 pts with a see-through net (flies out), 0 with stereo depth |
| `policies/comp_s2.npz` | real course, launch only at 134.6 deg | superseded: scores 0 facing north (the organizers' start) |

**Export** from dronehacks-sim (NumPy only, no torch on the drone):
`python -m dronehacks.export_policy <ckpt> --out policies/<name>.npz`.

**Observation contract** (`policynav/obs.py`, which must match training):
- Depth is 2 stacked frames, newest first, of z-depth / 5 m. Each frame is a 128x80 pinhole image, 72x50 deg,
  looking along body x.
- A policy pixel with fewer than 50% valid source pixels is INVALID and reads as near (0.0). An invalid pixel
  must never look like open space: the camera sits inside the OAK's 0.35 m minimum range at takeoff.
  - A simulator's +inf (no return within range) reads as far.
- Proprio inputs:
  - body velocity
  - roll and pitch
  - rangefinder
  - previous action

**Action:** `[fwd, right, down, yaw_rate]` in [-1, 1], scaled by `v_max`, in the yaw-aligned frame. It is sent
as a PX4 Offboard TrajectorySetpoint velocity (NED) plus yawspeed, with decisions at 10 Hz.

**Known gaps** (in training, dronehacks-sim):
- A see-through net with its frame is in training now (`netbars_*` runs). Taking off from the launch mark is
  the current blocker.
- Stereo depth (holes, occlusion, invalid floor) is evaluated in the realism suite but not yet trained on.
- The plant is a generic 1.5 kg quad, not our airframe (see HARDWARE.md).
