# Policies

| file | trained | what it can do |
|---|---|---|
| `policies/cur4_clean50_800.npz` | netrand2_675 fine-tuned with 50% of episodes on CLEAN depth (other effects on), randomized net returns, skip-terminal (dronehacks-sim-eval `cur4_nr2_clean50` @800, early: ~15M of 50M steps) | **Experimental, for FlyX/SITL.** From launch: 72.7% on `real` (stereo), 51.6% on `real` with clean depth (128 each). The first policy that works on both. 2026-11-07 19:30 UTC |
| `policies/cur4_netrand2_675.npz` | s5 @4450 fine-tuned with the stereo net-return rate drawn per episode from U(0, 0.15) (dronehacks-sim-eval `cur4_s5ft_netrand2` @675) | **Recommended.** From launch on `real`: 78.7% at the default 3% net returns (512 launches; collision 4.1%, floor 12.3%), 46.1% at 0%, 73.4% at 10% (128 each). Beats netrand_1300 at all three. **Fails with clean depth** (0-1% in the ideal sim or with every effect but stereo), so expect it to fail in Gazebo SITL, whose depth camera is clean. 2026-11-07 18:10 UTC |
| `policies/cur4_netrand_1300.npz` | s5_1550 fine-tuned with the stereo net-return rate drawn per episode from U(0, 0.15) (dronehacks-sim-eval `cur4_s5ft_netrand` @1300) | Superseded by netrand2_675 (robust to net returns, but lower). From launch on `real`: 68.9% at the default 3% net returns (512 launches), 44.5% at 0%, 68.8% at 10% (128 each). At 0% most extra failures fly through the unseen net. 2026-11-07 |
| `policies/cur4_real_s5_1550.npz` | same realism curriculum, seed 5 (dronehacks-sim-eval `cur4_realfull_s5` @1550) | **73% from launch** on the `real` preset (512 deterministic launches; collision 5.5%, floor 17%). Highest at the default 3% net returns, but brittle: 42% at 0% returns, 18% at 10%. numpy vs torch max diff 7.2e-7 |
| `policies/cur4_real_s2_700.npz` | realism curriculum: net+frame -> +stereo depth -> all effects at 640x400 stereo (dronehacks-sim-eval `cur4_realfull_s2` @700) | **57% from launch** on the `real` preset (all effects, eval-grade stereo; 128 deterministic launches); only 1-10% in the ideal sim (it expects realistic stereo). Superseded by s5_1550 |
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
