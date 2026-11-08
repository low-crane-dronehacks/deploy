# Handoff: getting the SITL rig running (read this first)

Written 2026-11-08 for a teammate's coding agent picking up `sitl/` on a different (more powerful) machine. The rig
was written and built on a Jetson Orin, but has never got past starting Gazebo. Your first job is to get it to READY
on your machine. The second is the classical-stack work in `sitl/CLASSICAL_STACK.md`.

Read in this order:
1. `AGENTS.md`, repo root: the rules. **Never arm without the team's OK for that specific flight, SITL included.**
   Never edit `../skybeta`.
2. This file.
3. `docs/SITL.md`: what the rig is, its commands and options.
4. `sitl/CLASSICAL_STACK.md`: what comes after, including A*, MPPI, iSAM2, cuVSLAM and recording data for replay.

## What has been verified (Jetson Orin, L4T R36.4.7, Docker 29.9.3 with the nvidia runtime)
- `sitl/run_rig.sh build` succeeds:
  - image `dronehacks-rig:humble-px4-v1.16.2`;
  - PX4 SITL v1.16.2 (`sitl/PX4-Autopilot/build/px4_sitl_default/bin/px4`);
  - workspace `sitl/.cache/ws` (`skybeta_flight_interface`, `oakd_vslam_bringup`, `policynav`).
- The image needed `libopencv-dev`: PX4's gz optical-flow plugin fails CMake configure without it. Already fixed in
  `sitl/Dockerfile`.
- `policynav: VPI not found, skipping vpi_stereo` during `build ws` is **expected**. The rig uses
  `stereo_backend:=none`, and depth comes from `sitl/gz_depth.py`.
- `run_rig.sh team` starts: `flight_interface` and `policynav_skybeta` come up. They then wait forever with
  `odom_age_s: null, depth_age_s: null`, because the rig below them never started.

## The open problem: Gazebo never becomes visible, so `up` fails
What `run_rig.sh up` gets to before failing:
- The container starts, `rig_entry.sh` runs, and the battery line is logged.
- `gz sim -r -s --headless-rendering --verbose=1 /tmp/rig/world/dronehacks.sdf` starts and **stays running** (about
  14% CPU after 2 minutes).
- `rig_entry.sh` polls `gz topic -l` for `/world/dronehacks/clock`, which never appears. After 60 tries it logs
  `[rig] ERROR: gz did not start (gz.log)` and the container exits. `gz.log` is empty, which is normal at
  `--verbose=1`.
- PX4, MicroXRCEAgent, `gz_vio.py`, `gz_depth.py` and skybeta's bridge are never started. `run_rig.sh pilot fly`
  then hangs in `wait_heartbeat`, because there is no PX4 to talk to.

What was observed:
- `gz topic -l` and `gz service -l` inside the container return **nothing**, with exit code 0, in about 2 s, even
  while the server process is alive.
- With `-v4` (in its own `GZ_PARTITION`), the server loads every system and logs `Serving world controls on
  [/world/dronehacks/control]`, `Found no publishers on /clock, adding root clock topic`, and the PostUpdate worker
  threads. The Sensors render thread logs `Waiting for init`. So the server gets well into startup.
- In a clean throwaway container, the **stock `empty.sdf`** behaved the same: `gz topic -l` was empty. In that run,
  no `gz` process was left after 15 s and the `-v2` log was empty. That test was run in a hurry and is
  **inconclusive**: it shows the problem isn't our world, but not whether the server crashed or hung.
- gz-sim 8.15.0 (Harmonic). The container runs as the host uid (`-u 1000:1000`), `HOME=/cache/home`, on docker's
  default bridge network (interfaces `lo` and `eth0`).

Two hypotheses, neither ruled out:
1. **gz-transport discovery fails inside the container.** The server runs, but other processes can't find it.
   - Test: start the server, then `GZ_IP=127.0.0.1 gz topic -l`.
   - Also try `GZ_IP=127.0.0.1` exported for the server too, and check whether multicast works on `eth0`.
   - Fix if confirmed: export `GZ_IP=127.0.0.1` in `rig_entry.sh` before anything gz starts. Every gz process and
     PX4's `gz_bridge` must share it.
2. **Headless rendering (ogre2 over EGL) hangs or crashes.** On the Jetson this goes through the nvidia container
   runtime, and the image writes `/usr/share/glvnd/egl_vendor.d/10_nvidia.json`.
   - Test: run the stock `empty.sdf` with the Sensors system removed (no rendering) and see if `gz topic -l` works.
     Run with `-v4` and capture the process's exit status.
   - Check `eglinfo` (or `EGL_LOG_LEVEL=debug`) inside the container.
   - This one may be **Jetson-specific**. An x86 machine with a desktop NVIDIA GPU may not show it at all.

Suggested order: run the stock `empty.sdf` with `-v4` and no Sensors system, then `gz topic -l` with and without
`GZ_IP`. Those two runs separate the hypotheses. Then fix, and run `run_rig.sh up` until it prints
`[rig] READY`.

## What your machine needs
- Docker with the **NVIDIA container toolkit**. `run_rig.sh` adds `--runtime nvidia` when `docker info` lists it.
  With snap Docker, use the runtime, not `--gpus all` (`AGENTS.md`).
- An NVIDIA GPU. Gazebo renders the depth camera with ogre2. Without a GPU it falls back to mesa, which is slow or
  crashes headless.
- The PX4 submodule: `git submodule update --init --recursive` (about 3 GB; pinned to v1.16.2, matching the drone's
  firmware).
- The read-only skybeta checkout next to this repo (`../skybeta`), or `SKYBETA_DIR=...`.
- Your own `sitl/run_rig.sh build`. Images are built per machine and are not shared between arm64 and x86.
  `JOBS=n` sets the compile parallelism; the default is 4, so raise it on a bigger machine.

## After READY
Phase 0 of `sitl/CLASSICAL_STACK.md`:
1. `run_rig.sh up`, then `run_rig.sh team`, then `run_rig.sh pilot fly`. That last one is a **dry run**; `--arm` only
   with the team's OK.
2. The policy log should show finite `odom_age_s` and `depth_age_s`.
3. Record the live PX4 topic names (`run_rig.sh sh`, then `ros2 topic list | grep fmu`), in particular whether they
   carry the `_v1` suffix. Later phases depend on it.
4. Expect more first-run fixes in `rig_entry.sh` (paths, gz topic names, READY detection) and in `pilot.py` (which
   MAVLink streams PX4's onboard link carries).

Then continue with Phase 1 of `sitl/CLASSICAL_STACK.md` (merging the `isam2` branch).

## Platform notes
- **x86 is fine for Phases 0–3**, which use the fake VIO. **Phase 4 (cuVSLAM)** uses skybeta's image, which is
  arm64-only (`FROM nvcr.io/nvidia/isaac/ros:aarch64-...`). On x86 it needs a separate x86 Isaac ROS base;
  see `CLASSICAL_STACK.md`.
- **Keep fixes portable.** The rig has to work on the Jetson and on x86 (`docs/SITL.md`). A fix made on x86 must be
  re-tested on the Jetson before calling the rig fixed. Say which machine each result came from.

## When you change things
- Update the Status sections of `docs/SITL.md` and this file with what you found and fixed: what ran, on which
  machine, and what did not. Don't claim something works if it hasn't run.
- Keep platform differences in launch arguments, `config/` and the rig scripts, never `if sim:` in nodes.
- Commit on `sitl-humble`, or a branch off it, and push, so the Jetson side can pull and re-test.
