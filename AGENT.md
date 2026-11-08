# Repository guide for agents

Read [AGENTS.md](AGENTS.md) first: it contains the operating rules. This file explains the code and
where to investigate changes; it does not authorize flight or override those rules.

This is **deployment software**, not the training environment. The learned controller is trained in
[dronesim](https://github.com/low-crane-dronehacks/dronesim), historically called `dronehacks-sim`.
Read that checkout's `AGENTS.md` and training implementation before changing the observation/action
contract. A training checkout is not guaranteed to be beside this repo.

## Start by identifying the branch and execution path

There are two controller families. Do not assume a directory missing on main does not exist elsewhere.

| Family | Branch / entry point | Purpose |
| --- | --- | --- |
| Learned policy on Skybeta | `main`, `launch/skybeta.launch.py` | Stereo depth + PX4 feedback -> recurrent policy -> organizer flight interface |
| Portable learned policy | `main`, `launch/policynav.launch.py` | Hardware profiles for drone, SITL, FlyX, and handheld inspection |
| Classical planning + MPPI + optional iSAM2 | `isam2`, `launch/maze_autonomy.launch.py` | Global path, local obstacle costs, sampled predictive control, output multiplexer |
| Earlier classical implementation | `isam2`, `launch/autonomy.launch.py` / `launch/ego_planner.launch.py` | Separate `ego_planner/` implementation with B-splines, primitives, RRT* and MPPI |

The branch sometimes called “isam” in conversation is named **`isam2`** in Git. This guide was checked
against main `122e31e` and `origin/isam2` `a593664` on 2026-11-08. Refresh refs and inspect actual source
before relying on defaults or old status sections. Branch-only paths below refer to the `isam2` tree.

```bash
git status --short
git branch -a
git log -1 --oneline
git show origin/isam2:docs/AUTONOMY_ARCHITECTURE.md
git show origin/isam2:launch/maze_autonomy.launch.py
```

Use `git show` to inspect another branch without switching the user's checkout. Use a separate worktree
when implementing branch-specific changes. Pull/fetch before documenting current remote behavior when
requested; do not merge the classical stack into main merely to inspect it.

## Main: source map

| Path | Responsibility |
| --- | --- |
| `policynav/obs.py` | Pure NumPy camera reprojection, depth filling/stacking, proprioception and action transforms |
| `policynav/policy_np.py` | Load exported `.npz` weights/metadata and execute CNN + GRU inference without PyTorch |
| `policynav/skybeta_node.py` | Real-drone policy lifecycle, input callbacks, startup diagnostics, ENU twist output |
| `policynav/px4_feedback.py` | ROS-independent PX4 input validation, timestamp/reset checks and filtered-range cache |
| `policynav/node.py`, `main_policy.py` | Portable policy ROS adapter and executable entry point |
| `policynav/mission.py` | Portable policy's pure mission state machine; distinct from Skybeta's lifecycle |
| `policynav/handheld_node.py`, `handheld.py`, `main_handheld.py` | Carried-camera inference, observation logging and frame helpers; no flight commands |
| `policynav/depth_node.py`, `stereo.py` | Stereo rectification and CPU OpenCV SGBM depth |
| `src/vpi_stereo_node.cpp` | C++ NVIDIA VPI CUDA stereo depth backend |
| `policynav/transport.py`, `config/fastdds_team.xml` | ROS QoS, TF subscriptions and DDS transport setup |
| `src/policy/`, `include/policynav/` | Standalone C++ policy/observation core and decision gate; not the Python Skybeta ROS adapter |
| `policies/` | Exported weights and metadata; inspect the selected export rather than guessing its contract |
| `launch/`, `config/` | Bringup and platform configuration; keep hardware differences here |
| `onboard/` | VIO wrapper, PX4 external-vision bridge, adapters, service ownership and standalone flight examples |
| `sitl/` | PX4/Gazebo rig, course export, depth/EV bridges and battery model |
| `tests/`, `tools/` | Pure Python tests, C++ parity tests and latency diagnostics |
| `scripts/`, `CMakeLists.txt`, `package.xml` | Installed ROS executables, build dependencies and installed launch/assets |

The ROS package is named **`policynav`**, even though the repository is named `deploy`.

## Learned-policy data flow and contracts

```text
stereo images + CameraInfo + TF
  -> rectification / VPI or SGBM -> metric depth
  -> training-camera reprojection -> depth stack -----------+
PX4 velocity + attitude + accepted downward distance ------+-> CNN + GRU
previous clipped action ----------------------------------+     |
                                                               v
                                clipped action -> yaw-frame velocity + yaw rate
```

- Training camera: 128 x 80, 72 x 50 degree FOV, looking along body +x. Optical axes, body axes and
  world axes are different; use CameraInfo and TF rather than assuming identity extrinsics.
- `decode_depth` handles image encoding. Invalid depth becomes near, not free space; simulator
  positive infinity represents a far/no-return observation. Preserve this distinction.
- Edge filling is export-controlled (`edge_fill_cols`, v5 uses 16), not a universal preprocessing rule.
  Python and C++ must agree on which invalid edge pixels can inherit inward neighbors.
- The depth stack is newest first and primed from its first image. The current Skybeta code pushes the
  latest depth at policy ticks; do not move stacking to camera callbacks without checking training.
- PX4 attitude is quaternion **w,x,y,z**, FRD body to the local NED-style world. Velocity is rotated into
  body FRD for the observation; roll/pitch come from the same quaternion. Local vision yaw need not
  be aligned to geographic north.
- Velocity observations divide by **`v_obs_scale`**, not action limits `v_max`. The v5 export uses
  `[5, 3, 2]` versus action limits `[1.2, 0.6, 0.6]`. Current loader fallback for missing observation
  scale is `[5, 3, 2]` with a warning; malformed scales are rejected. Some old comments still say v_max.
- Proprioception contains body velocity, downward range divided by 1.5, roll/pitch and previous action.
  Skybeta clamps range to `[0.04, rf_max]`; the selected export defines `rf_max`.
- Clip network output to `[-1, 1]` **before storing previous action**. Reset recurrent memory and previous
  action at mission start. Some exports have a fifth completion output; inspect `num_actions`.
- Actions are forward/right/down/yaw-rate in a yaw-aligned frame, with vertical gravity-aligned.
  Scale by metadata, rotate into PX4's local frame, then convert only at the transport boundary.
- Skybeta expects **ENU** twists. `handheld.ned_to_enu_twist` also negates yaw rate. Do not convert linear
  velocity while leaving the NED yaw-rate sign unchanged.

See [docs/POLICIES.md](docs/POLICIES.md) for exports and [docs/HARDWARE.md](docs/HARDWARE.md) for plant assumptions.

## Skybeta: actual drone execution

`launch/skybeta.launch.py` starts the organizer `flight_interface`, our stereo backend and
`skybeta_policy`. Its current default export is `policies/c5_s164_1575.npz` (v5), with VPI stereo.
Policy decisions/twists run at 10 Hz; the launch's interface stream rate is separately configured.
Do not blindly apply the portable stack's 20 Hz stream convention to this path.

The policy consumes PX4 local position, attitude and estimator flags, plus `/skybeta/state` and our
depth. It publishes `/skybeta/setpoint/twist`, not `/fmu/in/*`, and calls `/skybeta/hold` when required.
The pilot arms; the organizer interface handles its configured Position-to-Offboard switch behavior.
The policy itself controls takeoff once inputs pass and the vehicle is armed in Offboard.

`skybeta_node.py` has `wait`, `fly`, `hold`, and `yielded` phases. This is **not** `mission.py`'s
arm/fly/land state machine. Losing interface state, disarming or leaving Offboard yields permanently.
A HOLD request is not exclusive ownership: another mission can overwrite setpoints.

Input checks in `px4_feedback.py`:

- Velocity and attitude must be valid and fresh (0.5 s default). Duplicated samples do not renew age;
  resets/time discontinuities during flight latch faults.
- Heading passes with `heading_good_for_control`, **or** fresh estimator flags showing tilt alignment
  and active vision-yaw fusion without yaw fault/rejection. PX4 1.16 can deliberately leave north yaw
  alignment false while fusing healthy local-frame vision yaw.
- Estimator flags have a separate **2.5 s** timeout because unchanged flags publish at roughly 1 Hz.
- Range remains PX4's **filtered `dist_bottom`**. Accept finite positive distance with valid vertical
  velocity when terrain is valid, or when fresh flags confirm healthy range-height fusion. The latter
  supports the range-as-height setup even with `dist_bottom_valid=false`.
- One accepted range is required before starting. Later invalid range retains the last accepted reading,
  even through a long dropout. Do not silently replace this behavior with altitude, zero or a new 4 m value.
- Depth older than 0.5 s commands zero velocity while flying; loss beyond 2 s holds. A transient startup
  TF/flag warning that resolves differs from a persistent readiness failure.
- Startup blockers produce loud errors. Bright-green `POLICY INPUTS READY` means input checks pass,
  not flight authorization or proof of good obstacle perception. Status includes input ages, range source
  and cached-range status. `policy_ms: null` while waiting means inference has not run.

The organizer still runs cuVSLAM and may fuse it in EKF2; using PX4 feedback does not eliminate vision
from the estimator. Skybeta handheld mode uses a different state path and cannot validate flight readiness.

Observed deployment: host `skybeta-17`, ROS domain **25**, PX4 system ID **25**, PX4 1.16 and ROS Humble.
The portable `config/versions.yaml` currently targets PX4 1.17/Jazzy. This is a documented mismatch to
resolve explicitly, not permission to flash the aircraft. Verify live topic names/types: Skybeta uses
unsuffixed local position but `vehicle_status_v1`. Never infer suffixes from the major version alone.

Keep exactly one mission and flight interface. The organizer example flies a square automatically;
it previously commanded a climb and 5 m leg while our policy held. The example was stopped and its
host autostart set to `COMPETITOR_PKG=disabled` (a missing package leaves a shell). Recheck live state,
since host configuration can change. Do not edit the organizer `~/skybeta` source tree during team work.

Use [docs/DRONE_TESTING.md](docs/DRONE_TESTING.md) for build, manual launch and recorder commands, and
[docs/SKYBETA.md](docs/SKYBETA.md) for topic/transport contracts. Rebuild and source the team overlay:
editing this checkout does not update a running node or the installed drone package.

## Portable deployment, VIO and simulation

`launch/policynav.launch.py` selects `PROFILES` for `drone`, `sitl`, `flyx`, or `handheld`.
`main_policy.py` runs `policynav.node`; `mission.py` gates direct PX4 commands, arming authorization,
Offboard priming, flight, landing and completion. Keep platform differences in launch/config rather
than adding simulation branches to controller logic.

`onboard/services.py` manages declared processes in `onboard/config/services.json`, with ownership,
PID/log evidence under `.onboard/`. The cuVSLAM wrapper in `onboard/slam/` produces visual odometry;
`onboard/vio/ev_bridge.py` validates frame, covariance, reset and freshness before sending external
vision to PX4. `onboard/adapters/` translates PX4 state for other consumers. Read `onboard/README.md`
before changing this path: clock, calibration and heading-reference contracts matter.

`onboard/examples/` contains independent flight programs; some can arm and write PX4 inputs directly.
Do not treat “example” as synonymous with dry-run. The Skybeta climb/forward example uses the organizer
interface and measured PX4 velocity; its checks are separate from policy readiness.

`docs/SITL.md` describes the local PX4/Gazebo rig; `docs/FLYX_LOCAL.md` describes the organizer simulator.
Truth-based EV or synthetic sensors in a test are not evidence of hardware VIO performance. FlyX is
held out: do not tune policies or deployment parameters against it or distribute its proprietary assets.

## Classical branch: iSAM2, maps, global planning and MPPI

**iSAM2 is an estimator, not the controller. MPPI is the local controller.** The `isam2` branch contains
both, plus global planning and a command multiplexer. Its `src/` has a different purpose from main's
`src/policy/`; do not overwrite one tree with the other.

Current modular path in `launch/maze_autonomy.launch.py`:

```text
PX4 -> px4_local_odom_ned -> /odometry/local ------------------------+
prior/course map -> A* (default) or RRT* -> /plan/global_path         |
                          -> path transformer -> local reference --+-> MPPI
live obstacle cloud -> voxel costmap -------------------------------+     |
depth + PX4 -> optional iSAM2 -> corrected obstacle distance grid ---+     v
                                                /control/trajectory_setpoint
                                                             -> control mux -> PX4
```

| Branch-only paths | Role |
| --- | --- |
| `src/depth_isam2.cpp`, `depth_icp.cpp`, `include/isam2_depth/` | GTSAM incremental keyframe graph over depth ICP and PX4 motion, attitude, velocity and optional landmarks |
| `src/distance_grid.cpp`, `voxel_costmap.cpp`, `groundtruth_map.cpp` | Local distance field, live voxel obstacle representation and prior/course geometry |
| `src/astar_planner.cpp`, `src/rrt_star_planner.cpp`, `src/include/maze_planning/course_progress.hpp` | Global search and checkpoint progression |
| `src/path_transformer.cpp` | Transform/timestamp the map path into the local controller reference |
| `src/mppi_controller.cpp`, `quad_plant.cpp` | Sample and score predictive rollouts against the reference, dynamics and obstacles |
| `src/control_mux.cpp`, `src/ros/control_mux_node.cpp` | Select command pathway, enforce output/watchdog decisions and gate PX4 publications |
| `src/ros/` | Separate planner, transformer, MPPI, iSAM2, loopback and mux ROS adapters |
| `pkg_maze_planning/`, `pkg_maze_control/`, `src/CMakeLists.txt` | Modular ROS packages and standalone C++ build/test targets |
| `policynav/maze_planning_py.py`, `src/c_api.cpp`, `scripts/eval_rrt_mppi_sim.py` | Python/native interface and simulator evaluation harness |
| `ego_planner/` | Earlier separate architecture with its own MPPI, dynamics, interfaces and tests |

Frame and integration boundaries:

- The modular stack uses `map_ned`, `odom_ned`, `base_link_frd`, and `isam2_world_ned`. These are not
  interchangeable with ROS ENU `map`/`odom`/FLU `base_link`.
- `map_ned -> odom_ned` must have exactly one owner. Default static launch-mark alignment is a fixture
  assumption; external alignment requires the corresponding launch setting. Missing-TF identity fallback
  warnings are not proof of a valid map alignment. PX4 resets also require explicit handling.
- `/plan/global_path` is a latched map-frame path; pose stamps carry timing. The transformer produces
  `/plan/local_reference` and RL lookaheads. Replanning must preserve checkpoint progress.
- The current MPPI ROS node takes vehicle state from `/odometry/local`. `isam2:=true` starts the graph
  and feeds its obstacle cloud to MPPI's reconstructed local distance grid. This does **not** automatically
  replace the ROS node's PX4-derived vehicle state with corrected iSAM2 odometry.
- `src/include/isam2_depth/mppi_sink_adapter.hpp` provides an in-process state + distance-grid connection
  for a composed integration. Its existence does not mean every launch uses that connection.
- The graph consumes PX4's estimate as an input. **Never feed its corrected output back to PX4's EV
  input:** that creates an estimator feedback loop. PX4 retains its independent cuVSLAM/EV input path.
- The mux's `rl_policy` pathway expects rates setpoints. Main's learned policy emits velocity/yaw-rate
  commands through different adapters. The selector name alone does not make the two compatible.
- This launch's mux publishes native `/fmu/in/*`; it is not automatically a Skybeta-interface controller.
  Do not run it beside Skybeta's flight interface or assume it inherits the organizer safety envelope.

Current maze launch defaults include `planner:=astar`, `isam2:=false`, `mode:=classical`, and `arm:=false`.
Read its current arguments rather than copying historical examples. `arm:=false` gates mux PX4 output;
inspect the full launch and any separate simulated-pilot process before assuming nothing else can arm.

On this branch, read `docs/AUTONOMY_ARCHITECTURE.md`, `docs/ISAM2.md`, `docs/CUVSLAM_LOCAL.md`, and
`docs/SITL.md`. Native SITL additions include `sitl/run_native.sh`, simulated interface/pilot programs,
textured worlds and persistent ULogs. Isolate spoof/loopback DDS domains from the aircraft. iSAM2's
synthetic/spoof results do not establish Jetson timing or flight qualification. Known graph limitations
include persistent occupied voxels without free-space carving and no implemented prior-map registration
factor; do not assume the graph supplies absolute course alignment.

## Build, verification and evidence

For main's ROS-independent regression tests:

```bash
/usr/bin/python3 -m pytest -q
```

Use a Python environment with the declared dependencies; an unrelated activated environment may lack
pytest/NumPy. Main's `CMakeLists.txt` builds Python adapters and the C++ core; VPI targets require NVIDIA
VPI and are skipped when it is absent. A desktop Python pass does not verify VPI or ROS delivery.
For C++ observation/inference changes, use `tests/policy_core/` parity tests and read `run_golden.sh`'s
built-workspace and recorded-depth prerequisites. Keep Python/C++ preprocessing and metadata behavior aligned.

On the **isam2 checkout**, standalone core verification is:

```bash
cmake -S src -B /tmp/deploy-maze-build -DBUILD_TESTING=ON
cmake --build /tmp/deploy-maze-build -j2
ctest --test-dir /tmp/deploy-maze-build --output-on-failure
```

Eigen is required; GTSAM 4.2 enables the graph and its tests, so a passing build without GTSAM does not
validate iSAM2. ROS adapters need the corresponding ROS and px4_msgs dependencies. Read branch-specific
package/build docs before selecting colcon packages. No tests or commands in this guide authorize a flight.

Recordings and flight evidence:

- `docs/DRONE_TESTING.md` contains the copy-paste recorder command, including OAK mono images,
  calibration/IMU, PX4, policy, VIO and static-TF QoS, with 2 GB database splitting.
- Bags persist at `~/team_ws/logs/bags/` on the drone host, mounted as `/root/team_ws/logs/bags/` in the
  container. Stop the recorder cleanly and copy the entire folder, all `.db3` segments and `metadata.yaml`.
- `.onboard/` and other ignored evidence in one checkout are not automatically present in another worktree.
  Large recordings are evidence artifacts, not source files to commit.
- Missing metadata can be reconstructed with `ros2 bag reindex`; that does not repair a corrupt database.
  Preserve originals and explicitly disclose any segment omitted from a recovered index.
- Read-only telemetry and offline tests are different from live actuation. Always report what was actually
  tested, which inputs were synthetic, and what remains unverified.

Before changing behavior, identify the exact launch, executable, installed overlay, topic schema, frame
contract and command owner. Prefer source and current evidence over stale README status claims or an
architecture plan describing intended work. Update the relevant docs when those contracts change.
