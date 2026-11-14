# iSAM2 depth graph (src/)

C++ estimator that feeds an MPPI controller. It is a keyframe factor graph over depth ICP and PX4 odometry,
solved incrementally with iSAM2 (Kaess et al., ICRA 2011, `Kaess11icra.pdf`) through GTSAM 4.2. The MPPI
itself is not here.

**Planning only.** PX4 keeps flying on cuVSLAM through `onboard/vio/ev_bridge.py`. This graph uses PX4's
estimate as odometry, so its output must never be fed back into PX4 (that would be a feedback loop). It goes to
the planner and MPPI.

The estimator produces two things, both in PX4's local frame (NED world, FRD body), the same frames `policynav/obs.py` uses:
- **State:** the rollout start pose, velocity and covariance. It is the newest keyframe's estimate, carried
  forward by PX4's motion since that keyframe, and is updated at the PX4 rate.
- **Obstacle distance field:** a dense local Euclidean distance grid (O(1) lookups for rollout costs), rebuilt at
  every keyframe from the smoothed keyframe poses.

## Graph

| factor | between | from |
|---|---|---|
| prior | X0, V0 | PX4 at the first keyframe (anchors the graph in PX4's frame) |
| odometry `BetweenFactor` | X(k-1) to X(k) | PX4 relative motion; sigma grows with distance and rotation |
| depth ICP `BetweenFactor` | X(j) to X(k) | projective point-to-plane ICP of the keyframes' depth clouds, with the ICP information matrix and a Huber loss. j = k-1, plus up to 2 older keyframes within 1.5 m / 30 deg (revisits) |
| `Pose3AttitudeFactor` | X(k) | PX4 roll / pitch (gravity-referenced; neither ICP nor odometry observes them absolutely) |
| `BodyVelocityFactor` | X(k), V(k) | PX4 velocity in the body frame, so PX4's yaw drift does not leak in |
| `KinematicFactor` | X(k-1), V(k-1), X(k), V(k) | p_k - p_(k-1) = dt (v_(k-1) + v_k) / 2 |
| `PriorFactor<Pose3>` (CAD ESDF) | X(k) | projective point-to-CAD-plane registration of keyframe depth against `assets/cad_prior_esdf.bin` (`CADP` v1 5 cm exact closest-point + triangle-normal ESDF built from `assets/main_direction/dronehacks-course-v3-obj.zip` via `scripts/build_cad_prior.py`), normal-gated (`cos >= 0.90`), distance-gated (`<= 0.14 m`), Cauchy M-estimated (`k = 0.04 m`), and Huber-wrapped in GTSAM so shifted as-built walls in the real course are rejected while faithful CAD geometry anchors global drift |
| `BodyPointFactor` | X(k), L(id) | a 3D landmark measured in depth, e.g. the landing-pad ring centre (`addFrame(..., obs)`; no detector here) |

**Keyframes:** a new keyframe is taken after 0.25 m, 10 deg or 0.5 s of PX4 motion.

**iSAM2 settings** are the paper's: Gauss-Newton with wildfire threshold alpha = 0.001 (partial state updates,
Alg. 4) and relinearization threshold beta = 0.1 (fluid relinearization, Alg. 5). GTSAM orders affected variables
with constrained COLAMD, newest last (Sec. IV-A).

**Unobservable directions.** The arena is net walls and vertical posts. With the floor out of view (it enters
the 50 deg vertical FOV only beyond ~3 m at 1.4 m altitude), depth cannot observe vertical motion. ICP handles
this in two ways:
- It solves with a weak prior toward the odometry guess, which is excluded from the reported information, so it
  does not wander along that direction.
- Its information matrix says the direction is unobserved (about 50x lower than horizontal), so iSAM2 takes
  that axis from PX4.

**Rejected ICP results.** A result is dropped if it disagrees with the prediction by more than 0.3 m or 15 deg,
or if it lost more than 20% of its matches. The second check catches trimmed ICP "fitting" by gating the floor
out.

## Code

```
src/include/isam2_depth/   types, depth_icp (cloud + ICP), cad_prior (CADP v1 ESDF), prior_esdf_map (PriorEsdfMap 3D log-odds grid), factors, distance_grid, depth_isam2, mppi_interface
src/*.cpp                  core library: Eigen + GTSAM only, no ROS (includes VioIcpTracker baseline)
src/ros/isam2_depth_node.cpp   ROS 2 node (built only when px4_msgs is found)
src/test/                  synthetic test + scan-depth spoof benchmark (spoof_px4_depth.py, spoof_isam2.launch.py)
scripts/build_cad_prior.py builds assets/cad_prior_esdf.bin (CADP v1) + assets/arena_prior.bin from dronehacks-course-v3.obj
```

**`PriorEsdfMap` (`src/include/isam2_depth/prior_esdf_map.hpp`, `src/prior_esdf_map.cpp`):**
When `cad_prior_path` (`assets/cad_prior_esdf.bin`) is provided, `DepthIsam2` seeds a 3D log-odds occupancy grid (`0.08 m` voxels, floor `-z < 0.16 m` cropped) from the Course v3 CAD ESDF at `t = 0`. As smoothed keyframes arrive (replaying from prior if loop-closure smoothing shifts an older pose `> 4 cm` / `> 2 deg`), 3D Amanatides-Woo DDA raycasting carves free space (`-l_free`) where CAD obstacles moved or are absent in the real world, and fuses occupied hits (`+l_occ`) on real surfaces, feeding the exact 3D EDT (`DistanceGrid`) for MPPI/planners.

**Latched `PointCloud2` topics on `isam2_depth_node`:**
- `~/priors`: active CAD prior obstacle voxel centres (`prior_occ && log_odds > 0`), published immediately at startup (`t = 0`) and shrinking where depth rays clear empty space.
- `~/new_updated_obstacles`: newly discovered / shifted obstacle voxel centres (`!prior_occ && log_odds > 0.2`).
- `~/cleared_empty_space`: prebaked CAD prior voxels cleared by pass-through depth rays (`prior_occ && log_odds < -0.2`).
- `~/vio_icp_odometry`: benchmark VIO + sequential point-to-plane ICP baseline without factor-graph smoothing.

**MPPI hook, across processes (what `launch/maze_autonomy.launch.py isam2:=true` does):** `mppi_controller_node`
subscribes to `~/obstacles` (occupied voxel centres, frame `isam2_world_ned`), rebuilds the same `DistanceGrid`
around the vehicle and samples it alongside the voxel costmap. The launch file publishes the static TF
`odom_ned -> isam2_world_ned` (identity: the graph is anchored at PX4's estimate, so its world is PX4's local
frame). The grid is dropped if no cloud arrives for 1 s.

**MPPI hook, same process:** `isam2_depth::MppiControllerSink` (`include/isam2_depth/mppi_sink_adapter.hpp`)
implements `MppiStateSink::onEstimate(const MppiInput&)` for `maze_planning::MPPIController`; pass it to
`node->setSink(...)`. The MPPI core itself does not depend on GTSAM. Each `MppiInput` carries:
- the pose, velocity and covariance;
- a `shared_ptr<const DistanceGrid>`, which is immutable, so the MPPI thread can keep sampling the previous one;
- landmark estimates.

`DistanceGrid::distance(p)` saturates at 3 m. Outside its 10 x 10 x 4 m window it returns the saturated
value, so the MPPI must bound rollouts itself. The net may not be seen by stereo: keep the geofence.

## Build and run

```bash
sudo apt install ros-jazzy-gtsam                       # 4.2.0, pinned in config/versions.yaml
# workspace with px4_msgs (release/1.17) next to this package, e.g. ~/isam2_ws/src/{px4_msgs, isam2_depth -> src}
colcon build --packages-up-to isam2_depth --cmake-args -DCMAKE_BUILD_TYPE=Release   # builds px4_msgs first
ls install/isam2_depth/lib/isam2_depth/isam2_depth_node   # without px4_msgs the build still succeeds, minus the node
# Dev machines only, if needed: CMake not finding GTSAM -> -DGTSAM_DIR=<prefix>/lib/cmake/GTSAM (apt:
# /opt/ros/jazzy/lib/cmake/GTSAM); an active conda env breaking ament (catkin_pkg) -> conda deactivate, or
# -DPython3_EXECUTABLE=/usr/bin/python3
ros2 run isam2_depth isam2_depth_node --ros-args -p depth_topic:=/camera/depth/image_raw \
    -p depth_info_topic:=/camera/depth/camera_info     # drone: /oak/stereo/image_raw (launch PROFILES)
# core + test only, no ROS:
cmake -S src -B build && cmake --build build && ./build/test_depth_isam2
```

**Topics:**
- `~/odometry` (nav_msgs/Odometry at the PX4 rate; covariance reordered to ROS's [trans, rot])
- `~/vio_icp_odometry` (nav_msgs/Odometry at the PX4 rate): benchmark-only VIO + standard frame-to-frame point-to-point ICP (`VioIcpTracker`) without factor graph optimization, revisit loop closures, kinematic/attitude/body-velocity factors, or CAD prior
- `~/obstacles` (PointCloud2 of occupied voxels, per keyframe): the local window the MPPI sees
- `~/priors` (PointCloud2, latched): active CAD prior obstacle voxel centres (`prior_occ && log_odds > 0`), published immediately at startup (`t = 0`) and shrinking where depth rays clear empty space
- `~/new_updated_obstacles` (PointCloud2, latched): newly discovered / shifted obstacle voxel centres (`!prior_occ && log_odds > 0.2`)
- `~/cleared_empty_space` (PointCloud2, latched): prebaked CAD prior voxels cleared by pass-through depth rays (`prior_occ && log_odds < -0.2`)
- `~/keyframes` (nav_msgs/Path, latched): every keyframe pose at its current smoothed value, per keyframe. This is
  the paper's output, the whole trajectory re-estimated; past poses move when ICP or revisit links correct them.
- `~/map` (PointCloud2, latched): the whole corrected map. Every keyframe's 5 cm cloud is placed at its smoothed
  pose; cells with >= `grid_min_hits` points are kept. Rebuilt every `map_publish_period` s (default 1).
- `~/adapted_map` (PointCloud2 `xyz` + `rgb`, latched): global smoothed map classified against `CadPriorEsdf` (teal = agrees with CAD prior within `0.14 m`, warm amber = newly discovered / shifted as-built surfaces beyond the CAD prior).
- `~/prior_updates` (visualization_msgs/MarkerArray, latched): live factor-graph belief & prior adaptation per keyframe (`smoothed_keyframes`, `smoothing_shifts` initial-to-posterior correction vectors, `cad_prior_anchors` + `cad_anchor_points`, and `revisit_closures`).
- `~/landmarks` (visualization_msgs/MarkerArray, latched): estimated landmarks, sphere + id. Empty until something
  feeds landmark observations (e.g. a pad-ring detector); the graph supports them, the node does not produce any.

**Saving the map:** `-p map_save_path:=/path/name` writes it when the node shuts down (Ctrl-C), and
`ros2 service call /isam2_depth/save_map std_srvs/srv/Trigger` writes it any time (to `map_save_path`, or
`./isam2_map_<time>` if unset). `name.ply` holds the `~/map` voxels in ENU, z up (the phone-scan convention), and
`name_keyframes.csv` the smoothed keyframe poses (NED). Spoof test: `map_save:=/tmp/isam2_map`. Compare with the
course prior: `view_course_scan.py name.ply --mirror no --yaw 0 --tx 0 --ty 0` when the map is already in the course
frame (spoof test); otherwise `--mirror no` and the yaw/offset found for the PX4 frame. The map is too sparse for the
automatic search alone (it picked a mirrored fit on a spoof map); the map is right-handed, never pass `--mirror yes`.
Spoof map vs the prior: median 0.25 m, mostly the as-built differences the scan shows too (walls 0.4-0.7 m off).

All are published only while something subscribes. RViz: fixed frame `isam2_world_ned` (z points down), or a
`map` (ENU) -> `isam2_world_ned` static transform as the spoof test publishes; `src/test/spoof_isam2.rviz` has
displays for all of them.

What a keyframe is: a depth frame kept as graph variables X(k), V(k) once PX4 reports 0.25 m, 10 deg or 0.5 s of
motion since the last one. It keeps its full cloud for ICP (newest 150) and a 5 cm cloud for the maps (always).

The node sends nothing to PX4. It times PX4 samples by their arrival on the node clock (sim time in simulation)
and interpolates them to the depth header stamp.

## Spoof test: fake PX4 + depth rendered from a course scan

No PX4, Gazebo or vehicle. `src/test/spoof_px4_depth.py` takes a scan of the course (.ply, aligned with
`ego_planner/scripts/view_course_scan.py`), plans a 3D A* route through the physical scan's free space (`--route_mode scan`, zero CAD ESDF input used by the spoofer; `--route_mode cad_prior` also available for benchmarking), renders
OAK-like 16UC1 depth from it (z-buffered splats, noise 0.003 z^2, invalid edges, dropouts) and publishes a drifting
fake PX4 estimate on `/fmu/out/vehicle_local_position_v1` + `vehicle_attitude`, with `/clock`. It scores the node's
`~/odometry` against the truth and its `~/obstacles` against the scan. It publishes telemetry only, never a command.

```bash
source ~/isam2_ws/install/setup.bash                      # conda deactivated
export ROS_DOMAIN_ID=87                                   # your own, so the fake PX4 topics stay local
ros2 launch src/test/spoof_isam2.launch.py ply:=$PWD/assets/dronehacks-course-scan-2026-11-06.ply      # + RViz
ros2 launch src/test/spoof_isam2.launch.py ply:=... rviz:=false spoof_args:='--no_drift --csv /tmp/run.csv'
/usr/bin/python3 src/test/spoof_px4_depth.py scan.ply --plan_png /tmp/route.png    # route + depth frames, no ROS
```

`stereo:=true` also renders a stereo pair + IMU from the scan and runs cuVSLAM on it ([CUVSLAM_LOCAL.md](CUVSLAM_LOCAL.md)).

RViz (fixed frame `map`, ENU above the NED world): scan (RGB), truth path green, spoofed PX4 path red, iSAM2 path
blue, obstacle voxels orange, depth image. Drift knobs: `--yaw_drift_deg_s`, `--yaw_rw_deg`, `--scale`, `--pos_rw`.

Results (2026-11-13, CAD prior derived strictly from `assets/main_direction/dronehacks-course-v3-obj.zip` -> `assets/cad_prior_esdf.bin`, live depth sampled from `assets/dronehacks-course-scan-2026-11-06.ply` where ~47% of walls differ 0.2–0.6 m from CAD, 16.6 m joint-collision-free 3D route in 40 s):

| estimator (`0.5 deg/s` yaw bias, `4%` scale, `1 cm/sqrt(s)` walk) | position rms / final | yaw rms / final |
|---|---|---|
| Spoofed PX4 VIO (`/fmu/out/*`) | `0.377 m / 0.753 m` | `9.81 deg / 18.30 deg` |
| VIO + Standard ICP (`~/vio_icp_odometry`, no factor graph) | `0.621 m / 1.301 m` | `11.20 deg / 16.78 deg` |
| **iSAM2 Factor Graph + CAD ESDF Prior (`~/odometry`)** | **`0.143 m / 0.295 m`** (`0.130 m` smoothed keyframes) | **`3.11 deg / 5.16 deg`** (`2.83 deg` smoothed keyframes) |

Obstacle voxels (local window, ~16.8k voxels): median `0.054 m` from the scan surface (`73%` within 10 cm). Full corrected map (`~/map`, ~78k voxels after the route): median `0.038 m`, `82%` within 10 cm. On this 110-keyframe route, `CadPriorEsdf` accepted `32` alignments on faithful CAD geometry, gated out `64` views dominated by shifted as-built walls, and added `43` revisit loop closures (`13–30 ms` total keyframe update).

## Status (2026-11-07)

**Synthetic test** (`test_depth_isam2`, x86, 75 keyframes over a 40 s circle, stereo-like noise of 0.004 z^2):

| | final error |
|---|---|
| PX4 odometry drifting 1.1 deg/s in yaw with 4% scale error | 0.90 m / 46 deg |
| iSAM2 estimate, same run | 0.01 m / 0.9 deg |
| iSAM2 estimate with perfect odometry | 0.007 m (depth does not make it worse) |

Worst keyframe update: about 31 ms, of which ICP is about 18 ms, iSAM2 about 1.4 ms, and covariance plus grid about 7 ms.

**Not done yet:**
- Not run on SITL, FlyX or the Jetson (only the spoof test above, on x86). Expect several times slower on the Orin Nano; `cloud_stride`,
  `extra_links` and `IcpParams::point_step` are the knobs.
- When `cad_prior_path` is unset (`""`), `DistanceGrid` falls back to raw local-window hit counting (`grid_min_hits`, without log-odds raycast carving). When `cad_prior_path` is set, `PriorEsdfMap` performs full 3D Amanatides-Woo DDA free-space carving (`-l_free`) and hit fusion (`+l_occ`).
- Not in `launch/policynav.launch.py`; `launch/maze_autonomy.launch.py isam2:=true` runs it with the MPPI.
