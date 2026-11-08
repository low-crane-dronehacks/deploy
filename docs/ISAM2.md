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
src/include/isam2_depth/   types, depth_icp (cloud + ICP), factors, distance_grid, depth_isam2, mppi_interface
src/*.cpp                  core library: Eigen + GTSAM only, no ROS
src/ros/isam2_depth_node.cpp   ROS 2 node (built only when px4_msgs is found)
src/test/                  synthetic test: ray-cast depth in a 6 x 6 x 3 m room with pillars
```

**MPPI hook:** implement `MppiStateSink::onEstimate(const MppiInput&)` and call `node->setSink(...)` (same
process). Each `MppiInput` carries:
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
- `~/obstacles` (PointCloud2 of occupied voxels, per keyframe): the local window the MPPI sees
- `~/keyframes` (nav_msgs/Path, latched): every keyframe pose at its current smoothed value, per keyframe. This is
  the paper's output, the whole trajectory re-estimated; past poses move when ICP or revisit links correct them.
- `~/map` (PointCloud2, latched): the whole corrected map. Every keyframe's 5 cm cloud is placed at its smoothed
  pose; cells with >= `grid_min_hits` points are kept. Rebuilt every `map_publish_period` s (default 1).
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
`ego_planner/scripts/view_course_scan.py`), plans a route through its free space (launch mark, gates, pad), renders
OAK-like 16UC1 depth from it (z-buffered splats, noise 0.003 z^2, invalid edges, dropouts) and publishes a drifting
fake PX4 estimate on `/fmu/out/vehicle_local_position_v1` + `vehicle_attitude`, with `/clock`. It scores the node's
`~/odometry` against the truth and its `~/obstacles` against the scan. It publishes telemetry only, never a command.

```bash
source ~/isam2_ws/install/setup.bash                      # conda deactivated
export ROS_DOMAIN_ID=87                                   # your own, so the fake PX4 topics stay local
ros2 launch src/test/spoof_isam2.launch.py ply:=$PWD/dronehacks-course-scan-2026-11-06.ply      # + RViz
ros2 launch src/test/spoof_isam2.launch.py ply:=... rviz:=false spoof_args:='--no_drift --csv /tmp/run.csv'
/usr/bin/python3 src/test/spoof_px4_depth.py scan.ply --plan_png /tmp/route.png    # route + depth frames, no ROS
```

RViz (fixed frame `map`, ENU above the NED world): scan (RGB), truth path green, spoofed PX4 path red, iSAM2 path
blue, obstacle voxels orange, depth image. Drift knobs: `--yaw_drift_deg_s`, `--yaw_rw_deg`, `--scale`, `--pos_rw`.

Results (2026-11-07, scan of 2026-11-06, 14.6 m route in 36 s, no revisits after the first 20 s):

| fake PX4 estimate | PX4 error rms / final | iSAM2 error rms / final |
|---|---|---|
| perfect | 0 | 0.06 m / 0.09 m, 0.9 deg rms |
| 0.5 deg/s yaw bias, 4% scale, 1 cm/sqrt(s) walk | 0.37 m / 0.68 m, 15.5 deg final | 0.14-0.16 m / 0.28-0.32 m, 6.3-6.6 deg final |

Obstacle voxels (local window): median 0.05-0.07 m from the scan surface. Full corrected map (`~/map`, ~64k
voxels after the route): median 0.058 m, 77% within 10 cm. Smoothed keyframes (`~/keyframes`, the final estimate of
every past pose) with drift: 0.125-0.130 m rms, 3.0-3.1 deg rms, slightly better than the live estimate was at the
time (0.135-0.139 m), as smoothing should be. Keyframe update 12-20 ms (ICP 1-8 ms, iSAM2 < 0.5 ms).

Reading: depth cuts the drift by about 2.5x but cannot remove it on a route that does not revisit places; only
relative constraints exist (ICP between keyframes, revisit links). Absolute correction needs the prior-map factor
(register depth against `config/arena_prior.bin`), which is not written. With perfect odometry the graph adds
6 cm rms: ICP on splatted, noisy depth is slightly biased (the synthetic test's raycast depth gave 7 mm).

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
- No free-space carving: a voxel stays occupied once it has `grid_min_hits` points. Stereo speckle that repeats
  across keyframes, and anything that moves, persist.
- Not in `launch/policynav.launch.py` yet (there is no MPPI to pair it with).
