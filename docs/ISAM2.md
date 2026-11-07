# iSAM2 depth graph (src/)

C++ estimator that feeds an MPPI controller. It is a keyframe factor graph over depth ICP and PX4 odometry,
solved incrementally with iSAM2 (Kaess et al., ICRA 2011, `Kaess11icra.pdf`) through GTSAM 4.2. The MPPI
itself is not here.

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
colcon build --base-paths src                          # needs px4_msgs (release/1.17) in the workspace for the node
ros2 run isam2_depth isam2_depth_node --ros-args -p depth_topic:=/camera/depth/image_raw \
    -p depth_info_topic:=/camera/depth/camera_info     # drone: /oak/stereo/image_raw (launch PROFILES)
# core + test only, no ROS:
cmake -S src -B build && cmake --build build && ./build/test_depth_isam2
```

**Topics:**
- `~/odometry` (nav_msgs/Odometry at the PX4 rate; covariance reordered to ROS's [trans, rot])
- `~/obstacles` (PointCloud2 of occupied voxels, per keyframe, for RViz)

The node sends nothing to PX4. It times PX4 samples by their arrival on the node clock (sim time in simulation)
and interpolates them to the depth header stamp.

## Status (2026-11-07)

**Synthetic test** (`test_depth_isam2`, x86, 75 keyframes over a 40 s circle, stereo-like noise of 0.004 z^2):

| | final error |
|---|---|
| PX4 odometry drifting 1.1 deg/s in yaw with 4% scale error | 0.90 m / 46 deg |
| iSAM2 estimate, same run | 0.01 m / 0.9 deg |
| iSAM2 estimate with perfect odometry | 0.007 m (depth does not make it worse) |

Worst keyframe update: about 31 ms, of which ICP is about 18 ms, iSAM2 about 1.4 ms, and covariance plus grid about 7 ms.

**Not done yet:**
- Not run on SITL, FlyX or the Jetson. Expect several times slower on the Orin Nano; `cloud_stride`,
  `extra_links` and `IcpParams::point_step` are the knobs.
- The node was compile-checked against a stand-in px4_msgs with the same field names, not the real package.
- No free-space carving: a voxel stays occupied once it has `grid_min_hits` points. Stereo speckle that repeats
  across keyframes, and anything that moves, persist.
- Not in `launch/policynav.launch.py` yet (there is no MPPI to pair it with).
