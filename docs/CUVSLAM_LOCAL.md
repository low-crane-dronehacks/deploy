# cuVSLAM on a desktop (x86, no drone)

`onboard/slam/cuvslam_lib_node.py` wraps NVIDIA's PyCuVSLAM, pinned to cuVSLAM 17.0.0 in `config/versions.yaml`.
This page installs it on an x86 Ubuntu 24.04 machine and runs it on stereo rendered from a course scan, so the VIO
node can be exercised without the drone, FlyX or Gazebo. The drone runs cuVSLAM inside skybeta's Isaac ROS container
([SKYBETA.md](SKYBETA.md)); that is a related implementation, not this binary.

## Install (once)

Tested on automata-5: Ubuntu 24.04, Python 3.12, RTX 5080 (SM120), driver 580, no CUDA toolkit installed.

```bash
# 1. a venv that still sees ROS and the system numpy 1.26 / scipy
/usr/bin/python3 -m venv --system-site-packages ~/cuvslam-venv

# 2. the pinned wheel from NVIDIA's release (cp312 abi3 = Python 3.12, manylinux_2_39 = Ubuntu 24.04, CUDA 12)
mkdir -p ~/Downloads/cuvslam && cd ~/Downloads/cuvslam
W=cuvslam-17.0.0+cu12-cp312-abi3-manylinux_2_39_x86_64.whl
curl -L -o "$W" "https://github.com/nvidia-isaac/cuVSLAM/releases/download/v17.0.0/${W//+/%2B}"
sha256sum "$W"     # 39a6adfedbdf01f8e8e746f93095b1ba04c71e3efe258c1efc0251bab1310018
~/cuvslam-venv/bin/pip install "$W" "numpy<2"

# 3. the wheel bundles the CUDA runtime but needs cuBLAS / cuSOLVER / cuSPARSE (CUDA 12.6, as it was built)
~/cuvslam-venv/bin/pip install "nvidia-cublas-cu12==12.6.4.1" "nvidia-cusolver-cu12==11.7.1.2" \
    "nvidia-cusparse-cu12==12.5.4.2" "nvidia-nvjitlink-cu12==12.6.85" "numpy<2"

# 4. put those libraries on the loader path whenever the venv is active
cat >> ~/cuvslam-venv/bin/activate <<'EOF'

# cuVSLAM 17 (cu12): cuBLAS / cuSOLVER / cuSPARSE come from NVIDIA's PyPI wheels in this venv
_nv="$VIRTUAL_ENV/lib/python3.12/site-packages/nvidia"
export LD_LIBRARY_PATH="$_nv/cublas/lib:$_nv/cusolver/lib:$_nv/cusparse/lib:$_nv/nvjitlink/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
unset _nv
EOF

# 5. check
source ~/cuvslam-venv/bin/activate
ldd $VIRTUAL_ENV/lib/python3.12/site-packages/cuvslam/libcuvslam.so | grep "not found"   # prints nothing
python -c "import cuvslam; print(cuvslam.get_version())"                                  # ('17.0.0+57f42cc', 17, 0, 0)
```

Notes:
- The wheel is NVIDIA proprietary ("Other/Proprietary License"): keep it out of this repo.
- Why a venv: system Python stays untouched, and a `PYTHONNOUSERSITE=true` shell would hide `pip --user`
  installs. Keep numpy below 2: the system scipy 1.11 is built against numpy 1.x.
- `nanobind: leaked N instances!` at interpreter exit is a cosmetic message from the bindings, not an error.
- The wheel has no `[cuda]` extra in 17.0.0 (its only dependency is pyyaml), hence step 3.
- 17.0.1 exists; stay on the pin unless `config/versions.yaml` changes.

## Run the VIO node

Order matters: ROS, then px4_msgs (for `ev_bridge`), then the venv. Use the venv's `python`.
```bash
conda deactivate
source /opt/ros/jazzy/setup.bash
source ~/isam2_ws/install/setup.bash          # px4_msgs release/1.17 (docs/ISAM2.md)
source ~/cuvslam-venv/bin/activate
cd <deploy repo>
python -m onboard.slam.cuvslam_lib_node --ros-args --params-file onboard/config/vio.yaml
python -m onboard.vio.ev_bridge --ros-args --params-file onboard/config/vision_bridge.yaml   # -> PX4 EKF2
```
Inputs (defaults in the node, frames in `vio.yaml`): `/stereo/{left,right}/image` (mono8, both eyes with the same
stamp), `/stereo/{left,right}/camera_info`, `/stereo/imu` (sensor_msgs/Imu, `imu: true`), and static TF
`base_link` -> `oak_left`, `oak_right`, `oak_imu`. Output: `/visual_slam/tracking/odometry` = `base_link` relative to
its pose at start (ROS FLU), plus `/visual_slam/processing_status` and `/visual_slam/vis/*`. The optional Isaac ROS
`VisualSlamStatus` message is not installed here; the node runs without it.

## Stereo rendered from the course scan (no drone, no simulator)

The spoof test ([ISAM2.md](ISAM2.md)) renders, from the phone scan of the course, an OAK-like rectified mono pair
(640x400, fx 421, 7.5 cm baseline, grey from the scan colours) and a 200 Hz IMU from the true trajectory (white noise
at `vio.yaml`'s densities), and starts `cuvslam_lib_node` from the venv:
```bash
source /opt/ros/jazzy/setup.bash && source ~/isam2_ws/install/setup.bash     # conda deactivated
export ROS_DOMAIN_ID=87                                                      # your own
ros2 launch src/test/spoof_isam2.launch.py ply:=$PWD/assets/dronehacks-course-scan-2026-11-06.ply stereo:=true
#   vslam_imu:=false        stereo-only (Multicamera) instead of Inertial
#   cuvslam_venv:=<path>    if the venv is elsewhere
#   spoof_args:='--stereo_hz 10 --imu_hz 200 --pixel_noise 2'
```
The summary at the end scores cuVSLAM's odometry against the truth; RViz shows it as the orange path (and the left
eye under "Stereo left"). It runs slower than real time (rendering ~110 ms per stereo pair; sim time keeps it
consistent): the 36 s route takes ~4 min.

Results (2026-11-08, scan of 2026-11-06, 14.8 m route, 36 s, 15 Hz stereo):

| run | tracked | features/frame | final position error | rotation |
|---|---|---|---|---|
| offline, stereo only (no ROS) | 552/552 | p50 260, min 185 | 0.12 m (0.8%) | 0.8 deg |
| through ROS, Inertial (stereo + IMU) | 525/538 poses, 0 lost | p50 266, min 185 | 0.17 m (1.1%) | yaw 0.09 deg |

`track()` takes 1.3 ms per frame (p50) on the RTX 5080; gravity converges to (-0.02, -0.04, -9.81). 13 poses were
withheld because the SDK reported no covariance (the node never sends an unknown-uncertainty pose on).

**What this does and does not show.** It exercises the node end to end (topics, frames, stereo pairing, IMU
registration, covariance gating) on the real course geometry. It does not qualify cuVSLAM on real stereo: the images
are point splats (blocky up close, dark where the scan has no colour), there is no motion blur, exposure change,
rolling shutter or lens distortion, and the IMU is ideal apart from white noise (no bias, no vibration). Real-camera
accuracy needs a bag from the drone's OAK (handheld mode, props off) or FlyX.

Not wired yet: feeding cuVSLAM's pose into the fake PX4 estimate (the real chain is cuVSLAM -> `ev_bridge` -> EKF2,
and iSAM2 then corrects PX4's estimate). The spoofer's PX4 estimate is still the truth plus synthetic drift.
