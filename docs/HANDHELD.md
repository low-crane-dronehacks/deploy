# Handheld dry run / data collection

Carry the drone (props OFF) or just the OAK through the course while the policy runs on the live depth. It
reports what it would command and logs everything. **It never commands PX4**: the handheld node has no
publishers on `/fmu/in/*`.

```bash
ros2 launch launch/policynav.launch.py hardware:=handheld                          # Pixhawk on USB for state
ros2 launch launch/policynav.launch.py hardware:=handheld fc_device:=/dev/ttyACM0 record:=true
ros2 launch launch/policynav.launch.py hardware:=handheld state:=odom vio:=true    # cuVSLAM pose, no Pixhawk
ros2 launch launch/policynav.launch.py hardware:=handheld state:=none range:=fixed # depth only
ros2 topic pub --once /policynav/handheld/reset std_msgs/msg/Empty {}              # at the launch mark
```

## Options
- `policy:=policies/<name>.npz`. The default is the launch file's; use `cur4_clean50_800` for now.
- **`state`** (where velocity and attitude come from):
  - `px4`: a Pixhawk over USB, read only. Gives PX4's estimate plus the VL53L1X rangefinder.
  - `odom`: cuVSLAM odometry, with velocity differenced from the pose.
    - Note: `vio:=true` starts `onboard/services.py`, whose external-vision bridge writes
      `/fmu/in/vehicle_visual_odometry` when a PX4 is listening. That is an estimator input, not a command, and
      nothing arms.
  - `none`: zero velocity, level attitude.
- **`range`** (the policy's downward-range input):
  - `auto`: the rangefinder, else altitude above the reset point (plus `start_height_m`), else `range_fixed_m`
    (1.0 m).
  - Holding the drone at chest height is not the training distribution near the floor, so read the near-floor
    behaviour with care.
- **`memory`**:
  - `closed` (default): keeps the policy's recurrent state and feeds back its own previous action, as in flight.
    The person carrying it stands in for the controller.
  - `fresh`: shows what it would command on the first step of a flight from each pose.
- **`cam_pose:="x y z yaw pitch roll"`**: the camera mount, `base_link -> cam_frame` (default `oak-d-base-frame`).
  - Measure it.
  - Check the driver's frame name with `ros2 run tf2_tools view_frames`.
  - The node logs the optical axis error on the first frame; it should be near 0 deg for a forward camera.

## What you get
- Live: `/policynav/handheld/obs` (the policy's 128x80 depth, viewable in rqt_image_view),
  `/policynav/handheld/cmd` (would-be velocity as a TwistStamped in base_link, an rviz arrow),
  `/policynav/handheld/action`, `/policynav/status`. Plus a 1 Hz log line, e.g.
  `would command: fwd +1.80 right -0.20 up +0.40 m/s yaw +15 deg/s | range 1.02 m (rangefinder) | depth valid 87%`.
- `logs/handheld/handheld_<stamp>.npz`: every 10 Hz step, with the policy depth frames (float16), actions,
  setpoints, state and range. It is saved every 30 s and on exit.
- With `record:=true`, `logs/handheld/bag_<stamp>/`: the raw OAK depth, camera_info, IMU, TF, the policynav
  topics and PX4/VIO state. `record_stereo:=true` adds the raw mono pair (~15 MB/s).
  - The bag is what lets us check our stereo model against the real camera, and replay a walk offline with any
    policy.

Status: pure parts tested (`tests/test_handheld.py`). The ROS node has not yet been run on hardware.
