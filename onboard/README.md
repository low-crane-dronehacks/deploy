# Your onboard software

## October29 native application check

The identical `onboard/examples/takeoff_land.py` now has local isolated-companion
acceptance: VIO off safely refuses twice; actual cuVSLAM/EV services on complete
takeoff, hover, AUTO_LAND and confirmed disarm. The native flight grade passes,
but sustained timing fails (moving RTF 0.53). An earlier failed fixture is retained.
This is source/interface evidence, not an ARM binary, browser/cloud deployment
or initialized-inertial qualification. No physical aircraft was tested.


## October27 SDK calibration correction

The tracker rig is the body frame. Camera and IMU extrinsics are transformed
together from the recorded physical TF; returned odometry is already the body
pose. No post-hoc camera-frame conjugation is applied. SDK uncertainty is
xyz/Euler-rpy, and invalid covariance withholds the pose while still acknowledging
the capture. Runtime deployment supplies the actual simulated IMU cadence
(480 Hz on the240 Hz preset); hardware defaults remain500 Hz. See
[the correction report](../docs/IMU_FUSION_FIXES_2026-10-27.md).


These files are the actual source executed inside your Linux companion. The
installed NVIDIA cuVSLAM library is a dependency; the tracker ROS wrapper,
odometry conversion/health gates and process launch are all yours to edit.
Physics, PX4 SITL, virtual sensors and the simulation transport stay outside.

```bash
# Inspect normal Linux processes and native device topics.
ps -ef
ros2 topic echo /fmu/out/vehicle_status_v1 --once

# These commands run the source in THIS project. Startup never arms.
python3 onboard/services.py status
python3 onboard/services.py stop
python3 onboard/services.py start
```

The visible process list is `onboard/config/services.json`. Each process has its
own log, PID/start identity, exact command and source receipt under `.onboard/`.
Edit source/config, stop, then start to apply changes. A running Python process
is not automatically replaced when you save a file. Start refuses a second
launch; stop targets this launch's recorded ownership, never process names.

The simulator initially starts this project bringup for convenience. You can
stop it or replace it. A maze needs a working position estimator before PX4
can fly in position-controlled Offboard mode. Stopped/failed vision is reported
unhealthy; no hidden estimator takes over. Outdoor GNSS uses the visible PX4
health/odometry nodes without cuVSLAM.

`/etc/flyx/onboard.json` and `onboard.yaml` contain readable deployment data:
standard ROS clock selection, camera IMU rate and the declared heading reference.
Hardware uses the ROS system clock; simulation selects `use_sim_time: true` and
the standard `/clock` topic. This application imports no simulator clock,
latency model or grader. The managed transport maps clocks and models transport
delay; it does not estimate position, convert frames or make flight decisions.
Camera images, calibration, IMU and `/fmu/*` use their actual ROS message types.

`vio/ev_bridge.py`, `frames.py`, `validation.py`, `vio_health.py` and
`ownership.py` are ordinary onboard integration and safety code. With an unknown
startup heading, vision output is FRD. NED output requires `heading_enu_rad` and
a nonempty `heading_reference`; the code rotates position, orientation and
covariance using that reference. A real deployment needs its own valid surveyed
or otherwise established reference. The maze configuration declares its fixed
start pad; changing that fixture requires changing the calibration too.

The tracker publishes `/visual_slam/processing_status` as standard ROS
`diagnostic_msgs/DiagnosticArray`, including frames that produced no pose or
failed. This is useful onboard processing evidence; the simulator observes it
for lockstep without adding simulator calls to the application.

Start the same visible ROS modules directly when configuring your own robot:

```bash
python3 -m onboard.slam.cuvslam_lib_node --ros-args --params-file onboard/config/vio.yaml
python3 -m onboard.vio.ev_bridge --ros-args --params-file onboard/config/vision_bridge.yaml
```

Configure and run them in separate terminals or use the owned bringup above.
Keep the reviewed cuVSLAM/PX4 message versions and actual camera calibration.
The simulator's FC parameter presets are not hardware installation instructions.
This portable package changes frame/transport contracts and needs fresh moving
acceptance; older image flight results do not qualify this revision.

For a raw ROS program based on the supplied takeoff/hover/land example:

```bash
python3 onboard/examples/takeoff_land.py
```

That script directly creates native PX4 publishers/subscribers. It makes no lab
API calls and imports no mission controller. It requests Offboard and arms a
vehicle: run only with the required operator authorization and an appropriate
isolated simulator or reviewed aircraft setup. Its default height is1m.
It resolves native output topic versions from installed `px4_msgs` (`_v1` for
position/status on this image). Compared with the supplied example, stale
telemetry also stops landing retries, takeover yields before another setpoint,
completion requires fresh landed plus disarmed reports, and the process exits
when finished. These decisions are all visible in this script. The October29 direct-companion test exercised this native path; its flight
passed and its overall timing acceptance failed. Managed Run/browser/cloud and
hardware acceptance are separate.

The Micro XRCE agent equivalent is already running on the simulator side. Do not
start another agent or fake `/dev/ttyTHS1` in this container. On hardware the
agent uses the reviewed serial/UDP transport; this project uses the resulting
native ROS interface in either deployment.

The portal Run button remains an optional snapshot/recorded execution path.
Your terminal can run ordinary ROS code directly; direct programs are not
automatically admitted as a managed Run or guaranteed a mission rosbag.
PX4 ULogs and an explicit recorder remain the evidence sources in that case.

## Moving this application to a Jetson

Keep the flight Python file unchanged. Use ROS timers for flight intervals and
normal ROS deployment files for clock selection. Plain Python execution reads
`/etc/flyx/mission.yaml` (or `DRONE_PARAMS_FILE`); absent configuration uses the
standard ROS system clock. An explicitly missing configuration is an error.

On the drone, install the actual OAK ROS driver, measured CameraInfo/TF and
camera/IMU calibration, a compatible ARM ROS/CUDA/cuVSLAM build, matching
`px4_msgs`, and one reviewed uXRCE agent for the FC link. The simulator supplies
virtual devices behind those interfaces. It does not require a different mission
API, a fake serial device or a second agent inside your application container.

The current simulator library image is x86-64/Jazzy/SM120. Its binary cannot be
copied onto an ARM Orin; a matching target build and hardware commissioning are
still required. FlyX's existing aircraft Isaac ROS SLAM container and this
PyCuVSLAM wrapper are related implementations, not identical installed binaries.
Docker provides dependency/process isolation; it does not remove architecture,
driver, calibration or performance differences.

Vision publishes ordinary odometry. The visible EV bridge validates frames,
covariance, reset/freshness and health, then sends `VehicleOdometry` to
`/fmu/in/vehicle_visual_odometry`. This is external visual odometry, not optical
flow. PX4 still estimates state and controls the motors. Without GNSS or another
horizontal estimator, the position-controlled example refuses to arm; forcing
an attitude-only launch to produce drift would be a different program/test.

Real startup heading/frame calibration and PX4 fusion/motor/airframe parameters
need their own reviewed commissioning. Never copy a simulator parameter preset
onto a flight controller as installation instructions. RC safety and operator
arming authorization remain required on hardware.
