#!/usr/bin/env python3
"""ROS stereo/IMU node for the pinned NVIDIA cuVSLAM 17 library.

Consumes images, camera calibration, IMU and static TF. Publishes estimated
odometry, tracking status and per-frame processing diagnostics. Configuration
and code are shared across physical and simulated ROS sensor drivers.
"""
import base64
import json
import struct
import time

import numpy as np
import rclpy
from geometry_msgs.msg import Vector3Stamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from scipy.spatial.transform import Rotation as R
from sensor_msgs.msg import CameraInfo, Image, Imu
from tf2_ros import Buffer, TransformListener
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue

import cuvslam
from .pose_cov import transform_pose_covariance
from .stereo_pairing import add_half
from .vio_evidence import visually_tracked

try:
    from isaac_ros_visual_slam_interfaces.msg import VisualSlamStatus
except ImportError:                                    # status is optional
    VisualSlamStatus = None


def _T(rot_xyzw, t):
    M = np.eye(4)
    M[:3, :3] = R.from_quat(rot_xyzw).as_matrix()
    M[:3, 3] = t
    return M


def _tf_matrix(tf):
    q, p = tf.transform.rotation, tf.transform.translation
    return _T([q.x, q.y, q.z, q.w], [p.x, p.y, p.z])


def _pose(M):
    q = R.from_matrix(M[:3, :3]).as_quat()
    return cuvslam.Pose(rotation=list(q), translation=list(M[:3, 3]))


def _stamp_ns(h):
    return h.stamp.sec * 1_000_000_000 + h.stamp.nanosec


class CuvslamLibNode(Node):
    def __init__(self):
        super().__init__("visual_slam_node")
        from rcl_interfaces.msg import ParameterDescriptor
        loose = ParameterDescriptor(dynamic_typing=True)    # "500" and "500.0" both accepted
        dp = lambda n, v: self.declare_parameter(n, v, loose).value
        self.left_frame = dp("left_frame", "oak_left")
        self.right_frame = dp("right_frame", "oak_right")
        self.imu_frame = dp("imu_frame", "oak_imu")
        self.base_frame = dp("base_frame", "base_link")
        self.odom_frame = dp("odom_frame", "odom")
        self.use_imu = str(dp("imu", False)).lower() in ("1", "true", "yes")
        self.noise = dict(gyroscope_noise_density=float(dp("gyro_noise_density", 0.000244)),
                          gyroscope_random_walk=float(dp("gyro_random_walk", 0.000005)),
                          accelerometer_noise_density=float(dp("accel_noise_density", 0.001862)),
                          accelerometer_random_walk=float(dp("accel_random_walk", 0.0001)),
                          frequency=float(dp("imu_frequency", 500.0)))
        topics = dict(l=dp("left_image", "/stereo/left/image"), r=dp("right_image", "/stereo/right/image"),
                      li=dp("left_info", "/stereo/left/camera_info"), ri=dp("right_info", "/stereo/right/camera_info"),
                      imu=dp("imu_topic", "/stereo/imu"))
        self.tf_buf = Buffer()
        self.tf_listener = TransformListener(self.tf_buf, self)
        self.info = {"l": None, "r": None}
        self.half = {}                      # stamp ns -> {"l": img, "r": img}
        self.imu_q = []                     # (t_ns, accel, gyro), unregistered
        self.last_imu_ns = -1
        self.last_frame_ns = -1
        self.tracker = None
        self.T_base_left = None
        self.prev = None                    # (t_ns, odom_from_base)
        self.n = {"pairs": 0, "ok": 0, "lost": 0, "imu_only": 0, "imu": 0, "imu_dropped": 0, "unpaired": 0,
                  "errors": 0, "cov_bad": 0}
        # VISUAL vs AVAILABLE (external review 2026-10-16): in Inertial mode the library returns an
        # IMU-propagated pose when visual tracking fails, with no flag. A frame counts as visually
        # tracked only with at least min_visual_obs tracked features; otherwise vo_state = 2
        # (Failed) even though a pose exists. imu_only_policy: publish (the library's behaviour,
        # what the aircraft's Isaac ROS node does) | withhold (no odometry for such frames).
        from rcl_interfaces.msg import ParameterDescriptor as _PD
        self.min_visual_obs = int(self.declare_parameter("min_visual_obs", 10, _PD(dynamic_typing=True)).value)
        self.imu_only_policy = str(self.declare_parameter("imu_only_policy", "publish").value)
        if self.imu_only_policy not in ("publish", "withhold"):
            raise SystemExit("imu_only_policy: publish | withhold")
        # Per-frame processing diagnostics also describe frames that produced no pose.
        self.pub_processing = self.create_publisher(DiagnosticArray,
            dp("processing_status_topic", "/visual_slam/processing_status"), 10)
        self.track_ms = []
        self.pub_odom = self.create_publisher(Odometry, "/visual_slam/tracking/odometry", 10)
        self.pub_status = (self.create_publisher(VisualSlamStatus, "/visual_slam/status", 10)
                           if VisualSlamStatus else None)
        self.pub_grav = self.create_publisher(Vector3Stamped, "/visual_slam/vis/gravity", 10)
        # tracked-feature count per frame (std_msgs/Int32MultiArray [n_observations, frame stamp ns
        # low 32 bits]): a frame carried by the IMU alone shows as ~0 features while the pose
        # still reads "ok" - the evidence behind a position jump
        from std_msgs.msg import Int32MultiArray
        self._IntArr = Int32MultiArray
        self.pub_obs = self.create_publisher(Int32MultiArray, "/visual_slam/vis/observations_count", 10)
        self.obs_hist = []
        self.create_subscription(CameraInfo, topics["li"], lambda m: self._info("l", m), qos_profile_sensor_data)
        self.create_subscription(CameraInfo, topics["ri"], lambda m: self._info("r", m), qos_profile_sensor_data)
        self.create_subscription(Image, topics["l"], lambda m: self._image("l", m), qos_profile_sensor_data)
        self.create_subscription(Image, topics["r"], lambda m: self._image("r", m), qos_profile_sensor_data)
        if self.use_imu:
            # Five samples cover only 10 ms at 500 Hz. SDK work runs in this
            # executor too; retain a bounded second of IMU arrivals through a
            # processing burst. Best effort remains compatible with real drivers.
            self.create_subscription(Imu, topics["imu"], self._imu,
                                     QoSProfile(depth=512, reliability=ReliabilityPolicy.BEST_EFFORT))
        self.create_timer(10.0, self._log)
        self.get_logger().info(f"cuVSLAM library {cuvslam.get_version()} | mode "
                               f"{'Inertial' if self.use_imu else 'Multicamera'} | frames base={self.base_frame} "
                               f"left={self.left_frame} right={self.right_frame} imu={self.imu_frame} | {topics}")

    # ------------------------------------------------------------------ setup
    def _info(self, side, m):
        if self.info[side] is None and m.width:
            self.info[side] = m

    def _lookup(self, target, source):
        return _tf_matrix(self.tf_buf.lookup_transform(target, source, rclpy.time.Time()))

    def _try_init(self):
        if self.tracker is not None or None in self.info.values():
            return self.tracker is not None
        try:
            T_left_right = self._lookup(self.left_frame, self.right_frame)
            self.T_base_left = self._lookup(self.base_frame, self.left_frame)
            T_left_imu = self._lookup(self.left_frame, self.imu_frame) if self.use_imu else None
        except Exception as e:                           # static TF not here yet
            self.get_logger().info(f"waiting for the rig TF: {e}", throttle_duration_sec=5.0)
            return False
        # SDK rig/world coordinates are user-defined. Use the aircraft body
        # frame so an ordinary yaw does not become camera Euler pitch90deg.
        self.T_base_rig = np.eye(4)
        T_rig_imu = self.T_base_left @ T_left_imu if self.use_imu else None
        cams = []
        for side, M in (("l", self.T_base_left), ("r", self.T_base_left @ T_left_right)):
            ci = self.info[side]
            P = ci.p if any(ci.p) else [ci.k[0], 0, ci.k[2], 0, 0, ci.k[4], ci.k[5], 0, 0, 0, 1, 0]
            cam = cuvslam.Camera()
            cam.size = (int(ci.width), int(ci.height))
            cam.focal = (float(P[0]), float(P[5]))
            cam.principal = (float(P[2]), float(P[6]))
            cam.distortion = cuvslam.Distortion(cuvslam.Distortion.Model.Pinhole)
            cam.rig_from_camera = _pose(M)
            cams.append(cam)
        rig = cuvslam.Rig()
        rig.cameras = cams
        cfg = cuvslam.Tracker.OdometryConfig()
        cfg.rectified_stereo_camera = True
        cfg.use_gpu = True
        cfg.enable_landmarks_export = False
        cfg.enable_observations_export = True          # per-frame tracked features: the visual-strength evidence
        if self.use_imu:
            imu = cuvslam.ImuCalibration()
            imu.rig_from_imu = _pose(T_rig_imu)
            for k, v in self.noise.items():
                setattr(imu, k, v)
            rig.imus = [imu]
            cfg.odometry_mode = cuvslam.Tracker.OdometryMode.Inertial
        else:
            cfg.odometry_mode = cuvslam.Tracker.OdometryMode.Multicamera
        self.tracker = cuvslam.Tracker(rig, cfg)
        b = float(np.linalg.norm(T_left_right[:3, 3]))
        self.get_logger().info(f"tracker up: {cams[0].size} fx={cams[0].focal[0]:.1f} baseline {b:.4f} m"
                               + (f", IMU rig_from_imu R=\n{np.round(T_rig_imu[:3, :3], 3)}" if self.use_imu else ""))
        return True

    # ---------------------------------------------------------------- inputs
    def _imu(self, m):
        t = _stamp_ns(m.header)
        self.imu_q.append((t, [m.linear_acceleration.x, m.linear_acceleration.y, m.linear_acceleration.z],
                           [m.angular_velocity.x, m.angular_velocity.y, m.angular_velocity.z]))
        if len(self.imu_q) > 5000:
            del self.imu_q[:1000]

    def _image(self, side, m):
        if m.encoding.lower() not in ("mono8", "8uc1"):
            return
        t = _stamp_ns(m.header)
        img = np.frombuffer(m.data, np.uint8).reshape(m.height, m.step)[:, :m.width]
        h, dropped = add_half(self.half, t, side, img)
        self.n["unpaired"] += dropped
        if h is not None:
            self._track(t, h["l"], h["r"])

    # -------------------------------------------------------------- tracking
    def _track(self, t, left, right):
        started = time.perf_counter()
        self._processing_result = "skipped"
        self._processing_details = {}
        try:
            self._track_frame(t, left, right)
        except BaseException:
            self._processing_result = "exception"
            raise
        finally:
            self._complete(t, self._processing_result, (time.perf_counter()-started)*1000.)

    def _track_frame(self, t, left, right):
        if not self._try_init() or t <= self.last_frame_ns:
            return
        if self.use_imu:                                   # IMU up to this frame, in order
            self.imu_q.sort(key=lambda e: e[0])
            keep = []
            registered = []
            for e in self.imu_q:
                if e[0] > t:
                    keep.append(e)
                elif e[0] <= self.last_imu_ns or e[0] < self.last_frame_ns:
                    self.n["imu_dropped"] += 1
                else:
                    meas = cuvslam.ImuMeasurement()
                    meas.timestamp_ns, meas.linear_accelerations, meas.angular_velocities = int(e[0]), e[1], e[2]
                    self.tracker.register_imu_measurement(0, meas)
                    self.last_imu_ns = e[0]
                    self.n["imu"] += 1
                    registered.append(int(e[0]))
            self.imu_q = keep
            # A separate rosbag subscriber may receive samples that this node
            # did not. Preserve the actual SDK input boundary for honest replay.
            # Fixed-width int64 + base64 also bounds a full 5000-sample backlog
            # below the existing 64 KiB diagnostic transport limit. JSON epoch
            # nanoseconds could exceed that limit and lose the completion report.
            self._processing_details["imu_registered_ns_b64le"] = base64.b64encode(
                struct.pack(f"<{len(registered)}q", *registered)).decode("ascii")
            self._processing_details["imu_registered_count"] = str(len(registered))
        t0 = time.perf_counter()
        try:
            est, _ = self.tracker.track(int(t), [np.ascontiguousarray(left), np.ascontiguousarray(right)])
        except ValueError as e:
            self.n["errors"] += 1
            self.get_logger().warning(f"track refused: {e}", throttle_duration_sec=5.0)
            self._processing_result = "error"
            return
        ms = (time.perf_counter() - t0) * 1e3
        self.track_ms.append(ms)
        self.last_frame_ns = t
        self.n["pairs"] += 1
        ok = est is not None and est.world_from_rig is not None
        try:
            n_obs = len(self.tracker.get_last_observations(0)) if ok else 0
        except Exception:
            n_obs = -1
            self.n["obs_unknown"] = self.n.get("obs_unknown", 0) + 1
        # UNKNOWN evidence is not visual tracking (review 2026-10-17: a failed observation query used
        # to count as tracked); flights so far: 184-270 features per frame, the query never failed
        visual = visually_tracked(ok, n_obs, self.min_visual_obs)
        if not ok:
            self._status(t, False, ms)
            self.n["lost"] += 1
            self._processing_result = "no_pose"
            return
        if not visual:
            self.n["imu_only"] += 1
            if self.imu_only_policy == "withhold":
                self._status(t, False, ms)
                self._processing_result = "withheld"
                return
        self.obs_hist.append(n_obs)
        if len(self.obs_hist) > 600:
            del self.obs_hist[:300]
        m_obs = self._IntArr()
        m_obs.data = [int(n_obs), int(t % 2_147_483_648)]
        self.pub_obs.publish(m_obs)
        p = est.world_from_rig.pose
        W_rig = _T(list(p.rotation), list(p.translation))
        O = W_rig  # SDK rig is the body; world is the initial body frame
        if self._publish(t, O, est.world_from_rig, W_rig) is False:
            self._status(t, False, ms)
            self._processing_result = "withheld"
            return
        if visual:
            self.n["ok"] += 1
        self._status(t, visual, ms)
        self._processing_result = "visual" if visual else "inertial"
        if self.use_imu:
            g = self.tracker.get_last_gravity()
            if g is not None:
                v = Vector3Stamped()
                v.header.stamp.sec, v.header.stamp.nanosec = int(t // 1_000_000_000), int(t % 1_000_000_000)
                v.header.frame_id = self.base_frame  # SDK returns gravity in the current rig, which is the body
                v.vector.x, v.vector.y, v.vector.z = (float(x) for x in g)
                self.pub_grav.publish(v)

    def _complete(self, t, result, duration_ms):
        report = DiagnosticArray()
        report.header.stamp.sec, report.header.stamp.nanosec = divmod(int(t), 1_000_000_000)
        report.header.frame_id = self.odom_frame
        status = DiagnosticStatus()
        status.name = "visual_slam/frame"
        status.hardware_id = "cuvslam"
        status.level = (DiagnosticStatus.OK if result == "visual" else
                        DiagnosticStatus.ERROR if result in ("error", "exception") else DiagnosticStatus.WARN)
        status.message = result
        status.values = [KeyValue(key="capture_ns", value=str(int(t))),
                         KeyValue(key="duration_ms", value=str(float(duration_ms)))]
        details = getattr(self, "_processing_details", {})
        if "covariance_rejected" in details and status.level == DiagnosticStatus.OK:
            status.level = DiagnosticStatus.WARN
        status.values += [KeyValue(key=k, value=v) for k, v in details.items()]
        report.status = [status]
        self.pub_processing.publish(report)

    def _publish(self, t, O, pwc, W_rig):
        m = Odometry()
        m.header.stamp.sec, m.header.stamp.nanosec = int(t // 1_000_000_000), int(t % 1_000_000_000)
        m.header.frame_id, m.child_frame_id = self.odom_frame, self.base_frame
        q = R.from_matrix(O[:3, :3]).as_quat()
        m.pose.pose.position.x, m.pose.pose.position.y, m.pose.pose.position.z = (float(x) for x in O[:3, 3])
        (m.pose.pose.orientation.x, m.pose.pose.orientation.y,
         m.pose.pose.orientation.z, m.pose.pose.orientation.w) = (float(x) for x in q)
        raw_covariance = None
        try:                                               # the FULL 6x6 through O = A W A^-1 (pose_cov.py)
            raw_covariance = pwc.covariance_xyz_rpy
            if not np.any(np.asarray(raw_covariance, dtype=float)):
                raise ValueError('SDK covariance is all zero: uncertainty unavailable')
            C = transform_pose_covariance(raw_covariance, self.T_base_rig, W_rig)
            m.pose.covariance = [float(x) for x in C.ravel()]
        except Exception as e:                            # unavailable uncertainty: withhold the pose
            self.n["cov_bad"] += 1
            self._processing_details["covariance_rejected"] = str(e)[:512]
            # Preserve the SDK input on this capture, not just a throttled warning.
            # Completion still acknowledges this capture; no unknown-covariance
            # pose may reach PX4 as its small default variance. Nonfinite entries
            # remain explicit JSON strings in diagnostics, never JSON NaN.
            try:
                raw = np.asarray(raw_covariance, dtype=float)
                if raw.size == 36:                        # Preserve SDK order; flat binding is column-major.
                    entries = [float(x) if np.isfinite(x) else str(float(x)) for x in raw.ravel()]
                    self._processing_details["covariance_xyz_rpy"] = json.dumps(entries, allow_nan=False)
            except Exception:
                pass
            self.get_logger().warning(f"pose withheld: unavailable covariance: {e}", throttle_duration_sec=10.0)
            return False
        if self.prev is not None and t > self.prev[0]:      # twist in the BODY frame, by finite difference
            dt = (t - self.prev[0]) * 1e-9
            Op = self.prev[1]
            v_body = O[:3, :3].T @ (O[:3, 3] - Op[:3, 3]) / dt
            w_body = R.from_matrix(Op[:3, :3].T @ O[:3, :3]).as_rotvec() / dt
            m.twist.twist.linear.x, m.twist.twist.linear.y, m.twist.twist.linear.z = (float(x) for x in v_body)
            m.twist.twist.angular.x, m.twist.twist.angular.y, m.twist.twist.angular.z = (float(x) for x in w_body)
        self.prev = (t, O)
        self.pub_odom.publish(m)
        return True

    def _status(self, t, ok, ms):
        if self.pub_status is None:
            return
        s = VisualSlamStatus()
        s.header.stamp.sec, s.header.stamp.nanosec = int(t // 1_000_000_000), int(t % 1_000_000_000)
        s.header.frame_id = self.odom_frame
        s.vo_state = 1 if ok else 2
        s.track_execution_time = ms * 1e-3
        a = self.track_ms[-200:]
        s.track_execution_time_mean = (sum(a) / len(a)) * 1e-3
        s.track_execution_time_max = max(a) * 1e-3
        s.node_callback_execution_time = ms * 1e-3
        self.pub_status.publish(s)

    def _log(self):
        a = sorted(self.track_ms[-500:])
        tm = f"track ms p50 {a[len(a) // 2]:.1f} max {a[-1]:.1f}" if a else "no frames yet"
        g = self.tracker.get_last_gravity() if (self.tracker is not None and self.use_imu) else None
        ob = sorted(self.obs_hist[-300:])
        obs = f"features/frame p50 {ob[len(ob) // 2]} min {ob[0]}" if ob else "no frames"
        self.get_logger().info(f"{self.n} | {tm} | {obs} | gravity {'none yet' if g is None else np.round(list(g), 2)}")


def main():
    rclpy.init()
    node = CuvslamLibNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
