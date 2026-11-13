#!/usr/bin/env python3
"""Closed-loop scan sensor & PX4 VIO bridge for classical_demo.

Couples the closed-loop C++ QuadPlant (`sim_loopback_node` on `/truth/odometry`) with the
C++ `isam2_depth_node` + classical autonomy stack (`astar_planner_node`, `path_transformer_node`,
`mppi_controller_node`, `control_mux_node`) against the real physical course scan (.ply):
  - Renders 640x400 16UC1 OAK depth (`/camera/depth/image_raw` + `/camera/depth/camera_info`)
    from the TRUE drone pose against the physical `.ply` scan (reusing `Renderer` & `load_scene`
    directly from `src/test/spoof_px4_depth.py`).
  - Integrates realistic VIO drift onto incremental truth motion and publishes fake PX4 output
    telemetry (`/fmu/out/vehicle_local_position_v1` + `/fmu/out/vehicle_attitude`) for `isam2_depth_node`.
  - Bridges `isam2_depth_node`'s smoothed 50 Hz odometry (`/isam2_depth/odometry`) onto `/odometry/local`
    and `TF odom_ned -> base_link_frd` (bootstrapping from the initial launch pose for the first ~0.2 s
    before keyframe 0 initializes).
  - Never publishes commands to `/fmu/in/*` and never arms any vehicle.
"""
from __future__ import annotations

import argparse
import math
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

DEPLOY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DEPLOY_ROOT / "src" / "test"))
from spoof_px4_depth import (  # noqa: E402
    CAM_CX,
    CAM_CY,
    CAM_F,
    CAM_H,
    CAM_W,
    F_FLU,
    R_ENU_NED,
    R_FRD_OPT,
    T_FRD_CAM,
    Renderer,
    load_scene,
)

PAD_CENTER_XY = np.array([-0.7925, -2.0422], dtype=float)
LAUNCH_XY = np.array([0.9144, -2.5146], dtype=float)


def _rot_z(yaw: float) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=float)


class OpenSkyDepthRenderer(Renderer):
    """Point-splat depth renderer that distinguishes open-sky rays (+inf) from stereo dropouts (NaN).

    `Renderer` in `spoof_px4_depth.py` maps both stereo dropouts/occluding edges AND open-sky rays
    (such as rays looking upward through the physical elevated window above the 1.73 m walls where
    the `.ply` has no ceiling points) to `0`. In `isam2_depth_node`, `0` -> `NaN` is skipped by
    `PriorEsdfMap::extractRays()`, whereas `+inf` in `32FC1` means "no obstacle within range" and
    carves stale CAD wall voxels cleanly without inserting any fake endpoint hit (`AGENTS.md`).
    """

    def __init__(self, pts: np.ndarray, spacing: float, args: argparse.Namespace, rng: np.random.Generator) -> None:
        super().__init__(pts, spacing, args, rng)
        uu, vv = np.meshgrid(np.arange(self.w, dtype=np.float32), np.arange(self.h, dtype=np.float32))
        dirs = np.stack([(uu - self.cx) / self.f, (vv - self.cy) / self.f, np.ones_like(uu)], axis=-1)
        self._ray_dirs_cam = dirs / np.linalg.norm(dirs, axis=-1, keepdims=True)

    def render_32f(self, p_wb: np.ndarray, R_wb: np.ndarray) -> np.ndarray:
        """Render 640x400 float32 depth (m): finite = hit, +inf = open free space, NaN = stereo dropout."""
        R_wc = R_wb @ R_FRD_OPT
        t_wc = p_wb + R_wb @ T_FRD_CAM
        pc = (self.pts - t_wc) @ R_wc
        z = pc[:, 2]
        ok = (z > 0.15) & (z < 12.0)
        pc, z = pc[ok], z[ok]
        u = np.rint(self.f * pc[:, 0] / z + self.cx).astype(np.int32)
        v = np.rint(self.f * pc[:, 1] / z + self.cy).astype(np.int32)
        m = 4
        ok = (u >= -m) & (u < self.w + m) & (v >= -m) & (v < self.h + m)
        u, v, z = u[ok], v[ok], z[ok]
        r = np.clip(np.rint(0.5 * self.f * self.spacing / z + 0.25), 0, 3).astype(np.int32)
        us, vs, zs = [u], [v], [z]
        for rad in (1, 2, 3):
            sel = r >= rad
            if not sel.any():
                break
            ur, vr, zr = u[sel], v[sel], z[sel]
            for dv in range(-rad, rad + 1):
                for du in range(-rad, rad + 1):
                    if max(abs(du), abs(dv)) == rad:
                        us.append(ur + du)
                        vs.append(vr + dv)
                        zs.append(zr)
        u, v, z = np.concatenate(us), np.concatenate(vs), np.concatenate(zs)
        ok = (u >= 0) & (u < self.w) & (v >= 0) & (v < self.h)
        pix, z = v[ok] * self.w + u[ok], z[ok]
        order = np.argsort(z, kind="stable")
        pix, z = pix[order], z[order]
        first = np.unique(pix, return_index=True)[1]
        raw = np.zeros(self.w * self.h, np.float32)
        raw[pix[first]] = z[first]
        raw = raw.reshape(self.h, self.w)

        # Distinguish genuine open-sky / far rays (entire 3x3 decimated block has zero .ply hits and
        # the world ray does not intersect the floor z_ned=0 within 5.5 m) from 1-px splat cracks.
        pad0 = np.pad(raw, 1, mode="edge")
        max3 = np.max(np.stack([pad0[dy : dy + self.h, dx : dx + self.w] for dy in range(3) for dx in range(3)]), axis=0)
        d_world_z = self._ray_dirs_cam @ R_wc[2, :]  # world NED +z (downward) component per pixel
        z_end_ned = float(t_wc[2]) + 5.5 * d_world_z
        open_sky = (max3 == 0.0) & (z_end_ned < -0.08)

        a = self.args
        valid = raw > 0.0
        img = raw + valid * self.rng.normal(0.0, 1.0, raw.shape).astype(np.float32) * a.noise_k * raw**2
        pad = np.pad(np.where(valid, img, np.nan), 1, mode="edge")
        win = np.stack([pad[dy : dy + self.h, dx : dx + self.w] for dy in range(3) for dx in range(3)])
        with warnings.catch_warnings(), np.errstate(invalid="ignore"):
            warnings.simplefilter("ignore", RuntimeWarning)
            edge = (np.nanmax(win, 0) - np.nanmin(win, 0)) > a.edge_rel * img

        far = valid & ~edge & (img > a.max_depth)
        dropout_mask = self.rng.random(img.shape) < a.dropout
        bad = (~valid & ~open_sky) | edge | dropout_mask | (valid & (img < 0.35))

        out = img.astype(np.float32)
        out[open_sky | far] = np.float32(np.inf)
        out[bad] = np.float32(np.nan)
        return np.repeat(np.repeat(out, self.s, 0), self.s, 1)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "ply",
        nargs="?",
        type=Path,
        default=DEPLOY_ROOT / "dronehacks-course-scan-2026-11-06.ply",
        help="Aligned physical course scan (.ply)",
    )
    p.add_argument("--prior", type=Path, default=DEPLOY_ROOT / "config" / "arena_prior.bin")
    p.add_argument("--up", default="z", choices=("x", "y", "z", "-x", "-y", "-z"))
    p.add_argument("--auto_align", action="store_true")
    p.add_argument("--mirror", action="store_true")
    p.add_argument("--yaw", type=float, default=177.0)
    p.add_argument("--tx", type=float, default=-0.18)
    p.add_argument("--ty", type=float, default=-0.94)

    # OAK depth renderer parameters (matching spoof_px4_depth.py)
    p.add_argument("--depth_hz", type=float, default=15.0)
    p.add_argument("--max_depth", type=float, default=8.0)
    p.add_argument("--noise_k", type=float, default=0.004)
    p.add_argument("--edge_rel", type=float, default=0.08)
    p.add_argument("--dropout", type=float, default=0.02)
    p.add_argument("--seed", type=int, default=1)

    # Incremental VIO drift parameters (matching spoof_px4_depth.py)
    p.add_argument("--no_drift", action="store_true")
    p.add_argument("--yaw_drift_deg_s", type=float, default=0.25)
    p.add_argument("--yaw_rw_deg", type=float, default=0.35)
    p.add_argument("--scale", type=float, default=1.015)
    p.add_argument("--pos_rw", type=float, default=0.008)

    # Frame & lifecycle settings
    p.add_argument("--map_frame", default="map_ned")
    p.add_argument("--odom_frame", default="odom_ned")
    p.add_argument("--base_frame", default="base_link_frd")
    p.add_argument("--truth_flu_frame", default="truth_base_link")
    p.add_argument("--pad_hold_s", type=float, default=2.5, help="Exit 0 after holding near pad this long (0 = stay up)")
    p.add_argument("--max_duration_s", type=float, default=95.0, help="Safety wall-time cap in seconds (0 = unlimited)")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.ply.exists():
        raise FileNotFoundError(f"Course scan .ply not found: {args.ply}")

    pts, rgb, _tree, spacing, _prior = load_scene(args)
    rng = np.random.default_rng(args.seed)
    renderer = OpenSkyDepthRenderer(pts, spacing, args, rng)
    obs_pts = pts[(-pts[:, 2] >= 0.16) & (-pts[:, 2] <= 1.70)]
    obs_tree = cKDTree(obs_pts)

    import rclpy
    from geometry_msgs.msg import TransformStamped
    from nav_msgs.msg import Odometry
    from px4_msgs.msg import VehicleAttitude, VehicleLocalPosition
    from rclpy.executors import MultiThreadedExecutor
    from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
    from rclpy.node import Node
    from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
    from sensor_msgs.msg import CameraInfo, Image, PointCloud2, PointField
    from std_msgs.msg import Float32MultiArray, Header
    from tf2_ros import StaticTransformBroadcaster, TransformBroadcaster

    rclpy.init()
    node = Node("classical_scan_bridge")

    px4_qos = QoSProfile(
        reliability=ReliabilityPolicy.BEST_EFFORT,
        durability=DurabilityPolicy.TRANSIENT_LOCAL,
        history=HistoryPolicy.KEEP_LAST,
        depth=1,
    )
    latched_qos = QoSProfile(
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.TRANSIENT_LOCAL,
        history=HistoryPolicy.KEEP_LAST,
        depth=1,
    )

    pub_lpos = node.create_publisher(VehicleLocalPosition, "/fmu/out/vehicle_local_position_v1", px4_qos)
    pub_att = node.create_publisher(VehicleAttitude, "/fmu/out/vehicle_attitude", px4_qos)
    pub_img = node.create_publisher(Image, "/camera/depth/image_raw", qos_profile_sensor_data)
    pub_info = node.create_publisher(CameraInfo, "/camera/depth/camera_info", qos_profile_sensor_data)
    pub_scan = node.create_publisher(PointCloud2, "/spoof/scan", latched_qos)
    pub_local_odom = node.create_publisher(Odometry, "/odometry/local", qos_profile_sensor_data)
    pub_clr = node.create_publisher(Float32MultiArray, "/viz/scan_clearance", 10)

    static_tf = StaticTransformBroadcaster(node)
    dyn_tf = TransformBroadcaster(node)

    def make_tf(parent: str, child: str, Rm: np.ndarray, tv: np.ndarray | tuple[float, float, float], stamp_msg) -> TransformStamped:
        m = TransformStamped()
        m.header.stamp = stamp_msg
        m.header.frame_id = parent
        m.child_frame_id = child
        m.transform.translation.x = float(tv[0])
        m.transform.translation.y = float(tv[1])
        m.transform.translation.z = float(tv[2])
        qx, qy, qz, qw = Rotation.from_matrix(Rm).as_quat()
        m.transform.rotation.x = float(qx)
        m.transform.rotation.y = float(qy)
        m.transform.rotation.z = float(qz)
        m.transform.rotation.w = float(qw)
        return m

    zero_stamp = node.get_clock().now().to_msg()
    # Clean single-parent TF tree (no duplicate parents for camera_depth_optical!):
    #   map -> map_ned -> odom_ned -> base_link_frd
    #   map_ned -> truth_base_link (FLU)
    #   base_link (FLU) -> camera_depth_optical (used by isam2_depth_node; Rerun uses truth_base_link + static mount)
    static_tf.sendTransform(
        [
            make_tf("map", args.map_frame, R_ENU_NED, (0.0, 0.0, 0.0), zero_stamp),
            make_tf(args.map_frame, args.odom_frame, np.eye(3), (0.0, 0.0, 0.0), zero_stamp),
            make_tf("base_link", "camera_depth_optical", F_FLU @ R_FRD_OPT, F_FLU @ T_FRD_CAM, zero_stamp),
        ]
    )

    # Publish /spoof/scan once (latched) via zero-copy numpy buffer for colored .ply + ghost rescan visuals
    sel = (np.abs(pts[:, 0]) < 3.6) & (np.abs(pts[:, 1]) < 3.6) & (-pts[:, 2] < 3.0)
    idx = np.where(sel)[0]
    idx = idx[np.random.default_rng(3).choice(len(idx), min(250_000, len(idx)), replace=False)]
    c8 = (np.clip(rgb[idx], 0.0, 1.0) * 255.0).astype(np.uint32)
    packed = ((c8[:, 0] << 16) | (c8[:, 1] << 8) | c8[:, 2]).view(np.float32)
    cloud_arr = np.ascontiguousarray(
        np.concatenate([pts[idx].astype(np.float32), packed[:, None]], axis=1),
        dtype=np.float32,
    )
    scan_msg = PointCloud2()
    scan_msg.header = Header(stamp=zero_stamp, frame_id=args.map_frame)
    scan_msg.height = 1
    scan_msg.width = int(len(idx))
    scan_msg.fields = [
        PointField(name=n, offset=4 * k, datatype=PointField.FLOAT32, count=1)
        for k, n in enumerate(("x", "y", "z", "rgb"))
    ]
    scan_msg.is_bigendian = False
    scan_msg.point_step = 16
    scan_msg.row_step = 16 * scan_msg.width
    scan_msg.is_dense = True
    scan_msg.data = cloud_arr.tobytes()
    pub_scan.publish(scan_msg)

    # Prebuild CameraInfo template
    cam_info_template = CameraInfo()
    cam_info_template.width = CAM_W
    cam_info_template.height = CAM_H
    cam_info_template.distortion_model = "plumb_bob"
    cam_info_template.d = [0.0, 0.0, 0.0, 0.0, 0.0]
    cam_info_template.k = [CAM_F, 0.0, CAM_CX, 0.0, CAM_F, CAM_CY, 0.0, 0.0, 1.0]
    cam_info_template.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    cam_info_template.p = [CAM_F, 0.0, CAM_CX, 0.0, 0.0, CAM_F, CAM_CY, 0.0, 0.0, 0.0, 1.0, 0.0]

    # Mutable state
    drift_rng = np.random.default_rng(args.seed + 19)
    latest_truth: tuple[object, float, np.ndarray, np.ndarray] | None = None  # (stamp_msg, t_s, p_ned, R_wb)
    prev_t_s: float | None = None
    prev_p_truth: np.ndarray | None = None
    p_px4: np.ndarray | None = None
    yaw_err = 0.0
    has_isam2_odom = False
    latest_isam2_pos: np.ndarray | None = None
    isam2_errs: list[float] = []

    min_scan_clr = float("inf")
    in_contact = False
    contact_episodes = 0
    max_dist_from_launch = 0.0
    pad_dwell_s = 0.0
    t_wall_start = time.monotonic()
    last_log_wall = t_wall_start
    truth_tick = 0
    done_reason: str | None = None

    def publish_active_odom(msg: Odometry) -> None:
        out = Odometry()
        out.header.stamp = msg.header.stamp
        out.header.frame_id = args.odom_frame
        out.child_frame_id = args.base_frame
        out.pose = msg.pose
        out.twist = msg.twist
        pub_local_odom.publish(out)

        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        tf_m = TransformStamped()
        tf_m.header.stamp = msg.header.stamp
        tf_m.header.frame_id = args.odom_frame
        tf_m.child_frame_id = args.base_frame
        tf_m.transform.translation.x = p.x
        tf_m.transform.translation.y = p.y
        tf_m.transform.translation.z = p.z
        tf_m.transform.rotation = q
        dyn_tf.sendTransform(tf_m)

    def on_isam2_odom(msg: Odometry) -> None:
        nonlocal has_isam2_odom, latest_isam2_pos
        has_isam2_odom = True
        p = msg.pose.pose.position
        latest_isam2_pos = np.array([p.x, p.y, p.z], dtype=float)
        if latest_truth is not None:
            isam2_errs.append(float(np.linalg.norm(latest_isam2_pos - latest_truth[2])))
        publish_active_odom(msg)

    def on_truth_odom(msg: Odometry) -> None:
        nonlocal latest_truth, prev_t_s, prev_p_truth, p_px4, yaw_err
        nonlocal min_scan_clr, in_contact, contact_episodes, max_dist_from_launch, pad_dwell_s
        nonlocal last_log_wall, truth_tick, done_reason

        stamp_msg = msg.header.stamp
        t_s = float(stamp_msg.sec) + 1e-9 * float(stamp_msg.nanosec)
        if t_s <= 0.0:
            t_s = node.get_clock().now().nanoseconds * 1e-9

        pos_msg = msg.pose.pose.position
        ori_msg = msg.pose.pose.orientation
        p_ned = np.array([pos_msg.x, pos_msg.y, pos_msg.z], dtype=float)
        R_wb = Rotation.from_quat([ori_msg.x, ori_msg.y, ori_msg.z, ori_msg.w]).as_matrix()
        v_body = np.array([msg.twist.twist.linear.x, msg.twist.twist.linear.y, msg.twist.twist.linear.z], dtype=float)
        v_ned = R_wb @ v_body
        latest_truth = (stamp_msg, t_s, p_ned, R_wb)

        # Broadcast dynamic TF map_ned -> truth_base_link (FLU) for Rerun camera frustum + truth pose
        dyn_tf.sendTransform(make_tf(args.map_frame, args.truth_flu_frame, R_wb @ F_FLU, p_ned, stamp_msg))

        # Bootstrap /odometry/local before iSAM2 initializes on keyframe 0
        if not has_isam2_odom:
            publish_active_odom(msg)

        # Incremental VIO drift integration
        if p_px4 is None or prev_p_truth is None or prev_t_s is None:
            p_px4 = p_ned.copy()
            dt = 0.02
        else:
            dt = float(np.clip(t_s - prev_t_s, 1e-4, 0.2))
            dp = p_ned - prev_p_truth
            if args.no_drift:
                p_px4 = p_ned.copy()
            else:
                yaw_err += math.radians(args.yaw_drift_deg_s) * dt + drift_rng.normal(
                    0.0, math.radians(args.yaw_rw_deg) * math.sqrt(dt)
                )
                Rz = _rot_z(yaw_err)
                walk = drift_rng.normal(0.0, args.pos_rw * math.sqrt(dt), 3) * np.array([1.0, 1.0, 0.3])
                p_px4 = p_px4 + args.scale * (Rz @ dp) + walk

        prev_t_s = t_s
        prev_p_truth = p_ned.copy()

        Rz_now = np.eye(3) if args.no_drift else _rot_z(yaw_err)
        v_px4 = v_ned if args.no_drift else (args.scale * (Rz_now @ v_ned))
        R_px4 = R_wb if args.no_drift else (Rz_now @ R_wb)

        ts_us = int(round(t_s * 1e6))
        qx, qy, qz, qw = Rotation.from_matrix(R_px4).as_quat()
        att = VehicleAttitude()
        att.timestamp = ts_us
        att.timestamp_sample = ts_us
        att.q = [float(qw), float(qx), float(qy), float(qz)]
        pub_att.publish(att)

        lpos = VehicleLocalPosition()
        lpos.timestamp = ts_us
        lpos.timestamp_sample = ts_us
        lpos.xy_valid = True
        lpos.z_valid = True
        lpos.v_xy_valid = True
        lpos.v_z_valid = True
        lpos.x, lpos.y, lpos.z = float(p_px4[0]), float(p_px4[1]), float(p_px4[2])
        lpos.vx, lpos.vy, lpos.vz = float(v_px4[0]), float(v_px4[1]), float(v_px4[2])
        lpos.heading = float(math.atan2(R_px4[1, 0], R_px4[0, 0]))
        pub_lpos.publish(lpos)

        # Physical scan clearance & pad completion tracking
        d_scan = float(obs_tree.query(p_ned)[0])
        min_scan_clr = min(min_scan_clr, d_scan)
        if d_scan < 0.20:
            if not in_contact:
                in_contact = True
                contact_episodes += 1
        elif d_scan > 0.22:
            in_contact = False

        dist_launch = float(np.linalg.norm(p_ned[:2] - LAUNCH_XY))
        max_dist_from_launch = max(max_dist_from_launch, dist_launch)
        dist_pad_truth = float(np.linalg.norm(p_ned[:2] - PAD_CENTER_XY))
        dist_pad_est = (
            float(np.linalg.norm(latest_isam2_pos[:2] - PAD_CENTER_XY))
            if latest_isam2_pos is not None
            else dist_pad_truth
        )
        dist_pad = min(dist_pad_truth, dist_pad_est)
        if max_dist_from_launch > 4.0 and (dist_pad_truth <= 1.05 or dist_pad_est <= 0.85):
            pad_dwell_s += dt
        else:
            pad_dwell_s = 0.0

        truth_tick += 1
        if truth_tick % 5 == 0:
            clr_msg = Float32MultiArray()
            clr_msg.data = [float(d_scan), float(min_scan_clr), float(contact_episodes), float(dist_pad)]
            pub_clr.publish(clr_msg)

        wall_now = time.monotonic()
        elapsed = wall_now - t_wall_start
        if wall_now - last_log_wall >= 5.0:
            last_log_wall = wall_now
            isam_err = float(np.linalg.norm(latest_isam2_pos - p_ned)) if latest_isam2_pos is not None else float("nan")
            node.get_logger().info(
                f"t={elapsed:4.1f}s truth=({p_ned[0]:+.2f},{p_ned[1]:+.2f},{p_ned[2]:+.2f}) "
                f"scan_clr={d_scan:.2f}m (min={min_scan_clr:.2f}m, contacts={contact_episodes}) "
                f"isam2_err={isam_err:.3f}m pad_dist={dist_pad:.2f}m"
            )

        if args.pad_hold_s > 0.0 and pad_dwell_s >= args.pad_hold_s:
            done_reason = f"reached_pad (held {pad_dwell_s:.1f}s within 0.85m of pad)"
        elif args.max_duration_s > 0.0 and elapsed >= args.max_duration_s:
            done_reason = f"max_duration_s ({args.max_duration_s:.1f}s) reached"

    def on_depth_timer() -> None:
        if latest_truth is None:
            return
        stamp_at_pose, _t_s, p_ned, R_wb = latest_truth
        d_m = renderer.render_32f(p_ned, R_wb)

        # Stamp depth with the exact truth timestamp of the pose it was rendered at!
        # Since render_32f() takes ~15-25 ms, PX4 odometry at `stamp_at_pose` is already in
        # isam2_depth_node's ring buffer AND matches the rendered geometry with zero motion skew!
        hdr = Header(stamp=stamp_at_pose, frame_id="camera_depth_optical")
        cam_info_template.header = hdr
        pub_info.publish(cam_info_template)

        img = Image()
        img.header = hdr
        img.height = CAM_H
        img.width = CAM_W
        img.encoding = "32FC1"
        img.is_bigendian = 0
        img.step = CAM_W * 4
        img.data = np.ascontiguousarray(d_m, dtype=np.float32).tobytes()
        pub_img.publish(img)

    odom_cb_group = MutuallyExclusiveCallbackGroup()
    depth_cb_group = MutuallyExclusiveCallbackGroup()
    node.create_subscription(Odometry, "/truth/odometry", on_truth_odom, qos_profile_sensor_data, callback_group=odom_cb_group)
    node.create_subscription(Odometry, "/isam2_depth/odometry", on_isam2_odom, 50, callback_group=odom_cb_group)
    node.create_timer(1.0 / max(1.0, args.depth_hz), on_depth_timer, callback_group=depth_cb_group)

    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)

    node.get_logger().info(
        f"classical_scan_bridge ready (depth={args.depth_hz:.0f}Hz, drift={'off' if args.no_drift else 'on'})"
    )
    try:
        while rclpy.ok() and done_reason is None:
            executor.spin_once(timeout_sec=0.02)
    except KeyboardInterrupt:
        done_reason = "interrupted"
    finally:
        mean_err = float(np.mean(isam2_errs)) if isam2_errs else float("nan")
        print(
            f"[classical_scan_bridge] FINAL ({done_reason}): "
            f"max_travel={max_dist_from_launch:.2f}m, min_scan_clearance={min_scan_clr:.3f}m, "
            f"contacts={contact_episodes}, mean_isam2_pos_err={mean_err:.3f}m",
            flush=True,
        )
        executor.shutdown(timeout_sec=1.0)
        node.destroy_node()
        rclpy.try_shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
