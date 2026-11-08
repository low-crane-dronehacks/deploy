#!/usr/bin/env python3
"""Spoof PX4 + an OAK depth camera from a course scan, to run isam2_depth_node without PX4, Gazebo or a drone.

Nothing here talks to a real vehicle: it publishes fake PX4 *output* topics (/fmu/out/...) and never a command.

    # terminal 1 (or use spoof_isam2.launch.py, which starts all three):
    ros2 run isam2_depth isam2_depth_node --ros-args -p use_sim_time:=true
    # terminal 2:
    /usr/bin/python3 src/test/spoof_px4_depth.py dronehacks-course-scan-2026-11-06.ply
    # no ROS: plan the route + render sample depth frames to a PNG and exit
    /usr/bin/python3 src/test/spoof_px4_depth.py scan.ply --plan_png /tmp/route.png

What it does:
  scene   the .ply, aligned to the course NED frame with view_course_scan.py's transform (--yaw/--tx/--ty;
          --auto_align recomputes it).
  route   A* through the scan's free space at flight height, visiting the launch mark, the gates in order and
          the pad (goals snapped to the clearest spot nearby; unreachable ones are skipped), shortcut + smoothed,
          flown at --speed with lateral-acceleration and acceleration limits, heading along the path, tilt from
          the acceleration (as a multirotor would). Starts with a hover so the graph can initialise.
  depth   z-buffer splat of the scan into the 640x400 OAK camera (rendered at 160x100 and upsampled, 16UC1 mm,
          0 = invalid), stereo-like noise sigma = k z^2, invalid occluding edges, random dropouts, range limits.
          TF base_link (FLU) -> camera optical frame, 10 cm ahead (same as the C++ synthetic test).
  PX4     vehicle_local_position_v1 + vehicle_attitude from the truth with VIO-like drift: yaw bias + random
          walk, scale error, position random walk (tilt stays true: gravity keeps it observable).
  clock   /clock (sim time) paced at --rtf x real time; the node must run with use_sim_time:=true.
  score   compares /isam2_depth/odometry and the spoofed PX4 estimate with the truth at the same stamps (live
          every 5 s, summary at the end, --csv for the rows) and checks the node's obstacle voxels against the scan.
  rviz    /spoof/scan (RGB cloud), /spoof/{truth,px4,isam2}_path, TF map(ENU) -> isam2_world_ned -> base_link.
"""
from __future__ import annotations

import argparse
import heapq
import json
import math
import sys
import threading
import time
import warnings
from pathlib import Path

import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter1d
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

DEPLOY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(DEPLOY_ROOT / 'ego_planner' / 'scripts'))
import view_course_scan as vcs  # noqa: E402  (load_ply, to_ned, align, apply, Prior)

G = 9.81
# OAK-D S2-like camera (as src/test/test_depth_isam2.cpp) and its pose on the body
CAM_W, CAM_H, CAM_F, CAM_CX, CAM_CY = 640, 400, 421.0, 319.5, 199.5
R_FRD_OPT = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=float)   # optical (x right, y down, z fwd) -> FRD
T_FRD_CAM = np.array([0.10, 0.0, 0.0])
F_FLU = np.diag([1.0, -1.0, -1.0])                                      # FRD <-> FLU
R_ENU_NED = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]], dtype=float)


# ----------------------------------------------------------------------------------------------- scene + route
def load_scene(args):
    xyz, rgb = vcs.load_ply(args.ply)
    p = vcs.to_ned(xyz, args.up)
    prior = vcs.Prior(args.prior, args.prior.with_suffix('.json'))
    if args.auto_align:
        best = vcs.align(p, prior, (0.3, 1.5), 'no')
        _, yaw, t = best[False]
        mirror = False
    else:
        mirror, yaw, t = args.mirror, args.yaw, np.array([args.tx, args.ty])
    print(f'scan aligned: mirror={mirror} yaw={yaw:.1f} deg t=({t[0]:.2f}, {t[1]:.2f}) m')
    q = vcs.apply(p, mirror, yaw, t)
    keep = (np.abs(q[:, 0]) < 7) & (np.abs(q[:, 1]) < 7) & (-q[:, 2] < 4.5)
    q, rgb = q[keep], (rgb[keep] if rgb is not None else np.full((keep.sum(), 3), 0.7))
    tree = cKDTree(q)
    sample = q[np.random.default_rng(0).choice(len(q), min(20000, len(q)), replace=False)]
    spacing = float(np.median(tree.query(sample, k=2)[0][:, 1]))
    print(f'scene: {len(q)} points, median spacing {100 * spacing:.1f} cm')
    return q, rgb, tree, spacing, prior


def clearance_map(pts, half, res, band, net_margin):
    n = int(round(2 * half / res)) + 1
    h = -pts[:, 2]
    sel = (h >= band[0]) & (h <= band[1]) & (np.abs(pts[:, 0]) <= half) & (np.abs(pts[:, 1]) <= half)
    ij = np.rint((pts[sel, :2] + half) / res).astype(int)
    hits = np.zeros((n, n), np.int32)
    np.add.at(hits, (ij[:, 0], ij[:, 1]), 1)
    occ = hits >= 2
    axis = -half + res * np.arange(n)
    nn, ee = np.meshgrid(axis, axis, indexing='ij')
    occ |= np.maximum(np.abs(nn), np.abs(ee)) >= half - net_margin       # the net
    return distance_transform_edt(~occ) * res, axis


def astar(clr, start, goal, r_min):
    n = clr.shape[0]
    if clr[start] < r_min or clr[goal] < r_min:
        return None
    nbr = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
           (-1, -1, 1.414), (-1, 1, 1.414), (1, -1, 1.414), (1, 1, 1.414)]
    g = {start: 0.0}
    parent = {start: None}
    pq = [(0.0, start)]
    while pq:
        _, c = heapq.heappop(pq)
        if c == goal:
            path = []
            while c is not None:
                path.append(c)
                c = parent[c]
            return path[::-1]
        for di, dj, w in nbr:
            m = (c[0] + di, c[1] + dj)
            if not (0 <= m[0] < n and 0 <= m[1] < n) or clr[m] < r_min:
                continue
            cost = g[c] + w * (1.0 + 1.5 * max(0.0, 0.6 - clr[m]))       # prefer the middle of corridors
            if cost < g.get(m, 1e18):
                g[m] = cost
                parent[m] = c
                heapq.heappush(pq, (cost + math.hypot(goal[0] - m[0], goal[1] - m[1]), m))
    return None


def plan_route(pts, prior, args):
    res, half = 0.05, prior.meta['arena']['arena_half_m']
    clr, axis = clearance_map(pts, half, res, (args.alt - 0.5, args.alt + 0.5), 0.1)
    cell = lambda p: tuple(int(round((v + half) / res)) for v in p)
    pos = lambda c: np.array([axis[c[0]], axis[c[1]]])

    def snap(p, radius=0.8):
        c = cell(p)
        r = int(radius / res)
        i0, j0 = max(c[0] - r, 0), max(c[1] - r, 0)
        win = clr[i0:c[0] + r + 1, j0:c[1] + r + 1]
        k = np.unravel_index(int(np.argmax(win)), win.shape)
        return (i0 + k[0], j0 + k[1])

    lm = prior.meta['landmarks']
    goals = [('launch', lm['launch_mark_ned'][:2])]
    goals += [(g['name'], g['center_ned'][:2]) for g in sorted(prior.meta['gates'], key=lambda g: g['index'])]
    goals += [('pad', lm['pad_center_ned'][:2])]
    cells = []
    for name, p in goals:
        c = snap(np.array(p))
        print(f'  goal {name:13s} ({p[0]:+.2f}, {p[1]:+.2f}) -> ({pos(c)[0]:+.2f}, {pos(c)[1]:+.2f}), '
              f'clearance {clr[c]:.2f} m')
        cells.append((name, c))
    path, cur = [cells[0][1]], cells[0][1]
    for name, c in cells[1:]:
        seg = None
        for r_min in (args.clearance, 0.75 * args.clearance):
            seg = astar(clr, cur, c, r_min)
            if seg:
                break
        if not seg:
            print(f'  WARNING: no path to {name} with >= {0.75 * args.clearance:.2f} m clearance; skipped')
            continue
        path += seg[1:]
        cur = c
    xy = np.array([pos(c) for c in path])

    def free_line(a, b, r_min):
        n = max(2, int(np.linalg.norm(b - a) / 0.025))
        for s in np.linspace(0, 1, n):
            c = cell(a + s * (b - a))
            if clr[c] < r_min:
                return False
        return True

    short, i = [xy[0]], 0                                                   # line-of-sight shortcut
    while i < len(xy) - 1:
        j = len(xy) - 1
        while j > i + 1 and not free_line(xy[i], xy[j], args.clearance):
            j -= 1
        short.append(xy[j])
        i = j
    poly = np.array(short)
    for _ in range(4):                                                      # Chaikin corner cutting
        q = [poly[0]]
        for a, b in zip(poly[:-1], poly[1:]):
            q += [0.75 * a + 0.25 * b, 0.25 * a + 0.75 * b]
        q.append(poly[-1])
        poly = np.array(q)
    seg = np.linalg.norm(np.diff(poly, axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(seg)])
    ss = np.arange(0, s[-1], 0.02)
    route = np.stack([np.interp(ss, s, poly[:, 0]), np.interp(ss, s, poly[:, 1])], 1)
    c_along = np.array([clr[cell(p)] for p in route])
    print(f'route: {s[-1]:.1f} m, min clearance to the scan {c_along.min():.2f} m '
          f'(drone radius {prior.meta["arena"]["collision_radius_m"]} m)')
    return route, clr, axis


def trajectory(route, args, dt):
    """Time-parametrise the 2D route -> truth arrays at dt: t, p (NED), v, R (world NED <- body FRD)."""
    d = np.linalg.norm(np.diff(route, axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(d)])
    t1 = np.gradient(route, s, axis=0)
    t2 = np.gradient(t1, s, axis=0)
    kappa = np.abs(t1[:, 0] * t2[:, 1] - t1[:, 1] * t2[:, 0])
    kappa = gaussian_filter1d(kappa, 5)
    v = np.minimum(args.speed, np.sqrt(args.lat_acc / np.maximum(kappa, 1e-6)))
    v[0] = v[-1] = 0.0
    for i in range(1, len(v)):
        v[i] = min(v[i], math.sqrt(v[i - 1] ** 2 + 2 * args.acc * d[i - 1]))
    for i in range(len(v) - 2, -1, -1):
        v[i] = min(v[i], math.sqrt(v[i + 1] ** 2 + 2 * args.acc * d[i]))
    tt = np.concatenate([[0], np.cumsum(2 * d / np.maximum(v[:-1] + v[1:], 0.02))])
    hover0, hover1 = args.hover_s, 2.0
    t = np.arange(0, hover0 + tt[-1] + hover1, dt)
    tau = np.clip(t - hover0, 0, tt[-1])
    n, e = np.interp(tau, tt, route[:, 0]), np.interp(tau, tt, route[:, 1])
    alt = args.alt + args.alt_wiggle * np.sin(2 * np.pi * np.clip(t - hover0, 0, None) / 7.0)
    p = np.stack([n, e, -alt], 1)
    p = gaussian_filter1d(p, 0.15 / dt, axis=0, mode='nearest')
    vel = np.gradient(p, dt, axis=0)
    acc = gaussian_filter1d(np.gradient(vel, dt, axis=0), 0.1 / dt, axis=0, mode='nearest')
    head = np.unwrap(np.arctan2(t1[:, 1], t1[:, 0]))
    psi = gaussian_filter1d(np.interp(tau, tt, head), 0.4 / dt, mode='nearest')
    f = acc - np.array([0, 0, G])                       # specific force; body z (down) = -f / |f|
    zb = -f / np.linalg.norm(f, axis=1, keepdims=True)
    xc = np.stack([np.cos(psi), np.sin(psi), np.zeros_like(psi)], 1)
    yb = np.cross(zb, xc)
    yb /= np.linalg.norm(yb, axis=1, keepdims=True)
    xb = np.cross(yb, zb)
    R = np.stack([xb, yb, zb], 2)                      # columns = body axes in NED
    print(f'trajectory: {t[-1]:.1f} s (hover {hover0:.0f} s), peak speed {np.linalg.norm(vel, axis=1).max():.2f} '
          f'm/s, peak tilt {np.degrees(np.arccos(np.clip(zb[:, 2], -1, 1))).max():.1f} deg')
    return t, p, vel, R


def drift(t, p, vel, R, args, rng):
    """PX4-like estimate: yaw bias + random walk and scale error applied to the motion, position random walk."""
    dt = t[1] - t[0]
    yaw_err = np.cumsum(np.full(len(t), math.radians(args.yaw_drift_deg_s) * dt)
                        + rng.normal(0, math.radians(args.yaw_rw_deg) * math.sqrt(dt), len(t)))
    yaw_err -= yaw_err[0]
    c, s = np.cos(yaw_err), np.sin(yaw_err)
    Rz = np.zeros((len(t), 3, 3))
    Rz[:, 0, 0], Rz[:, 0, 1], Rz[:, 1, 0], Rz[:, 1, 1], Rz[:, 2, 2] = c, -s, s, c, 1
    dp = np.diff(p, axis=0, prepend=p[:1])
    walk = np.cumsum(rng.normal(0, args.pos_rw * math.sqrt(dt), (len(t), 3)), axis=0) * np.array([1, 1, 0.3])
    p_e = p[0] + np.cumsum(np.einsum('nij,nj->ni', Rz, args.scale * dp), axis=0) + walk
    v_e = np.einsum('nij,nj->ni', Rz, args.scale * vel)
    R_e = Rz @ R
    return p_e, v_e, R_e, yaw_err


# ----------------------------------------------------------------------------------------------- depth render
class Renderer:
    def __init__(self, pts, spacing, args, rng):
        self.pts, self.spacing, self.args, self.rng = pts, spacing, args, rng
        self.s = 4
        self.w, self.h = CAM_W // self.s, CAM_H // self.s
        self.f = CAM_F / self.s
        self.cx, self.cy = (CAM_CX - 1.5) / self.s, (CAM_CY - 1.5) / self.s

    def render(self, p_wb, R_wb):
        """Depth (m, 640x400 float32, 0 = invalid) seen from body pose (world NED <- body FRD)."""
        R_wc = R_wb @ R_FRD_OPT
        t_wc = p_wb + R_wb @ T_FRD_CAM
        pc = (self.pts - t_wc) @ R_wc                   # = R_wc^T (p - t)
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
                        us.append(ur + du); vs.append(vr + dv); zs.append(zr)
        u, v, z = np.concatenate(us), np.concatenate(vs), np.concatenate(zs)
        ok = (u >= 0) & (u < self.w) & (v >= 0) & (v < self.h)
        pix, z = v[ok] * self.w + u[ok], z[ok]
        order = np.argsort(z, kind='stable')
        pix, z = pix[order], z[order]
        first = np.unique(pix, return_index=True)[1]
        img = np.zeros(self.w * self.h, np.float32)
        img[pix[first]] = z[first]
        img = img.reshape(self.h, self.w)
        # stereo-like artefacts
        a = self.args
        valid = img > 0
        img = img + valid * self.rng.normal(0, 1, img.shape).astype(np.float32) * a.noise_k * img ** 2
        pad = np.pad(np.where(valid, img, np.nan), 1, mode='edge')
        win = np.stack([pad[dy:dy + self.h, dx:dx + self.w] for dy in range(3) for dx in range(3)])
        with warnings.catch_warnings(), np.errstate(invalid='ignore'):
            warnings.simplefilter('ignore', RuntimeWarning)
            edge = (np.nanmax(win, 0) - np.nanmin(win, 0)) > a.edge_rel * img
        bad = edge | (self.rng.random(img.shape) < a.dropout) | (img < 0.35) | (img > a.max_depth)
        img[bad] = 0.0
        return np.repeat(np.repeat(img, self.s, 0), self.s, 1)


def save_plan_png(path, pts, rgb, clr, axis, route, traj, renderer):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    t, p, _, R = traj
    fig = plt.figure(figsize=(18, 8), constrained_layout=True)
    gs = fig.add_gridspec(2, 4)
    ax = fig.add_subplot(gs[:, :2])
    ext = [axis[0], axis[-1], axis[0], axis[-1]]
    ax.imshow(np.minimum(clr, 1.0), origin='lower', extent=[ext[2], ext[3], ext[0], ext[1]], cmap='Greys_r',
              alpha=0.6)
    h = -pts[:, 2]
    sel = (h > 0.3) & (h < 1.5) & (np.abs(pts[:, :2]).max(1) < 3.3)
    sel = np.where(sel)[0][::3]
    ax.scatter(pts[sel, 1], pts[sel, 0], s=0.2, c=rgb[sel], lw=0, rasterized=True)
    ax.plot(route[:, 1], route[:, 0], 'tab:blue', lw=2, label='route')
    ax.plot(route[0, 1], route[0, 0], 'go', label='start')
    ax.plot(route[-1, 1], route[-1, 0], 'rs', label='end')
    ax.set_aspect('equal')
    ax.set_xlabel('east [m]')
    ax.set_ylabel('north [m]')
    ax.legend()
    ax.set_title('route through the scan (grey = clearance at flight height)')
    for k, frac in enumerate(np.linspace(0.05, 0.9, 4)):
        i = int(frac * (len(t) - 1))
        d = renderer.render(p[i], R[i])
        a2 = fig.add_subplot(gs[k // 2, 2 + k % 2])
        a2.imshow(np.where(d > 0, d, np.nan), cmap='turbo', vmin=0, vmax=6)
        a2.set_title(f't = {t[i]:.1f} s, ({p[i, 0]:+.1f}, {p[i, 1]:+.1f}), {100 * np.mean(d > 0):.0f}% valid')
        a2.axis('off')
        ax.plot(p[i, 1], p[i, 0], 'k^')
        ax.annotate(f'{t[i]:.0f}s', (p[i, 1], p[i, 0]), xytext=(4, 4), textcoords='offset points')
    fig.savefig(path, dpi=110)
    print(f'wrote {path}')


# ----------------------------------------------------------------------------------------------- ROS
def yaw_of(R):
    return math.atan2(R[1, 0], R[0, 0])


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def run_ros(args, pts, rgb, tree, traj, est, renderer):
    import rclpy
    from geometry_msgs.msg import PoseStamped, TransformStamped
    from nav_msgs.msg import Odometry, Path as PathMsg
    from px4_msgs.msg import VehicleAttitude, VehicleLocalPosition
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.node import Node
    from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
    from rosgraph_msgs.msg import Clock
    from sensor_msgs.msg import CameraInfo, Image, PointCloud2, PointField
    from sensor_msgs_py import point_cloud2
    from std_msgs.msg import Header
    from tf2_ros import StaticTransformBroadcaster, TransformBroadcaster

    t_arr, p, vel, R = traj
    p_e, v_e, R_e, _ = est
    dt = t_arr[1] - t_arr[0]
    world = 'isam2_world_ned'

    rclpy.init()
    node = Node('spoof_px4_depth')
    px4_qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                         history=HistoryPolicy.KEEP_LAST, depth=1)
    latched = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                         history=HistoryPolicy.KEEP_LAST, depth=1)
    pub_clock = node.create_publisher(Clock, '/clock', 10)
    pub_lpos = node.create_publisher(VehicleLocalPosition, '/fmu/out/vehicle_local_position_v1', px4_qos)
    pub_att = node.create_publisher(VehicleAttitude, '/fmu/out/vehicle_attitude', px4_qos)
    pub_img = node.create_publisher(Image, '/camera/depth/image_raw', qos_profile_sensor_data)
    pub_info = node.create_publisher(CameraInfo, '/camera/depth/camera_info', qos_profile_sensor_data)
    pub_scan = node.create_publisher(PointCloud2, '/spoof/scan', latched)
    pub_paths = {k: node.create_publisher(PathMsg, f'/spoof/{k}_path', latched) for k in ('truth', 'px4', 'isam2')}
    static_tf, tf = StaticTransformBroadcaster(node), TransformBroadcaster(node)

    def stamp(ts):
        from builtin_interfaces.msg import Time
        sec = int(ts)
        return Time(sec=sec, nanosec=int(round((ts - sec) * 1e9)) % 1_000_000_000)

    def tfmsg(parent, child, Rm, tv, ts):
        m = TransformStamped()
        m.header.stamp, m.header.frame_id, m.child_frame_id = stamp(ts), parent, child
        m.transform.translation.x, m.transform.translation.y, m.transform.translation.z = map(float, tv)
        x, y, z, w = Rotation.from_matrix(Rm).as_quat()
        m.transform.rotation.x, m.transform.rotation.y, m.transform.rotation.z, m.transform.rotation.w = x, y, z, w
        return m

    def pose(ts, pv, Rm):
        ps = PoseStamped()
        ps.header.stamp, ps.header.frame_id = stamp(ts), world
        ps.pose.position.x, ps.pose.position.y, ps.pose.position.z = map(float, pv)
        x, y, z, w = Rotation.from_matrix(Rm).as_quat()
        ps.pose.orientation.x, ps.pose.orientation.y, ps.pose.orientation.z, ps.pose.orientation.w = x, y, z, w
        return ps

    # static TF: RViz-friendly ENU map above the NED world; camera on base_link (FLU) as the node expects
    static_tf.sendTransform([
        tfmsg('map', world, R_ENU_NED, (0, 0, 0), 0.0),
        tfmsg('base_link', 'camera_depth_optical', F_FLU @ R_FRD_OPT, F_FLU @ T_FRD_CAM, 0.0)])

    sel = (np.abs(pts[:, 0]) < 3.6) & (np.abs(pts[:, 1]) < 3.6) & (-pts[:, 2] < 3.0)
    idx = np.where(sel)[0]
    idx = idx[np.random.default_rng(3).choice(len(idx), min(250_000, len(idx)), replace=False)]
    c8 = (np.clip(rgb[idx], 0, 1) * 255).astype(np.uint32)
    packed = ((c8[:, 0] << 16) | (c8[:, 1] << 8) | c8[:, 2]).view(np.float32)
    fields = [PointField(name=n, offset=4 * k, datatype=PointField.FLOAT32, count=1)
              for k, n in enumerate(('x', 'y', 'z', 'rgb'))]
    cloud = np.concatenate([pts[idx].astype(np.float32), packed[:, None]], 1)
    pub_scan.publish(point_cloud2.create_cloud(Header(frame_id=world), fields, cloud.tolist()))
    truth_path = PathMsg(header=Header(frame_id=world))
    truth_path.poses = [pose(t_arr[i], p[i], R[i]) for i in range(0, len(t_arr), 10)]
    pub_paths['truth'].publish(truth_path)

    info = CameraInfo()
    info.header.frame_id = 'camera_depth_optical'
    info.width, info.height, info.distortion_model = CAM_W, CAM_H, 'plumb_bob'
    info.k = [CAM_F, 0.0, CAM_CX, 0.0, CAM_F, CAM_CY, 0.0, 0.0, 1.0]
    info.p = [CAM_F, 0.0, CAM_CX, 0.0, 0.0, CAM_F, CAM_CY, 0.0, 0.0, 0.0, 1.0, 0.0]
    info.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    info.d = [0.0] * 5

    rows, obstacle_cloud, lock = [], [None], threading.Lock()
    isam_path = PathMsg(header=Header(frame_id=world))
    px4_path = PathMsg(header=Header(frame_id=world))

    def interp_truth(ts):
        i = int(np.clip(round(ts / dt), 0, len(t_arr) - 1))
        return i

    def on_odom(m: Odometry):
        ts = m.header.stamp.sec + 1e-9 * m.header.stamp.nanosec
        i = interp_truth(ts)
        q = m.pose.pose.orientation
        Rm = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
        pv = np.array([m.pose.pose.position.x, m.pose.pose.position.y, m.pose.pose.position.z])
        with lock:
            rows.append((ts, *p[i], yaw_of(R[i]), *p_e[i], yaw_of(R_e[i]), *pv, yaw_of(Rm)))
            if len(rows) % 5 == 0:
                isam_path.poses.append(pose(ts, pv, Rm))

    def on_obst(m: PointCloud2):
        obstacle_cloud[0] = m

    node.create_subscription(Odometry, '/isam2_depth/odometry', on_odom, 50)
    node.create_subscription(PointCloud2, '/isam2_depth/obstacles', on_obst, 1)
    exe = SingleThreadedExecutor()
    exe.add_node(node)
    spin = threading.Thread(target=exe.spin, daemon=True)
    spin.start()

    def report(final=False):
        with lock:
            a = np.array(rows) if rows else None
        n_pub = node.count_publishers('/isam2_depth/odometry')
        if n_pub > 1:
            print(f'  WARNING: {n_pub} nodes publish /isam2_depth/odometry (a leftover isam2_depth_node?); '
                  'the scores mix their outputs')
        if a is None or not len(a):
            print(f'  t={t_now:6.1f} s  no iSAM2 odometry yet (is isam2_depth_node running with use_sim_time:=true?)')
            return
        e_px4 = np.linalg.norm(a[:, 5:8] - a[:, 1:4], axis=1)
        e_isam = np.linalg.norm(a[:, 9:12] - a[:, 1:4], axis=1)
        y_px4 = np.degrees(np.abs([wrap(x) for x in a[:, 8] - a[:, 4]]))
        y_isam = np.degrees(np.abs([wrap(x) for x in a[:, 12] - a[:, 4]]))
        if not final:
            print(f'  t={t_now:6.1f} s  error now: iSAM2 {e_isam[-1]:.3f} m / {y_isam[-1]:.1f} deg   '
                  f'PX4 {e_px4[-1]:.3f} m / {y_px4[-1]:.1f} deg')
            return
        rms = lambda x: float(np.sqrt(np.mean(np.square(x))))
        print('\nsummary over %d iSAM2 odometry messages:' % len(a))
        print(f'  position  iSAM2 rms {rms(e_isam):.3f}  max {e_isam.max():.3f}  final {e_isam[-1]:.3f} m')
        print(f'            PX4   rms {rms(e_px4):.3f}  max {e_px4.max():.3f}  final {e_px4[-1]:.3f} m')
        print(f'  yaw       iSAM2 rms {rms(y_isam):.2f}  max {y_isam.max():.2f}  final {y_isam[-1]:.2f} deg')
        print(f'            PX4   rms {rms(y_px4):.2f}  max {y_px4.max():.2f}  final {y_px4[-1]:.2f} deg')
        if args.csv:
            np.savetxt(args.csv, a, delimiter=',', fmt='%.5f', comments='',
                       header='t,truth_n,truth_e,truth_d,truth_yaw,px4_n,px4_e,px4_d,px4_yaw,'
                              'isam2_n,isam2_e,isam2_d,isam2_yaw')
            print(f'  rows written to {args.csv}')
        oc = obstacle_cloud[0]
        if oc is not None:
            xyz = point_cloud2.read_points_numpy(oc, field_names=('x', 'y', 'z'))
            if len(xyz):
                d = tree.query(xyz.astype(np.float64))[0]
                print(f'  obstacle voxels: {len(xyz)}, distance to the scan median {np.median(d):.3f} m, '
                      f'{100 * np.mean(d < 0.1):.0f}% within 10 cm (voxel {0.05:.2f} m)')

    print(f'spoofing {t_arr[-1]:.0f} s of sim time at {args.rtf}x real time; Ctrl-C to stop')
    depth_period, px4_every = 1.0 / args.depth_hz, max(1, int(round(1.0 / (args.px4_hz * dt))))
    next_depth, next_report = 0.0, 5.0
    wall0 = time.monotonic()
    t_now = 0.0
    try:
        for i, t_now in enumerate(t_arr):
            target = wall0 + t_now / args.rtf
            lag = target - time.monotonic()
            if lag > 0:
                time.sleep(lag)
            elif lag < -0.05:            # rendering fell behind: slow sim time down rather than bursting
                wall0 -= lag
            pub_clock.publish(Clock(clock=stamp(t_now)))
            if i % px4_every == 0:
                time.sleep(0.0005)      # let /clock land first: the node stamps PX4 samples on arrival
                us = int(t_now * 1e6)
                att = VehicleAttitude()
                att.timestamp = att.timestamp_sample = us
                x, y, z, w = Rotation.from_matrix(R_e[i]).as_quat()
                att.q = [float(w), float(x), float(y), float(z)]
                pub_att.publish(att)
                lp = VehicleLocalPosition()
                lp.timestamp = lp.timestamp_sample = us
                lp.xy_valid = lp.z_valid = lp.v_xy_valid = lp.v_z_valid = True
                lp.x, lp.y, lp.z = map(float, p_e[i])
                lp.vx, lp.vy, lp.vz = map(float, v_e[i])
                lp.heading = float(yaw_of(R_e[i]))
                pub_lpos.publish(lp)
                tf.sendTransform(tfmsg(world, 'base_link', R[i] @ F_FLU, p[i], t_now))
                if i % (px4_every * 10) == 0:
                    px4_path.poses.append(pose(t_now, p_e[i], R_e[i]))
            if t_now >= next_depth:
                next_depth += depth_period
                d = renderer.render(p[i], R[i])
                img = Image()
                img.header.stamp, img.header.frame_id = stamp(t_now), 'camera_depth_optical'
                img.height, img.width, img.encoding, img.is_bigendian = CAM_H, CAM_W, '16UC1', 0
                img.step = CAM_W * 2
                img.data = np.clip(np.rint(d * 1000), 0, 65535).astype('<u2').tobytes()
                info.header.stamp = img.header.stamp
                pub_info.publish(info)
                pub_img.publish(img)
                with lock:
                    pub_paths['px4'].publish(px4_path)
                    pub_paths['isam2'].publish(isam_path)
            if t_now >= next_report:
                next_report += 5.0
                report()
        time.sleep(1.0)
        report(final=True)
    except KeyboardInterrupt:
        report(final=True)
    exe.shutdown()
    node.destroy_node()
    rclpy.try_shutdown()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('ply', type=Path)
    ap.add_argument('--prior', type=Path, default=DEPLOY_ROOT / 'config' / 'arena_prior.bin')
    ap.add_argument('--up', choices=('z', 'y'), default='z')
    ap.add_argument('--mirror', action='store_true', help='scan alignment (view_course_scan.py output)')
    ap.add_argument('--yaw', type=float, default=177.0)
    ap.add_argument('--tx', type=float, default=-0.18)
    ap.add_argument('--ty', type=float, default=-0.94)
    ap.add_argument('--auto_align', action='store_true', help='recompute the alignment instead (~10 s)')
    g = ap.add_argument_group('route')
    g.add_argument('--alt', type=float, default=0.8, help='flight height [m]')
    g.add_argument('--alt_wiggle', type=float, default=0.1, help='height oscillation amplitude [m]')
    g.add_argument('--speed', type=float, default=0.5)
    g.add_argument('--acc', type=float, default=0.5)
    g.add_argument('--lat_acc', type=float, default=0.6)
    g.add_argument('--clearance', type=float, default=0.3, help='min distance to scan walls [m]')
    g.add_argument('--hover_s', type=float, default=3.0)
    g = ap.add_argument_group('fake PX4 drift')
    g.add_argument('--yaw_drift_deg_s', type=float, default=0.5)
    g.add_argument('--yaw_rw_deg', type=float, default=0.3, help='yaw random walk [deg/sqrt(s)]')
    g.add_argument('--scale', type=float, default=1.04)
    g.add_argument('--pos_rw', type=float, default=0.01, help='position random walk [m/sqrt(s)]')
    g.add_argument('--no_drift', action='store_true', help='PX4 estimate = truth')
    g = ap.add_argument_group('depth')
    g.add_argument('--noise_k', type=float, default=0.003, help='sigma = k z^2 [1/m]')
    g.add_argument('--edge_rel', type=float, default=0.15, help='invalidate jumps > this x depth in 3x3')
    g.add_argument('--dropout', type=float, default=0.02)
    g.add_argument('--max_depth', type=float, default=10.0)
    g = ap.add_argument_group('run')
    g.add_argument('--rtf', type=float, default=1.0, help='sim time / wall time')
    g.add_argument('--depth_hz', type=float, default=15.0)
    g.add_argument('--px4_hz', type=float, default=50.0)
    g.add_argument('--seed', type=int, default=0)
    g.add_argument('--csv', type=Path, help='write truth / PX4 / iSAM2 rows here at the end')
    g.add_argument('--plan_png', type=Path, help='no ROS: save the route + sample depth frames and exit')
    args = ap.parse_args(argv)
    if args.no_drift:
        args.yaw_drift_deg_s = args.yaw_rw_deg = args.pos_rw = 0.0
        args.scale = 1.0

    rng = np.random.default_rng(args.seed)
    pts, rgb, tree, spacing, prior = load_scene(args)
    route, clr, axis = plan_route(pts, prior, args)
    dt = 0.01
    traj = trajectory(route, args, dt)
    est = drift(*traj, args, rng)
    e = np.linalg.norm(est[0] - traj[1], axis=1)
    print(f'fake PX4 drift: final {e[-1]:.2f} m / {math.degrees(est[3][-1]):.1f} deg yaw '
          f'(bias {args.yaw_drift_deg_s} deg/s, scale {args.scale})')
    renderer = Renderer(pts, spacing, args, rng)
    if args.plan_png:
        save_plan_png(args.plan_png, pts, rgb, clr, axis, route, traj, renderer)
        return 0
    run_ros(args, pts, rgb, tree, traj, est, renderer)
    return 0


if __name__ == '__main__':
    sys.exit(main())
