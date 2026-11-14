#!/usr/bin/env python3
"""Unmodelled-obstacle demo for the maze stack in RViz (sim_loopback; no Gazebo, no PX4, nothing armed).

    # terminal 1 FIRST (the simulated drone starts flying as soon as the stack is up): the unmodelled objects
    /usr/bin/python3 scripts/unmodeled_obstacles.py
    # terminal 2: the stack against the simulated vehicle
    ros2 launch launch/maze_autonomy.launch.py sim_loopback:=true rviz:=true rviz_config:=config/maze_unmodeled.rviz

Point cloud A = the designed course (/viz/groundtruth_map_cloud, published by astar_planner_node): the ONLY map A*
plans on. Point cloud B = A + objects the design does not have (poles, crates). A* never sees B; MPPI does, through
its live obstacle input (obstacle_cloud_topic, default /oak/points/clamped -> VoxelCostmap dynamic voxels, the path a
real depth camera's points take). This node fakes that camera: each object's surface points that lie within the
camera's range and field of view are published in the body frame at 10 Hz, so objects appear as the drone approaches
(and decay 4 s after they leave the view, as in the costmap).

out /viz/cloud_b               PointCloud2 (map_ned, latched): designed walls + unmodelled objects, all red
    /viz/unmodeled_objects     MarkerArray (map_ned, latched): the objects as solid shapes
    /oak/points/clamped        PointCloud2 (base_link_frd): what the fake camera sees of the objects -> MPPI
    /viz/unmodeled_status      Marker: the drone's clearance to the nearest object, min so far, contacts
in  /odometry/local            the simulated vehicle (sim_loopback: odom == map)
Edit OBJECTS below to move / add objects. Contact = drone centre within REACH (0.25 m, rotors + 5 in props) of an
object surface.
"""
from __future__ import annotations

import math
import struct
from pathlib import Path

import numpy as np
import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import ColorRGBA, Header
from visualization_msgs.msg import Marker, MarkerArray

ROOT = Path(__file__).resolve().parents[1]
# (kind, north, east, size, height): pole = radius (m); crate = square side (m); standing on the floor.
# Each one dead on the default A* path (cruise 0.5 m/s, A* defaults), where the designed corridor leaves >= 0.55 m to
# the walls either side.
OBJECTS = [
    ('pole', 2.05, 1.05, 0.06, 1.60),      # leaving the maze towards gate A
    ('pole', 0.72, 2.30, 0.06, 1.60),      # after gate A, towards the tunnel
    ('crate', -0.08, 2.29, 0.15, 1.20),    # tunnel exit side (0.15 m: ~0.31 m clear either side in the 1.44 m tube)
    ('crate', -1.44, -1.36, 0.30, 1.00),   # run-in to the pad
]
REACH = 0.25            # m, drone centre to prop tip (sitl/models/make_dronehacks_quad.py)
CAM_RANGE = 3.5         # m, the costmap's disparity clamp
CAM_HFOV_DEG = 80.0     # OAK-D S2 mono ~ 75 deg; 0 = all around
FLOOR, TOP = 0.15, 1.65  # heights the costmap keeps


def surface_points(kind, n, e, size, height, step=0.02):
    hs = np.arange(FLOOR, min(height, TOP) + 1e-9, step)
    if kind == 'pole':
        a = np.linspace(0, 2 * np.pi, max(8, int(2 * np.pi * size / step)), endpoint=False)
        ring = np.stack([n + size * np.cos(a), e + size * np.sin(a)], 1)
        pts = [np.column_stack([ring, np.full(len(ring), -h)]) for h in hs]
    else:
        s = size / 2
        u = np.arange(-s, s + 1e-9, step)
        edge = np.concatenate([np.column_stack([u, np.full_like(u, -s)]), np.column_stack([u, np.full_like(u, s)]),
                               np.column_stack([np.full_like(u, -s), u]), np.column_stack([np.full_like(u, s), u])])
        pts = [np.column_stack([edge + [n, e], np.full(len(edge), -h)]) for h in hs]
        uu, vv = np.meshgrid(u, u)
        pts.append(np.column_stack([uu.ravel() + n, vv.ravel() + e, np.full(uu.size, -min(height, TOP))]))
    return np.concatenate(pts)


def designed_walls(step=0.06):
    """Wall cells of the designed course (assets/arena_prior.bin distance grid) in the flight band, for cloud B."""
    b = open(ROOT / 'assets' / 'arena_prior.bin', 'rb').read()
    _, _, nx, ny, nz, res, lx, ly, lz = struct.unpack('<4sIiiiffff', b[:36])
    D = np.frombuffer(b[36:], np.uint8).reshape(nx, ny, nz) / 200.0
    i, j, k = np.nonzero(D < 0.6 * res)
    p = np.column_stack([lx + i * res, ly + j * res, lz + k * res])
    h = -p[:, 2]
    p = p[(h > 0.1) & (h < 1.8) & (np.abs(p[:, 0]) < 3.0) & (np.abs(p[:, 1]) < 3.0)]
    key = np.unique(np.floor(p / step).astype(np.int64), axis=0)
    return (key + 0.5) * step


def cloud(points, frame, stamp, rgb=None):
    fields = [PointField(name=c, offset=4 * i, datatype=PointField.FLOAT32, count=1) for i, c in enumerate('xyz')]
    data = points.astype(np.float32)
    if rgb is not None:
        fields.append(PointField(name='rgb', offset=12, datatype=PointField.FLOAT32, count=1))
        c = (np.clip(rgb, 0, 1) * 255).astype(np.uint32)
        data = np.column_stack([data, ((c[:, 0] << 16) | (c[:, 1] << 8) | c[:, 2]).view(np.float32)])
    step = data.shape[1] * 4
    return PointCloud2(header=Header(frame_id=frame, stamp=stamp), height=1, width=len(points), fields=fields,
                       is_bigendian=False, point_step=step, row_step=step * len(points), is_dense=True,
                       data=np.ascontiguousarray(data).tobytes())


class Unmodeled(Node):
    def __init__(self):
        super().__init__('unmodeled_obstacles')
        self.map_frame = self.declare_parameter('map_frame', 'map_ned').value
        self.body_frame = self.declare_parameter('body_frame', 'base_link_frd').value
        self.obj = np.concatenate([surface_points(*o) for o in OBJECTS])
        self.tree = cKDTree(self.obj)
        latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.pub_b = self.create_publisher(PointCloud2, '/viz/cloud_b', latched)
        self.pub_mk = self.create_publisher(MarkerArray, '/viz/unmodeled_objects', latched)
        self.pub_cam = self.create_publisher(PointCloud2, '/oak/points/clamped', qos_profile_sensor_data)
        self.pub_status = self.create_publisher(Marker, '/viz/unmodeled_status', 1)
        walls = designed_walls()
        stamp = self.get_clock().now().to_msg()
        self.pub_b.publish(cloud(np.concatenate([walls, self.obj]), self.map_frame, stamp,
                                 np.tile([0.9, 0.12, 0.1], (len(walls) + len(self.obj), 1))))
        self.pub_mk.publish(self._markers(stamp))
        self.create_subscription(Odometry, '/odometry/local', self._odom, qos_profile_sensor_data)
        self.create_timer(0.1, self._camera)
        self.pose = None
        self.min_clear, self.contacts, self.in_contact = math.inf, 0, False
        self.get_logger().info(f'{len(OBJECTS)} unmodelled objects ({len(self.obj)} surface points); cloud B = '
                               f'{len(walls)} designed wall points + objects. A* does not see them; MPPI does via '
                               f'/oak/points/clamped (camera range {CAM_RANGE} m, HFOV {CAM_HFOV_DEG:.0f} deg).')

    def _markers(self, stamp):
        arr = MarkerArray()
        for i, (kind, n, e, size, height) in enumerate(OBJECTS):
            m = Marker(header=Header(frame_id=self.map_frame, stamp=stamp), ns='unmodeled', id=i, action=Marker.ADD,
                       type=Marker.CYLINDER if kind == 'pole' else Marker.CUBE)
            m.pose.position.x, m.pose.position.y, m.pose.position.z = n, e, -height / 2
            m.pose.orientation.w = 1.0
            m.scale.x = m.scale.y = 2 * size if kind == 'pole' else size
            m.scale.z = height
            m.color = ColorRGBA(r=0.9, g=0.15, b=0.1, a=0.85)
            arr.markers.append(m)
        return arr

    def _odom(self, m: Odometry):
        p, q = m.pose.pose.position, m.pose.pose.orientation
        self.pose = (np.array([p.x, p.y, p.z]), Rotation.from_quat([q.x, q.y, q.z, q.w]))
        d = float(self.tree.query(self.pose[0])[0])
        self.min_clear = min(self.min_clear, d)
        if d < REACH and not self.in_contact:
            self.contacts += 1
            self.get_logger().warn(f'CONTACT #{self.contacts}: {d:.3f} m from an object at NED ({p.x:+.2f}, {p.y:+.2f})')
            self.in_contact = True
        elif d > REACH + 0.02:            # hysteresis: one contact per approach, not one per message at 0.25 m
            self.in_contact = False
        mk = Marker(header=Header(frame_id=self.map_frame, stamp=m.header.stamp), ns='status', id=0,
                    type=Marker.TEXT_VIEW_FACING, action=Marker.ADD)
        mk.pose.position.x, mk.pose.position.y, mk.pose.position.z = p.x, p.y, p.z - 0.45
        mk.pose.orientation.w = 1.0
        mk.scale.z = 0.14
        mk.color = ColorRGBA(r=1.0, g=0.25 if self.in_contact else 1.0, b=0.25 if self.in_contact else 1.0, a=1.0)
        mk.text = f'nearest unmodelled object {d:.2f} m (min {self.min_clear:.2f})  contacts {self.contacts}'
        self.pub_status.publish(mk)

    def _camera(self):
        if self.pose is None:
            return
        pos, R = self.pose
        body = R.inv().apply(self.obj - pos)                    # map/odom NED -> body FRD
        rng = np.linalg.norm(body, axis=1)
        keep = (rng < CAM_RANGE) & (rng > 0.05)
        if CAM_HFOV_DEG > 0:
            keep &= (body[:, 0] > 0) & (np.abs(np.arctan2(body[:, 1], body[:, 0])) < math.radians(CAM_HFOV_DEG / 2))
        self.pub_cam.publish(cloud(body[keep], self.body_frame, self.get_clock().now().to_msg()))


def main():
    rclpy.init()
    node = Unmodeled()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info(f'final: nearest approach to an unmodelled object {node.min_clear:.3f} m, '
                               f'contacts {node.contacts}')
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
