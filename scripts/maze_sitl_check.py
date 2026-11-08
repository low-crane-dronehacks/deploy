#!/usr/bin/env python3
"""Stage-0 check + overlay for the maze stack on the native sim (sitl/run_native.sh + launch/maze_autonomy.launch.py).

    sitl/run_native.sh up                                   # EV_SOURCE=truth_odom recommended (see below)
    ros2 launch launch/maze_autonomy.launch.py hardware:=sitl arm:=false rviz:=true rviz_config:=sitl/maze_sitl.rviz
    /usr/bin/python3 scripts/maze_sitl_check.py             # report every 5 s + RViz overlay (Ctrl-C to stop)
    /usr/bin/python3 scripts/maze_sitl_check.py --once 20   # one report after 20 s, exit 0 = all checks pass

Checks (stage 0, arm:=false):
  odometry   /odometry/local from the PX4 adapter, rate
  plan       /plan/global_rrt received (poses, length), /plan/local_reference rate
  MPPI       /control/trajectory_setpoint rate (~50 Hz) and its velocity
  PX4 input  /fmu/in/{trajectory_setpoint, offboard_control_mode, vehicle_command} SILENT (arm:=false forwards nothing)
  PX4 state  disarmed
  frames     TF map_ned -> base_link_frd (where the stack thinks the drone is) vs Gazebo's truth
             (/sitl/truth_odometry, NED in the Gazebo/course frame = map_ned): < 0.15 m and < 5 deg
Overlay (frame map_ned, NED like the stack): /maze_viz/truth_path (Gazebo truth), /maze_viz/course (course mesh).
Frames: the stack's static map_ned -> odom_ned is the launch mark with yaw 0, i.e. PX4's origin at the launch mark and a
true-north heading. That holds for run_native.sh EV_SOURCE=truth_odom or cuvslam with the default spawn and heading;
NOT for EV_SOURCE=truth (PX4 origin = Gazebo origin) or SKYBETA=1 (PX4 frame rotated: map_odom_yaw:=-1.5708).
Publishes only /maze_viz/*.
"""
from __future__ import annotations

import argparse
import collections
import math
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

HERE = Path(__file__).resolve().parent


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--once', type=float, default=0.0, help='report once after this many seconds and exit')
    ap.add_argument('--course', default=str(HERE.parent / 'sitl' / 'world' / 'course.obj'))
    ap.add_argument('--no_sim_time', action='store_true')
    args = ap.parse_args(argv)

    import rclpy
    from geometry_msgs.msg import Point, PoseStamped, TwistStamped
    from nav_msgs.msg import Odometry, Path as PathMsg
    from px4_msgs.msg import OffboardControlMode, TrajectorySetpoint, VehicleCommand, VehicleOdometry, VehicleStatus
    from rclpy.node import Node
    from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
    from std_msgs.msg import ColorRGBA, Header
    from tf2_ros import Buffer, TransformListener
    from visualization_msgs.msg import Marker

    px4_qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                         history=HistoryPolicy.KEEP_LAST, depth=1)
    any_qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, durability=DurabilityPolicy.VOLATILE,
                         history=HistoryPolicy.KEEP_LAST, depth=10)
    latched = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                         history=HistoryPolicy.KEEP_LAST, depth=1)
    px4_out = lambda n, t: '/fmu/out/' + n + (f'_v{t.MESSAGE_VERSION}' if getattr(t, 'MESSAGE_VERSION', 0) else '')
    rclpy.init(args=['maze_sitl_check', '--ros-args', '-p', f'use_sim_time:={str(not args.no_sim_time).lower()}'])
    node = Node('maze_sitl_check')
    tf_buf = Buffer()
    TransformListener(tf_buf, node)
    arr = collections.defaultdict(lambda: collections.deque(maxlen=400))
    st = {'plan': None, 'sp': None, 'truth': None, 'armed': None, 'nav': None, 'course': False}
    truth_path = PathMsg(header=Header(frame_id='map_ned'))
    pub_truth = node.create_publisher(PathMsg, '/maze_viz/truth_path', latched)
    pub_course = node.create_publisher(Marker, '/maze_viz/course', latched)

    def hit(k):
        arr[k].append(time.monotonic())

    def rate(k):
        q = arr.get(k)
        if not q or len(q) < 2 or time.monotonic() - q[-1] > 2.0:
            return 0.0
        return (len(q) - 1) / max(q[-1] - q[0], 1e-6)

    def on_plan(m):
        hit('plan')
        st['plan'] = m

    def on_sp(m):
        hit('sp')
        st['sp'] = m

    def on_truth(m):
        hit('truth')
        if not all(math.isfinite(v) for v in m.position):
            return
        q = m.q
        st['truth'] = (np.array(m.position), Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix())
        added = arr['truth_add']
        if not added or time.monotonic() - added[-1] > 0.1:                 # path point every 0.1 s
            hit('truth_add')
            ps = PoseStamped()
            ps.header.frame_id = 'map_ned'
            ps.header.stamp = node.get_clock().now().to_msg()
            ps.pose.position.x, ps.pose.position.y, ps.pose.position.z = (float(v) for v in m.position)
            ps.pose.orientation.w, ps.pose.orientation.x, ps.pose.orientation.y, ps.pose.orientation.z = (float(v) for v in q)
            truth_path.poses.append(ps)
            del truth_path.poses[:-4000]
            truth_path.header.stamp = ps.header.stamp
            pub_truth.publish(truth_path)

    def on_status(m):
        st['armed'] = m.arming_state == VehicleStatus.ARMING_STATE_ARMED
        st['nav'] = m.nav_state

    node.create_subscription(Odometry, '/odometry/local', lambda m: hit('odom'), qos_profile_sensor_data)
    node.create_subscription(PathMsg, '/plan/global_rrt', on_plan, latched)
    node.create_subscription(PathMsg, '/plan/local_reference', lambda m: hit('local_ref'), 10)
    node.create_subscription(TrajectorySetpoint, '/control/trajectory_setpoint', on_sp, px4_qos)
    node.create_subscription(TwistStamped, '/control/cmd_vel', lambda m: hit('cmd_vel'), 10)
    for t, typ in (('/fmu/in/trajectory_setpoint', TrajectorySetpoint),
                   ('/fmu/in/offboard_control_mode', OffboardControlMode), ('/fmu/in/vehicle_command', VehicleCommand)):
        node.create_subscription(typ, t, lambda m, t=t: hit(t), any_qos)
    node.create_subscription(VehicleOdometry, '/sitl/truth_odometry', on_truth, px4_qos)
    node.create_subscription(VehicleStatus, px4_out('vehicle_status', VehicleStatus), on_status, px4_qos)

    def publish_course():
        if st['course'] or not Path(args.course).exists():
            return
        v, f = [], []
        for line in open(args.course):
            if line.startswith('v '):
                v.append([float(x) for x in line.split()[1:4]])
            elif line.startswith('f '):
                idx = [int(t.split('/')[0]) for t in line.split()[1:]]
                f += [(idx[0], idx[k], idx[k + 1]) for k in range(1, len(idx) - 1)]
        enu = np.asarray(v)[np.asarray(f) - 1].reshape(-1, 3)
        ned = np.stack([enu[:, 1], enu[:, 0], -enu[:, 2]], 1)          # course.obj is ENU; map_ned is NED
        mk = Marker(header=Header(frame_id='map_ned'), ns='course', id=0, type=Marker.TRIANGLE_LIST)
        mk.pose.orientation.w = 1.0
        mk.scale.x = mk.scale.y = mk.scale.z = 1.0
        mk.color = ColorRGBA(r=0.75, g=0.75, b=0.8, a=0.3)
        mk.points = [Point(x=float(a), y=float(b), z=float(c)) for a, b, c in ned]
        pub_course.publish(mk)
        st['course'] = True

    def report():
        rows, ok = [], True

        def check(name, good, text):
            nonlocal ok
            ok &= bool(good)
            rows.append(f"  {'PASS' if good else 'FAIL'}  {name:10s} {text}")

        check('odometry', rate('odom') > 20, f"/odometry/local {rate('odom'):.0f} Hz")
        p = st['plan']
        if p is not None and len(p.poses) > 1:
            xy = np.array([[q.pose.position.x, q.pose.position.y, q.pose.position.z] for q in p.poses])
            check('plan', True, f"/plan/global_rrt {len(p.poses)} poses, {np.sum(np.linalg.norm(np.diff(xy, axis=0), axis=1)):.1f} m, "
                                f"frame {p.header.frame_id}; local reference {rate('local_ref'):.0f} Hz")
        else:
            check('plan', False, '/plan/global_rrt not received')
        sp = st['sp']
        vtxt = '' if sp is None else '  v = (' + ', '.join(f'{c:+.2f}' for c in sp.velocity) + ') NED'
        check('MPPI', rate('sp') > 30, f"/control/trajectory_setpoint {rate('sp'):.0f} Hz{vtxt}")
        loud = {t: rate(t) for t in ('/fmu/in/trajectory_setpoint', '/fmu/in/offboard_control_mode', '/fmu/in/vehicle_command')}
        check('PX4 input', all(r == 0 for r in loud.values()),
              'silent' if all(r == 0 for r in loud.values()) else 'PUBLISHING: ' + ', '.join(f'{t} {r:.0f} Hz' for t, r in loud.items() if r))
        check('PX4 state', st['armed'] is False, f"armed={st['armed']} nav_state={st['nav']}")
        try:
            t = tf_buf.lookup_transform('map_ned', 'base_link_frd', rclpy.time.Time())
            pm = np.array([t.transform.translation.x, t.transform.translation.y, t.transform.translation.z])
            qr = t.transform.rotation
            Rm = Rotation.from_quat([qr.x, qr.y, qr.z, qr.w]).as_matrix()
            if st['truth'] is not None:
                pt, Rt = st['truth']
                dyaw = math.degrees(abs(math.remainder(math.atan2(Rm[1, 0], Rm[0, 0]) - math.atan2(Rt[1, 0], Rt[0, 0]),
                                                       2 * math.pi)))
                e = np.linalg.norm(pm - pt)
                hint = ('  <-- ~90 deg: a SKYBETA=1 session (PX4 frame = odom read as ENU); restart run_native.sh '
                        'without SKYBETA, or launch with map_odom_yaw:=-1.5708' if abs(dyaw - 90) < 10 else '')
                check('frames', e < 0.15 and dyaw < 5, f"stack {np.round(pm, 2)} vs truth {np.round(pt, 2)} NED: "
                                                       f"{e:.2f} m, {dyaw:.1f} deg{hint}")
            else:
                check('frames', True, f"stack thinks {np.round(pm, 2)} NED (no /sitl/truth_odometry to compare)")
        except Exception as ex:
            check('frames', False, f"no TF map_ned -> base_link_frd ({str(ex).splitlines()[0][:70]})")
        print(time.strftime('%H:%M:%S') + ('  ALL PASS' if ok else '  some checks FAIL'))
        print('\n'.join(rows), flush=True)
        return ok

    t0, last = time.monotonic(), time.monotonic()
    result = True
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.1)
            publish_course()
            now = time.monotonic()
            if args.once > 0 and now - t0 >= args.once:
                result = report()
                break
            if args.once <= 0 and now - last >= 5.0:
                last = now
                report()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
    return 0 if result else 1


if __name__ == '__main__':
    sys.exit(main())
