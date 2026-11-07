#!/usr/bin/env python3
"""ROS vision odometry -> native PX4 external vision, on hardware or ROS /clock.

The capture timestamp stays the measurement timestamp. Middleware owns clock
transport; this node owns frame conversion, validity, source resets and health.
"""
import math

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from px4_msgs.msg import VehicleOdometry
from std_msgs.msg import Bool

from .frames import WORLDS,initial_reset_counter,odometry_to_px4,quat_xyzw_to_wxyz
from .ownership import PublisherGuard
from .ros_common import PX4_QOS,SENSOR_QOS,declare,fill_vehicle_odometry,now_us,stamp_of
from .validation import CaptureGate,sample_problem
from .vio_health import VioHealth


class EvBridge(Node):
    def __init__(self):
        super().__init__('px4_vision_bridge')
        self.world=str(declare(self,'world','flu'))
        if self.world not in WORLDS:raise ValueError('world must be enu or flu')
        heading=float(declare(self,'heading_enu_rad',math.nan))
        reference=str(declare(self,'heading_reference',''))
        self.heading=None if math.isnan(heading) else heading
        if self.heading is not None and (not math.isfinite(heading) or not reference.strip() or self.world!='flu'):
            raise ValueError('heading alignment needs a finite surveyed FLU-to-ENU reference and its description')
        self.required=set(declare(self,'required_fields',['position','orientation']))
        if not self.required or not self.required<={'position','orientation','velocity'}:
            raise ValueError('required_fields must name the position/orientation/velocity fields consumed by PX4')
        self.in_topic=str(declare(self,'odom_topic','/visual_slam/tracking/odometry'))
        self.out_topic=str(declare(self,'out_topic','/fmu/in/vehicle_visual_odometry'))
        self.velocity_in_body=bool(declare(self,'velocity_in_body',True))
        self.quality=int(declare(self,'quality',0))
        self.jump_m=float(declare(self,'jump_reset_m',1.))
        self.health_stale_s=float(declare(self,'health_stale_s',0.5))
        self.max_speed=float(declare(self,'max_speed_mps',5.))
        self.quiet_s=float(declare(self,'health_quiet_s',5.))
        if not all(math.isfinite(v) and v>0 for v in (self.jump_m,self.health_stale_s,self.max_speed,self.quiet_s)):
            raise ValueError('health limits must be positive finite numbers')
        self.gate=CaptureGate(float(declare(self,'max_sample_age_s',.3)),float(declare(self,'future_tol_s',.005)))
        self.health=self._health()
        self.floors={name:[float(v) for v in declare(self,name,[0.,0.,0.])]
                     for name in ('position_floor','orientation_floor','velocity_floor')}
        if any(len(v)!=3 or not all(math.isfinite(x) and x>=0 for x in v) for v in self.floors.values()):
            raise ValueError('variance floors need three finite nonnegative values')
        self.pub=self.create_publisher(VehicleOdometry,self.out_topic,PX4_QOS)
        self.ok_pub=self.create_publisher(Bool,str(declare(self,'health_topic','/estimator/healthy')),10)
        self.guard=PublisherGuard(self,self.out_topic) if self.out_topic.startswith('/fmu/in/') else None
        self.sub=self.create_subscription(Odometry,self.in_topic,self._on_odom,SENSOR_QOS)
        self.reset_counter=initial_reset_counter();self.last_pos=None;self.frame_id=None
        self.last_now=None;self.last_good=None;self.ok=False;self.reason='no samples'
        self.counts={'input':0,'published':0,'invalid':0,'withheld':0,'unowned':0,'clock_resets':0}
        self.create_timer(.2,self._publish_health)
        self.create_timer(5.,self._log)
        self.get_logger().info(f'{self.in_topic} -> {self.out_topic}; frame={self.world}; heading reference={reference or "unreferenced (FRD)"}')

    def _health(self):
        return VioHealth(max_speed_mps=self.max_speed,jump_m=self.jump_m,quiet_s=self.quiet_s)

    def _observe_clock(self,now):
        if self.last_now is not None and now<self.last_now:
            self.gate.reset();self.health=self._health();self.last_pos=None;self.frame_id=None
            self.reset_counter=(self.reset_counter+1)&255;self.last_good=None;self.ok=False
            self.reason='ROS clock restarted';self.counts['clock_resets']+=1
        self.last_now=now

    def _healthy_now(self):
        now=now_us(self);self._observe_clock(now)
        return bool(self.ok and self.last_good is not None and 0<=now-self.last_good<self.health_stale_s*1e6)

    def _publish_health(self):
        self.ok_pub.publish(Bool(data=self._healthy_now()))

    def _on_odom(self,msg):
        self.counts['input']+=1
        now=now_us(self);self._observe_clock(now);capture=stamp_of(msg.header)
        accepted,reason,_=self.gate.check(capture,now)
        if not accepted:
            if reason not in ('duplicate','reordered'):self.ok=False
            self.reason=reason;return
        p,q=msg.pose.pose.position,msg.pose.pose.orientation
        v,w=msg.twist.twist.linear,msg.twist.twist.angular
        pos=(p.x,p.y,p.z);quat=(q.x,q.y,q.z,q.w);vel=(v.x,v.y,v.z)
        problem=sample_problem(self.required,pos,quat,vel)
        if problem:
            self.ok=False;self.reason=problem;self.counts['invalid']+=1;return
        sane,reason=self.health.update(capture*1e-6,pos)
        if self.guard is not None and not self.guard.may_publish():
            self.ok=False;self.reason='proving sole vision ownership';self.counts['unowned']+=1;return
        self.ok,self.reason=sane,reason
        if not sane:self.counts['withheld']+=1;return
        changed=(self.frame_id is not None and self.frame_id!=msg.header.frame_id)
        jumped=self.last_pos is not None and math.dist(self.last_pos,pos)>self.jump_m
        if changed or jumped:self.reset_counter=(self.reset_counter+1)&255
        self.frame_id=msg.header.frame_id;self.last_pos=pos
        norm=math.sqrt(sum(x*x for x in quat));quat=tuple(x/norm for x in quat) if norm>0 else quat
        value=odometry_to_px4(stamp_sample_us=capture,now_us=now,position=pos,
            q_wxyz=quat_xyzw_to_wxyz(*quat),world=self.world,heading_enu_rad=self.heading,
            velocity=vel,velocity_in_body=self.velocity_in_body,angular_velocity=(w.x,w.y,w.z),
            pose_cov=list(msg.pose.covariance),twist_cov=list(msg.twist.covariance),
            reset_counter=self.reset_counter,quality=self.quality,**self.floors)
        self.pub.publish(fill_vehicle_odometry(VehicleOdometry(),value))
        self.last_good=now;self.counts['published']+=1

    def _log(self):
        self.get_logger().info(f'counts={self.counts} captures={self.gate.counts} healthy={self._healthy_now()} reason={self.reason}')

    def destroy_node(self):
        guard=getattr(self,'guard',None)
        if guard is not None:guard.close()
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args);node=None
    try:
        node=EvBridge();rclpy.spin(node)
    except KeyboardInterrupt:pass
    finally:
        if node is not None:node.destroy_node()
        rclpy.shutdown()


if __name__=='__main__':main()
