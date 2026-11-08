#!/usr/bin/env python3
"""Generate the Gazebo models of OUR quad from PX4's x500, sized and weighted like the training plant.

    python3 sitl/models/make_dronehacks_quad.py [--px4_models ~/PX4-v1.17/Tools/simulation/gz/models]

Writes sitl/models/dronehacks_quad/ (airframe + PX4's sensors + motor plugins) and sitl/models/dronehacks_stereo/
(dronehacks_quad + OAK depth camera + stereo pair + IMU at the nose). Re-run after changing a number below.

Why: PX4's x500 reaches ~0.39 m from its centre (rotors 0.246 m out, 0.14 m props); the planner and MPPI plan for our
drone (arena_prior.json collision_radius_m 0.2), and the course's corridor_end pass has ~0.24 m clearance, so an x500
clips walls the real drone clears (2026-11-08 crash, docs/SITL.md).

From the training plant (dronehacks-sim/dronehacks/quad.py), so Gazebo, training and the planner describe one drone:
  mass 1.5 kg; inertia (0.012, 0.012, 0.020) kg m^2; rotors X at +-0.13 m on each body axis (0.184 m from centre);
  max thrust 10 N per rotor (2.7:1); motor time constant 25-35 ms (0.03 s used).
Kept from the x500 on purpose, and why:
  moment constant 0.016 (training: 0.07). With PX4's x500 tuning, 0.07 and half the yaw inertia would give ~9x the
    yaw response; yaw authority is not what the course tests.
  propeller: quad.py has no prop size. 5 in (0.127 m) props are assumed: reach = 0.184 + 0.064 = 0.25 m, which
    is ALREADY above the planner's 0.2 m / 0.24 m corridor_end pass. The real drone's measured reach decides it.
  everything else (sensors, landing gear height, drag coefficients, motor plugin type, ESC range 150-1000 rad/s).
"""
from __future__ import annotations

import argparse
import copy
import os
import xml.etree.ElementTree as ET
from pathlib import Path

HERE = Path(__file__).resolve().parent

X500_ROTOR = 0.174                 # x500 rotor offset on each axis, m
ARM = 0.13                         # ours (quad.py ARM)
S = ARM / X500_ROTOR               # horizontal scale of the airframe geometry
MASS = 1.5                         # kg, total
ROTOR_MASS = 0.010                 # kg each (prop + bell)
INERTIA = (0.012, 0.012, 0.020)    # kg m^2, total (rotor contribution is ~2e-4: folded into the base)
MAX_THRUST_N = 10.0                # per rotor
MAX_ROT_VEL = 1000.0               # rad/s at max thrust (same as the x500 / its ESC range)
MOTOR_CONSTANT = MAX_THRUST_N / MAX_ROT_VEL ** 2   # thrust = k w^2
MOTOR_TAU = 0.030                  # s, up and down
PROP_DIAMETER = 0.127              # m (5 in), assumed
X500_PROP_DIAMETER = 0.2792307692307692   # x500 prop collision box length (its 13 in mesh at 0.846 scale)
CAMERA_X = 0.12                    # m ahead of the centre, on the nose (base_link frame)


def fmt(v):
    return ' '.join(f'{x:.6g}' for x in v)


def nums(el):
    return [float(x) for x in el.text.split()]


def scale_pose_xy(el, s=S):
    v = nums(el)
    v[0] *= s
    v[1] *= s
    el.text = fmt(v)


def build_quad(px4_models: Path) -> str:
    base = ET.parse(px4_models / 'x500_base' / 'model.sdf').getroot()
    plugins = ET.parse(px4_models / 'x500' / 'model.sdf').getroot().find('model').findall('plugin')
    model = base.find('model')
    model.set('name', 'dronehacks_quad')
    for link in model.findall('link'):
        name = link.get('name')
        inertial = link.find('inertial')
        if name == 'base_link':
            inertial.find('mass').text = f'{MASS - 4 * ROTOR_MASS:.4f}'
            ine = inertial.find('inertia')
            for k, v in zip(('ixx', 'iyy', 'izz'), INERTIA):
                ine.find(k).text = f'{v:.6g}'
            for vis in link.findall('visual'):
                scale_pose_xy(vis.find('pose'))
                for sc in vis.iter('scale'):
                    sc.text = fmt([x * S for x in nums(sc)])
            for col in link.findall('collision'):
                scale_pose_xy(col.find('pose'))
                for size in col.iter('size'):
                    v = nums(size)
                    if len(v) == 3:                       # boxes: footprint shrinks, height kept
                        v[0] *= S
                        v[1] *= S
                    size.text = fmt(v)
        elif name.startswith('rotor_'):
            scale_pose_xy(link.find('pose'))
            inertial.find('mass').text = f'{ROTOR_MASS:.4f}'
            k = PROP_DIAMETER / X500_PROP_DIAMETER
            for vis in link.findall('visual'):
                uri = vis.find('.//uri')
                if uri is not None and 'prop' in uri.text:
                    vis.find('pose').text = fmt([x * k for x in nums(vis.find('pose'))])
                    sc = vis.find('.//scale')
                    sc.text = fmt([x * k for x in nums(sc)])
            for col in link.findall('collision'):
                for size in col.iter('size'):
                    v = nums(size)
                    size.text = fmt([v[0] * k, v[1] * k, v[2]])
    for p in plugins:
        p = copy.deepcopy(p)
        for tag, val in (('motorConstant', f'{MOTOR_CONSTANT:.6g}'), ('maxRotVelocity', f'{MAX_ROT_VEL:g}'),
                         ('timeConstantUp', f'{MOTOR_TAU:g}'), ('timeConstantDown', f'{MOTOR_TAU:g}')):
            el = p.find(tag)
            if el is not None:
                el.text = val
        model.append(p)
    ET.indent(base, '  ')
    head = ('<?xml version="1.0" encoding="UTF-8"?>\n<!-- GENERATED by sitl/models/make_dronehacks_quad.py from '
            'PX4 v1.17 x500_base + x500: our airframe (quad.py: 1.5 kg, rotors +-0.13 m, 10 N each). Do not edit. -->\n')
    return head + ET.tostring(base, encoding='unicode') + '\n'


STEREO = f'''<?xml version="1.0" encoding="UTF-8"?>
<!-- GENERATED by sitl/models/make_dronehacks_quad.py: dronehacks_quad + the OAK on its nose. Do not edit.
  depth  PX4's OakD-Lite model (depth 640x480 30 Hz, as x500_depth) at base_link ({CAMERA_X}, 0.03, 0): its depth sensor
         sits at ({CAMERA_X + 0.01233:.5f}, 0, 0.01878) in base_link (the OakD-Lite sensor offset is (.01233, -.03, .01878))
  stereo OAK-D S2-like mono pair (640x400 L8, 74.7 deg, 7.5 cm, 30 Hz, identical stamps) + 200 Hz IMU, as x500_stereo,
         stereo_link at base_link ({CAMERA_X}, 0, 0.02); eyes at y = +-0.0375
  sitl/run_native.sh publishes the matching static TF for MODEL=dronehacks_stereo. -->
<sdf version="1.9">
  <model name="dronehacks_stereo">
    <include merge="true"><uri>dronehacks_quad</uri></include>
    <include merge="true">
      <uri>model://OakD-Lite</uri>
      <pose relative_to="base_link">{CAMERA_X} 0.03 0 0 0 0</pose>
    </include>
    <joint name="CameraJoint" type="fixed">
      <parent>base_link</parent>
      <child>camera_link</child>
      <pose relative_to="base_link">{CAMERA_X} 0.03 0 0 0 0</pose>
    </joint>
    <link name="stereo_link">
      <pose relative_to="base_link">{CAMERA_X} 0 0.02 0 0 0</pose>
      <inertial><mass>0.01</mass><inertia><ixx>1e-6</ixx><iyy>1e-6</iyy><izz>1e-6</izz><ixy>0</ixy><ixz>0</ixz><iyz>0</iyz></inertia></inertial>
      <sensor name="stereo_left" type="camera">
        <pose>0 0.0375 0 0 0 0</pose>
        <camera><horizontal_fov>1.3038</horizontal_fov><image><width>640</width><height>400</height><format>L8</format></image>
          <clip><near>0.1</near><far>40</far></clip></camera>
        <always_on>1</always_on><update_rate>30</update_rate><topic>/stereo/left/image_gz</topic>
      </sensor>
      <sensor name="stereo_right" type="camera">
        <pose>0 -0.0375 0 0 0 0</pose>
        <camera><horizontal_fov>1.3038</horizontal_fov><image><width>640</width><height>400</height><format>L8</format></image>
          <clip><near>0.1</near><far>40</far></clip></camera>
        <always_on>1</always_on><update_rate>30</update_rate><topic>/stereo/right/image_gz</topic>
      </sensor>
      <sensor name="stereo_imu" type="imu">
        <always_on>1</always_on><update_rate>200</update_rate><topic>/stereo/imu_gz</topic>
        <imu>
          <angular_velocity>
            <x><noise type="gaussian"><mean>0</mean><stddev>0.0345</stddev></noise></x>
            <y><noise type="gaussian"><mean>0</mean><stddev>0.0345</stddev></noise></y>
            <z><noise type="gaussian"><mean>0</mean><stddev>0.0345</stddev></noise></z>
          </angular_velocity>
          <linear_acceleration>
            <x><noise type="gaussian"><mean>0</mean><stddev>0.263</stddev></noise></x>
            <y><noise type="gaussian"><mean>0</mean><stddev>0.263</stddev></noise></y>
            <z><noise type="gaussian"><mean>0</mean><stddev>0.263</stddev></noise></z>
          </linear_acceleration>
        </imu>
      </sensor>
    </link>
    <joint name="stereo_joint" type="fixed"><parent>base_link</parent><child>stereo_link</child></joint>
  </model>
</sdf>
'''


def config(name, desc):
    return (f'<?xml version="1.0"?>\n<model>\n  <name>{name}</name>\n  <version>1.0</version>\n'
            f'  <sdf version="1.9">model.sdf</sdf>\n  <description>{desc}</description>\n</model>\n')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--px4_models', type=Path,
                    default=Path(os.path.expanduser('~/PX4-v1.17/Tools/simulation/gz/models')))
    args = ap.parse_args()
    for name, sdf, desc in (
            ('dronehacks_quad', build_quad(args.px4_models),
             'Our quad (training plant: 1.5 kg, rotors +-0.13 m, 10 N each) on PX4 x500 sensors/plugins.'),
            ('dronehacks_stereo', STEREO, 'dronehacks_quad + OAK depth + stereo pair + IMU at the nose.')):
        d = HERE / name
        d.mkdir(exist_ok=True)
        (d / 'model.sdf').write_text(sdf)
        (d / 'model.config').write_text(config(name, desc))
        print(f'wrote {d}/model.sdf')
    print(f'airframe scale {S:.4f}; motorConstant {MOTOR_CONSTANT:.3g}; hover thrust fraction '
          f'{MASS * 9.81 / (4 * MAX_THRUST_N):.3f}; reach {ARM * 2 ** 0.5 + PROP_DIAMETER / 2:.3f} m')


if __name__ == '__main__':
    main()
