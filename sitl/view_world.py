#!/usr/bin/env python3
"""Look at the Gazebo world (sitl/world) without Gazebo: course mesh, net shell, floor, spawn pose. VTK only.

    /usr/bin/python3 sitl/view_world.py                                   # interactive window
    /usr/bin/python3 sitl/view_world.py --scan assets/dronehacks-course-scan-2026-11-06.ply   # + the as-built scan
    /usr/bin/python3 sitl/view_world.py --png /tmp/world.png              # render to a file instead

Mouse: left = rotate, middle / shift+left = pan, wheel = zoom; 'r' resets the view, 'w'/'s' wireframe/surface, q quits.
Frames: Gazebo's ENU (x east, y north, z up), as in course.obj / dronehacks.sdf / spawn.txt.
--scan overlays a phone scan aligned with view_course_scan.py's transform (--yaw/--tx/--ty, NED course frame),
converted to ENU, so the designed world (grey) and the built course (points) can be compared directly.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import vtk
from vtk.util import numpy_support

HERE = Path(__file__).resolve().parent
WORLD = HERE / 'world'


def obj_actor(path: Path, color, opacity=1.0, wire=False):
    r = vtk.vtkOBJReader()
    r.SetFileName(str(path))
    r.Update()
    m = vtk.vtkPolyDataMapper()
    m.SetInputConnection(r.GetOutputPort())
    a = vtk.vtkActor()
    a.SetMapper(m)
    a.GetProperty().SetColor(*color)
    a.GetProperty().SetOpacity(opacity)
    if wire:
        a.GetProperty().SetRepresentationToWireframe()
    return a, r.GetOutput().GetNumberOfPoints(), r.GetOutput().GetNumberOfCells()


def floor_actor(size=8.0):
    p = vtk.vtkPlaneSource()
    p.SetOrigin(-size / 2, -size / 2, 0)
    p.SetPoint1(size / 2, -size / 2, 0)
    p.SetPoint2(-size / 2, size / 2, 0)
    p.SetResolution(int(size), int(size))
    m = vtk.vtkPolyDataMapper()
    m.SetInputConnection(p.GetOutputPort())
    a = vtk.vtkActor()
    a.SetMapper(m)
    a.GetProperty().SetColor(0.55, 0.5, 0.45)            # dronehacks.sdf ground colour
    a.GetProperty().EdgeVisibilityOn()                   # 1 m grid lines
    a.GetProperty().SetEdgeColor(0.4, 0.37, 0.33)
    return a


def spawn_actor():
    line = next(l for l in (WORLD / 'spawn.txt').read_text().splitlines() if l.strip() and not l.startswith('#'))
    x, y, z, _, _, yaw = (float(v) for v in line.split(','))
    arrow = vtk.vtkArrowSource()
    t = vtk.vtkTransform()
    t.Translate(x, y, z + 0.05)
    t.RotateZ(math.degrees(yaw))
    t.Scale(0.5, 0.5, 0.5)
    f = vtk.vtkTransformPolyDataFilter()
    f.SetTransform(t)
    f.SetInputConnection(arrow.GetOutputPort())
    m = vtk.vtkPolyDataMapper()
    m.SetInputConnection(f.GetOutputPort())
    a = vtk.vtkActor()
    a.SetMapper(m)
    a.GetProperty().SetColor(0.6, 0.2, 0.8)
    return a, (x, y, z, yaw)


def scan_actor(args):
    sys.path.insert(0, str(HERE.parent / 'ego_planner' / 'scripts'))
    import view_course_scan as vcs                       # load_ply / to_ned / apply
    xyz, rgb = vcs.load_ply(args.scan)
    q = vcs.apply(vcs.to_ned(xyz, 'z'), args.mirror, args.yaw, np.array([args.tx, args.ty]))
    keep = (np.abs(q[:, 0]) < 4) & (np.abs(q[:, 1]) < 4) & (-q[:, 2] < 3.0)   # arena + margin, below the net top
    q = q[keep]
    rgb = rgb[keep] if rgb is not None else np.full((len(q), 3), 0.2)
    if len(q) > args.max_points:
        sel = np.random.default_rng(0).choice(len(q), args.max_points, replace=False)
        q, rgb = q[sel], rgb[sel]
    enu = np.stack([q[:, 1], q[:, 0], -q[:, 2]], 1)     # NED -> ENU
    pts = vtk.vtkPoints()
    pts.SetData(numpy_support.numpy_to_vtk(enu.astype(np.float32), deep=True))
    poly = vtk.vtkPolyData()
    poly.SetPoints(pts)
    colors = numpy_support.numpy_to_vtk((np.clip(rgb, 0, 1) * 255).astype(np.uint8), deep=True)
    colors.SetName('rgb')
    poly.GetPointData().SetScalars(colors)
    glyph = vtk.vtkVertexGlyphFilter()
    glyph.SetInputData(poly)
    m = vtk.vtkPolyDataMapper()
    m.SetInputConnection(glyph.GetOutputPort())
    a = vtk.vtkActor()
    a.SetMapper(m)
    a.GetProperty().SetPointSize(2)
    return a, len(q)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--scan', type=Path, help='overlay a .ply scan (aligned with --yaw/--tx/--ty)')
    ap.add_argument('--mirror', action='store_true')
    ap.add_argument('--yaw', type=float, default=177.0, help='scan alignment (view_course_scan.py output)')
    ap.add_argument('--tx', type=float, default=-0.18)
    ap.add_argument('--ty', type=float, default=-0.94)
    ap.add_argument('--max_points', type=int, default=300_000)
    ap.add_argument('--no_net', action='store_true', help='hide the (invisible in Gazebo) net shell')
    ap.add_argument('--png', type=Path, help='render to this file and exit')
    args = ap.parse_args(argv)

    ren = vtk.vtkRenderer()
    ren.SetBackground(0.16, 0.16, 0.18)
    course, nv, nf = obj_actor(WORLD / 'course.obj', (0.7, 0.7, 0.72), opacity=0.55 if args.scan else 1.0)
    ren.AddActor(course)
    print(f'course.obj: {nv} vertices, {nf} triangles')
    if not args.no_net:
        net, _, _ = obj_actor(WORLD / 'net.obj', (0.3, 0.55, 0.9), opacity=0.12)
        ren.AddActor(net)
    ren.AddActor(floor_actor())
    sp, (x, y, z, yaw) = spawn_actor()
    ren.AddActor(sp)
    print(f'spawn (ENU): ({x:.2f}, {y:.2f}, {z:.2f}), yaw {math.degrees(yaw):.0f} deg (purple arrow)')
    if args.scan:
        sa, n = scan_actor(args)
        ren.AddActor(sa)
        print(f'scan overlay: {n} points (course mesh drawn translucent)')
    axes = vtk.vtkAxesActor()                            # x east (red), y north (green), z up (blue) at the origin
    axes.SetTotalLength(1, 1, 1)
    axes.SetAxisLabels(False)
    ren.AddActor(axes)

    cam = ren.GetActiveCamera()
    cam.SetFocalPoint(0, 0, 0.8)
    cam.SetPosition(-6.5, -8.0, 7.0)
    cam.SetViewUp(0, 0, 1)
    ren.ResetCameraClippingRange()

    win = vtk.vtkRenderWindow()
    win.AddRenderer(ren)
    win.SetSize(1400, 950)
    win.SetWindowName('DroneHacks Gazebo world (sitl/world)')
    if args.png:
        win.SetOffScreenRendering(1)
        win.Render()
        w2i = vtk.vtkWindowToImageFilter()
        w2i.SetInput(win)
        w2i.Update()
        png = vtk.vtkPNGWriter()
        png.SetFileName(str(args.png))
        png.SetInputConnection(w2i.GetOutputPort())
        png.Write()
        print(f'wrote {args.png}')
        return 0
    iren = vtk.vtkRenderWindowInteractor()
    iren.SetRenderWindow(win)
    iren.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())
    win.Render()
    iren.Start()
    return 0


if __name__ == '__main__':
    sys.exit(main())
