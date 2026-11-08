#!/usr/bin/env python3
"""Golden data for the C++ policy core (offline tool; run inside the skybeta image, never needed at runtime).

Feeds identical inputs through the Python code (policynav/obs.py, node.py:decode_depth, policy_np.py and the
nodes' per-decision step) and saves inputs + outputs to an .npz that tests/policy_core/golden_test.cpp replays.

  docker run --rm --network none -e OPENBLAS_NUM_THREADS=1 -v <repo>:/w:ro -v <scratch>:/data:ro -v <out>:/out \\
      skybeta:0.3.2 bash -lc 'python3 -I /w/tests/policy_core/make_golden.py /w/policies/cur4_clean50_800.npz \\
      /out/golden_cur4.npz /data/depth_vpi.npz /data/depth_sgbm.npz'
"""
from __future__ import annotations

import math
import os
import sys
import time
import types

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
from policynav import obs as O                       # noqa: E402
from policynav.policy_np import NumpyPolicy          # noqa: E402



def _load_decode_depth():
    """node.py:decode_depth, taken from the source itself (node.py imports rclpy, absent under python3 -I)."""
    import ast
    src = open(os.path.join(REPO, 'policynav', 'node.py')).read()
    fn = next(n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == 'decode_depth')
    ns = {'np': np}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), 'node.py', 'exec'), ns)
    return ns['decode_depth']


decode_depth = _load_decode_depth()


def quat_axis_angle(axis, ang):
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    s = math.sin(ang / 2)
    return (math.cos(ang / 2), *(axis * s))


def qmul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return (w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2)


def main(policy_path, out_path, *depth_files):
    rng = np.random.default_rng(20261107)
    pol = NumpyPolicy(policy_path)
    meta = pol.meta
    n_act, (Hh, Ww) = meta['num_actions'], meta['depth_hw']
    G = {}

    # ── raw depth frames (NaN = invalid) ──
    frames = np.concatenate([np.load(f)['depth'].astype(np.float32) for f in depth_files])
    G['depth_raw'] = frames
    Hs, Ws = frames.shape[1:]

    # ── reprojector configs: (K, R_body_cam as TF quaternion, t, stride) ──
    q_opt = (0.5, -0.5, 0.5, -0.5)                         # optical (x right, y down, z fwd) -> FLU, level
    assert np.allclose(O.quat_to_mat(*q_opt), [[0, 0, 1], [-1, 0, 0], [0, -1, 0]])
    tilt = qmul(qmul(quat_axis_angle((0, 1, 0), math.radians(12.0)), quat_axis_angle((0, 0, 1), math.radians(-3.0))),
                quat_axis_angle((1, 0, 0), math.radians(1.5)))
    cfgs = [((500.0, 500.0, 319.5, 199.5), q_opt, (0.08, 0.0, -0.02), 2),
            ((421.0, 421.0, 320.0, 200.0), qmul(tilt, q_opt), (0.10, 0.03, 0.01), 2),
            ((452.3, 451.7, 316.2, 203.9), qmul(quat_axis_angle((0, 1, 0), math.radians(-20.0)), q_opt), (0.0, 0.0, 0.0), 1)]
    K = np.array([c[0] for c in cfgs]); R = np.array([O.quat_to_mat(*c[1]) for c in cfgs])
    T = np.array([c[2] for c in cfgs]); S = np.array([c[3] for c in cfgs], np.int64)
    reproj = np.zeros((len(cfgs), len(frames), Hh, Ww), np.float32)
    axis_err = np.zeros(len(cfgs))
    reps = []
    for i in range(len(cfgs)):
        rp = O.DepthReprojector(tuple(K[i]), (Hs, Ws), R[i], T[i], out_hw=(Hh, Ww), hfov_deg=meta['hfov_deg'],
                                vfov_deg=meta['vfov_deg'], far=meta['far_clip'], stride=int(S[i]), edge_fill_cols=meta['edge_fill_cols'])
        reps.append(rp)
        axis_err[i] = rp.axis_err_deg
        for f in range(len(frames)):
            reproj[i, f] = rp(frames[f])
    G.update(reproj_K=K, reproj_R=R, reproj_t=T, reproj_stride=S, reproj_out=reproj, reproj_axis_err=axis_err,
             reproj_far=np.float64(meta['far_clip']), reproj_hfov=np.float64(meta['hfov_deg']),
             reproj_vfov=np.float64(meta['vfov_deg']), reproj_out_hw=np.array([Hh, Ww], np.int64))

    # ── decode_depth on cropped synthetic messages (row padding, invalid / inf values) ──
    crop = frames[0, 100:164, 200:280].copy()                                  # 64 x 80
    mm = np.where(np.isfinite(crop), np.clip(np.round(crop * 1000.0), 0, 65535), 0).astype(np.uint16)
    mm[0, :5] = 0
    step16 = 2 * mm.shape[1] + 6
    buf16 = np.zeros((mm.shape[0], step16), np.uint8)
    buf16[:, :2 * mm.shape[1]] = mm.view(np.uint8).reshape(mm.shape[0], -1)
    buf16[:, 2 * mm.shape[1]:] = 0xAB
    f32 = crop.copy()
    f32[1, :4] = [np.inf, -np.inf, 0.0, -1.0]
    step32 = 4 * f32.shape[1] + 8
    buf32 = np.zeros((f32.shape[0], step32), np.uint8)
    buf32[:, :4 * f32.shape[1]] = f32.view(np.uint8).reshape(f32.shape[0], -1)
    buf32[:, 4 * f32.shape[1]:] = 0x7F
    msg = lambda b, enc, step: types.SimpleNamespace(data=b.tobytes(), encoding=enc, height=b.shape[0],  # noqa: E731
                                                     width=crop.shape[1], step=step)
    G.update(dec16_buf=buf16, dec16_step=np.int64(step16), dec16_out=decode_depth(msg(buf16, '16UC1', step16)),
             dec32_buf=buf32, dec32_step=np.int64(step32), dec32_out=decode_depth(msg(buf32, '32FC1', step32)))

    # ── proprio / action_to_setpoint on synthetic states ──
    N = 256
    quats = []
    for i in range(N):
        if i == 0:
            q = (1.0, 0.0, 0.0, 0.0)
        elif i == 1:
            q = quat_axis_angle((0, 0, 1), math.pi / 2)
        elif i in (2, 3):
            q = quat_axis_angle((0, 1, 0), math.radians(89.99 if i == 2 else -89.99))
        elif i == 4:
            q = tuple(-c for c in quat_axis_angle((1, 2, 3), 0.7))           # negative w
        else:
            q = rng.normal(size=4)
            q = tuple(q / np.linalg.norm(q))
        quats.append(q)
    quats = np.array(quats, float)
    vel = rng.normal(0, 1.5, (N, 3))
    rngm = rng.uniform(0.04, 4.0, N)
    prev = rng.uniform(-1.2, 1.2, (N, n_act)).astype(np.float32)
    act = rng.uniform(-1.5, 1.5, (N, n_act)).astype(np.float32)
    v_max = meta['v_max']
    v_obs_scale = meta['v_obs_scale']
    G['pp_q'], G['pp_vel'], G['pp_range'], G['pp_prev'], G['pp_act'] = quats, vel, rngm, prev, act
    G['pp_proprio'] = np.stack([O.proprio(vel[i], quats[i], rngm[i], prev[i], v_obs_scale) for i in range(N)])
    sps = [O.action_to_setpoint(act[i, :4], quats[i], v_max, meta['yaw_rate_max']) for i in range(N)]
    G['pp_vel_sp'] = np.stack([s[0] for s in sps])
    G['pp_yawspeed'] = np.array([s[1] for s in sps])

    # ── single policy steps from a fresh state (inputs: Python reprojection, cfg 0) ──
    nf = len(frames)
    stacks = [np.stack([reproj[0, i], reproj[0, (i + 1) % nf]]) for i in range(nf)]
    stacks += [np.zeros((2, Hh, Ww), np.float32), np.ones((2, Hh, Ww), np.float32)]
    stacks = np.stack(stacks)[:, :meta['frame_stack']]
    pros = G['pp_proprio'][:len(stacks)]
    outs = []
    for i in range(len(stacks)):
        pol.reset()
        outs.append(pol(stacks[i], pros[i]))
    G['single_depth'], G['single_proprio'], G['single_out'] = stacks, pros, np.stack(outs)

    # ── multi-step sequence exactly as skybeta_node._step / handheld_node._control (GRU state carried) ──
    T_ = 40
    resets = {0, 23}
    yaw0 = 0.3
    seq_q, seq_vel, seq_rng, seq_f = [], [], [], []
    for k in range(T_):
        yaw, pitch, roll = yaw0 + 0.05 * k, 0.08 * math.sin(0.3 * k), 0.05 * math.cos(0.2 * k)
        q = qmul(qmul(quat_axis_angle((0, 0, 1), yaw), quat_axis_angle((0, 1, 0), pitch)), quat_axis_angle((1, 0, 0), roll))
        seq_q.append(q)
        seq_vel.append((1.2 * math.cos(0.1 * k), 0.4 * math.sin(0.15 * k), -0.2 + (0.05 * k) % 0.3))
        seq_rng.append(0.08 + 0.05 * k if k < 30 else 1.6)
        seq_f.append((3 * k) % nf)
    pol.reset()
    stack = O.FrameStack(meta['frame_stack'])
    prev_action = np.zeros(n_act, np.float32)
    rec = {k: [] for k in ('action', 'proprio', 'vel_sp', 'yawspeed', 'h')}
    for k in range(T_):
        if k in resets:
            pol.reset(); stack.reset(); prev_action[:] = 0.0
        img = stack.push(reproj[0, seq_f[k]])
        pro = O.proprio(seq_vel[k], seq_q[k], seq_rng[k], prev_action, v_obs_scale)
        a = np.clip(pol(img, pro), -1.0, 1.0).astype(np.float32)
        prev_action = a
        vel_sp, yawspeed = O.action_to_setpoint(a[:4], seq_q[k], v_max, meta['yaw_rate_max'])
        rec['action'].append(a.copy()); rec['proprio'].append(pro); rec['vel_sp'].append(vel_sp)
        rec['yawspeed'].append(yawspeed); rec['h'].append(pol.h.copy())
    G.update(seq_q=np.array(seq_q), seq_vel=np.array(seq_vel), seq_range=np.array(seq_rng),
             seq_frame=np.array(seq_f, np.int64), seq_reset=np.array(sorted(resets), np.int64),
             **{'seq_' + k: np.array(v) for k, v in rec.items()})

    # ── timings (median of 50; OPENBLAS_NUM_THREADS=1 as in the launch files) ──
    def med(fn, n=50):
        fn()
        ts = []
        for _ in range(n):
            t0 = time.perf_counter(); fn(); ts.append(time.perf_counter() - t0)
        return 1e3 * float(np.median(ts))
    pol.reset()
    stack.reset()
    G['py_ms_reproj'] = np.float64(med(lambda: reps[0](frames[0])))
    G['py_ms_policy'] = np.float64(med(lambda: pol(stacks[0], pros[0])))

    def full_step():
        img = stack.push(reproj[0, 0])
        pro = O.proprio(seq_vel[0], seq_q[0], 1.0, prev_action, v_obs_scale)
        a = np.clip(pol(img, pro), -1.0, 1.0).astype(np.float32)
        O.action_to_setpoint(a[:4], seq_q[0], v_max, meta['yaw_rate_max'])
    G['py_ms_step'] = np.float64(med(full_step))
    print(f"python: reproject {G['py_ms_reproj']:.2f} ms, policy forward {G['py_ms_policy']:.2f} ms, "
          f"full step {G['py_ms_step']:.2f} ms (median of 50, OPENBLAS_NUM_THREADS="
          f"{os.environ.get('OPENBLAS_NUM_THREADS', 'unset')})")

    np.savez_compressed(out_path, **G)
    print(f'wrote {out_path}: {len(G)} arrays, {os.path.getsize(out_path) / 1e6:.1f} MB')


if __name__ == '__main__':
    if len(sys.argv) < 4:
        sys.exit(__doc__)
    main(*sys.argv[1:])
