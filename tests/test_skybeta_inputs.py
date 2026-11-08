"""Exercise real node methods without ROS middleware, which is absent on dev hosts.

Load the class AST only; transport imports and ROS Node construction are excluded.
The policy, input validation, observation and command conversion code remain real.
"""
import ast
import json
import math
from pathlib import Path
import time
from types import SimpleNamespace as Msg

import numpy as np
import pytest

from policynav import handheld as H, obs as O
from policynav.px4_feedback import Px4Feedback
from test_px4_feedback import lpos, att, flags


@pytest.fixture
def node():
    source = Path(__file__).parents[1] / 'policynav/skybeta_node.py'
    tree = ast.parse(source.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Skybeta')
    scope = dict(Node=object, np=np, math=math, json=json, time=time, H=H, O=O)
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(source), 'exec'), scope)
    n = object.__new__(scope['Skybeta'])
    n._now = lambda: 10.1
    n.feedback = Px4Feedback()
    n.feedback.on_lpos(lpos(), 10.)
    n.feedback.on_att(att(), 10.)
    n.feedback_lost_s = n.state_lost_s = .5
    n.phase, n.why, n.t_fly = 'fly', 'policy', 10.
    n.state, n.state_t = dict(armed=True, offboard_active=True), 10.
    n.depth_t, n.depth_stale_s, n.depth_lost_s = 10., .5, 2.
    n.max_flight_s = 120.
    n.cmd = ((1., 0., 0.), 0.)
    n.get_logger = lambda: Msg(info=lambda *a, **k: None, warning=lambda *a, **k: None)
    n.hold_cli = Msg(service_is_ready=lambda: False)
    n._publish = lambda: None
    n.steps = []
    n._step = lambda t: n.steps.append(t)
    return n


@pytest.mark.parametrize('kind', ['state', 'attitude', 'disarm', 'offboard'])
def test_bad_input_prevents_policy_step(node, kind):
    if kind == 'state':
        node.state_t = 9.
    elif kind == 'attitude':
        node.feedback.att_t = 9.
    elif kind == 'disarm':
        node.state['armed'] = False
    else:
        node.state['offboard_active'] = False
    node._tick()
    assert not node.steps
    assert node.phase in ('hold', 'yielded')
    assert node.cmd == ((0., 0., 0.), 0.)


def test_valid_inputs_allow_policy(node):
    node._tick()
    assert node.steps == [10.1]


def test_range_dropout_keeps_policy_running_with_last_distance(node):
    node._on_lpos(lpos(timestamp_sample=1_100_000, dist_bottom_valid=False))
    node._tick()
    assert node.steps == [10.1] and node.feedback.range_m == 1.2


def test_hold_yields_permanently_on_pilot_takeover(node):
    node._hold('test')
    node.state['offboard_active'] = False
    node._tick()
    assert node.phase == 'yielded'
    node.state['offboard_active'] = True
    node._tick()
    assert node.phase == 'yielded' and not node.steps


@pytest.mark.parametrize('data', ['[]', 'null', '{', '{"armed":"false","offboard_active":true}'])
def test_malformed_state_cannot_refresh_watchdog(node, data):
    node._on_state(Msg(data=data))
    assert node.state_t == 10.


def test_no_altitude_or_constant_range_fallback(node):
    node.feedback.range_m = None
    assert node._range() is None


def test_px4_inputs_reach_real_policy_step_and_enu_output(node):
    seen = []
    node.meta = dict(rf_max=4., v_obs_scale=[1., 1., 1.])
    node.airborne = False
    node.stack = O.FrameStack(2)
    node.depth_img = np.ones((80, 128), np.float32)
    node.prev_action = np.zeros(4)
    def policy(img, pro):
        seen.append(pro)
        return np.array([.5, 0., 0., .2])
    node.policy = policy
    node.v_max, node.yaw_max = [1., 1., 1.], 1.
    node.speed_cap, node.yaw_cap, node.timing = 1.45, 1., []
    type(node)._step(node, 10.1)
    assert seen[0][:6] == pytest.approx([1., 0., -.2, .8, 0., 0.], abs=1e-6)
    assert node.cmd[0] == pytest.approx([.5, 0., 0.], abs=1e-6)
    assert node.cmd[1] == pytest.approx(-.2)
    # The startup rf_max default must not falsely mark a grounded vehicle airborne.
    node.airborne = False
    node.feedback.range_m, node.feedback.range_t = 4., None
    type(node)._step(node, 10.2)
    assert not node.airborne


def test_local_vision_and_filtered_range_start_policy_from_ground(node):
    node.phase = 'wait'
    node.last_startup_blocker = None
    node.feedback = Px4Feedback()
    node.feedback.on_estimator(flags(), 10.)
    node.feedback.on_lpos(lpos(heading_good_for_control=False, dist_bottom_valid=False,
                              dist_bottom=.10), 10.)
    node.feedback.on_att(att(), 10.)
    node.policy = Msg(reset=lambda: None)
    node.stack = O.FrameStack(2)
    node.prev_action = np.ones(4)
    node._tick()
    assert node.phase == 'fly' and node.steps == [10.1]
    assert node.feedback.range_m == .10
    assert np.all(node.prev_action == 0.)


def test_no_initial_range_logs_error_before_offboard_and_blocks_inference(node):
    node.phase = 'wait'
    node.last_startup_blocker = None
    node.feedback.range_t = None
    node.state['armed'] = False
    errors = []
    node.get_logger = lambda: Msg(error=lambda text, **kw: errors.append(text),
                                 info=lambda *a, **kw: None, warning=lambda *a, **kw: None)
    node._tick()
    assert errors and 'range' in errors[0] and 'BLOCKED' in errors[0]
    assert node.phase == 'wait' and not node.steps
    node.state['armed'] = True
    node._tick()
    assert node.phase == 'hold' and not node.steps


@pytest.mark.parametrize('missing', [None, 'depth', 'state', 'range'])
def test_green_readiness_requires_all_startup_inputs(node, missing):
    node.phase = 'wait'
    node.state['armed'] = False
    messages = []
    node.get_logger = lambda: Msg(info=lambda text, **kw: messages.append(text),
                                 error=lambda *a, **kw: None)
    if missing == 'depth':
        node.depth_t = None
    elif missing == 'state':
        node.state_t = None
    elif missing == 'range':
        node.feedback.range_t = None
    node._tick()
    green = [m for m in messages if '\033[1;92m' in m]
    assert bool(green) == (missing is None)
    if green:
        assert green[0].endswith('\033[0m')
    assert not node.steps


def test_ftw0_policy_uses_training_velocity_scale_in_real_step(node):
    from policynav.policy_np import NumpyPolicy
    policy = NumpyPolicy(Path(__file__).parents[1] / 'policies/ftw0_s330_1627.npz')
    node.meta = policy.meta
    node.feedback.vel = np.array([1.2, .3, -.3])
    node.feedback.q = np.array([1., 0., 0., 0.])
    node.airborne = False
    node.stack = O.FrameStack(node.meta['frame_stack'])
    node.depth_img = np.ones((80, 128), np.float32)
    node.prev_action = np.zeros(4, np.float32)
    seen = []
    def run_policy(depth, proprio):
        seen.append(proprio.copy())
        return policy(depth, proprio)
    node.policy = run_policy
    node.v_max, node.yaw_max = node.meta['v_max'], node.meta['yaw_rate_max']
    node.speed_cap, node.yaw_cap, node.timing = 1.45, 1., []
    type(node)._step(node, 10.1)
    assert seen[0][:3] == pytest.approx([.24, .10, -.15])
    assert node.meta['v_max'] == [1.2, .6, .6]
    assert np.isfinite(node.prev_action).all()
    assert np.max(np.abs(node.prev_action)) <= 1.
    # Identity attitude: ENU x=east/right, y=north/forward, z=up.
    assert node.cmd[0] == pytest.approx([node.prev_action[1] * .6,
                                        node.prev_action[0] * 1.2,
                                        -node.prev_action[2] * .6])
    assert node.cmd[1] == pytest.approx(-node.prev_action[3])
