"""Execute launch construction without starting ROS nodes."""
import ast
from pathlib import Path

import pytest


@pytest.mark.parametrize('swap,left,right', [('false', '/oak/left', '/oak/right'),
                                          ('true', '/oak/right', '/oak/left')])
def test_eye_swap_routes_images_without_swapping_calibration(swap, left, right):
    path = Path(__file__).parents[1] / 'launch/skybeta.launch.py'
    tree = ast.parse(path.read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'nodes')
    class Config:
        def __init__(self, name): self.name = name
        def perform(self, context): return context[self.name]
    scope = dict(LaunchConfiguration=Config, dds_env=lambda ctx: ({}, ''), cpu_prefix=lambda ctx: None,
                 ONE_BLAS_THREAD={}, Node=lambda **kw: kw, LogInfo=lambda **kw: kw,
                 LaunchConfigurationEquals=lambda *a: a, LaunchConfigurationNotEquals=lambda *a: a,
                 PythonExpression=lambda a: a)
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(path), 'exec'), scope)
    nodes = scope['nodes']({'swap_eyes': swap})
    for name in ('stereo_depth', 'vpi_stereo'):
        node = next(n for n in nodes if n.get('executable') == name)
        params = node['parameters'][0]
        assert params['left_ns'] == '/oak/left' and params['right_ns'] == '/oak/right'
        assert params['left_image_ns'] == left and params['right_image_ns'] == right
