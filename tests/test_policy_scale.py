import json
from pathlib import Path

import numpy as np
import pytest

from policynav.policy_np import NumpyPolicy


def export_with_meta(tmp_path, **changes):
    path = Path(__file__).parents[1] / 'policies/anneal_slow_b128_3255.npz'
    with np.load(path, allow_pickle=False) as z:
        data = dict(z)
    meta = json.loads(str(data['meta']))
    meta.pop('v_obs_scale', None)
    meta.update(changes)
    data['meta'] = json.dumps(meta)
    out = tmp_path / 'policy.npz'
    np.savez(out, **data)
    return out


def test_missing_scale_does_not_use_curriculum_final_speed(tmp_path):
    with pytest.warns(UserWarning, match='v_obs_scale'):
        policy = NumpyPolicy(export_with_meta(tmp_path))
    assert policy.meta['v_obs_scale'] == [5., 3., 2.]
    assert policy.meta['v_max'] == [1.2, .6, .6]


def test_explicit_training_scale_takes_precedence(tmp_path):
    policy = NumpyPolicy(export_with_meta(tmp_path, v_obs_scale=[6., 4., 3.]))
    assert policy.meta['v_obs_scale'] == [6., 4., 3.]


@pytest.mark.parametrize('scale', [None, [1., 2.], [0., 3., 2.], [-1., 3., 2.], [float('nan'), 3., 2.]])
def test_invalid_scale_rejected_before_inference(tmp_path, scale):
    with pytest.raises(ValueError, match='v_obs_scale'):
        NumpyPolicy(export_with_meta(tmp_path, v_obs_scale=scale))
