import numpy as np
import pytest

from policynav import obs as O


def test_edge_bands_fill_inward_but_interior_holes_stay_near():
    a = np.array([[0., 0., .2, 0., .6, .8, 0., 0.]])
    filled, valid = O.edge_fill(a, a > 0, 2)
    assert filled[0] == pytest.approx([.2, .2, .2, 0., .6, .8, .8, .8])
    assert not valid[0, 3]
    assert a[0, 0] == 0.  # input is not mutated


def test_no_valid_seed_and_band_wider_than_fill_remain_near():
    a = np.array([[0., 0., 0., .4, .8, 0., 0., 0.], [0.] * 8])
    filled, valid = O.edge_fill(a, a > 0, 2)
    np.testing.assert_array_equal(filled, a)
    np.testing.assert_array_equal(valid, a > 0)


@pytest.mark.parametrize('width', [1, 2, 8])
def test_fill_width_clamps_to_half_image(width):
    a = np.ones((2, width))
    a[1] = 0.
    filled, valid = O.edge_fill(a, a > 0, 16)
    np.testing.assert_array_equal(filled, a)


def test_reprojection_preserves_legacy_default_and_fills_when_enabled():
    h, w = 80, 128
    fx = .5*w/np.tan(np.radians(72)/2)
    fy = .5*h/np.tan(np.radians(50)/2)
    R = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]])
    depth = np.full((h, w), 2., np.float32)
    depth[:, :3] = np.nan
    args = ((fx, fy, w/2, h/2), (h, w), R, (0, 0, 0))
    legacy = O.DepthReprojector(*args)(depth)
    filled = O.DepthReprojector(*args, edge_fill_cols=16)(depth)
    assert np.all(legacy[:, :3] == 0.)
    assert np.allclose(filled[:, :3], .4)
