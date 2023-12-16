import plenoirf
import numpy as np


def test_edge():
    img = np.arange(32**2).reshape(32, 32)
    r = 1

    o = plenoirf.utils.copy_square_selection_from_2D_array(
        img=img,
        ix=0,
        iy=0,
        r=r,
        fill=np.nan,
    )

    assert o.shape[0] == 2 * r + 1
    assert o.shape[1] == 2 * r + 1

    assert np.isnan(o[0, 0])
    assert np.isnan(o[0, 1])
    assert np.isnan(o[0, 2])

    assert np.isnan(o[1, 0])
    assert o[1, 1] == 0.0
    assert o[1, 2] == 1.0

    assert np.isnan(o[2, 0])
    assert o[2, 1] == 32.0
    assert o[2, 2] == 33.0


def test_full():
    img = np.arange(17**2).reshape(17, 17)
    r = 8
    omg = plenoirf.utils.copy_square_selection_from_2D_array(
        img=img,
        ix=r,
        iy=r,
        r=r,
        fill=0,
    )
    np.testing.assert_array_equal(img, omg)
