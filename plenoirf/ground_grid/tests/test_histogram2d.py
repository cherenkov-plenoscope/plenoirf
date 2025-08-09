import plenoirf
import numpy as np
import pytest


def test_histogram2d_bins_unique():
    hist_ok = np.recarray(
        dtype=plenoirf.ground_grid.histogram2d.make_dtype(), shape=2
    )
    (hist_ok["x_bin"][0], hist_ok["y_bin"][0]) = (0, 0)
    (hist_ok["x_bin"][1], hist_ok["y_bin"][1]) = (1, 1)

    hist_bad = np.recarray(
        dtype=plenoirf.ground_grid.histogram2d.make_dtype(), shape=2
    )
    (hist_bad["x_bin"][0], hist_bad["y_bin"][0]) = (2, 3)
    (hist_bad["x_bin"][1], hist_bad["y_bin"][1]) = (2, 3)

    plenoirf.ground_grid.histogram2d.assert_bins_unique(hist_ok)

    with pytest.raises(AssertionError):
        plenoirf.ground_grid.histogram2d.assert_bins_unique(hist_bad)


def test_histogram2d_bins_in_limits():
    hist_ok = np.recarray(
        dtype=plenoirf.ground_grid.histogram2d.make_dtype(), shape=2
    )
    (hist_ok["x_bin"][0], hist_ok["y_bin"][0]) = (0, 2)
    (hist_ok["x_bin"][1], hist_ok["y_bin"][1]) = (0, 3)

    hist_bad = np.recarray(
        dtype=plenoirf.ground_grid.histogram2d.make_dtype(), shape=2
    )
    (hist_bad["x_bin"][0], hist_bad["y_bin"][0]) = (2, 0)
    (hist_bad["x_bin"][1], hist_bad["y_bin"][1]) = (2, 4)

    plenoirf.ground_grid.histogram2d.assert_bins_in_limits(
        hist_ok, num_bins_each_axis=4
    )

    with pytest.raises(AssertionError):
        plenoirf.ground_grid.histogram2d.assert_bins_in_limits(
            hist_bad, num_bins_each_axis=4
        )
