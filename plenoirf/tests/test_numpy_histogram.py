import numpy as np


def test_what_is_np_histogram2d_doing():
    num_bins_radius = 3
    num_bins_diameter = num_bins_radius * 2
    xy_bin_edges = np.linspace(-30, 30, 2 * num_bins_radius + 1)

    num_points = 100
    xs = np.ones(num_points)
    ys = 21 * np.ones(num_points)

    i = np.histogram2d(xs, ys, bins=(xy_bin_edges, xy_bin_edges))[0]

    assert num_bins_diameter == i.shape[0]
    assert num_bins_diameter == i.shape[1]

    lidx = np.where(i)

    assert lidx[0].shape[0] == 1
    assert lidx[1].shape[0] == 1

    xidx = lidx[0][0]
    yidx = lidx[1][0]

    assert xidx == 3
    assert yidx == 5

    assert xy_bin_edges[xidx] == 0
    assert xy_bin_edges[xidx + 1] == 10

    assert xy_bin_edges[yidx] == 20
    assert xy_bin_edges[yidx + 1] == 30

    x_bin_idxs = np.digitize(xs, bins=xy_bin_edges)
    y_bin_idxs = np.digitize(ys, bins=xy_bin_edges)

    assert x_bin_idxs[0] - 1 == xidx
    assert y_bin_idxs[0] - 1 == yidx
