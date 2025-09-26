import numpy as np
import binning_utils


def make_bin_edges_and_cumsum_with_sliding_bin_width(x, num_steps=None):
    """
    Parameters
    ----------
    x : array like
        samples

    Returns
    -------
    bin_edges : array like
        Not equally spaced. Spacing is according to distribution of samples.
    cumsum : array like
        Cumulative sum (max = 1.0) of the sample density in the 'bin_edges'.
    """
    if num_steps is None:
        num_steps = int(np.round(np.sqrt(len(x))))

    x = np.sort(x)
    x_x = [0]
    x_c = []
    for i in range(0, len(x), num_steps):
        _count = int(num_steps)
        _left = len(x) - i
        _count = min([_count, _left])
        x_c.append(_count)
        i_stop = i + _count
        x_x.append(x[i_stop - 1])
    x_x = np.array(x_x)
    x_c = np.array(x_c)
    x_d = x_c / binning_utils.widths(x_x)
    x_d /= np.sum(x_d)
    x_x_cumsum = np.zeros(shape=x_x.shape)
    x_x_cumsum[1:] = np.cumsum(x_d)

    return x_x, x_x_cumsum


def make_sample_weights(x, bin_edges):
    """
    Estimate weights for samples 'x' inverse proportinal to their density in
    the 'bin_edges'.

    Parameters
    ----------
    x : array like
        samples
    bin_edges : array like
        Edges to bin 'x' in.

    Returns
    -------
    sample_weights : array like
        Weights for sampeles 'x'.
    """
    bin_population = np.histogram(x, bins=bin_edges)[0]
    bin_weights = (1.0 / bin_population) ** 0.5
    bin_assignment = -1 + np.digitize(x, bins=bin_edges)
    sample_weights = bin_weights[bin_assignment]
    return sample_weights
