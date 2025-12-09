import binning_utils
import numpy as np


def init_from_bin_edges(bin_edges):
    assert len(bin_edges) >= 2
    assert np.all(bin_edges > 0.0)
    assert np.all(np.gradient(bin_edges) > 0.0)
    return binning_utils.Binning(bin_edges=bin_edges)


def init_from_decades(
    start_decade,
    start_bin,
    stop_decade,
    stop_bin,
    num_bins_per_decade,
):
    bin_edges = binning_utils.power10.space(
        start_decade=start_decade,
        start_bin=start_bin,
        stop_decade=stop_decade,
        stop_bin=stop_bin,
        num_bins_per_decade=num_bins_per_decade,
    )
    return init_from_bin_edges(bin_edges=bin_edges)


def init_from_analysis_config(analysis_config, key):
    enebin = analysis_config["energy_binning"]
    multi = enebin["fine"][key]
    return init_from_decades(
        start_decade=enebin["start"]["decade"],
        start_bin=enebin["start"]["bin"] * multi,
        stop_decade=enebin["stop"]["decade"],
        stop_bin=enebin["stop"]["bin"] * multi,
        num_bins_per_decade=enebin["num_bins_per_decade"] * multi,
    )
