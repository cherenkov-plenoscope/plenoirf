import binning_utils
import solid_angle_utils
import numpy as np


def init_from_analysis_config(analysis_config, key):
    zb_cfg = analysis_config["pointing_binning"]["zenith_binning"]

    num_bins = zb_cfg["num_bins"] * zb_cfg["fine"][key]
    return init_from_start_stop_num(
        start_half_angle_rad=zb_cfg["start_half_angle_rad"],
        stop_half_angle_rad=zb_cfg["stop_half_angle_rad"],
        num_bins=num_bins,
    )


def init_from_start_stop_num(
    start_half_angle_rad, stop_half_angle_rad, num_bins
):
    bin_edges = solid_angle_utils.cone.half_angle_space(
        start_half_angle_rad=start_half_angle_rad,
        stop_half_angle_rad=stop_half_angle_rad,
        num=num_bins + 1,
    )
    return init_zenith_binning_from_bin_edges(bin_edges=bin_edges)


def init_from_bin_edges(bin_edges):
    z = binning_utils.Binning(bin_edges=bin_edges)

    # apply spacing to centers
    for i in range(z["num"]):
        z["centers"][i] = solid_angle_utils.cone.half_angle_space(
            start_half_angle_rad=z["edges"][i],
            stop_half_angle_rad=z["edges"][i + 1],
            num=3,
        )[1]

    # add solid angles
    z["solid_angles"] = np.zeros(z["num"])
    for i in range(z["num"]):
        outer = solid_angle_utils.cone.solid_angle(z["edges"][i + 1])
        inner = solid_angle_utils.cone.solid_angle(z["edges"][i])
        z["solid_angles"][i] = outer - inner

    return z
