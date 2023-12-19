import numpy as np
import json_utils


def where_bin_idxs_within_radius(groundgrid, center_x_m, center_y_m, radius_m):
    """
    Returns the bin idxs of the ground grid where the bin is within a circle
    of radius and center_x/y.
    Same format as np.where()
    """
    bin_idxs = []
    for ix in range(groundgrid["x_bin"]["num"]):
        x_m = groundgrid["x_bin"]["centers"][ix]
        for iy in range(groundgrid["y_bin"]["num"]):
            y_m = groundgrid["y_bin"]["centers"][iy]

            delta_m = np.hypot(x_m - center_x_m, y_m - center_y_m)
            if delta_m <= radius_m:
                bin_idx = (ix, iy)
                bin_idxs.append(bin_idx)
    return bin_idxs


def intersection_of_bin_idxs(a_bin_idxs, b_bin_idxs):
    return list(set(a_bin_idxs).intersection(set(b_bin_idxs)))


def apply_bin_limitation_and_warn(
    bin_idxs_above_threshold, bin_idxs_limitation
):
    bin_idxs_above_threshold_and_in_limits = intersection_of_bin_idxs(
        a_bin_idxs=bin_idxs_above_threshold, b_bin_idxs=bin_idxs_limitation
    )
    msg = {}
    msg["artificial_core_limitation"] = {
        "num_grid_bins_in_limits": len(bin_idxs_limitation[0]),
        "num_grid_bins_above_threshold": len(bin_idxs_above_threshold[0]),
        "num_grid_bins_above_threshold_and_in_limits": len(
            bin_idxs_above_threshold_and_in_limits[0]
        ),
    }
    print(json_utils.dumps(msg))

    return bin_idxs_above_threshold_and_in_limits
