import ray_voxel_overlap
import numpy as np
import binning_utils
import corsika_primary as cpw
import spherical_coordinates
from . import artificial_core_limitation


def make_ground_grid_config(
    bin_width_m,
    num_bins_each_axis,
    cherenkov_pool_median_x_m,
    cherenkov_pool_median_y_m,
    prng,
):
    assert bin_width_m > 0.0
    assert num_bins_each_axis > 0

    random_radius_m = (1/2) * bin_width_m

    random_shift_x_m = prng.uniform(-random_radius_m, random_radius_m)
    random_shift_y_m = prng.uniform(-random_radius_m, random_radius_m)

    center_x_m = cherenkov_pool_median_x_m + random_shift_x_m
    center_y_m = cherenkov_pool_median_y_m + random_shift_y_m

    return {
        "bin_width_m": bin_width_m,
        "num_bins_each_axis": num_bins_each_axis,
        "center_x_m": center_x_m,
        "center_y_m": center_y_m,
        "cherenkov_pool_median_x_m": cherenkov_pool_median_x_m,
        "cherenkov_pool_median_y_m": cherenkov_pool_median_y_m,
        "random_shift_x_m": random_shift_x_m,
        "random_shift_y_m": random_shift_y_m,
    }


def GroundGrid(
    bin_width_m,
    num_bins_each_axis,
    center_x_m,
    center_y_m,
):
    assert bin_width_m > 0.0
    assert num_bins_each_axis > 0

    gg = {}
    gg["bin_width_m"] = float(bin_width_m)
    gg["num_bins_each_axis"] = int(num_bins_each_axis)
    gg["center_x_m"] = float(center_x_m)
    gg["center_y_m"] = float(center_y_m)

    _width = gg["bin_width_m"] * gg["num_bins_each_axis"]
    _num = 1 + gg["num_bins_each_axis"]
    for ax in ["x", "y"]:
        _center = gg["center_{:s}_m".format(ax)]
        _start = (-1/2) * _width + _center
        _stop = (1/2) * _width + _center
        _bin_edges = np.linspace(start=_start, stop=_stop, num=_num)
        gg["{:s}_bin".format(ax)] = binning_utils.Binning(bin_edges=_bin_edges)

    z_bin_edges = np.linspace(
        start=-(1/2) * gg["bin_width_m"],
        stop=(1/2) * gg["bin_width_m"],
        num=2,
    )
    gg["z_bin"] = binning_utils.Binning(bin_edges=z_bin_edges)

    # area
    # ----
    gg["total_num_bins"] = gg["num_bins_each_axis"] ** 2
    gg["bin_area_m2"] = gg["bin_width_m"] ** 2
    gg["total_area_m2"] = gg["total_num_bins"] * gg["bin_area_m2"]
    return gg


def histogram(groundgrid, cherenkov_bunches):
    bin_photon_assignment = {}
    num_overflow = 0

    for ibunch, bunch in enumerate(cherenkov_bunches):
        cer_x_m = cpw.CM2M * bunch[cpw.I.BUNCH.X_CM]
        cer_y_m = cpw.CM2M * bunch[cpw.I.BUNCH.Y_CM]
        cer_z_m = 0.0 # by definition of ground / observation-level

        cer_cx = bunch[cpw.I.BUNCH.CX_RAD]
        cer_cy = bunch[cpw.I.BUNCH.CY_RAD]
        cer_cz = spherical_coordinates.restore_cz(cx=cer_cx, cy=cer_cy)

        cer_w = bunch[cpw.I.BUNCH.BUNCH_SIZE_1]

        overlap = ray_voxel_overlap.estimate_overlap_of_ray_with_voxels(
            support=[cer_x_m, cer_y_m, cer_z_m],
            direction=[cer_cx, cer_cy, cer_cz],
            x_bin_edges=groundgrid["x_bin"]["edges"],
            y_bin_edges=groundgrid["y_bin"]["edges"],
            z_bin_edges=groundgrid["z_bin"]["edges"],
        )

        num_overlaps = len(overlap["x"])
        if num_overlaps == 0:
            num_overflow += 1
        else:
            for n in range(num_overlaps):
                bin_idx = (overlap["x"][n], overlap["y"][n])
                if bin_idx in bin_photon_assignment:
                    bin_photon_assignment[bin_idx][0].append(ibunch)
                    bin_photon_assignment[bin_idx][1].append(cer_w)
                else:
                    bin_photon_assignment[bin_idx] = ([ibunch], [cer_w])

    return bin_photon_assignment, num_overflow


def assign(
    groundgrid,
    cherenkov_bunches,
    threshold_num_photons,
    prng,
    bin_idxs_limitation=None,
):
    bin_photon_assignment, num_photons_overflow = histogram(
        groundgrid=groundgrid,
        cherenkov_bunches=cherenkov_bunches,
    )

    bin_idxs_above_threshold = find_bin_idxs_above_threshold(
        bin_photon_assignment=bin_photon_assignment,
        threshold_num_photons=threshold_num_photons,
    )

    if bin_idxs_limitation:
        bin_idxs_above_threshold = apply_bin_limitation_and_warn(
            bin_idxs_above_threshold=bin_idxs_above_threshold,
            bin_idxs_limitation=bin_idxs_limitation,
        )

    out = {}
    out["num_photons_overflow"] = num_photons_overflow
    out["num_bins_above_threshold"] = len(bin_idxs_above_threshold)

    if out["num_bins_above_threshold"]:
        out["choice"] = None
    else:
        out["choice"] = draw_random_bin_choice(
            groundgrid=groundgrid,
            cherenkov_bunches=cherenkov_bunches,
            bin_photon_assignment=bin_photon_assignment,
            bin_idxs_above_threshold=bin_idxs_above_threshold,
            prng=prng,
        )

    return out


def find_bin_idxs_above_threshold(bin_photon_assignment, threshold_num_photons):
    bin_idxs = []
    for bin_idx in bin_photon_assignment:
        weights = bin_photon_assignment[bin_idx][1]
        if np.sum(weights) >= threshold_num_photons:
            bin_idxs.append(bin_idx)
    return bin_idxs


def draw_bin_idx(bin_idxs, prng):
    return bin_idxs[prng.choice(a=len(bin_idxs))]


def draw_random_bin_choice(
    groundgrid,
    cherenkov_bunches,
    bin_photon_assignment,
    bin_idxs_above_threshold,
    prng,
):
    bin_idx = draw_bin_idx(bin_idxs=bin_idxs_above_threshold, prng=prng)

    cc = {}
    cc["bin_idx_x"] = bin_idx[0]
    cc["bin_idx_y"] = bin_idx[1]
    cc["core_x_m"] = groundgrid["x_bin"]["centers"][cc["bin_idx_x"]]
    cc["core_y_m"] = groundgrid["y_bin"]["centers"][cc["bin_idx_y"]]
    bin_photon_mask = np.array(bin_photon_assignment[bin_idx][0])

    cc["cherenkov_bunches"] = cherenkov_bunches[bin_photon_mask, :].copy()
    cc["cherenkov_bunches"][:, cpw.I.BUNCH.X_CM] -= cpw.M2CM * cc["core_x_m"]
    cc["cherenkov_bunches"][:, cpw.I.BUNCH.Y_CM] -= cpw.M2CM * cc["core_y_m"]

    return cc
