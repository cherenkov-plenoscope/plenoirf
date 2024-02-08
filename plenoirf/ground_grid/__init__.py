import ray_voxel_overlap
import numpy as np
import binning_utils
import corsika_primary as cpw
import spherical_coordinates
import tempfile
import json_utils
import os
import subprocess
import json_line_logger

from . import io


def make_ground_grid_config(
    bin_width_m,
    num_bins_each_axis,
    cherenkov_pool_median_x_m,
    cherenkov_pool_median_y_m,
    prng,
):
    assert bin_width_m > 0.0
    assert num_bins_each_axis > 0

    random_radius_m = (1 / 2) * bin_width_m

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
        _start = (-1 / 2) * _width + _center
        _stop = (1 / 2) * _width + _center
        _bin_edges = np.linspace(start=_start, stop=_stop, num=_num)
        gg["{:s}_bin".format(ax)] = binning_utils.Binning(bin_edges=_bin_edges)

    z_bin_edges = np.linspace(
        start=-(1 / 2) * gg["bin_width_m"],
        stop=(1 / 2) * gg["bin_width_m"],
        num=2,
    )
    gg["z_bin"] = binning_utils.Binning(bin_edges=z_bin_edges)

    # area
    # ----
    gg["num_bins_thrown"] = gg["num_bins_each_axis"] ** 2
    gg["bin_area_m2"] = gg["bin_width_m"] ** 2
    gg["area_thrown_m2"] = gg["num_bins_thrown"] * gg["bin_area_m2"]
    return gg


def _make_fake_runh():
    runh = np.zeros(273, dtype=np.float32)
    runh[cpw.I.RUNH.MARKER] = cpw.I.RUNH.MARKER_FLOAT32
    runh[cpw.I.RUNH.RUN_NUMBER] = 1
    runh[cpw.I.RUNH.NUM_EVENTS] = 1
    return runh


def _make_fake_evth():
    runh = np.zeros(273, dtype=np.float32)
    runh[cpw.I.EVTH.MARKER] = cpw.I.EVTH.MARKER_FLOAT32
    runh[cpw.I.EVTH.RUN_NUMBER] = 1
    runh[cpw.I.EVTH.EVENT_NUMBER] = 1
    return runh


def assign_cherenkov_bunches2(groundgrid, cherenkov_bunch_storage_path):
    exe = "/home/relleums/Desktop/starter_kit/merlict/merlict/c89/merlict_c89/build/bin/ground_grid"

    #log = json_line_logger.LoggerStdout()

    with tempfile.TemporaryDirectory() as tmp:
        cpath = os.path.join(tmp, "config")
        opath = os.path.join(tmp, "assignment.jsonl")

        #with json_line_logger.TimeDelta(log, "assign write config"):
        with open(cpath, "wt") as f:
            for dim in ["x", "y", "z"]:
                dimbin = "{:s}_bin".format(dim)
                num_edges = 1 + groundgrid[dimbin]["num"]
                f.write("{:d}\n".format(num_edges))
                for i in range(num_edges):
                    f.write(
                        "{:f}\n".format(
                            1e2 * groundgrid[dimbin]["edges"][i]
                        )
                    )

        #with json_line_logger.TimeDelta(log, "assign call merlict c89"):
        rc = subprocess.call(
            [exe, cherenkov_bunch_storage_path, opath, cpath]
        )
        assert rc == 0

        #with json_line_logger.TimeDelta(log, "assign read json"):
        ass = json_utils.lines.read(opath)[0]

       # with json_line_logger.TimeDelta(log, "assign convert json"):
        num_overflow = ass["num_overflow"]
        bin_photon_assignment = {}
        for key in ass["assignment"]:
            xstr, ystr = str.split(key, ",")
            pay = ass["assignment"][key]
            bin_idx = (int(xstr), int(ystr))
            bin_photon_assignment[bin_idx] = (pay["idx"], pay["weight"])

    return bin_photon_assignment, num_overflow


def assign_cherenkov_bunches(groundgrid, cherenkov_bunch_storage_path):
    bin_photon_assignment = {}
    num_overflow = 0

    ibunch = 0
    with cpw.cherenkov.CherenkovEventTapeReader(
        path=cherenkov_bunch_storage_path
    ) as _run:
        _evth, cherenkov_reader  = next(_run)
        for cherenkov_block in cherenkov_reader:
            for bunch in cherenkov_block:
                cer_x_m = cpw.CM2M * bunch[cpw.I.BUNCH.X_CM]
                cer_y_m = cpw.CM2M * bunch[cpw.I.BUNCH.Y_CM]
                cer_z_m = 0.0  # by definition of ground / observation-level

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
                            bin_photon_assignment[bin_idx][1] += cer_w
                        else:
                            bin_photon_assignment[bin_idx] = [[ibunch], cer_w]
                ibunch += 1

    return bin_photon_assignment, num_overflow


def assign2(
    groundgrid,
    cherenkov_bunch_storage_path,
    threshold_num_photons,
    prng,
):
    #log = json_line_logger.LoggerStdout()
    #log.info("---------start-------------")
    #with json_line_logger.TimeDelta(log, "py-loop"):
    #    bin_photon_assignment, num_photons_overflow = assign_cherenkov_bunches(
    #        groundgrid=groundgrid,
    #        cherenkov_bunch_storage_path=cherenkov_bunch_storage_path,
    #    )

    #with json_line_logger.TimeDelta(log, "c-loop"):
    bin_photon_assignment, num_photons_overflow = assign_cherenkov_bunches2(
        groundgrid=groundgrid,
        cherenkov_bunch_storage_path=cherenkov_bunch_storage_path,
    )

    bin_idxs_above_threshold = find_bin_idxs_above_or_equal_threshold(
        bin_photon_assignment=bin_photon_assignment,
        threshold_num_photons=threshold_num_photons,
    )

    out = {}
    out["num_photons_overflow"] = num_photons_overflow
    out["num_bins_above_threshold"] = len(bin_idxs_above_threshold)

    if out["num_bins_above_threshold"] == 0:
        out["choice"] = None
    else:
        out["choice"] = draw_random_bin_choice(
            groundgrid=groundgrid,
            bin_photon_assignment=bin_photon_assignment,
            bin_idxs_above_threshold=bin_idxs_above_threshold,
            prng=prng,
        )

    # compare to classic scatter algorithm as used in the
    # Cherenkov-Telescpe-Array
    # ---------------------------------------------------
    out["scatter_histogram"] = histogram_bins_in_scatter_radius(
        groundgrid=groundgrid,
        bin_idxs=bin_idxs_above_threshold,
    )
    debug = {}
    debug["bin_photon_assignment"] = bin_photon_assignment
    return out, debug


def find_bin_idxs_above_or_equal_threshold(
    bin_photon_assignment, threshold_num_photons
):
    bin_idxs = []
    for bin_idx in bin_photon_assignment:
        summed_weights = bin_photon_assignment[bin_idx][1]
        if summed_weights >= threshold_num_photons:
            bin_idxs.append(bin_idx)
    return bin_idxs


def draw_bin_idx(bin_idxs, prng):
    return bin_idxs[prng.choice(a=len(bin_idxs))]


def draw_random_bin_choice(
    groundgrid,
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
    cc["cherenkov_bunches_idxs"] = np.array(bin_photon_assignment[bin_idx][0])
    return cc


def histogram_bins_in_scatter_radius(groundgrid, bin_idxs):
    NUM_BINS = 16
    scatter_radius_bin_edges_m = radii_for_area_power_space(num_bins=NUM_BINS)

    scatter_radii_of_bins_above_threshold = []
    for bin_idx in bin_idxs:
        bin_x_m = groundgrid["x_bin"]["centers"][bin_idx[0]]
        bin_y_m = groundgrid["y_bin"]["centers"][bin_idx[1]]
        bin_radius_m = np.hypot(bin_x_m, bin_y_m)
        scatter_radii_of_bins_above_threshold.append(bin_radius_m)
    bin_counts = np.histogram(
        scatter_radii_of_bins_above_threshold,
        bins=scatter_radius_bin_edges_m,
    )[0]
    out = {
        "scatter_radius_bin_edges_m": scatter_radius_bin_edges_m,
        "bin_counts": bin_counts,
    }
    return out


def radii_for_area_power_space(start=1e6, factor=2.0, num_bins=16):
    radii = [0]
    for i in range(num_bins):
        area = start * factor**i
        r = np.sqrt(area) / np.pi
        radii.append(r)
    return np.array(radii)


def bin_photon_assignment_to_array_roi(
    bin_photon_assignment, x_bin, y_bin, r_bin, dtype=np.float32
):
    x_bin = int(x_bin)
    y_bin = int(y_bin)
    r_bin = int(r_bin)
    assert r_bin >= 0
    dia = 2 * r_bin + 1
    out = np.zeros(shape=(dia, dia), dtype=dtype)
    for bin_idx in bin_photon_assignment:
        _x = int(bin_idx[0])
        _y = int(bin_idx[1])
        ox = _x - x_bin + r_bin
        if 0 <= ox < dia:
            oy = _y - y_bin + r_bin
            if 0 <= oy < dia:
                summed_weights = bin_photon_assignment[bin_idx][1]
                i = summed_weights
                out[ox, oy] = i
    return out


def bin_photon_assignment_to_array(
    bin_photon_assignment, num_bins_each_axis, dtype=np.float32
):
    dia = num_bins_each_axis
    out = np.zeros(shape=(dia, dia), dtype=dtype)
    for bin_idx in bin_photon_assignment:
        x, y = bin_idx
        summed_weights = bin_photon_assignment[bin_idx][1]
        out[x, y] = summed_weights
    return out
