import json_utils
import numpy as np
import rename_after_writing as rnw
import os
from os import path as op
from os.path import join as opj

from .. import bookkeeping
from . import version_control


def read(plenoirf_dir):
    """
    Returns the configuration of a plenoirf directory.
    """
    cfg = json_utils.tree.read(opj(plenoirf_dir, "config"))
    cfg["sites"] = compile_sites(sites=cfg["sites"])
    return cfg


def write_default(plenoirf_dir, build_dir):
    pdir = plenoirf_dir
    os.makedirs(opj(pdir, "config"), exist_ok=False)
    with rnw.open(opj(pdir, "config", "executables.json"), "wt") as f:
        f.write(
            json_utils.dumps(
                make_executables_paths(build_dir=build_dir), indent=4
            )
        )

    with rnw.open(opj(pdir, "config", "sites.json"), "wt") as f:
        f.write(json_utils.dumps(make_sites(), indent=4))
    with rnw.open(opj(pdir, "config", "particles.json"), "wt") as f:
        f.write(json_utils.dumps(make_particles(), indent=4))

    with rnw.open(opj(pdir, "config", "magnetic_deflection.json"), "wt") as f:
        f.write(json_utils.dumps(make_magnetic_deflection(), indent=4))

    with rnw.open(opj(pdir, "config", "plenoptics.json"), "wt") as f:
        f.write(json_utils.dumps(make_plenoptics(), indent=4))

    with rnw.open(opj(pdir, "config", "instruments.json"), "wt") as f:
        f.write(json_utils.dumps(make_instruments(), indent=4))

    with rnw.open(opj(pdir, "config", "pointing.json"), "wt") as f:
        f.write(json_utils.dumps(make_pointing(), indent=4))

    with rnw.open(opj(pdir, "config", "sum_trigger.json"), "wt") as f:
        f.write(json_utils.dumps(make_sum_trigger(), indent=4))

    with rnw.open(opj(pdir, "config", "ground_grid.json"), "wt") as f:
        f.write(json_utils.dumps(make_groundgrid(), indent=4))


def make_executables_paths(build_dir="build"):
    return {
        "corsika_primary_path": opj(
            build_dir,
            "corsika",
            "modified",
            "corsika-75600",
            "run",
            "corsika75600Linux_QGSII_urqmd",
        ),
        "merlict_plenoscope_propagator_path": opj(
            build_dir, "merlict", "merlict-plenoscope-propagation"
        ),
        "merlict_plenoscope_calibration_map_path": opj(
            build_dir, "merlict", "merlict-plenoscope-calibration-map"
        ),
        "merlict_plenoscope_calibration_reduce_path": opj(
            build_dir, "merlict", "merlict-plenoscope-calibration-reduce"
        ),
        "merlict_plenoscope_raw_photon_propagation_path": opj(
            build_dir, "merlict", "merlict-plenoscope-raw-photon-propagation"
        ),
    }


def make_sites():
    out = {
        "instruemnt_response": ["namibia", "chile"],
        "only_magnetic_deflection": ["lapalma", "namibiaOff"],
    }
    return out


def compile_sites(sites):
    sites["magnetic_deflection"] = list(
        set(sites["instruemnt_response"] + sites["only_magnetic_deflection"])
    )
    # assert only_magnetic_deflection does not lie
    for key in sites["only_magnetic_deflection"]:
        assert key not in sites["instruemnt_response"]
    return sites


def make_particles():
    return ["gamma", "electron", "proton", "helium"]


def make_magnetic_deflection():
    return {
        "energy_stop_GeV": 64.0,
        "num_showers_target": 2 * 1000 * 1000,
        "run": {
            "num_runs": 192,
            "num_showers_per_run": 1280,
        },
    }


def make_plenoptics():
    return {"random_seed": 42, "minimal": False}


def make_instruments():
    """
    Returns the list of instrumtns which are simulated.
    The instruments here are keys (str) which point to an instrument in
    the plenoptic package.

    The Portal Cherenkov plenoscope in its default geometry without
    deformations and without misalignments is called 'diag9_default_default'.
    """
    return ["diag9_default_default"]


def make_pointing():
    return {
        "model": "cable_robo_mount",
        "range": {
            "max_zenith_distance_rad": np.deg2rad(60.0),
            "run_half_angle_rad": np.deg2rad(5.0),
        },
    }


def make_sum_trigger():
    return {
        "object_distances_m": [
            5000.0,
            6164.0,
            7600.0,
            9369.0,
            11551.0,
            14240.0,
            17556.0,
            21644.0,
            26683.0,
            32897.0,
            40557.0,
            50000.0,
        ],
        "threshold_pe": 105,
        "integration_time_slices": 10,
        "image": {
            "image_outer_radius_rad": np.deg2rad(3.25 - 0.033335),
            "pixel_spacing_rad": np.deg2rad(0.06667),
            "pixel_radius_rad": np.deg2rad(0.146674),
            "max_number_nearest_lixel_in_pixel": 7,
        },
    }


def make_ground_grid():
    return {
        "geometry": {
            "bin_width_m": 1e2,
            "num_bins_each_axis": 1024,
        },
        "threshold_num_photons": 10,
    }
