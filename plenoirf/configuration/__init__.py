import json_utils
import numpy as np
import rename_after_writing as rnw
import merlict_development_kit_python as mlidev
import gamma_ray_reconstruction as gamrec
import os
import atmospheric_cherenkov_response
import binning_utils
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


def write_default(plenoirf_dir):
    pdir = plenoirf_dir
    os.makedirs(opj(pdir, "config"), exist_ok=False)

    with rnw.open(opj(pdir, "config", "sites.json"), "wt") as f:
        f.write(json_utils.dumps(make_sites(), indent=4))
    with rnw.open(opj(pdir, "config", "particles.json"), "wt") as f:
        f.write(json_utils.dumps(make_particles(), indent=4))
    with rnw.open(
        opj(pdir, "config", "particles_scatter_cone.json"), "wt"
    ) as f:
        f.write(json_utils.dumps(make_particles_scatter_cone(), indent=4))

    with rnw.open(opj(pdir, "config", "energy_range.json"), "wt") as f:
        f.write(json_utils.dumps(make_energy_range(), indent=4))

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
        f.write(json_utils.dumps(make_ground_grid(), indent=4))

    with rnw.open(opj(pdir, "config", "debugging.json"), "wt") as f:
        f.write(json_utils.dumps(make_debugging(), indent=4))

    with rnw.open(
        opj(pdir, "config", "cherenkov_classification.json"), "wt"
    ) as f:
        f.write(json_utils.dumps(make_cherenkov_classification(), indent=4))

    with rnw.open(opj(pdir, "config", "reconstruction.json"), "wt") as f:
        f.write(json_utils.dumps(make_reconstruction(), indent=4))

    with rnw.open(
        opj(pdir, "config", "merlict_plenoscope_propagator_config.json"), "wt"
    ) as f:
        f.write(
            json_utils.dumps(
                make_merlict_plenoscope_propagator_config(), indent=4
            )
        )


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
    out = {}
    out["energy_stop_GeV_power10"] = {
        "decade": 1,
        "bin": 4,
        "num_bins_per_decade": 5,
    }
    out["energy_stop_GeV"] = binning_utils.power10.lower_bin_edge(
        **out["energy_stop_GeV_power10"]
    )
    out["num_showers_target"] = 2 * 1000 * 1000
    out["run"] = {
        "num_runs": 192,
        "num_showers_per_run": 1280,
    }
    out["query_mode"] = "cone"
    return out


def make_energy_range():
    out = {}
    out["energy_start_GeV_power10"] = {
        "decade": -1,
        "bin": 2,
        "num_bins_per_decade": 5,
    }
    out["energy_start_GeV"] = binning_utils.power10.lower_bin_edge(
        **out["energy_start_GeV_power10"]
    )
    out["energy_stop_GeV_power10"] = {
        "decade": 3,
        "bin": 2,
        "num_bins_per_decade": 5,
    }
    out["energy_stop_GeV"] = binning_utils.power10.lower_bin_edge(
        **out["energy_stop_GeV_power10"]
    )
    return out


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
        "model": "cable_robot",
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


def make_debugging():
    return {
        "run": {
            "min_num_events": 1,
            "fraction_of_events": 1e-2,
        }
    }


def make_merlict_plenoscope_propagator_config():
    return mlidev.plenoscope_propagator.make_plenoscope_propagator_config(
        night_sky_background_ligth_key="nsb_la_palma_2013_benn",
        photo_electric_converter_key="hamamatsu_r11920_100_05",
    )


def make_cherenkov_classification():
    return {
        "region_of_interest": {
            "time_offset_start_s": -10e-9,
            "time_offset_stop_s": 10e-9,
            "direction_radius_deg": 2.0,
            "object_distance_offsets_m": [
                4000.0,
                2000.0,
                0.0,
                -2000.0,
            ],
        },
        "min_num_photons": 17,
        "neighborhood_radius_deg": 0.075,
        "direction_to_time_mixing_deg_per_s": 0.375e9,
    }


def make_reconstruction():
    return {
        "trajectory": gamrec.trajectory.v2020dec04iron0b.config.make_example_config_for_71m_plenoscope(
            fov_radius_deg=3.25
        ),
    }


def make_particles_scatter_cone():
    acr = atmospheric_cherenkov_response
    out = {}
    for particle_key in make_particles():
        out[particle_key] = acr.particles.scatter_cone(key=particle_key)
    return out
