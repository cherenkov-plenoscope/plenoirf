import json_utils
import numpy as np
import rename_after_writing as rnw
import merlict_development_kit_python as mlidev
import gamma_ray_reconstruction as gamrec
import os
import copy
import atmospheric_cherenkov_response
import binning_utils
import solid_angle_utils
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


def read_if_None(plenoirf_dir, config):
    if config:
        return config
    else:
        return read(plenoirf_dir)


def write_default(plenoirf_dir):
    pdir = plenoirf_dir
    os.makedirs(opj(pdir, "config"), exist_ok=False)

    def jdumps(o):
        return json_utils.dumps(o, indent=4)

    with rnw.open(opj(pdir, "config", "sites.json"), "wt") as f:
        f.write(jdumps(make_sites()))
    with rnw.open(opj(pdir, "config", "particles.json"), "wt") as f:
        f.write(jdumps(make_particles()))

    with rnw.open(
        opj(pdir, "config", "particles_simulated_energy_distribution.json"),
        "wt",
    ) as f:
        f.write(jdumps(make_particles_simulated_energy_distribution()))

    with rnw.open(
        opj(pdir, "config", "particles_scatter_solid_angle.json"), "wt"
    ) as f:
        f.write(jdumps(make_particles_scatter_solid_angle()))

    with rnw.open(opj(pdir, "config", "magnetic_deflection.json"), "wt") as f:
        f.write(jdumps(make_magnetic_deflection()))

    with rnw.open(opj(pdir, "config", "plenoptics.json"), "wt") as f:
        f.write(jdumps(make_plenoptics()))

    with rnw.open(opj(pdir, "config", "instruments.json"), "wt") as f:
        f.write(jdumps(make_instruments()))

    with rnw.open(opj(pdir, "config", "pointing.json"), "wt") as f:
        f.write(jdumps(make_pointing()))

    with rnw.open(opj(pdir, "config", "sum_trigger.json"), "wt") as f:
        f.write(jdumps(make_sum_trigger()))

    with rnw.open(opj(pdir, "config", "ground_grid.json"), "wt") as f:
        f.write(jdumps(make_ground_grid()))

    with rnw.open(opj(pdir, "config", "debugging.json"), "wt") as f:
        f.write(jdumps(make_debugging()))

    with rnw.open(
        opj(pdir, "config", "cherenkov_classification.json"), "wt"
    ) as f:
        f.write(jdumps(make_cherenkov_classification()))

    with rnw.open(opj(pdir, "config", "reconstruction.json"), "wt") as f:
        f.write(jdumps(make_reconstruction()))

    with rnw.open(
        opj(pdir, "config", "merlict_plenoscope_propagator_config.json"), "wt"
    ) as f:
        f.write(jdumps(make_merlict_plenoscope_propagator_config()))

    with rnw.open(opj(pdir, "config", "population_target.json"), "wt") as f:
        f.write(jdumps(make_population_target()))

    with rnw.open(
        opj(pdir, "config", "population_partitioning.json"), "wt"
    ) as f:
        f.write(jdumps(make_population_partitioning()))


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


def _suggested_ratio_of_thrown_particle_types():
    o = {"gamma": 35, "electron": 15, "proton": 40, "helium": 10}
    s = 0
    for key in o:
        s += o[key]
    for key in o:
        o[key] /= s
    return o


def make_magnetic_deflection():
    out = {}
    out["energy_start_GeV_power10"] = {
        "decade": -1,
        "bin": 2,
        "num_bins_per_decade": 5,
    }
    out["energy_stop_GeV_power10"] = {
        "decade": 1,
        "bin": 4,
        "num_bins_per_decade": 5,
    }
    out["num_showers_target"] = 2 * 1000 * 1000
    out["run"] = {
        "num_runs": 192,
        "num_showers_per_run": 1280,
    }
    return out


def make_particles_scatter_solid_angle():
    acr = atmospheric_cherenkov_response
    out = {}
    for particle_key in make_particles():
        cone = acr.particles.scatter_cone(key=particle_key)
        out[particle_key] = {}
        out[particle_key]["energy_GeV"] = cone["energy_GeV"]
        out[particle_key]["solid_angle_sr"] = (
            solid_angle_utils.cone.solid_angle(
                half_angle_rad=cone["half_angle_rad"]
            )
        )
    return out


def make_particles_simulated_energy_distribution():
    common_energy_stop_GeV_power10 = {
        "decade": 3,
        "bin": 2,
        "num_bins_per_decade": 5,
    }

    out = {}
    for pk in ["gamma", "electron", "proton", "helium"]:
        out[pk] = {}
        out[pk]["spectrum"] = {}
        out[pk]["spectrum"]["type"] = "power_law"
        out[pk]["spectrum"]["power_slope"] = -1.5
        out[pk]["energy_stop_GeV_power10"] = copy.deepcopy(
            common_energy_stop_GeV_power10
        )

    out["gamma"]["energy_start_GeV_power10"] = {
        "decade": -1,
        "bin": 2,
        "num_bins_per_decade": 5,
    }
    out["electron"]["energy_start_GeV_power10"] = {
        "decade": -1,
        "bin": 2,
        "num_bins_per_decade": 5,
    }
    out["proton"]["energy_start_GeV_power10"] = {
        "decade": 0,
        "bin": 7,
        "num_bins_per_decade": 10,
    }
    out["helium"]["energy_start_GeV_power10"] = {
        "decade": 1,
        "bin": 0,
        "num_bins_per_decade": 5,
    }

    for pk in out:
        particle = atmospheric_cherenkov_response.particles.init(pk)
        asseret_energy_start_GeV_is_valid(
            particle=particle,
            energy_start_GeV=binning_utils.power10.lower_bin_edge(
                **out[pk]["energy_start_GeV_power10"]
            ),
        )
    return out


def asseret_energy_start_GeV_is_valid(particle, energy_start_GeV):
    if particle["corsika"]["min_energy_GeV"] is not None:
        if energy_start_GeV < particle["corsika"]["min_energy_GeV"]:
            msg = "The energy {:f}GeV is too low for particle '{:s}'. ".format(
                energy_start_GeV, particle["key"]
            )
            msg += "Minimum energy is {:f}GeV.".format(
                particle["corsika"]["min_energy_GeV"]
            )
            raise AssertionError(msg)


def make_plenoptics():
    return {"random_seed": 42, "minimal": False}


def make_instruments():
    """
    Returns the list of instrumtns which are simulated.
    The instruments here are keys (str) which point to an instrument in
    the plenoptics package.

    The Portal Cherenkov plenoscope in its default geometry without
    deformations and without misalignments is called 'diag9_default_default'.
    """
    return ["diag9_default_default"]


def make_pointing():
    return {
        "model": "cable_robot",
        "range": {
            "max_zenith_distance_rad": np.deg2rad(45.0),
        },
    }


def make_sum_trigger_object_distance_geomspace_binning(start_m, stop_m, num):
    """
    make a dict similar to binning_utils.Binning
    """
    assert start_m > 0.0
    assert stop_m > 0.0
    assert start_m < stop_m
    assert num > 1

    out = {}

    out["num"] = num
    out["centers"] = np.geomspace(start_m, stop_m, out["num"])

    out["_decade_step"] = np.sqrt(out["centers"][1] / out["centers"][0])
    out["start"] = start_m / out["_decade_step"]
    out["stop"] = stop_m * out["_decade_step"]
    out["limits"] = [out["start"], out["stop"]]

    _edges = np.geomspace(
        out["start"],
        out["stop"],
        len(out["centers"]) * 2 + 1,
    )
    _mask = np.arange(0, len(_edges), 2)
    out["edges"] = _edges[_mask]
    return out


def make_sum_trigger():
    out = {}
    out["object_distances"] = {"start_m": 5e3, "stop_m": 30e3, "num": 12}
    out["object_distances_m"] = (
        make_sum_trigger_object_distance_geomspace_binning(
            **out["object_distances"]
        )["centers"]
    )
    out["threshold_pe"] = 105
    out["integration_time_slices"] = 10
    out["image"] = {
        "image_outer_radius_rad": np.deg2rad(3.25 - 0.033335),
        "pixel_spacing_rad": np.deg2rad(0.06667),
        "pixel_radius_rad": np.deg2rad(0.146674),
        "max_number_nearest_lixel_in_pixel": 7,
    }
    return out


def make_ground_grid():
    return {
        "geometry": {
            "bin_width_m": 1e2,
            "num_bins_each_axis": 1024,
        },
        "threshold_num_photons": 25,
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


def make_population_partitioning():
    o = {}
    o["num_showers_per_corsika_run"] = 1000

    # adds about 10MB per event to the temporary directory.
    o["num_showers_per_merlict_run"] = 50
    return o


def make_population_target():
    o = {}
    sugg = _suggested_ratio_of_thrown_particle_types()

    num_per_site_and_instrument = 4 * 1e6
    for ikey in make_instruments():
        o[ikey] = {}
        for skey in make_sites()["instruemnt_response"]:
            o[ikey][skey] = {}
            for pkey in make_particles():
                o[ikey][skey][pkey] = {
                    "num_showers_thrown": int(
                        sugg[pkey] * num_per_site_and_instrument
                    )
                }
    return o
