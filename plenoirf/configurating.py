import os
import json_utils
from os import path as op
from os.path import join as opj


def read(run_dir):
    cfg = json_utils.tree.read(opj(run_dir, "config"))
    cfg["sites"] = compile_sites(sites=cfg["sites"])
    return cfg


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

    # assert
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
