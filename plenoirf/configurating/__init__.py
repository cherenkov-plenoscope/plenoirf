import json_utils
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
    cfg["particles"] = compile_particles(particles=cfg["particles"])
    return cfg


def write_default(plenoirf_dir, build_dir):
    os.makedirs(opj(plenoirf_dir, "config"), exist_ok=False)
    with rnw.open(opj(plenoirf_dir, "config", "executables.json"), "wt") as f:
        f.write(
            json_utils.dumps(
                make_executables_paths(build_dir=build_dir), indent=4
            )
        )

    with rnw.open(opj(plenoirf_dir, "config", "sites.json"), "wt") as f:
        f.write(json_utils.dumps(make_sites(), indent=4))
    with rnw.open(opj(plenoirf_dir, "config", "particles.json"), "wt") as f:
        f.write(json_utils.dumps(make_particles(), indent=4))

    with rnw.open(
        opj(plenoirf_dir, "config", "magnetic_deflection.json"), "wt"
    ) as f:
        f.write(json_utils.dumps(make_magnetic_deflection(), indent=4))

    with rnw.open(opj(plenoirf_dir, "config", "plenoptics.json"), "wt") as f:
        f.write(json_utils.dumps(make_plenoptics(), indent=4))

    with rnw.open(opj(plenoirf_dir, "config", "instruments.json"), "wt") as f:
        f.write(json_utils.dumps(make_instruments(), indent=4))


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
        "instruemnt_response": {
            "namibia": {"random_seed_offset": 0},
            "chile": {"random_seed_offset": 1},
        },
        "only_magnetic_deflection": ["lapalma", "namibiaOff"],
    }
    return out


def compile_sites(sites):
    sites["magnetic_deflection"] = list(
        set(
            list(sites["instruemnt_response"].keys())
            + sites["only_magnetic_deflection"]
        )
    )

    # assert only_magnetic_deflection does not lie
    for key in sites["only_magnetic_deflection"]:
        assert key not in sites["instruemnt_response"]

    bookkeeping.random_seed_offset.assert_valid_dict(
        obj=sites["instruemnt_response"]
    )
    return sites


def make_particles():
    return {
        "gamma": {
            "random_seed_offset": 0,
        },
        "electron": {
            "random_seed_offset": 1,
        },
        "proton": {
            "random_seed_offset": 2,
        },
        "helium": {
            "random_seed_offset": 3,
        },
    }


def compile_particles(particles):
    bookkeeping.random_seed_offset.assert_valid_dict(obj=particles)
    return particles


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
