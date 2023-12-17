import json_utils
import os
import rename_after_writing as rnw
from os import path as op
from os.path import join as opj

from . import bookkeeping


def read(plenoirf_dir):
    """
    Returns the configuration of a plenoirf directory.
    """
    cfg = json_utils.tree.read(opj(plenoirf_dir, "config"))
    cfg["sites"] = compile_sites(sites=cfg["sites"])
    cfg["particles"] = compile_particles(particles=cfg["particles"])
    assert_config_random_seed_offsets_did_not_change_since_first_seen(
        plenoirf_dir=plenoirf_dir, config=cfg
    )
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
            "namibia": {"random_seed_offset": 6},
            "chile": {"random_seed_offset": 4},
        },
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

    bookkeeping.random_seed_offset.assert_valid_dict(
        obj=sites["instruemnt_response"]
    )
    return sites


def make_particles():
    return {
        "gamma": {
            "random_seed_offset": 1,
        },
        "electron": {
            "random_seed_offset": 3,
        },
        "proton": {
            "random_seed_offset": 14,
        },
        "helium": {
            "random_seed_offset": 402,
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


def assert_config_random_seed_offsets_did_not_change_since_first_seen(
    plenoirf_dir,
    config,
):
    fresh = {}
    fresh["sites_instruemnt_response"] = config["sites"]["instruemnt_response"]
    fresh["particles"] = config["particles"]

    path = opj(plenoirf_dir, ".random_seed_offsets_when_first_seen.json")
    if op.exists(path):
        with open(path, "rt") as f:
            last = json_utils.loads(f.read())

        union = {}
        union[
            "sites_instruemnt_response"
        ] = bookkeeping.random_seed_offset.combine_into_valid_union(
            last=last["sites_instruemnt_response"],
            fresh=fresh["sites_instruemnt_response"],
        )
        union[
            "particles"
        ] = bookkeeping.random_seed_offset.combine_into_valid_union(
            last=last["particles"], fresh=fresh["particles"]
        )
        with open(path, "wt") as f:
            f.write(json_utils.dumps(union))

    else:
        with open(path, "wt") as f:
            f.write(json_utils.dumps(fresh))
