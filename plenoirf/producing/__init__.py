import os
from os import path as op
from os.path import join as opj
import tempfile
import copy
import numpy as np
import tarfile
import gzip

import magnetic_deflection
import json_utils
import json_line_logger
import merlict_development_kit_python
import spherical_coordinates
import atmospheric_cherenkov_response as acr
import rename_after_writing as rnw
import sparse_numeric_table as spt
import corsika_primary as cpw
import homogeneous_transformation

from .. import outer_telescope_array
from .. import bookkeeping
from .. import configurating
from .. import ground_grid
from .. import event_table
from .. import tar_append
from .. import debugging
from .. import constants

from . import job_to_json
from . import sum_trigger
from . import transform_cherenkov_bunches
from . import draw_primaries_and_pointings
from . import draw_pointing_range
from . import corsika_and_grid


def make_example_job(
    plenoirf_dir,
    run_id=1337,
    site_key="chile",
    particle_key="electron",
    instrument_key="diag9_default_default",
    num_events=128,
    max_num_events_in_merlict_run=12,
):
    job = {}
    job["run_id"] = run_id
    job["plenoirf_dir"] = plenoirf_dir
    job["site_key"] = site_key
    job["particle_key"] = particle_key
    job["instrument_key"] = instrument_key
    job["num_events"] = num_events
    job["max_num_events_in_merlict_run"] = max_num_events_in_merlict_run
    return job


def run_job(job):
    with tempfile.TemporaryDirectory(suffix="-plenoirf") as tmp_dir:
        return run_job_in_dir(job=job, tmp_dir=tmp_dir)


def run_job_in_dir(job, tmp_dir):
    job = compile_job_paths_and_unique_identity(job=job, tmp_dir=tmp_dir)

    os.makedirs(job["paths"]["stage_dir"], exist_ok=True)

    job["logger"] = json_line_logger.LoggerFile(
        path=job["paths"]["logger_tmp"]
    )
    job["logger"].info("starting")

    job["logger"].debug("making tmp_dir: {:s}".format(job["paths"]["tmp_dir"]))
    os.makedirs(job["paths"]["tmp_dir"], exist_ok=True)

    job["logger"].debug(
        "initializing prng with seed: {:d}".format(job["run_id"])
    )
    job["prng"] = np.random.Generator(np.random.PCG64(seed=job["run_id"]))
    job["run"] = {}

    job["run"][
        "event_ids_for_debug"
    ] = debugging.draw_event_ids_for_debug_output(
        num_events_in_run=job["num_events"],
        min_num_events=job["config"]["debug_output"]["run"]["min_num_events"],
        fraction_of_events=job["config"]["debug_output"]["run"][
            "fraction_of_events"
        ],
        prng=job["prng"],
    )
    job["logger"].debug(
        "event-ids for debugging: {:s}.".format(
            str(job["run"]["event_ids_for_debug"].tolist())
        )
    )

    with json_line_logger.TimeDelta(job["logger"], "draw_pointing_range"):
        job = draw_pointing_range.run_job(job=job)

    with json_line_logger.TimeDelta(
        job["logger"], "draw_primaries_and_pointings"
    ):
        job = draw_primaries_and_pointings.run_job(job=job)

    job["event_table"] = event_table.structure.init_table_dynamicsizerecarray()

    with json_line_logger.TimeDelta(job["logger"], "corsika_and_grid"):
        job = corsika_and_grid.run_job(job=job)

    with rnw.open(opj(job["paths"]["tmp_dir"], "job.json"), "wt") as f:
        f.write(json_utils.dumps(job, indent=4))

    job["logger"].info("ending")
    rnw.move(job["paths"]["logger_tmp"], job["paths"]["logger"])
    return job


def read_light_field_camera_config(plenoirf_dir, instrument_key):
    return merlict_development_kit_python.plenoscope_propagator.read_plenoscope_geometry(
        merlict_scenery_path=opj(
            plenoirf_dir,
            "plenoptics",
            "instruments",
            instrument_key,
            "light_field_geometry",
            "input",
            "scenery",
            "scenery.json",
        )
    )


def compile_job_paths_and_unique_identity(job, tmp_dir):
    """
    Adds all kind of static information to the job dict.
    Items in the job dict are not meant to change!
    It is meant to be read only.
    """
    assert job["max_num_events_in_merlict_run"] > 0

    job["run_id_str"] = bookkeeping.uid.make_run_id_str(run_id=job["run_id"])
    job["paths"] = compile_paths(job=job, tmp_dir=tmp_dir)
    job["config"] = configurating.read(
        plenoirf_dir=job["paths"]["plenoirf_dir"]
    )

    _allskycfg = json_utils.tree.read(
        opj(job["paths"]["magnetic_deflection_allsky"], "config")
    )
    job["site"] = copy.deepcopy(_allskycfg["site"])
    job["particle"] = copy.deepcopy(_allskycfg["particle"])

    job["light_field_camera_config"] = read_light_field_camera_config(
        plenoirf_dir=job["paths"]["plenoirf_dir"],
        instrument_key=job["instrument_key"],
    )
    job["instrument"] = {}
    job["instrument"]["field_of_view_half_angle_rad"] = 0.5 * (
        np.deg2rad(job["light_field_camera_config"]["max_FoV_diameter_deg"])
    )
    job["instrument"]["local_speed_of_light_m_per_s"] = (
        constants.speed_of_light_in_vacuum_m_per_s()
        / job["site"]["atmosphere_refractive_index_at_observation_level"]
    )
    return job


def compile_paths(job, tmp_dir):
    paths = {}
    paths["plenoirf_dir"] = job["plenoirf_dir"]

    paths["stage_dir"] = opj(
        job["plenoirf_dir"],
        "response",
        job["instrument_key"],
        job["site_key"],
        job["particle_key"],
        "stage",
    )

    # logger
    # ------
    paths["logger"] = opj(paths["stage_dir"], job["run_id_str"] + "_log.jsonl")
    paths["logger_tmp"] = paths["logger"] + ".tmp"

    # input
    # -----
    paths["magnetic_deflection_allsky"] = opj(
        job["plenoirf_dir"],
        "magnetic_deflection",
        job["site_key"],
        job["particle_key"],
    )
    paths["light_field_calibration"] = opj(
        job["plenoirf_dir"],
        "plenoptics",
        "instruments",
        job["instrument_key"],
        "light_field_geometry",
    )

    # temporary
    # ---------
    paths["tmp_dir"] = tmp_dir
    paths["tmp"] = {}

    paths["tmp"]["cherenkov_pools"] = opj(tmp_dir, "cherenkov_pools.tar")
    paths["tmp"]["cherenkov_pools_block_fmt"] = opj(
        tmp_dir, "cherenkov_pools_{block:06d}.tar"
    )
    paths["tmp"]["particle_pools_dat"] = opj(tmp_dir, "particle_pools.dat")
    paths["tmp"]["particle_pools_tar"] = opj(tmp_dir, "particle_pools.tar.gz")

    paths["tmp"]["ground_grid_intensity"] = opj(tmp_dir, "ground_grid.tar")
    paths["tmp"]["ground_grid_intensity_roi"] = opj(
        tmp_dir, "ground_grid_roi.tar"
    )

    paths["tmp"]["corsika_stdout"] = opj(tmp_dir, "corsika.stdout")
    paths["tmp"]["corsika_stderr"] = opj(tmp_dir, "corsika.stderr")
    paths["tmp"]["merlict_stdout"] = opj(tmp_dir, "merlict.stdout")
    paths["tmp"]["merlict_stderr"] = opj(tmp_dir, "merlict.stderr")

    # debug cache
    # -----------
    paths["cache"] = {}
    paths["cache"]["primary"] = opj(
        tmp_dir, job["run_id_str"] + "_corsika_primary_steering.tar"
    )

    # debug output
    # ------------
    paths["debug"] = {}
    paths["debug"]["draw_primary_and_pointing"] = opj(
        paths["stage_dir"],
        job["run_id_str"] + "_debug_" + "draw_primary_and_pointing" + ".tar",
    )

    return paths
