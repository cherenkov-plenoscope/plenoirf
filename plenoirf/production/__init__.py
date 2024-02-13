import os
from os import path as op
from os.path import join as opj
import tempfile
import copy
import numpy as np

import json_utils
import json_line_logger as jll
import merlict_development_kit_python as mlidev
import rename_after_writing as rnw
import plenopy

from .. import bookkeeping
from .. import configuration
from .. import ground_grid
from .. import event_table
from .. import constants

from . import job_io
from . import sum_trigger
from . import draw_event_uids_for_debugging
from . import draw_pointing_range
from . import draw_primaries_and_pointings
from . import simulate_shower_and_collect_cherenkov_light_in_grid
from . import split_event_tape_into_blocks
from . import inspect_particle_pool
from . import simulate_hardware
from . import simulate_loose_trigger
from . import classify_cherenkov_photons


def make_example_job(
    plenoirf_dir,
    run_id=1234,
    site_key="chile",
    particle_key="gamma",
    instrument_key="diag9_default_default",
    num_events=1000,
    max_num_events_in_merlict_run=100,
    cache=True,
):
    job = {}
    job["run_id"] = run_id
    job["plenoirf_dir"] = plenoirf_dir
    job["site_key"] = site_key
    job["particle_key"] = particle_key
    job["instrument_key"] = instrument_key
    job["num_events"] = num_events
    job["max_num_events_in_merlict_run"] = max_num_events_in_merlict_run
    job["cache"] = cache
    return job


def run_job(job):
    with tempfile.TemporaryDirectory(suffix="-plenoirf") as tmp_dir:
        return run_job_in_dir(job=job, tmp_dir=tmp_dir)


def run_job_in_dir(job, tmp_dir):
    job = compile_job_paths_and_unique_identity(job=job, tmp_dir=tmp_dir)

    os.makedirs(job["paths"]["stage_dir"], exist_ok=True)

    logger = jll.LoggerFile(path=job["paths"]["logger_tmp"])
    logger.info("starting")

    logger.debug("making tmp_dir: {:s}".format(job["paths"]["tmp_dir"]))
    os.makedirs(job["paths"]["tmp_dir"], exist_ok=True)

    logger.info("initializing prng(seed={:d})".format(job["run_id"]))
    job["prng"] = np.random.Generator(np.random.PCG64(seed=job["run_id"]))
    job["run"] = {}

    with jll.TimeDelta(logger, "draw_event_uids_for_debugging"):
        job = draw_event_uids_for_debugging.run_job(job=job, logger=logger)

    with jll.TimeDelta(logger, "draw_pointing_range"):
        job = draw_pointing_range.run_job(job=job, logger=logger)

    with jll.TimeDelta(logger, "draw_primaries_and_pointings"):
        job = draw_primaries_and_pointings.run_job(job=job, logger=logger)

    job["event_table"] = event_table.structure.init_table_dynamicsizerecarray()

    with jll.TimeDelta(
        logger, "simulate_shower_and_collect_cherenkov_light_in_grid"
    ):
        job = simulate_shower_and_collect_cherenkov_light_in_grid.run_job(
            job=job, logger=logger
        )

    with jll.TimeDelta(logger, "inspect_particle_pool"):
        job = inspect_particle_pool.run_job(job=job, logger=logger)

    with jll.TimeDelta(logger, "split_event_tape_into_blocks"):
        job = split_event_tape_into_blocks.run_job(job=job, logger=logger)

    blk = {}
    with jll.TimeDelta(logger, "read light_field_calibration"):
        blk["light_field_geometry"] = plenopy.LightFieldGeometry(
            path=job["paths"]["light_field_calibration"]
        )

    with jll.TimeDelta(logger, "read trigger_geometry"):
        blk["trigger_geometry"] = plenopy.trigger.geometry.read(
            path=job["paths"]["trigger_geometry"]
        )

    blocks_dir = os.path.join(job["paths"]["tmp_dir"], "blocks")
    os.makedirs(blocks_dir, exist_ok=True)

    with jll.TimeDelta(logger, "run blocks"):
        for block_id_str in job["run"]["uids_in_cherenkov_pool_blocks"]:
            block_dir = os.path.join(blocks_dir, block_id_str)
            os.makedirs(block_dir, exist_ok=True)

            block_id = int(block_id_str)
            job = _run_job_block(
                job=job, blk=blk, block_id=block_id, logger=logger
            )

    logger.info("ending")
    rnw.move(job["paths"]["logger_tmp"], job["paths"]["logger"])

    return job


def _run_job_block(job, blk, block_id, logger):
    with jll.TimeDelta(
        logger, "simulate_hardware_block{:06d}".format(block_id)
    ):
        job = simulate_hardware.run_job_block(
            job=job, block_id=block_id, logger=logger
        )

    with jll.TimeDelta(
        logger, "simulate_loose_trigger_block{:06d}".format(block_id)
    ):
        job = simulate_loose_trigger.run_job_block(
            job=job, blk=blk, block_id=block_id, logger=logger
        )

    """
    with jll.TimeDelta(
        logger, "classify_cherenkov_photons_block{:06d}".format(block_id)
    ):
        job = classify_cherenkov_photons.run_job_block(
            job=job, blk=blk, block_id=block_id, logger=logger
        )
    """

    return job


def compile_job_paths_and_unique_identity(job, tmp_dir):
    """
    Adds static information to the job dict.
    """
    assert job["max_num_events_in_merlict_run"] > 0

    job["run_id_str"] = bookkeeping.uid.make_run_id_str(run_id=job["run_id"])
    job["paths"] = compile_job_paths(job=job, tmp_dir=tmp_dir)
    job["config"] = configuration.read(
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


def compile_job_paths(job, tmp_dir):
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
    paths["trigger_geometry"] = opj(
        job["plenoirf_dir"],
        "trigger_geometry",
        job["instrument_key"],
    )

    # temporary
    # ---------
    paths["tmp_dir"] = tmp_dir
    """
    paths["tmp"] = {}

    paths["tmp"]["cherenkov_pools"] = opj(tmp_dir, "cherenkov_pools.tar")
    paths["tmp"]["cherenkov_pools_block_fmt"] = opj(
        tmp_dir, "cherenkov_pools_block{block_id:06d}.tar"
    )
    paths["tmp"]["particle_pools_dat"] = opj(tmp_dir, "particle_pools.dat")
    paths["tmp"]["particle_pools_tar"] = opj(tmp_dir, "particle_pools.tar.gz")

    paths["tmp"]["ground_grid_intensity"] = opj(tmp_dir, "ground_grid.tar")
    paths["tmp"]["ground_grid_intensity_roi"] = opj(
        tmp_dir, "ground_grid_roi.tar"
    )

    paths["tmp"]["corsika_stdout"] = opj(tmp_dir, "corsika.stdout")
    paths["tmp"]["corsika_stderr"] = opj(tmp_dir, "corsika.stderr")
    paths["tmp"]["merlict_stdout_block_fmt"] = opj(
        tmp_dir, "merlict_block{block_id:06d}.stdout"
    )
    paths["tmp"]["merlict_stderr_block_fmt"] = opj(
        tmp_dir, "merlict_block{block_id:06d}.stderr"
    )
    paths["tmp"]["merlict_output_block_fmt"] = opj(
        tmp_dir, "merlict_block{block_id:06d}"
    )

    paths["tmp"]["past_loose_trigger_block_fmt"] = opj(
        tmp_dir, "past_loose_trigger_block{block_id:06d}"
    )
    """

    # debug output
    # ------------
    paths["debug"] = {}
    paths["debug"]["draw_primary_and_pointing"] = opj(
        paths["stage_dir"],
        job["run_id_str"] + "_debug_" + "draw_primary_and_pointing" + ".tar",
    )

    return paths


def read_light_field_camera_config(plenoirf_dir, instrument_key):
    return mlidev.plenoscope_propagator.read_plenoscope_geometry(
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
