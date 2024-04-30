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
import sparse_numeric_table

from .. import seeding
from .. import bookkeeping
from .. import configuration
from .. import ground_grid
from .. import event_table
from .. import constants

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
from . import inspect_cherenkov_pool


def make_example_job(
    plenoirf_dir,
    run_id=1221,
    site_key="chile",
    particle_key="gamma",
    instrument_key="diag9_default_default",
    num_events=1000,
    max_num_events_in_merlict_run=100,
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
    with tempfile.TemporaryDirectory(suffix="-plenoirf") as work_dir:
        return run_job_in_dir(job=job, work_dir=work_dir)


def run_job_in_dir(job, work_dir):
    # important directories
    # plenoirf_dir / work_dir / stage_dir

    job["work_dir"] = work_dir
    job["stage_dir"] = opj(
        job["plenoirf_dir"],
        "response",
        job["instrument_key"],
        job["site_key"],
        job["particle_key"],
        "stage",
    )

    job = read_config_into_job(job=job)

    os.makedirs(job["stage_dir"], exist_ok=True)

    logger_path = opj(job["stage_dir"], job["run_id_str"] + "_log.jsonl")
    logger = jll.LoggerFile(path=logger_path + ".part")
    logger.info("starting")

    logger.debug("making work_dir: {:s}".format(job["work_dir"]))
    os.makedirs(job["work_dir"], exist_ok=True)

    logger.info("initializing random seeds (seed={:d})".format(job["run_id"]))
    named_random_seeds = seeding.make_named_random_seeds(
        run_id=job["run_id"],
        names=[
            "draw_event_uids_for_debugging",
            "draw_pointing_range",
            "draw_primaries_and_pointings",
            "simulate_shower_and_collect_cherenkov_light_in_grid",
            "inspect_cherenkov_pool",
            "inspect_particle_pool",
        ],
    )
    seeding.write(
        path=opj(job["work_dir"], "named_random_seeds.json"),
        named_random_seeds=named_random_seeds,
    )

    with jll.TimeDelta(logger, "draw_event_uids_for_debugging"):
        draw_event_uids_for_debugging.run_job(job=job, logger=logger)

    with jll.TimeDelta(logger, "draw_pointing_range"):
        draw_pointing_range.run_job(job=job, logger=logger)

    with jll.TimeDelta(logger, "draw_primaries_and_pointings"):
        draw_primaries_and_pointings.run_job(job=job, logger=logger)

    job["event_table"] = sparse_numeric_table.init(
        dtypes=event_table.structure.dtypes()
    )

    with jll.TimeDelta(
        logger, "simulate_shower_and_collect_cherenkov_light_in_grid"
    ):
        simulate_shower_and_collect_cherenkov_light_in_grid.run_job(
            job=job, logger=logger
        )

    with jll.TimeDelta(logger, "inspect_cherenkov_pool"):
        inspect_cherenkov_pool.run_job(job=job, logger=logger)

    with jll.TimeDelta(logger, "inspect_particle_pool"):
        job = inspect_particle_pool.run_job(job=job, logger=logger)

    with jll.TimeDelta(logger, "split_event_tape_into_blocks"):
        job = split_event_tape_into_blocks.run_job(job=job, logger=logger)

    blk = {}
    with jll.TimeDelta(logger, "read light_field_calibration"):
        light_field_calibration_path = opj(
            job["plenoirf_dir"],
            "plenoptics",
            "instruments",
            job["instrument_key"],
            "light_field_geometry",
        )
        blk["light_field_geometry"] = plenopy.LightFieldGeometry(
            path=light_field_calibration_path
        )

    with jll.TimeDelta(logger, "read trigger_geometry"):
        trigger_geometry_path = opj(
            job["plenoirf_dir"],
            "trigger_geometry",
            job["instrument_key"]
            + plenopy.trigger.geometry.suggested_filename_extension(),
        )

        blk["trigger_geometry"] = plenopy.trigger.geometry.read(
            path=trigger_geometry_path
        )

    blocks_dir = os.path.join(job["work_dir"], "blocks")
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
    rnw.move(logger_path + ".part", logger_path)

    return job


def _run_job_block(job, blk, block_id, logger):
    with jll.TimeDelta(
        logger, "simulate_hardware_block{:06d}".format(block_id)
    ):
        simulate_hardware.run_job_block(
            job=job, blk=blk, block_id=block_id, logger=logger
        )

    with jll.TimeDelta(
        logger, "simulate_loose_trigger_block{:06d}".format(block_id)
    ):
        simulate_loose_trigger.run_job_block(
            job=job, blk=blk, block_id=block_id, logger=logger
        )

    with jll.TimeDelta(
        logger, "classify_cherenkov_photons_block{:06d}".format(block_id)
    ):
        classify_cherenkov_photons.run_job_block(
            job=job, blk=blk, block_id=block_id, logger=logger
        )

    return job


def read_config_into_job(job):
    """
    Adds static information to the job dict.
    """
    assert job["max_num_events_in_merlict_run"] > 0

    job["run_id_str"] = bookkeeping.uid.make_run_id_str(run_id=job["run_id"])
    job["config"] = configuration.read(plenoirf_dir=job["plenoirf_dir"])
    _skymap_cfg = json_utils.tree.read(
        opj(
            job["plenoirf_dir"],
            "magnetic_deflection",
            job["site_key"],
            job["particle_key"],
            "config",
        )
    )
    job["site"] = copy.deepcopy(_skymap_cfg["site"])
    job["particle"] = copy.deepcopy(_skymap_cfg["particle"])

    job["light_field_camera_config"] = read_light_field_camera_config(
        plenoirf_dir=job["plenoirf_dir"],
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
