import os
from os import path as op
from os.path import join as opj
import tempfile
import copy
import numpy as np

import json_utils
import json_line_logger as jll
from json_line_logger import TimeDelta
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
    env = compile_environment_for_job(job=job, work_dir=work_dir)

    os.makedirs(env["stage_dir"], exist_ok=True)

    logger_path = opj(env["stage_dir"], env["run_id_str"] + "_log.jsonl")
    logger = jll.LoggerFile(path=logger_path + ".part")
    logger.info("starting")

    logger.debug("making work_dir: {:s}".format(env["work_dir"]))
    os.makedirs(env["work_dir"], exist_ok=True)

    run_id = env["run_id"]

    with seeding.Section(run_id, draw_event_uids_for_debugging, logger) as sec:
        sec.module.run(env=env, seed=sec.seed, logger=logger)

    with seeding.Section(run_id, draw_pointing_range, logger) as sec:
        sec.module.run(env=env, seed=sec.seed, logger=logger)

    with seeding.Section(run_id, draw_primaries_and_pointings, logger) as sec:
        sec.module.run(env=env, seed=sec.seed, logger=logger)

    env["event_table"] = sparse_numeric_table.init(
        dtypes=event_table.structure.dtypes()
    )

    with seeding.Section(
        run_id, simulate_shower_and_collect_cherenkov_light_in_grid, logger
    ) as sec:
        sec.module.run(env=env, seed=sec.seed, logger=logger)

    with seeding.Section(run_id, inspect_cherenkov_pool, logger) as sec:
        sec.module.run(env=env, seed=sec.seed, logger=logger)

    with seeding.Section(run_id, inspect_particle_pool, logger) as sec:
        sec.module.run(env=env, seed=sec.seed, logger=logger)

    with seeding.Section(run_id, split_event_tape_into_blocks, logger) as sec:
        sec.module.run(env=env, seed=sec.seed, logger=logger)

    blk = {}
    with TimeDelta(logger, "read light_field_calibration"):
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

    with TimeDelta(logger, "read trigger_geometry"):
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

    with TimeDelta(logger, "run blocks"):
        for block_id_str in job["run"]["uids_in_cherenkov_pool_blocks"]:
            block_dir = os.path.join(blocks_dir, block_id_str)
            os.makedirs(block_dir, exist_ok=True)

            block_id = int(block_id_str)
            job = _run_job_block(
                env=env, blk=blk, block_id=block_id, logger=logger
            )

    logger.info("ending")
    rnw.move(logger_path + ".part", logger_path)

    return job


def _run_job_block(job, blk, block_id, logger):
    with TimeDelta(logger, "simulate_hardware_block{:06d}".format(block_id)):
        simulate_hardware.run_job_block(
            env=env, blk=blk, block_id=block_id, logger=logger
        )

    with TimeDelta(
        logger, "simulate_loose_trigger_block{:06d}".format(block_id)
    ):
        simulate_loose_trigger.run_job_block(
            env=env, blk=blk, block_id=block_id, logger=logger
        )

    with TimeDelta(
        logger, "classify_cherenkov_photons_block{:06d}".format(block_id)
    ):
        classify_cherenkov_photons.run_job_block(
            env=env, blk=blk, block_id=block_id, logger=logger
        )

    return job


def compile_environment_for_job(job, work_dir):
    """
    Adds static information to the job dict.
    """
    assert job["max_num_events_in_merlict_run"] > 0
    env = copy.deepcopy(job)

    env["work_dir"] = work_dir
    env["stage_dir"] = opj(
        env["plenoirf_dir"],
        "response",
        env["instrument_key"],
        env["site_key"],
        env["particle_key"],
        "stage",
    )

    env["run_id_str"] = bookkeeping.uid.make_run_id_str(run_id=env["run_id"])
    env["config"] = configuration.read(plenoirf_dir=env["plenoirf_dir"])
    _skymap_cfg = json_utils.tree.read(
        opj(
            env["plenoirf_dir"],
            "magnetic_deflection",
            env["site_key"],
            env["particle_key"],
            "config",
        )
    )
    env["site"] = copy.deepcopy(_skymap_cfg["site"])
    env["particle"] = copy.deepcopy(_skymap_cfg["particle"])

    env["light_field_camera_config"] = read_light_field_camera_config(
        plenoirf_dir=env["plenoirf_dir"],
        instrument_key=env["instrument_key"],
    )
    env["instrument"] = {}
    env["instrument"]["field_of_view_half_angle_rad"] = 0.5 * (
        np.deg2rad(env["light_field_camera_config"]["max_FoV_diameter_deg"])
    )
    env["instrument"]["local_speed_of_light_m_per_s"] = (
        constants.speed_of_light_in_vacuum_m_per_s()
        / env["site"]["atmosphere_refractive_index_at_observation_level"]
    )
    return env


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


class TimeDeltaModuleName:
    def __init__(self, logger, module):
        self.module = module
        self.time_delta = TimeDelta(logger=logger, name=self.module.__name__)

    def __enter__(self):
        self.time_delta.__enter__()
        return self.module

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.time_delta.__exit__()
