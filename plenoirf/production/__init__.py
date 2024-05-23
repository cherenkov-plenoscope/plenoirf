import os
from os import path as op
from os.path import join as opj
import tempfile
import copy
import numpy as np

import json_utils
import json_line_logger
from json_line_logger import TimeDelta
import merlict_development_kit_python as mlidev
import rename_after_writing as rnw
import plenopy
import sparse_numeric_table
import gamma_ray_reconstruction as gamrec

from .. import debugging
from .. import seeding
from .. import bookkeeping
from .. import configuration
from .. import ground_grid
from .. import event_table
from .. import constants

from . import sum_trigger
from . import draw_event_uids_for_debugging
from . import draw_primaries_and_pointings
from . import simulate_shower_and_collect_cherenkov_light_in_grid
from . import split_event_tape_into_blocks
from . import inspect_particle_pool
from . import simulate_hardware
from . import simulate_loose_trigger
from . import classify_cherenkov_photons
from . import inspect_cherenkov_pool
from . import extract_features_from_light_field
from . import estimate_primary_trajectory


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
    logger = json_line_logger.LoggerFile(path=logger_path + ".part")
    logger.info("starting")

    logger.debug("making work_dir: {:s}".format(env["work_dir"]))
    os.makedirs(env["work_dir"], exist_ok=True)

    run_id = env["run_id"]

    with seeding.SeedSection(
        run_id=run_id,
        module=draw_event_uids_for_debugging,
        logger=logger,
    ) as sec:
        sec.module.run(
            env=env,
            seed=sec.seed,
            logger=logger,
        )

    with seeding.SeedSection(
        run_id=run_id,
        module=draw_primaries_and_pointings,
        logger=logger,
    ) as sec:
        sec.module.run(
            env=env,
            seed=sec.seed,
            logger=logger,
        )

    with seeding.SeedSection(
        run_id=run_id,
        module=simulate_shower_and_collect_cherenkov_light_in_grid,
        logger=logger,
    ) as sec:
        sec.module.run(
            env=env,
            seed=sec.seed,
            logger=logger,
        )

    with seeding.SeedSection(
        run_id=run_id,
        module=inspect_cherenkov_pool,
        logger=logger,
    ) as sec:
        sec.module.run(
            env=env,
            logger=logger,
        )

    with seeding.SeedSection(
        run_id=run_id,
        module=inspect_particle_pool,
        logger=logger,
    ) as sec:
        sec.module.run(
            env=env,
            logger=logger,
        )

    blk = {}
    blk["blocks_dir"] = os.path.join(env["work_dir"], "blocks")

    with seeding.SeedSection(
        run_id=run_id,
        module=split_event_tape_into_blocks,
        logger=logger,
    ) as sec:
        sec.module.run(
            env=env,
            logger=logger,
        )

    _blkp = os.path.join(
        env["work_dir"], "blocks", "event_uid_strs_in_block.json"
    )
    with open(_blkp, "rt") as fin:
        blk["event_uid_strs_in_block"] = json_utils.loads(fin.read())

    with TimeDelta(logger, "read light_field_calibration"):
        light_field_calibration_path = opj(
            env["plenoirf_dir"],
            "plenoptics",
            "instruments",
            env["instrument_key"],
            "light_field_geometry",
        )
        blk["light_field_geometry"] = plenopy.LightFieldGeometry(
            path=light_field_calibration_path
        )

    with TimeDelta(logger, "make light_field_calibration addon"):
        blk[
            "light_field_geometry_addon"
        ] = plenopy.features.make_light_field_geometry_addon(
            light_field_geometry=blk["light_field_geometry"]
        )

    with TimeDelta(logger, "read trigger_geometry"):
        trigger_geometry_path = opj(
            env["plenoirf_dir"],
            "trigger_geometry",
            env["instrument_key"]
            + plenopy.trigger.geometry.suggested_filename_extension(),
        )
        blk["trigger_geometry"] = plenopy.trigger.geometry.read(
            path=trigger_geometry_path
        )

    with TimeDelta(logger, "init trajectory reconstruction config"):
        blk["trajectory_reconstruction"] = {}
        blk["trajectory_reconstruction"][
            "fuzzy_config"
        ] = gamrec.trajectory.v2020nov12fuzzy0.config.compile_user_config(
            user_config=env["config"]["reconstruction"]["trajectory"][
                "fuzzy_method"
            ]
        )
        blk["trajectory_reconstruction"][
            "model_fit_config"
        ] = gamrec.trajectory.v2020dec04iron0b.config.compile_user_config(
            user_config=env["config"]["reconstruction"]["trajectory"][
                "core_axis_fit"
            ]
        )

    # estimate memory footprint of env and blk
    # ----------------------------------------
    with TimeDelta(logger, "estimate size of block-environment 'blk'."):
        blk_size_bytes = debugging.estimate_memory_size_in_bytes_of_anything(
            nything=blk
        )
        logger.info("<blk size_bytes='{:d}'/>".format(blk_size_bytes))

    with TimeDelta(logger, "estimate size of environment 'env'."):
        blk_size_bytes = debugging.estimate_memory_size_in_bytes_of_anything(
            nything=env
        )
        logger.info("<env size_bytes='{:d}'/>".format(blk_size_bytes))

    # loop over blocks
    # ----------------
    for block_id_str in blk["event_uid_strs_in_block"]:
        run_job_block(
            env=env, blk=blk, block_id=int(block_id_str), logger=logger
        )

    with TimeDelta(logger, "estimate disk usage in work_dir."):
        disk_usage = debugging.estimate_disk_usage_in_bytes(
            path=env["work_dir"]
        )
        with gzip.open(
            opj(env["work_dir"], "disk_usage.json.gz", "wt")
        ) as fout:
            fout.write(json_utils.dumps(disk_usage, indent=4))

    logger.info("ending")
    json_line_logger.shutdown(logger=logger)
    rnw.move(logger_path + ".part", logger_path)

    return 1


def run_job_block(env, blk, block_id, logger):
    run_id = env["run_id"]

    with seeding.SeedSection(
        run_id=run_id,
        module=simulate_hardware,
        block_id=block_id,
        logger=logger,
    ) as sec:
        sec.module.run_block(
            env=env,
            blk=blk,
            block_id=block_id,
            logger=logger,
        )

    with seeding.SeedSection(
        run_id=run_id,
        module=simulate_loose_trigger,
        block_id=block_id,
        logger=logger,
    ) as sec:
        sec.module.run_block(
            env=env,
            blk=blk,
            block_id=block_id,
            logger=logger,
        )

    with seeding.SeedSection(
        run_id=run_id,
        module=classify_cherenkov_photons,
        block_id=block_id,
        logger=logger,
    ) as sec:
        sec.module.run_block(
            env=env, blk=blk, block_id=block_id, logger=logger
        )

    with seeding.SeedSection(
        run_id=run_id,
        module=extract_features_from_light_field,
        block_id=block_id,
        logger=logger,
    ) as sec:
        sec.module.run_block(
            env=env, blk=blk, seed=sec.seed, block_id=block_id, logger=logger
        )

    with seeding.SeedSection(
        run_id=run_id,
        module=estimate_primary_trajectory,
        block_id=block_id,
        logger=logger,
    ) as sec:
        sec.module.run_block(
            env=env, blk=blk, block_id=block_id, logger=logger
        )

    # remove the merlict events to free temporary diskspace.
    block_dir = os.path.join(
        env["work_dir"], "blocks", "{:06d}".format(block_id)
    )
    merlict_events_path = os.path.join(block_dir, "merlict")
    if os.path.exists(merlict_events_path):
        logger.info(
            "removing merlict events: '{:s}'".format(merlict_events_path)
        )
        shutil.rmtree(merlict_events_path)

    return 1


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
