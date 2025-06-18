import os
from os.path import join as opj
import tempfile
import copy
import shutil
import numpy as np
import zipfile
import gzip

import json_utils
import json_line_logger
from json_line_logger import TimeDelta
from json_line_logger import xml
import merlict_development_kit_python as mlidev
import rename_after_writing as rnw
import plenopy
import sparse_numeric_table as snt
import gamma_ray_reconstruction as gamrec

from .. import provenance
from .. import debugging
from .. import seeding
from .. import bookkeeping
from .. import configuration
from .. import ground_grid
from .. import event_table
from .. import constants
from .. import utils

from . import zipfileutils as zfu

from . import gather_and_export_provenance
from . import sum_trigger
from . import draw_event_uids_for_debugging
from . import draw_primaries_and_pointings
from . import simulate_shower_and_collect_cherenkov_light_in_grid
from . import (
    simulate_shower_again_and_cut_cherenkov_light_falling_into_instrument,
)

from . import inspect_particle_pool

from . import inspect_cherenkov_pool
from . import extract_features_from_light_field
from . import estimate_primary_trajectory
from . import benchmark_compute_environment
from . import simulate_instrument_and_reconstruct_cherenkov


def make_example_jobs(
    plenoirf_dir,
    run_ids=[1221],
    site_key="chile",
    particle_key="gamma",
    instrument_key="diag9_default_default",
    num_events=1000,
    max_num_events_in_merlict_run=100,
    debugging_figures=True,
):
    jobs = []
    for run_id in run_ids:
        job = {}
        job["run_id"] = run_id
        job["plenoirf_dir"] = plenoirf_dir
        job["site_key"] = site_key
        job["particle_key"] = particle_key
        job["instrument_key"] = instrument_key
        job["num_events"] = num_events
        job["max_num_events_in_merlict_run"] = max_num_events_in_merlict_run
        job["debugging_figures"] = debugging_figures
        jobs.append(job)
    return jobs


def run_job(job):
    tmpDir = tempfile.TemporaryDirectory

    env = compile_environment_for_job(job=job)
    result_path = opj(
        env["stage_dir"], "{part:s}", env["run_id_str"] + ".{part:s}.zip"
    )

    if not os.path.exists(result_path.format(part="prm2cer")):
        with tmpDir(prefix="plenoirf-prm2cer-") as w:
            run_job_prm2cer_in_dir(job=job, work_dir=w)

    if not os.path.exists(result_path.format(part="cer2cls")):
        with tmpDir(prefix="plenoirf-cer2cls-") as w:
            run_job_cer2cls_in_dir(job=job, work_dir=w)

    if not os.path.exists(result_path.format(part="cls2rec")):
        with tmpDir(prefix="plenoirf-cls2rec-") as w:
            run_job_cls2rec_in_dir(job=job, work_dir=w)


def run_job_prm2cer_in_dir(job, work_dir):
    PART = "prm2cer"
    env = compile_environment_for_job(job=job, work_dir=work_dir)

    stage_part_dir = opj(env["stage_dir"], PART)
    os.makedirs(stage_part_dir, exist_ok=True)

    logger_path = opj(
        stage_part_dir, env["run_id_str"] + f".{PART:s}.log.jsonl"
    )
    logger = json_line_logger.LoggerFile(path=logger_path)
    logger.info(f"starting {PART:s}.")

    logger.debug("making work_dir: {:s}".format(env["work_dir"]))
    os.makedirs(env["work_dir"], exist_ok=True)

    with seeding.SeedSection(
        run_id=env["run_id"],
        module=gather_and_export_provenance,
        logger=logger,
    ) as sec:
        sec.module.run(env=env, part=PART)

    with seeding.SeedSection(
        run_id=env["run_id"],
        module=benchmark_compute_environment,
        logger=logger,
    ) as sec:
        sec.module.run(env=env, part=PART)

    with seeding.SeedSection(
        run_id=env["run_id"],
        module=draw_event_uids_for_debugging,
        logger=logger,
    ) as sec:
        sec.module.run(env=env, part=PART, seed=sec.seed)

    with seeding.SeedSection(
        run_id=env["run_id"],
        module=draw_primaries_and_pointings,
        logger=logger,
    ) as sec:
        sec.module.run(env=env, part=PART, seed=sec.seed)

    with seeding.SeedSection(
        run_id=env["run_id"],
        module=simulate_shower_and_collect_cherenkov_light_in_grid,
        logger=logger,
    ) as sec:
        sec.module.run(env=env, part=PART, seed=sec.seed)

    with seeding.SeedSection(
        run_id=env["run_id"],
        module=simulate_shower_again_and_cut_cherenkov_light_falling_into_instrument,
        logger=logger,
    ) as sec:
        sec.module.run(env=env, part=PART, seed=sec.seed)

    with seeding.SeedSection(
        run_id=env["run_id"],
        module=inspect_cherenkov_pool,
        logger=logger,
    ) as sec:
        sec.module.run(env=env, part=PART)

    with seeding.SeedSection(
        run_id=env["run_id"],
        module=inspect_particle_pool,
        logger=logger,
    ) as sec:
        sec.module.run(env=env, part=PART)

    logger.info("shuting down logger.")
    json_line_logger.shutdown(logger=logger)
    rnw.copy(src=logger_path, dst=opj(env["work_dir"], PART, "log.jsonl"))
    utils.gzip_file(opj(env["work_dir"], PART, "log.jsonl"))

    zipfileutils.archive_dir(
        path=opj(stage_part_dir, env["run_id_str"] + f".{PART:s}.zip"),
        dir_path=opj(env["work_dir"], PART),
        base_dir_path=opj(env["run_id_str"], PART),
    )

    os.remove(logger_path)
    return 1


def run_job_cer2cls_in_dir(job, work_dir):
    PART = "cer2cls"
    env = compile_environment_for_job(job=job, work_dir=work_dir)

    stage_part_dir = opj(env["stage_dir"], PART)
    os.makedirs(stage_part_dir, exist_ok=True)

    logger_path = opj(
        stage_part_dir, env["run_id_str"] + f".{PART:s}.log.jsonl"
    )
    logger = json_line_logger.LoggerFile(path=logger_path)
    logger.info(f"starting {PART:s}.")

    logger.debug("making work_dir: {:s}".format(env["work_dir"]))
    os.makedirs(env["work_dir"], exist_ok=True)

    env = load_instrument_geometry_into_environment(env=env, logger=logger)

    with json_line_logger.TimeDelta(logger, "Load prm2cer."):
        load_part(env=env, part="prm2cer")

    with seeding.SeedSection(
        run_id=env["run_id"],
        module=gather_and_export_provenance,
        logger=logger,
    ) as sec:
        sec.module.run(env=env, part=PART)

    with seeding.SeedSection(
        run_id=env["run_id"],
        module=simulate_instrument_and_reconstruct_cherenkov,
        logger=logger,
    ) as sec:
        sec.module.run(env=env, part=PART, seed=sec.seed)

    logger.info("shuting down logger.")
    json_line_logger.shutdown(logger=logger)
    rnw.copy(src=logger_path, dst=opj(env["work_dir"], PART, "log.jsonl"))
    utils.gzip_file(opj(env["work_dir"], PART, "log.jsonl"))

    zipfileutils.archive_dir(
        path=opj(stage_part_dir, env["run_id_str"] + f".{PART:s}.zip"),
        dir_path=opj(env["work_dir"], PART),
        base_dir_path=opj(env["run_id_str"], PART),
    )

    os.remove(logger_path)
    return 1


def run_job_cls2rec_in_dir(job, work_dir):
    PART = "cls2rec"
    env = compile_environment_for_job(job=job, work_dir=work_dir)

    stage_part_dir = opj(env["stage_dir"], PART)
    os.makedirs(stage_part_dir, exist_ok=True)

    logger_path = opj(
        stage_part_dir, env["run_id_str"] + f".{PART:s}.log.jsonl"
    )
    logger = json_line_logger.LoggerFile(path=logger_path)
    logger.info(f"starting {PART:s}.")

    logger.debug("making work_dir: {:s}".format(env["work_dir"]))
    os.makedirs(env["work_dir"], exist_ok=True)

    env = load_instrument_geometry_into_environment(env=env, logger=logger)

    with json_line_logger.TimeDelta(logger, "Load cer2cls."):
        load_part(env=env, part="cer2cls")

    with seeding.SeedSection(
        run_id=env["run_id"],
        module=gather_and_export_provenance,
        logger=logger,
    ) as sec:
        sec.module.run(env=env, part=PART)

    with seeding.SeedSection(
        run_id=env["run_id"],
        module=extract_features_from_light_field,
        logger=logger,
    ) as sec:
        sec.module.run(env=env, part=PART, seed=sec.seed)

    with seeding.SeedSection(
        run_id=env["run_id"],
        module=estimate_primary_trajectory,
        logger=logger,
    ) as sec:
        sec.module.run(env=env, part=PART)

    logger.info("shuting down logger.")
    json_line_logger.shutdown(logger=logger)
    rnw.copy(src=logger_path, dst=opj(env["work_dir"], PART, "log.jsonl"))
    utils.gzip_file(opj(env["work_dir"], PART, "log.jsonl"))

    zipfileutils.archive_dir(
        path=opj(stage_part_dir, env["run_id_str"] + f".{PART:s}.zip"),
        dir_path=opj(env["work_dir"], PART),
        base_dir_path=opj(env["run_id_str"], PART),
    )

    os.remove(logger_path)
    return 1


def load_part(env, part):
    part_dir = opj(env["work_dir"], part)
    if not os.path.exists(part_dir):
        shutil.unpack_archive(
            filename=opj(
                env["stage_dir"], part, env["run_id_str"] + f".{part:s}.zip"
            ),
            extract_dir=opj(env["work_dir"]),
        )
        os.rename(
            opj(env["work_dir"], env["run_id_str"], part),
            opj(env["work_dir"], part),
        )
        shutil.rmtree(opj(env["work_dir"], env["run_id_str"]))


def run_job_in_dir(job, work_dir):

    # collect output
    # ==============
    logger.info("Collecting results in output file.")

    # bundle event_table
    # ------------------
    with TimeDelta(logger, "bundling event_table.zip"):
        evttab = snt.SparseNumericTable(index_key="uid")
        evttab = event_table.add_levels_from_path(
            evttab=evttab,
            path=opj(
                env["work_dir"],
                "plenoirf.production.simulate_shower_and_collect_cherenkov_light_in_grid",
                "event_table.snt.zip",
            ),
        )
        evttab = event_table.add_levels_from_path(
            evttab=evttab,
            path=opj(
                env["work_dir"],
                "plenoirf.production.simulate_shower_again_and_cut_cherenkov_light_falling_into_instrument",
                "event_table.snt.zip",
            ),
        )
        evttab = event_table.add_levels_from_path(
            evttab=evttab,
            path=opj(
                env["work_dir"],
                "plenoirf.production.inspect_particle_pool",
                "event_table.snt.zip",
            ),
        )
        for block_id_str in blk["event_uid_strs_in_block"]:
            evttab = event_table.append_to_levels_from_path(
                evttab=evttab,
                path=opj(
                    env["work_dir"],
                    "blocks",
                    block_id_str,
                    "plenoirf.production.simulate_loose_trigger",
                    "event_table.snt.zip",
                ),
            )
            evttab = event_table.append_to_levels_from_path(
                evttab=evttab,
                path=opj(
                    env["work_dir"],
                    "blocks",
                    block_id_str,
                    "plenoirf.production.classify_cherenkov_photons",
                    "event_table.snt.zip",
                ),
            )
            evttab = event_table.append_to_levels_from_path(
                evttab=evttab,
                path=opj(
                    env["work_dir"],
                    "blocks",
                    block_id_str,
                    "plenoirf.production.extract_features_from_light_field",
                    "event_table.snt.zip",
                ),
            )
            evttab = event_table.append_to_levels_from_path(
                evttab=evttab,
                path=opj(
                    env["work_dir"],
                    "blocks",
                    block_id_str,
                    "plenoirf.production.estimate_primary_trajectory",
                    "event_table.snt.zip",
                ),
            )
        event_table.write_all_levels_to_path(
            evttab=evttab,
            path=opj(env["work_dir"], "event_table.snt.zip"),
        )


def compile_environment_for_job(job, work_dir=None):
    """
    Adds static information to the job dict.
    """
    assert job["max_num_events_in_merlict_run"] > 0
    env = copy.deepcopy(job)
    env["run_id_str"] = bookkeeping.uid.make_run_id_str(run_id=env["run_id"])

    env["stage_dir"] = opj(
        env["plenoirf_dir"],
        "response",
        env["instrument_key"],
        env["site_key"],
        env["particle_key"],
        "stage",
    )

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

    if work_dir is not None:
        env["work_dir"] = work_dir
        env["run_dir"] = opj(work_dir, env["run_id_str"])

    return env


def load_instrument_geometry_into_environment(env, logger):
    with json_line_logger.TimeDelta(logger, "read light_field_calibration"):
        light_field_calibration_path = opj(
            env["plenoirf_dir"],
            "plenoptics",
            "instruments",
            env["instrument_key"],
            "light_field_geometry",
        )
        env["light_field_geometry"] = plenopy.LightFieldGeometry(
            path=light_field_calibration_path
        )

    with json_line_logger.TimeDelta(
        logger, "make light_field_calibration addon"
    ):
        env["light_field_geometry_addon"] = (
            plenopy.features.make_light_field_geometry_addon(
                light_field_geometry=env["light_field_geometry"]
            )
        )

    with json_line_logger.TimeDelta(logger, "read trigger_geometry"):
        trigger_geometry_path = opj(
            env["plenoirf_dir"],
            "trigger_geometry",
            env["instrument_key"]
            + plenopy.trigger.geometry.suggested_filename_extension(),
        )
        env["trigger_geometry"] = plenopy.trigger.geometry.read(
            path=trigger_geometry_path
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
