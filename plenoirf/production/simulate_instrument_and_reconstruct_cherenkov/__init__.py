import os
from os.path import join as opj
import tarfile
import numpy as np
import gzip
import hashlib
import shutil
import zipfile

import json_utils
import json_line_logger
import plenopy
import rename_after_writing as rnw

from ... import seeding
from ... import utils

from . import split_event_tape_into_blocks
from . import simulate_hardware
from . import simulate_loose_trigger
from . import classify_cherenkov_photons


def run(env, seed):
    module_work_dir = opj(env["work_dir"], __name__)

    """
    if os.path.exists(module_work_dir):
        return
    """

    os.makedirs(module_work_dir, exist_ok=True)
    logger = json_line_logger.LoggerFile(opj(module_work_dir, "log.jsonl"))
    logger.info(__name__)
    logger.info(f"seed: {seed:d}")

    prng = np.random.Generator(np.random.PCG64(seed))

    blk = {}
    blk["blocks_dir"] = os.path.join(module_work_dir, "blocks")

    with json_line_logger.TimeDelta(logger, "split_event_tape_into_blocks"):
        split_event_tape_into_blocks.run(env=env, blk=blk)

    with gzip.open(
        opj(blk["blocks_dir"], "event_uid_strs_in_block.json.gz"), "rt"
    ) as fin:
        blk["event_uid_strs_in_block"] = json_utils.loads(fin.read())

    with json_line_logger.TimeDelta(logger, "read light_field_calibration"):
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

    with json_line_logger.TimeDelta(
        logger, "make light_field_calibration addon"
    ):
        blk["light_field_geometry_addon"] = (
            plenopy.features.make_light_field_geometry_addon(
                light_field_geometry=blk["light_field_geometry"]
            )
        )

    with json_line_logger.TimeDelta(logger, "read trigger_geometry"):
        trigger_geometry_path = opj(
            env["plenoirf_dir"],
            "trigger_geometry",
            env["instrument_key"]
            + plenopy.trigger.geometry.suggested_filename_extension(),
        )
        blk["trigger_geometry"] = plenopy.trigger.geometry.read(
            path=trigger_geometry_path
        )

    # loop over blocks
    # ----------------
    for block_id_str in blk["event_uid_strs_in_block"]:
        run_job_block(
            env=env, blk=blk, block_id=int(block_id_str), logger=logger
        )

    logger.info("bundle_merlict_events_from_blocks.")
    bundle_merlict_events_from_blocks(module_work_dir=module_work_dir, blk=blk)

    """
    # bundle reconstructed cherenkov light (loph)
    # -------------------------------------------
    with json_line_logger.TimeDelta(logger, "bundling reconstructed_cherenkov.loph.tar"):
        loph_in_paths = []
        for block_id_str in blk["event_uid_strs_in_block"]:
            loph_in_path = opj(
                blk["blocks_dir"],
                block_id_str,
                "reconstructed_cherenkov.loph.tar",
            )
            loph_in_paths.append(loph_in_path)

        plenopy.photon_stream.loph.concatenate_tars(
            in_paths=loph_in_paths,
            out_path=opj(module_work_dir, "reconstructed_cherenkov.loph.tar"),
        )
    """

    logger.info("done.")
    json_line_logger.shutdown(logger=logger)

    # tidy up and compress
    utils.gzip_file(opj(module_work_dir, "log.jsonl"))


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

    # remove the merlict events to free temporary diskspace
    # -----------------------------------------------------
    # This removal is whole reason behind the block structure with merlict.
    # We must not flood the temporary drives with too many merlict events at
    # once.
    block_id_str = "{:06d}".format(block_id)
    block_dir = opj(blk["blocks_dir"], block_id_str)
    merlict_events_path = os.path.join(
        block_dir, "simulate_hardware", "merlict"
    )
    if os.path.isdir(merlict_events_path):
        logger.info(f"removing merlict events: '{merlict_events_path:s}'")
        shutil.rmtree(merlict_events_path)

    # remove the cherenkov photon block
    # ---------------------------------
    if os.path.isfile(opj(block_dir, "cherenkov_pools.tar")):
        logger.info("removing block's cherenkov_pools.tar")
        os.remove(opj(block_dir, "cherenkov_pools.tar"))
    return 1


def bundle_merlict_events_from_blocks(module_work_dir, blk):
    out_path = opj(module_work_dir, "merlict_events.debug.zip")
    if not os.path.exists(out_path):
        in_paths = []
        for block_id_str in blk["event_uid_strs_in_block"]:
            in_paths.append(
                opj(
                    blk["blocks_dir"],
                    block_id_str,
                    "simulate_hardware",
                    "merlict_events.debug.zip",
                )
            )
        concatenate_zip_files(
            in_paths=in_paths,
            out_path=out_path,
        )
        for in_path in in_paths:
            os.remove(in_path)


def concatenate_zip_files(in_paths, out_path):
    with rnw.Path(out_path) as tmp_out_path:
        with zipfile.ZipFile(tmp_out_path, "w") as zout:
            for in_path in in_paths:
                with zipfile.ZipFile(in_path, "r") as zin:
                    for item in zin.filelist:
                        with zout.open(item.filename, "w") as fout:
                            fout.write(zin.read(item))
