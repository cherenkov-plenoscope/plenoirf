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
import sparse_numeric_table as snt

from ... import seeding
from ... import utils
from ... import event_table

from . import split_event_tape_into_blocks
from . import simulate_hardware
from . import simulate_loose_trigger
from . import classify_cherenkov_photons


def run(env, part, seed):
    name = __name__.split(".")[-1]
    module_work_dir = opj(env["work_dir"], part, name)

    if os.path.exists(module_work_dir):
        return

    os.makedirs(module_work_dir, exist_ok=True)
    logger = json_line_logger.LoggerFile(opj(module_work_dir, "log.jsonl"))
    logger.info(__name__)
    logger.info(f"seed: {seed:d}")

    prng = np.random.Generator(np.random.PCG64(seed))

    blk = {}
    blk["blocks_dir"] = os.path.join(module_work_dir, "blocks")

    with json_line_logger.TimeDelta(logger, "split_event_tape_into_blocks"):
        split_event_tape_into_blocks.run(env=env, blk=blk, logger=logger)

    with gzip.open(
        opj(blk["blocks_dir"], "event_uid_strs_in_block.json.gz"), "rt"
    ) as fin:
        blk["event_uid_strs_in_block"] = json_utils.loads(fin.read())

    # loop over blocks
    # ----------------
    for block_id_str in blk["event_uid_strs_in_block"]:
        run_job_block(
            env=env, blk=blk, block_id=int(block_id_str), logger=logger
        )

    with json_line_logger.TimeDelta(
        logger, "bundle_merlict_events_from_blocks"
    ):
        bundle_merlict_events_from_blocks(
            module_work_dir=module_work_dir, blk=blk
        )

    with json_line_logger.TimeDelta(
        logger, "bundle_reconstructed_cherenkov_from_blocks"
    ):
        bundle_reconstructed_cherenkov_from_blocks(
            module_work_dir=module_work_dir, blk=blk
        )

    with json_line_logger.TimeDelta(logger, "bundle_event_table_from_blocks"):
        bundle_event_table_from_blocks(module_work_dir, blk)

    logger.info("done.")
    json_line_logger.shutdown(logger=logger)
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


def bundle_reconstructed_cherenkov_from_blocks(module_work_dir, blk):
    out_path = opj(module_work_dir, "reconstructed_cherenkov.loph.tar")
    if not os.path.exists(out_path):
        in_paths = []
        for block_id_str in blk["event_uid_strs_in_block"]:
            in_paths.append(
                opj(
                    blk["blocks_dir"],
                    block_id_str,
                    "classify_cherenkov_photons",
                    "reconstructed_cherenkov.loph.tar",
                )
            )
        plenopy.photon_stream.loph.concatenate_tars(
            in_paths=in_paths,
            out_path=out_path,
        )
        for in_path in in_paths:
            os.remove(in_path)


def bundle_event_table_from_blocks(module_work_dir, blk):
    out_path = opj(module_work_dir, "event_table.snt.zip")
    if not os.path.exists(out_path):
        in_paths = []
        for block_id_str in blk["event_uid_strs_in_block"]:
            in_paths.append(
                opj(
                    blk["blocks_dir"],
                    block_id_str,
                    "simulate_loose_trigger",
                    "event_table.snt.zip",
                )
            )
            in_paths.append(
                opj(
                    blk["blocks_dir"],
                    block_id_str,
                    "classify_cherenkov_photons",
                    "event_table.snt.zip",
                )
            )

        evttab = snt.SparseNumericTable(index_key="uid")
        for in_path in in_paths:
            evttab = event_table.append_to_levels_from_path(
                evttab=evttab, path=in_path
            )
        event_table.write_all_levels_to_path(evttab=evttab, path=out_path)

        for in_path in in_paths:
            os.remove(in_path)
