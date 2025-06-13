import os
from os.path import join as opj
import merlict_development_kit_python as mlidev
import rename_after_writing as rnw
import json_utils
import numpy as np
import plenopy
import corsika_primary as cpw
import zipfile

from .. import bookkeeping
from .. import utils


def run_block(env, blk, block_id, logger):
    logger.info(__name__ + ": start ...")

    block_dir = opj(env["work_dir"], "blocks", "{:06d}".format(block_id))
    output_path = opj(block_dir, "merlict")

    if os.path.exists(output_path):
        logger.info(__name__ + ": already done. skip computation.")
        return

    mlidev_cfg_path = opj(
        env["work_dir"], "merlict_plenoscope_propagator_config.json"
    )
    write_mlidev_config(env=env, path=mlidev_cfg_path)

    light_field_geometry_path = opj(
        env["plenoirf_dir"],
        "plenoptics",
        "instruments",
        env["instrument_key"],
        "light_field_geometry",
    )

    rc = mlidev.plenoscope_propagator.plenoscope_propagator(
        corsika_run_path=opj(block_dir, "cherenkov_pools.tar"),
        output_path=output_path,
        light_field_geometry_path=light_field_geometry_path,
        merlict_plenoscope_propagator_config_path=mlidev_cfg_path,
        random_seed=env["run_id"],
        photon_origins=True,
        stdout_path=opj(block_dir, "merlict.stdout.txt"),
        stderr_path=opj(block_dir, "merlict.stderr.txt"),
    )

    """
    2025-03-22: 1 out 25,000 merlict calls returned non zero. This was added in
    the hope of finding out why.
    """
    errmsg = f"Expected merlict's return code to be zero, but it is '{rc:d}'."
    if rc != 0:
        logger.critical(__name__ + errmsg)
        logger.critical(__name__ + ": Rescue merlict stdout and stderr.")
        filename = f"{env['run_id_str']:s}.block_{block_id:03d}.merlict"
        for extension in [".stdout.txt", ".stderr.txt"]:
            rnw.copy(
                src=opj(block_dir, "merlict" + extension),
                dst=opj(env["stage_dir"], filename + extension),
            )

    assert rc == 0, errmsg

    logger.info(__name__ + ": make debug output.")
    make_debug_output(env=env, blk=blk, block_id=block_id, logger=logger)
    logger.info(__name__ + ": ... done.")


def write_mlidev_config(env, path):
    if not os.path.exists(path):
        with rnw.open(path, "wt") as f:
            f.write(
                json_utils.dumps(
                    env["config"]["merlict_plenoscope_propagator_config"],
                    indent=4,
                )
            )


def make_debug_output(env, blk, block_id, logger):
    with open(
        opj(
            env["work_dir"],
            "plenoirf.production.draw_event_uids_for_debugging",
            "event_uids_for_debugging.json",
        ),
        "rt",
    ) as fin:
        event_uids_for_debugging = json_utils.loads(fin.read())

    block_id_str = "{:06d}".format(block_id)
    debug_out_path = opj(env["work_dir"], "merlict_events.debug.zip")
    event_uid_strs_in_block = blk["event_uid_strs_in_block"][block_id_str]

    if not os.path.exists(debug_out_path):
        with zipfile.ZipFile(debug_out_path, "w") as zout:
            pass

    for ii, event_uid_str in enumerate(event_uid_strs_in_block):
        merlict_event_id = ii + 1
        event_uid = int(event_uid_str)
        if event_uid in event_uids_for_debugging:
            logger.info(
                __name__
                + " exporting merlict uid:{:s} for debugging.".format(
                    event_uid_str
                )
            )
            merlict_event_path = opj(
                env["work_dir"],
                "blocks",
                block_id_str,
                "merlict",
                "{:d}".format(merlict_event_id),
            )

            assert_merlict_event_has_uid(
                merlict_event_path=merlict_event_path,
                event_uid=event_uid,
            )

            plenopy.tools.acp_format.compress_event_in_place(
                merlict_event_path
            )

            with zipfile.ZipFile(file=debug_out_path, mode="a") as zout:
                utils.zipfile_write_dir_recursively(
                    zipfile=zout,
                    filename=merlict_event_path,
                    arcname=bookkeeping.uid.make_uid_str(uid=event_uid),
                )


def assert_merlict_event_has_uid(merlict_event_path, event_uid):
    evth_path = opj(
        merlict_event_path, "simulation_truth", "corsika_event_header.bin"
    )
    with open(evth_path, "rb") as fin:
        corsika_evth = np.frombuffer(fin.read(), dtype=np.float32)
    event_uid_from_evth = bookkeeping.uid.make_uid(
        run_id=int(corsika_evth[cpw.I.EVTH.RUN_NUMBER]),
        event_id=int(corsika_evth[cpw.I.EVTH.EVENT_NUMBER]),
    )
    assert event_uid == event_uid_from_evth


def make_merlict_event_id(event_uid, event_uid_strs_in_block):
    for ii, i_event_uid_str in enumerate(event_uid_strs_in_block):
        merlict_event_id = ii + 1
        i_event_uid = int(i_event_uid_str)
        if i_event_uid == event_uid:
            return merlict_event_id
    assert False


def assert_plenopy_event_has_uid(event, event_uid):
    evth = event.simulation_truth.event.corsika_event_header.raw
    r = int(evth[cpw.I.EVTH.RUN_NUMBER])
    e = int(evth[cpw.I.EVTH.EVENT_NUMBER])
    actual_event_uid = bookkeeping.uid.make_uid(run_id=r, event_id=e)
    assert (
        actual_event_uid == event_uid
    ), "Actual {:d} vs expected {:d}".format(actual_event_uid, event_uid)
