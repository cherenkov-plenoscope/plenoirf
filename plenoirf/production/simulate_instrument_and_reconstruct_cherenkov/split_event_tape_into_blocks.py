import copy
import os
from os.path import join as opj
import numpy as np
import json_utils
import corsika_primary as cpw
import rename_after_writing as rnw
import json_line_logger
import gzip

from ... import bookkeeping
from ... import utils


def run(env, blk, logger):
    name = __name__.split(".")[-1]
    logger.info(name + ": start ...")

    if os.path.exists(blk["blocks_dir"]):
        logger.info(name + ": already done. skip computation.")
        return
    os.makedirs(blk["blocks_dir"])

    uid_map = split_event_tape_into_blocks(
        inpath=opj(
            env["work_dir"],
            "prm2cer",
            "simulate_shower_again_and_cut_cherenkov_light_falling_into_instrument",
            "cherenkov_pools.tar.gz",
        ),
        outpath_block_fmt=opj(
            blk["blocks_dir"], "{block_id:06d}", "cherenkov_pools.tar"
        ),
        num_events=env["max_num_events_in_merlict_run"],
    )

    logger.info(name + "write uids in blocks.")
    with rnw.Path(
        opj(blk["blocks_dir"], "event_uid_strs_in_block.json.gz")
    ) as opath:
        with gzip.open(opath, "wt") as fout:
            fout.write(json_utils.dumps(uid_map))

    logger.info(name + ": ... done.")


def split_event_tape_into_blocks(inpath, outpath_block_fmt, num_events):
    Writer = cpw.cherenkov.CherenkovEventTapeWriter
    Reader = cpw.cherenkov.CherenkovEventTapeReader
    block_ids = []

    uid_map = {}

    orun = None
    block_id = 0
    event_counter = 0
    with Reader(inpath) as irun:
        runh = copy.deepcopy(irun.runh)

        for event in irun:
            evth, cherenkov_reader = event
            cherenkov_bunches = read_all_cherenkov_bunches(cherenkov_reader)
            uid_str = bookkeeping.uid.make_uid_str(
                run_id=int(evth[cpw.I.EVTH.RUN_NUMBER]),
                event_id=int(evth[cpw.I.EVTH.EVENT_NUMBER]),
            )

            if event_counter % num_events == 0:
                block_id += 1
                block_id_str = "{:06d}".format(block_id)
                if orun is not None:
                    orun.close()
                outpath = os.path.join(
                    outpath_block_fmt.format(block_id=block_id)
                )
                outdir = os.path.dirname(outpath)
                os.makedirs(outdir, exist_ok=True)
                orun = Writer(outpath)
                orun.write_runh(runh)
                uid_map[block_id_str] = []

            uid_map[block_id_str].append(uid_str)
            orun.write_evth(evth=evth)
            orun.write_payload(payload=cherenkov_bunches)
            event_counter += 1

        if orun is not None:
            orun.close()

    return uid_map


def read_all_cherenkov_bunches(cherenkov_reader):
    return np.vstack([b for b in cherenkov_reader])
