import copy
import os
import numpy as np
from os import path as op
from os.path import join as opj

import corsika_primary as cpw
from .. import bookkeeping


def run_job(job, logger):
    logger.info("split_event_tape_into_blocks, split")

    blocks_dir = os.path.join(job["paths"]["work_dir"], "blocks")
    os.makedirs(blocks_dir, exist_ok=True)

    job["run"]["uids_in_cherenkov_pool_blocks"] = split_event_tape_into_blocks(
        inpath=os.path.join(
            job["paths"]["work_dir"],
            "simulate_shower_and_collect_cherenkov_light_in_grid",
            "cherenkov_pools.tar",
        ),
        outpath_block_fmt=os.path.join(
            blocks_dir, "{block_id:06d}", "cherenkov_pools.tar"
        ),
        num_events=job["max_num_events_in_merlict_run"],
    )

    return job


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
            uid = bookkeeping.uid.make_uid(
                run_id=int(evth[cpw.I.EVTH.RUN_NUMBER]),
                event_id=int(evth[cpw.I.EVTH.EVENT_NUMBER]),
            )

            if event_counter % num_events == 0:
                block_id += 1
                block_id_str = "{:06d}".format(block_id)
                if orun is not None:
                    orun.close()
                outpath = opj(outpath_block_fmt.format(block_id=block_id))
                outdir = os.path.dirname(outpath)
                os.makedirs(outdir, exist_ok=True)
                orun = Writer(outpath)
                orun.write_runh(runh)
                uid_map[block_id_str] = []

            uid_map[block_id_str].append(uid)
            orun.write_evth(evth=evth)
            orun.write_payload(payload=cherenkov_bunches)
            event_counter += 1

        if orun is not None:
            orun.close()

    return uid_map


def read_all_cherenkov_bunches(cherenkov_reader):
    return np.vstack([b for b in cherenkov_reader])
