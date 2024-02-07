import copy
import os
from os import path as op
from os.path import join as opj

import corsika_primary as cpw
from .simulate_shower_and_collect_cherenkov_light_in_grid import (
    read_all_cherenkov_bunches,
)
from .. import bookkeeping


def run_job(job, logger):
    logger.info("split_event_tape_into_blocks, split")
    job["run"]["uids_in_cherenkov_pool_blocks"] = split_event_tape_into_blocks(
        inpath=job["paths"]["tmp"]["cherenkov_pools"],
        outpath_block_fmt=job["paths"]["tmp"]["cherenkov_pools_block_fmt"],
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
