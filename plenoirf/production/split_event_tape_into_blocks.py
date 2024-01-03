import copy
import os
from os import path as op
from os.path import join as opj

import corsika_primary as cpw
from .corsika_and_grid import read_all_cherenkov_bunches


def run_job(job, logger):
    logger.info("split_event_tape_into_blocks, split")
    job["run"]["cherenkov_pools"] = split_event_tape_into_blocks(
        inpath=job["paths"]["tmp"]["cherenkov_pools"],
        outpath_block_fmt=job["paths"]["tmp"]["cherenkov_pools_block_fmt"],
        num_events=job["max_num_events_in_merlict_run"],
    )

    return job


def split_event_tape_into_blocks(inpath, outpath_block_fmt, num_events):
    Writer = cpw.cherenkov.CherenkovEventTapeWriter
    Reader = cpw.cherenkov.CherenkovEventTapeReader
    outpaths = {}

    orun = None
    block = 0
    event_counter = 0
    with Reader(inpath) as irun:
        runh = copy.deepcopy(irun.runh)

        for event in irun:
            evth, cherenkov_reader = event
            cherenkov_bunches = read_all_cherenkov_bunches(cherenkov_reader)

            if event_counter % num_events == 0:
                block += 1
                if orun is not None:
                    orun.close()
                outpaths[block] = opj(outpath_block_fmt.format(block=block))
                orun = Writer(outpaths[block])
                orun.write_runh(runh)

            orun.write_evth(evth=evth)
            orun.write_payload(payload=cherenkov_bunches)
            event_counter += 1

        if orun is not None:
            orun.close()

    return outpaths
