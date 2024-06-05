import copy
import os
import numpy as np
import json_utils
import corsika_primary as cpw
import rename_after_writing as rnw

from .. import bookkeeping


def run(env, logger):
    logger.info(__name__ + ": start ...")

    blocks_dir = os.path.join(env["work_dir"], "blocks")
    os.makedirs(blocks_dir, exist_ok=True)

    result_path = os.path.join(blocks_dir, "event_uid_strs_in_block.json")

    if os.path.exists(result_path):
        logger.info(__name__ + ": already done. skip computation.")
        return

    uid_map = split_event_tape_into_blocks(
        inpath=os.path.join(
            env["work_dir"],
            "plenoirf.production.simulate_shower_again_and_cut_cherenkov_light_falling_into_instrument",
            "cherenkov_pools.tar",
        ),
        outpath_block_fmt=os.path.join(
            blocks_dir, "{block_id:06d}", "cherenkov_pools.tar"
        ),
        num_events=env["max_num_events_in_merlict_run"],
    )
    with rnw.open(result_path, "wt") as fout:
        fout.write(json_utils.dumps(uid_map))

    logger.info(__name__ + ": ... done.")


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
