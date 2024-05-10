from .. import debugging
from .. import bookkeeping

import numpy as np
import rename_after_writing
import json_utils
import os


def run(env, seed, logger):
    opj = os.path.join
    logger.info(__name__ + ": start ...")

    result_path = opj(env["work_dir"], "event_uids_for_debugging.json")
    if os.path.exists(result_path):
        logger.info(__name__ + ": already done. skip computation.")
        return

    prng = np.random.Generator(np.random.PCG64(seed))
    event_ids_for_debugging = debugging.draw_event_ids_for_debugging(
        num_events_in_run=env["num_events"],
        min_num_events=env["config"]["debugging"]["run"]["min_num_events"],
        fraction_of_events=env["config"]["debugging"]["run"][
            "fraction_of_events"
        ],
        prng=prng,
    )
    event_uids_for_debugging = [
        bookkeeping.uid.make_uid(run_id=env["run_id"], event_id=event_id)
        for event_id in event_ids_for_debugging
    ]

    with rename_after_writing.open(result_path, "wt") as fout:
        fout.write(json_utils.dumps(event_uids_for_debugging))

    logger.info(__name__ + ": ... done.")
