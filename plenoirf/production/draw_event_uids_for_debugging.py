from .. import debugging
from .. import bookkeeping
from .. import seeding

import numpy as np
import rename_after_writing as rnw
import json_utils
import os
from os.path import join as opj


def run(env, seed, logger):
    result_path = opj(env["work_dir"], "event_uids_for_debugging.json")
    if os.path.exists(result_path):
        logger.info("draw_pointing_range: already exists")
        return

    logger.info("draw_event_uids_for_debugging: ...")

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

    logger.info(
        "draw_event_uids_for_debugging: {:s}.".format(
            str(event_uids_for_debugging)
        )
    )

    with rnw.open(result_path, "wt") as fout:
        fout.write(json_utils.dumps(event_uids_for_debugging))

    logger.info("draw_event_uids_for_debugging: ... done.")
