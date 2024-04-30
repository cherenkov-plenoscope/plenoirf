from .. import debugging
from .. import bookkeeping
from .. import seeding

import numpy as np
import rename_after_writing as rnw
import json_utils
import os


def run_job(job, logger):
    opj = os.path.join

    prng = seeding.init_numpy_random_Generator_PCG64_from_path_and_name(
        path=opj(job["work_dir"], "named_random_seeds.json"),
        name="draw_event_uids_for_debugging",
    )

    event_ids_for_debugging = debugging.draw_event_ids_for_debugging(
        num_events_in_run=job["num_events"],
        min_num_events=job["config"]["debugging"]["run"]["min_num_events"],
        fraction_of_events=job["config"]["debugging"]["run"][
            "fraction_of_events"
        ],
        prng=prng,
    )
    event_uids_for_debugging = [
        bookkeeping.uid.make_uid(run_id=job["run_id"], event_id=event_id)
        for event_id in event_ids_for_debugging
    ]

    logger.info(
        "event uids for debugging: {:s}.".format(str(event_uids_for_debugging))
    )

    with rnw.open(
        opj(job["work_dir"], "event_uids_for_debugging.json"), "wt"
    ) as fout:
        fout.write(json_utils.dumps(event_uids_for_debugging))
