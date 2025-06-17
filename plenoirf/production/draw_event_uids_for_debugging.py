from .. import debugging
from .. import bookkeeping
from .. import utils

import numpy as np
import rename_after_writing as rnw
import json_utils
import json_line_logger
import os
from os.path import join as opj


def run(env, part, seed):
    name = __name__.split(".")[-1]
    module_work_dir = opj(env["work_dir"], part, name)

    if os.path.exists(module_work_dir):
        return

    os.makedirs(module_work_dir)
    logger = json_line_logger.LoggerFile(opj(module_work_dir, "log.jsonl"))
    logger.info(__name__)
    logger.info(f"seed: {seed:d}")

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

    path = os.path.join(module_work_dir, "event_uids_for_debugging.json")
    with rnw.open(path, "wt") as fout:
        fout.write(json_utils.dumps(event_uids_for_debugging))

    logger.info("done.")
    json_line_logger.shutdown(logger=logger)
    utils.gzip_file(opj(module_work_dir, "log.jsonl"))
