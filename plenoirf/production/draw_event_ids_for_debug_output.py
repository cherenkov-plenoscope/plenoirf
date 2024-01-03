from .. import debugging
from .. import bookkeeping


def run_job(job, logger):
    event_ids_for_debugging = debugging.draw_event_ids_for_debug_output(
        num_events_in_run=job["num_events"],
        min_num_events=job["config"]["debug_output"]["run"]["min_num_events"],
        fraction_of_events=job["config"]["debug_output"]["run"][
            "fraction_of_events"
        ],
        prng=job["prng"],
    )

    job["run"]["uids_for_debugging"] = []
    for event_id in event_ids_for_debugging:
        uid = bookkeeping.uid.make_uid(run_id=job["run_id"], event_id=event_id)
        job["run"]["uids_for_debugging"].append(uid)
    job["run"]["uids_for_debugging"] = np.array(
        job["run"]["uids_for_debugging"]
    )

    logger.info(
        "event uids for debugging: {:s}.".format(
            str(job["run"]["uids_for_debugging"].tolist())
        )
    )
    return job