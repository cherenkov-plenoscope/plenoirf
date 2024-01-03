from .. import debugging


def run_job(job, logger):
    job["run"][
        "event_ids_for_debug"
    ] = debugging.draw_event_ids_for_debug_output(
        num_events_in_run=job["num_events"],
        min_num_events=job["config"]["debug_output"]["run"]["min_num_events"],
        fraction_of_events=job["config"]["debug_output"]["run"][
            "fraction_of_events"
        ],
        prng=job["prng"],
    )
    logger.debug(
        "event-ids for debugging: {:s}.".format(
            str(job["run"]["event_ids_for_debug"].tolist())
        )
    )
    return job
