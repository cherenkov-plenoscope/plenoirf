import os
from os import path as op
from os.path import join as opj

import merlict_development_kit_python
from . import job_io


def run_job_block(job, block_id, logger):
    cache_path = opj(
        job["paths"]["tmp_dir"],
        "simulate_hardware_block{block_id:06d}".format(block_id=block_id),
    )

    if os.path.exists(cache_path) and job["cache"]:
        logger.info(
            "simulate_hardware block{:06d}, read cache".format(block_id)
        )
        return job_io.read(path=cache_path)
    else:
        job = simulate_hardware(job=job, block_id=block_id)
        make_debug_output(job=job, block_id=block_id)

        if job["cache"]:
            logger.info(
                "simulate_hardware block{:06d}, write cache".format(block_id)
            )
            job_io.write(path=cache_path, job=job)

    return job


def simulate_hardware(job, block_id):
    detector_responses_path = op.join(tmp_dir, "detector_responses")

    rc = merlict_development_kit_python.plenoscope_propagator.plenoscope_propagator(
        corsika_run_path=job["paths"][
            "cherenkov_pools_block_fmt".format(block_id=block_id)
        ],
        output_path=job["paths"][
            "merlict_output_block_fmt".format(block_id=block_id)
        ],
        light_field_geometry_path=job["paths"]["light_field_calibration"],
        merlict_plenoscope_propagator_path=job["config"]["executables"][
            "merlict_plenoscope_propagator_path"
        ],
        merlict_plenoscope_propagator_config_path=job[
            "merlict_plenoscope_propagator_config_path"
        ],
        random_seed=job["run_id"],
        photon_origins=True,
        stdout_path=job["paths"][
            "merlict_stdout_block_fmt".format(block_id=block_id)
        ],
        stderr_path=job["paths"][
            "merlict_stderr_block_fmt".format(block_id=block_id)
        ],
    )
    assert rc == 0, "Expected merlict's return code to be zero."

    return job


def make_debug_output(job, block_id):
    uids_in_block = job["run"]["uids_in_cherenkov_pool_blocks"][str(block_id)]
    for event_id in job["run"]["event_ids_for_debug"]:
        uid = bookkeeping.uid.make_uid(run_id=job["run_id"], event_id=event_id)
        if uid in uids_in_block:
            print("Do some debug I guess?", uid, block_id)
