import os
import numpy as np
import tarfile

import plenopy
import corsika_primary as cpw
import sparse_numeric_table as spt
import rename_after_writing as rnw
import json_utils

from .. import bookkeeping
from . import job_io


def run_job_block(job, blk, block_id, logger):
    opj = os.path.join
    block_dir = opj(
        job["paths"]["work_dir"], "blocks", "{:06d}".format(block_id)
    )
    work_dir = opj(block_dir, "simulate_loose_trigger")
    os.makedirs(work_dir, exist_ok=True)
    cache_path = os.path.join(work_dir, "__job_cache__")

    if os.path.exists(cache_path) and job["cache"]:
        logger.info(
            "simulate_hardware block{:06d}, read cache".format(block_id)
        )
        return job_io.read(path=cache_path)
    else:
        job = simulate_loose_trigger(
            job=job, blk=blk, block_id=block_id, logger=logger
        )

        if job["cache"]:
            logger.info(
                "simulate_hardware block{:06d}, write cache".format(block_id)
            )
            job_io.write(path=cache_path, job=job)

    return job


def simulate_loose_trigger(
    job,
    blk,
    block_id,
    logger,
):
    opj = os.path.join
    block_dir = opj(
        job["paths"]["work_dir"], "blocks", "{:06d}".format(block_id)
    )
    work_dir = opj(block_dir, "simulate_loose_trigger")

    # loop over sensor responses
    # --------------------------
    merlict_run = plenopy.Run(
        path=opj(block_dir, "merlict"),
        light_field_geometry=blk["light_field_geometry"],
    )
    table_past_trigger = []
    os.makedirs(work_dir, exist_ok=True)

    for event in merlict_run:
        # id
        # --
        cevth = event.simulation_truth.event.corsika_event_header.raw
        run_id = int(cevth[cpw.I.EVTH.RUN_NUMBER])
        event_id = int(cevth[cpw.I.EVTH.EVENT_NUMBER])
        uidrec = {
            spt.IDX: bookkeeping.uid.make_uid(run_id=run_id, event_id=event_id)
        }

        # export instrument's time relative to CORSIKA's time
        # ---------------------------------------------------
        insrec = uidrec.copy()
        insrec[
            "start_time_of_exposure_s"
        ] = event.simulation_truth.photon_propagator.nsb_exposure_start_time()
        job["event_table"]["instrument"].append_record(insrec)

        # apply loose trigger
        # -------------------
        (
            trigger_responses,
            max_response_in_focus_vs_timeslices,
        ) = plenopy.trigger.estimate.first_stage(
            raw_sensor_response=event.raw_sensor_response,
            light_field_geometry=blk["light_field_geometry"],
            trigger_geometry=blk["trigger_geometry"],
            integration_time_slices=(
                job["config"]["sum_trigger"]["integration_time_slices"]
            ),
        )

        trg_resp_path = opj(event._path, "refocus_sum_trigger.json")
        with rnw.open(trg_resp_path, "wt") as f:
            f.write(json_utils.dumps(trigger_responses, indent=4))

        trg_maxr_path = opj(
            event._path, "refocus_sum_trigger.focii_x_time_slices.uint32"
        )
        with rnw.open(trg_maxr_path, "wb") as f:
            f.write(max_response_in_focus_vs_timeslices.tobytes())

        # export trigger-truth
        # --------------------
        trgtru = uidrec.copy()
        trgtru["num_cherenkov_pe"] = int(
            event.simulation_truth.detector.number_air_shower_pulses()
        )
        trgtru["response_pe"] = int(
            np.max([focus["response_pe"] for focus in trigger_responses])
        )
        for o in range(len(trigger_responses)):
            trgtru["focus_{:02d}_response_pe".format(o)] = int(
                trigger_responses[o]["response_pe"]
            )
        job["event_table"]["trigger"].append_record(trgtru)

        # passing loose trigger
        # ---------------------
        if (
            trgtru["response_pe"]
            >= job["config"]["sum_trigger"]["threshold_pe"]
        ):
            ptp = uidrec.copy()
            ptp["tmp_path"] = event._path
            ptp["uid_str"] = bookkeeping.uid.UID_FOTMAT_STR.format(
                ptp[spt.IDX]
            )
            table_past_trigger.append(ptp)

            patrec = uidrec.copy()
            job["event_table"]["pasttrigger"].append_record(patrec)

            # export past loose trigger
            # -------------------------
            if uidrec[spt.IDX] in job["run"]["event_uids_for_debugging"]:
                plenopy.tools.acp_format.compress_event_in_place(
                    ptp["tmp_path"]
                )
                final_tarname = ptp["uid_str"] + ".tar"
                plenoscope_event_dir_to_tar(
                    event_dir=ptp["tmp_path"],
                    output_tar_path=opj(work_dir, final_tarname),
                )

    return job


def plenoscope_event_dir_to_tar(event_dir, output_tar_path=None):
    if output_tar_path is None:
        output_tar_path = event_dir + ".tar"
    with tarfile.open(output_tar_path, "w") as tarfout:
        tarfout.add(event_dir, arcname=".")
