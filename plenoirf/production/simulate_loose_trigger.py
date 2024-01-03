import os
from os import path as op
from os.path import join as opj

import plenopy
import corsika_primary as cpw
import sparse_numeric_table as spt
import rename_after_writing as rnw

from .. import bookkeeping


def _run_job_block(job, blk, block_id, logger):
    job = simulate_loose_trigger(
        job=job, blk=blk, block_id=block_id, logger=logger
    )
    return job


def simulate_loose_trigger(
    job,
    blk,
    block_id,
    logger,
    # tabrec,
    # detector_responses_path,
    # light_field_geometry,
    # trigger_geometry,
    # tmp_dir,
):
    # loop over sensor responses
    # --------------------------
    merlict_run = plenopy.Run(detector_responses_path)
    table_past_trigger = []
    tmp_past_trigger_dir = op.join(tmp_dir, "past_trigger")
    os.makedirs(tmp_past_trigger_dir, exist_ok=True)
    RAW_SKIP = int(job["raw_sensor_response"]["skip_num_events"])
    assert RAW_SKIP > 0

    for event in merlict_run:
        # id
        # --
        cevth = event.simulation_truth.event.corsika_event_header.raw
        run_id = int(cevth[cpw.I.EVTH.RUN_NUMBER])
        event_id = int(cevth[cpw.I.EVTH.EVENT_NUMBER])
        ide = {spt.IDX: unique.make_uid(run_id=run_id, event_id=event_id)}

        # export instrument's time relative to CORSIKA's time
        # ---------------------------------------------------
        ttabs = ide.copy()
        ttabs[
            "start_time_of_exposure_s"
        ] = event.simulation_truth.photon_propagator.nsb_exposure_start_time()
        tabrec["instrument"].append(ttabs)

        # apply loose trigger
        # -------------------
        (
            trigger_responses,
            max_response_in_focus_vs_timeslices,
        ) = plenopy.trigger.estimate.first_stage(
            raw_sensor_response=event.raw_sensor_response,
            light_field_geometry=light_field_geometry,
            trigger_geometry=trigger_geometry,
            integration_time_slices=(
                job["sum_trigger"]["integration_time_slices"]
            ),
        )

        trg_resp_path = op.join(event._path, "refocus_sum_trigger.json")
        with rnw.open(trg_resp_path, "wt") as f:
            f.write(json_utils.dumps(trigger_responses, indent=4))

        trg_maxr_path = op.join(
            event._path, "refocus_sum_trigger.focii_x_time_slices.uint32"
        )
        with rnw.open(trg_maxr_path, "wb") as f:
            f.write(max_response_in_focus_vs_timeslices.tobytes())

        # export trigger-truth
        # --------------------
        trgtru = ide.copy()
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
        tabrec["trigger"].append(trgtru)

        # passing loose trigger
        # ---------------------
        if trgtru["response_pe"] >= job["sum_trigger"]["threshold_pe"]:
            ptp = ide.copy()
            ptp["tmp_path"] = event._path
            ptp["unique_id_str"] = unique.UID_FOTMAT_STR.format(ptp[spt.IDX])
            table_past_trigger.append(ptp)

            ptrg = ide.copy()
            tabrec["pasttrigger"].append(ptrg)

            # export past loose trigger
            # -------------------------
            if ide[spt.IDX] % RAW_SKIP == 0:
                plenopy.tools.acp_format.compress_event_in_place(
                    ptp["tmp_path"]
                )
                final_tarname = ptp["unique_id_str"] + ".tar"
                plenoscope_event_dir_to_tar(
                    event_dir=ptp["tmp_path"],
                    output_tar_path=op.join(
                        tmp_past_trigger_dir, final_tarname
                    ),
                )
                rnw.copy(
                    src=op.join(tmp_past_trigger_dir, final_tarname),
                    dst=op.join(job["past_trigger_dir"], final_tarname),
                )

    return tabrec, table_past_trigger, tmp_past_trigger_dir
