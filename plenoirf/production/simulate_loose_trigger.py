import os
import numpy as np
import tarfile

import plenopy
import corsika_primary as cpw
import sparse_numeric_table as snt
import rename_after_writing as rnw
import json_utils
from json_line_logger import xml
import sebastians_matplotlib_addons as sebplt

from .. import bookkeeping
from .. import event_table


def run_block(env, blk, block_id, logger):
    opj = os.path.join
    logger.info(__name__ + ": start ...")

    block_dir = opj(env["work_dir"], "blocks", "{:06d}".format(block_id))
    sub_work_dir = opj(block_dir, __name__)

    if os.path.exists(sub_work_dir):
        logger.info(__name__ + ": already done. skip computation.")
        return

    event_uids_for_debugging = json_utils.read(
        path=os.path.join(
            env["work_dir"],
            "plenoirf.production.draw_event_uids_for_debugging.json",
        )
    )
    visible_cherenkov_photon_size = json_utils.read(
        path=os.path.join(
            env["work_dir"],
            "plenoirf.production.inspect_cherenkov_pool",
            "visible_cherenkov_photon_size.json",
        )
    )

    evttab = {}
    evttab = event_table.add_levels_from_path(
        evttab=evttab,
        path=opj(
            env["work_dir"],
            "plenoirf.production.simulate_shower_and_collect_cherenkov_light_in_grid",
            "event_table.tar",
        ),
    )
    evttab = event_table.add_levels_from_path(
        evttab=evttab,
        path=opj(
            env["work_dir"],
            "plenoirf.production.inspect_particle_pool",
            "event_table.tar",
        ),
    )
    evttab = event_table.add_empty_level(evttab, "instrument")
    evttab = event_table.add_empty_level(evttab, "trigger")
    evttab = event_table.add_empty_level(evttab, "pasttrigger")

    evttab = simulate_loose_trigger(
        env=env,
        blk=blk,
        block_id=block_id,
        work_dir=sub_work_dir,
        evttab=evttab,
        event_uids_for_debugging=event_uids_for_debugging,
        visible_cherenkov_photon_size=visible_cherenkov_photon_size,
        logger=logger,
        write_figures=env["debugging_figures"],
    )

    event_table.write_certain_levels_to_path(
        evttab=evttab,
        path=opj(sub_work_dir, "event_table.tar"),
        level_keys=["instrument", "trigger", "pasttrigger"],
    )

    logger.info(__name__ + ": ... done.")


def simulate_loose_trigger(
    env,
    blk,
    block_id,
    work_dir,
    evttab,
    event_uids_for_debugging,
    visible_cherenkov_photon_size,
    logger,
    write_figures,
):
    opj = os.path.join
    block_dir = opj(env["work_dir"], "blocks", "{:06d}".format(block_id))

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
            snt.IDX: bookkeeping.uid.make_uid(run_id=run_id, event_id=event_id)
        }
        uid_str = bookkeeping.uid.make_uid_str(
            run_id=run_id,
            event_id=event_id,
        )

        # export instrument's time relative to CORSIKA's time
        # ---------------------------------------------------
        insrec = uidrec.copy()
        insrec[
            "start_time_of_exposure_s"
        ] = event.simulation_truth.photon_propagator.nsb_exposure_start_time()
        evttab["instrument"].append_record(insrec)

        # apply loose trigger
        # -------------------
        if write_figures:
            if visible_cherenkov_photon_size[uid_str] > 100:
                foci_trigger_image_sequences = (
                    plenopy.trigger.estimate.estimate_trigger_image_sequences(
                        raw_sensor_response=event.raw_sensor_response,
                        light_field_geometry=blk["light_field_geometry"],
                        trigger_geometry=blk["trigger_geometry"],
                        integration_time_slices=(
                            env["config"]["sum_trigger"][
                                "integration_time_slices"
                            ]
                        ),
                    )
                )

                plot_foci_trigger_image_sequences(
                    out_dir=os.path.join(work_dir, uid_str),
                    foci_trigger_image_sequences=foci_trigger_image_sequences,
                )

        logger.debug(xml("EventTime", uid=uid_str, status="trigger_start"))

        (
            trigger_responses,
            max_response_in_focus_vs_timeslices,
        ) = plenopy.trigger.estimate.first_stage(
            raw_sensor_response=event.raw_sensor_response,
            light_field_geometry=blk["light_field_geometry"],
            trigger_geometry=blk["trigger_geometry"],
            integration_time_slices=(
                env["config"]["sum_trigger"]["integration_time_slices"]
            ),
        )

        logger.debug(xml("EventTime", uid=uid_str, status="trigger_stop"))

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
        evttab["trigger"].append_record(trgtru)

        logger.debug(xml("EventTime", uid=uid_str, status="trigger_exported"))

        # passing loose trigger
        # ---------------------
        if (
            trgtru["response_pe"]
            >= env["config"]["sum_trigger"]["threshold_pe"]
        ):
            ptp = uidrec.copy()
            ptp["tmp_path"] = event._path
            ptp["uid_str"] = bookkeeping.uid.UID_FOTMAT_STR.format(
                ptp[snt.IDX]
            )
            table_past_trigger.append(ptp)

            patrec = uidrec.copy()
            evttab["pasttrigger"].append_record(patrec)

    return evttab


def plot_foci_trigger_image_sequences(out_dir, foci_trigger_image_sequences):
    os.makedirs(out_dir, exist_ok=True)

    num_foci, num_time_slices, num_pixel = foci_trigger_image_sequences.shape
    time_slices_bin_edges = np.arange(num_time_slices + 1)
    pixel_bin_edges = np.arange(num_pixel + 1)
    vmax = np.max(foci_trigger_image_sequences)
    vmin = 0.0

    for focus in range(num_foci):
        image = foci_trigger_image_sequences[focus]

        fig = sebplt.figure(
            style={"rows": 720, "cols": 2560, "fontsize": 1}, dpi=240
        )
        ax_img = sebplt.add_axes(fig=fig, span=[0.12, 0.12, 0.75, 0.8])
        ax_cm = sebplt.add_axes(fig=fig, span=[0.9, 0.12, 0.03, 0.8])

        pcm_img = ax_img.pcolormesh(
            pixel_bin_edges,
            time_slices_bin_edges,
            image,
            cmap="viridis",
            norm=sebplt.plt_colors.PowerNorm(gamma=1, vmin=vmin, vmax=vmax),
        )

        sebplt.plt.colorbar(pcm_img, cax=ax_cm, extend="max")

        fig.savefig(os.path.join(out_dir, "{:06d}.jpg".format(focus)))
        sebplt.close(fig)
