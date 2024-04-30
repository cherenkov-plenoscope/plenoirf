import os
import numpy as np
import tarfile

import plenopy
import corsika_primary as cpw
import sparse_numeric_table as spt
import rename_after_writing as rnw
import json_utils
import sebastians_matplotlib_addons as sebplt

from .. import bookkeeping


def run_block(env, blk, block_id, logger):
    env = simulate_loose_trigger(
        env=env, blk=blk, block_id=block_id, logger=logger
    )
    return env


def simulate_loose_trigger(
    env,
    blk,
    block_id,
    logger,
):
    opj = os.path.join
    block_dir = opj(env["work_dir"], "blocks", "{:06d}".format(block_id))
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
        env["event_table"]["instrument"].append_record(insrec)

        # apply loose trigger
        # -------------------
        if True:
            uid_str = bookkeeping.uid.make_uid_str(
                run_id=run_id,
                event_id=event_id,
            )
            visible_cherenkov_photon_size = json_utils.read(
                path=os.path.join(
                    env["work_dir"],
                    "inspect_cherenkov_pool",
                    "visible_cherenkov_photon_size.json",
                )
            )
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
                    out_dir=os.path.join(work_dir, "{:06d}".format(event_id)),
                    foci_trigger_image_sequences=foci_trigger_image_sequences,
                )

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
        env["event_table"]["trigger"].append_record(trgtru)

        # passing loose trigger
        # ---------------------
        if (
            trgtru["response_pe"]
            >= env["config"]["sum_trigger"]["threshold_pe"]
        ):
            ptp = uidrec.copy()
            ptp["tmp_path"] = event._path
            ptp["uid_str"] = bookkeeping.uid.UID_FOTMAT_STR.format(
                ptp[spt.IDX]
            )
            table_past_trigger.append(ptp)

            patrec = uidrec.copy()
            env["event_table"]["pasttrigger"].append_record(patrec)

            # export past loose trigger
            # -------------------------
            if uidrec[spt.IDX] in env["run"]["event_uids_for_debugging"]:
                plenopy.tools.acp_format.compress_event_in_place(
                    ptp["tmp_path"]
                )
                final_tarname = ptp["uid_str"] + ".tar"
                plenoscope_event_dir_to_tar(
                    event_dir=ptp["tmp_path"],
                    output_tar_path=opj(work_dir, final_tarname),
                )

    return env


def plenoscope_event_dir_to_tar(event_dir, output_tar_path=None):
    if output_tar_path is None:
        output_tar_path = event_dir + ".tar"
    with tarfile.open(output_tar_path, "w") as tarfout:
        tarfout.add(event_dir, arcname=".")


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
