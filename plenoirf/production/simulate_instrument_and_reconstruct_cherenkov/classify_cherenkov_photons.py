import numpy as np
import os
from os.path import join as opj
import plenopy as pl
import rename_after_writing as rnw
import sparse_numeric_table as snt
import corsika_primary as cpw
from .. import bookkeeping
from .. import event_table
from . import simulate_hardware


def run_block(env, blk, block_id, logger):
    name = __name__.split(".")[-1]
    logger.info(name + ": start ...")

    block_id_str = "{:06d}".format(block_id)
    block_dir = opj(blk["blocks_dir"], block_id_str)
    sub_work_dir = opj(block_dir, name)

    if os.path.exists(sub_work_dir):
        logger.info(name + ": already done. skip computation.")
        return

    os.makedirs(sub_work_dir)

    evttab = snt.SparseNumericTable(index_key="uid")
    evttab = event_table.add_levels_from_path(
        evttab=evttab,
        path=opj(
            block_dir,
            "simulate_loose_trigger",
            "event_table.snt.zip",
        ),
    )
    additional_level_keys = ["cherenkovclassification"]
    for key in additional_level_keys:
        evttab = event_table.add_empty_level(evttab, key)

    evttab = classify_cherenkov_photons(
        evttab=evttab,
        reconstructed_cherenkov_path=os.path.join(
            sub_work_dir, "reconstructed_cherenkov.loph.tar"
        ),
        config_cherenkov_classification_region_of_interest=env["config"][
            "cherenkov_classification"
        ]["region_of_interest"],
        config_cherenkov_classification=env["config"][
            "cherenkov_classification"
        ],
        light_field_geometry=env["light_field_geometry"],
        trigger_geometry=env["trigger_geometry"],
        event_uid_strs_in_block=blk["event_uid_strs_in_block"][block_id_str],
        block_id=block_id,
        block_dir=block_dir,
        logger=logger,
    )

    event_table.write_certain_levels_to_path(
        evttab=evttab,
        path=opj(sub_work_dir, "event_table.snt.zip"),
        level_keys=additional_level_keys,
    )

    logger.info(name + ": ... done.")


def classify_cherenkov_photons(
    evttab,
    reconstructed_cherenkov_path,
    config_cherenkov_classification_region_of_interest,
    config_cherenkov_classification,
    light_field_geometry,
    trigger_geometry,
    event_uid_strs_in_block,
    block_id,
    block_dir,
    logger,
):
    opj = os.path.join

    roi_cfg = config_cherenkov_classification_region_of_interest
    dbscan_cfg = config_cherenkov_classification

    with pl.photon_stream.loph.LopfTarWriter(
        path=reconstructed_cherenkov_path,
        uid_num_digits=bookkeeping.uid.UID_NUM_DIGITS,
    ) as cer_phs_run:
        for ptp in evttab["pasttrigger"]:
            event_uid = ptp["uid"]
            merlict_event_id = simulate_hardware.make_merlict_event_id(
                event_uid=event_uid,
                event_uid_strs_in_block=event_uid_strs_in_block,
            )

            event_path = opj(
                block_dir,
                "simulate_hardware",
                "merlict",
                "{:d}".format(merlict_event_id),
            )

            event = pl.Event(
                path=event_path,
                light_field_geometry=light_field_geometry,
            )
            simulate_hardware.assert_plenopy_event_has_uid(
                event=event, event_uid=event_uid
            )
            trigger_responses = pl.trigger.io.read_trigger_response_from_path(
                path=os.path.join(event._path, "refocus_sum_trigger.json")
            )
            roi = pl.trigger.region_of_interest.from_trigger_response(
                trigger_response=trigger_responses,
                trigger_geometry=trigger_geometry,
                time_slice_duration=event.raw_sensor_response[
                    "time_slice_duration"
                ],
            )
            photons = pl.classify.RawPhotons.from_event(event)
            (
                cherenkov_photons,
                roi_settings,
            ) = pl.classify.cherenkov_photons_in_roi_in_image(
                roi=roi,
                photons=photons,
                roi_time_offset_start=roi_cfg["time_offset_start_s"],
                roi_time_offset_stop=roi_cfg["time_offset_stop_s"],
                roi_cx_cy_radius=np.deg2rad(roi_cfg["direction_radius_deg"]),
                roi_object_distance_offsets=roi_cfg[
                    "object_distance_offsets_m"
                ],
                dbscan_epsilon_cx_cy_radius=np.deg2rad(
                    dbscan_cfg["neighborhood_radius_deg"]
                ),
                dbscan_min_number_photons=dbscan_cfg["min_num_photons"],
                dbscan_deg_over_s=dbscan_cfg[
                    "direction_to_time_mixing_deg_per_s"
                ],
            )
            pl.classify.write_dense_photon_ids_to_event(
                event_path=os.path.abspath(event._path),
                photon_ids=cherenkov_photons.photon_ids,
                settings=roi_settings,
            )
            crcl = pl.classify.benchmark(
                pulse_origins=event.simulation_truth.detector.pulse_origins,
                photon_ids_cherenkov=cherenkov_photons.photon_ids,
            )
            crcl["uid"] = event_uid
            evttab["cherenkovclassification"].append_record(crcl)

            # export reconstructed Cherenkov photons
            # --------------------------------------
            cer_phs = pl.photon_stream.loph.raw_sensor_response_to_photon_stream_in_loph_repr(
                raw_sensor_response=event.raw_sensor_response,
                cherenkov_photon_ids=cherenkov_photons.photon_ids,
            )
            cer_phs_run.add(uid=event_uid, phs=cer_phs)

    return evttab
