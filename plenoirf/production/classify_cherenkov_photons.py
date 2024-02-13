import numpy as np
import os
import plenopy as pl


def run_job_block(job, blk, block_id, logger):
    opj = os.path.join
    block_dir = opj(
        job["paths"]["tmp_dir"], "blocks", "{:06d}".format(block_id)
    )
    work_dir = opj(block_dir, "classify_cherenkov_photons")
    os.makedirs(work_dir, exist_ok=True)
    cache_path = os.path.join(work_dir, "__job_cache__")

    if os.path.exists(cache_path) and job["cache"]:
        logger.info(
            "classify_cherenkov_photons block{:06d}, read cache".format(
                block_id
            )
        )
        return job_io.read(path=cache_path)
    else:
        job = classify_cherenkov_photons(
            job=job, blk=blk, block_id=block_id, logger=logger
        )

        if job["cache"]:
            logger.info(
                "classify_cherenkov_photons block{:06d}, write cache".format(
                    block_id
                )
            )
            job_io.write(path=cache_path, job=job)

    return job


def classify_cherenkov_photons(
    job,
    blk,
    block_id,
    logger,
    # tabrec,
    # tmp_dir,
    # table_past_trigger,
):
    opj = os.path.join
    block_dir = opj(
        job["paths"]["tmp_dir"], "blocks", "{:06d}".format(block_id)
    )

    roi_cfg = job["cherenkov_classification"]["region_of_interest"]
    dbscan_cfg = job["cherenkov_classification"]

    with pl.photon_stream.loph.LopfTarWriter(
        path=os.path.join(block_dir, "reconstructed_cherenkov.tar"),
        uid_num_digits=unique.UID_NUM_DIGITS,
    ) as cer_phs_run:
        for ptp in table_past_trigger:
            event = pl.Event(
                path=ptp["tmp_path"],
                light_field_geometry=blk["light_field_geometry"],
            )
            trigger_responses = pl.trigger.io.read_trigger_response_from_path(
                path=os.path.join(event._path, "refocus_sum_trigger.json")
            )
            roi = pl.trigger.region_of_interest.from_trigger_response(
                trigger_response=trigger_responses,
                trigger_geometry=blk["trigger_geometry"],
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
                event_path=op.abspath(event._path),
                photon_ids=cherenkov_photons.photon_ids,
                settings=roi_settings,
            )
            crcl = pl.classify.benchmark(
                pulse_origins=event.simulation_truth.detector.pulse_origins,
                photon_ids_cherenkov=cherenkov_photons.photon_ids,
            )
            crcl[spt.IDX] = ptp[spt.IDX]
            tabrec["cherenkovclassification"].append(crcl)

            # export reconstructed Cherenkov photons
            # --------------------------------------
            cer_phs = pl.photon_stream.loph.raw_sensor_response_to_photon_stream_in_loph_repr(
                raw_sensor_response=event.raw_sensor_response,
                cherenkov_photon_ids=cherenkov_photons.photon_ids,
            )
            cer_phs_run.add(uid=ptp[spt.IDX], phs=cer_phs)

    rnw.copy(
        src=op.join(tmp_dir, "reconstructed_cherenkov.tar"),
        dst=op.join(
            job["past_trigger_reconstructed_cherenkov_dir"],
            _run_id_str(job=job) + "_reconstructed_cherenkov.tar",
        ),
    )
    return tabrec
