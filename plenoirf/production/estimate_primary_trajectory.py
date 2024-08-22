import gamma_ray_reconstruction as gamrec
import numpy as np
import os
import sparse_numeric_table as snt
import plenopy
from .. import bookkeeping
from .. import event_table
from . import simulate_hardware


def run_block(env, blk, block_id, logger):
    opj = os.path.join
    logger.info(__name__ + ": start ...")

    block_id_str = "{:06d}".format(block_id)
    block_dir = opj(env["work_dir"], "blocks", block_id_str)
    sub_work_dir = opj(block_dir, __name__)

    if os.path.exists(sub_work_dir):
        logger.info(__name__ + ": already done. skip computation.")
        return

    os.makedirs(sub_work_dir)

    evttab = {}
    evttab = event_table.add_levels_from_path(
        evttab=evttab,
        path=opj(
            block_dir,
            "plenoirf.production.extract_features_from_light_field",
            "event_table.tar",
        ),
    )
    evttab = event_table.add_empty_level(evttab, "reconstructed_trajectory")

    evttab = estimate_primary_trajectory(
        evttab=evttab,
        fuzzy_config=blk["trajectory_reconstruction"]["fuzzy_config"],
        model_fit_config=blk["trajectory_reconstruction"]["model_fit_config"],
        reconstructed_cherenkov_path=opj(
            block_dir, "reconstructed_cherenkov.tar"
        ),
        light_field_geometry=blk["light_field_geometry"],
        logger=logger,
    )

    event_table.write_certain_levels_to_path(
        evttab=evttab,
        path=opj(sub_work_dir, "event_table.tar"),
        level_keys=["reconstructed_trajectory"],
    )

    logger.info(__name__ + ": ... done.")


def get_column_as_dict_by_index(table, level_key, column_key, index_key):
    level = table[level_key]
    out = {}
    for ii in range(level.shape[0]):
        out[level[index_key][ii]] = level[column_key][ii]
    return out


def estimate_primary_trajectory(
    evttab,
    fuzzy_config,
    model_fit_config,
    reconstructed_cherenkov_path,
    light_field_geometry,
    logger,
):
    shower_maximum_object_distance = get_column_as_dict_by_index(
        table=evttab,
        level_key="features",
        column_key="image_smallest_ellipse_object_distance",
        index_key="idx",
    )

    run = plenopy.photon_stream.loph.LopfTarReader(
        reconstructed_cherenkov_path
    )

    for event in run:
        uid, loph_record = event

        if uid in shower_maximum_object_distance:
            estimate, debug = gamrec.trajectory.v2020dec04iron0b.estimate(
                loph_record=loph_record,
                light_field_geometry=light_field_geometry,
                shower_maximum_object_distance=shower_maximum_object_distance[
                    uid
                ],
                fuzzy_config=fuzzy_config,
                model_fit_config=model_fit_config,
            )

            if gamrec.trajectory.v2020dec04iron0b.is_valid_estimate(
                estimate=estimate
            ):
                rec = {}
                rec[snt.IDX] = uid

                rec["cx_rad"] = estimate["primary_particle_cx"]
                rec["cy_rad"] = estimate["primary_particle_cy"]
                rec["x_m"] = estimate["primary_particle_x"]
                rec["y_m"] = estimate["primary_particle_y"]

                rec["fuzzy_cx_rad"] = debug["fuzzy_result"]["reco_cx"]
                rec["fuzzy_cy_rad"] = debug["fuzzy_result"]["reco_cy"]
                rec["fuzzy_main_axis_support_cx_rad"] = debug["fuzzy_result"][
                    "main_axis_support_cx"
                ]
                rec["fuzzy_main_axis_support_cy_rad"] = debug["fuzzy_result"][
                    "main_axis_support_cy"
                ]
                rec["fuzzy_main_axis_support_uncertainty_rad"] = debug[
                    "fuzzy_result"
                ]["main_axis_support_uncertainty"]
                rec["fuzzy_main_axis_azimuth_rad"] = debug["fuzzy_result"][
                    "main_axis_azimuth"
                ]
                rec["fuzzy_main_axis_azimuth_uncertainty_rad"] = debug[
                    "fuzzy_result"
                ]["main_axis_azimuth_uncertainty"]

                evttab["reconstructed_trajectory"].append_record(rec)

    return evttab
