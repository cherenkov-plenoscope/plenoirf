import numpy as np
import os
from os.path import join as opj

import gamma_ray_reconstruction as gamrec
import plenopy
import sparse_numeric_table as snt
import json_line_logger

from .. import event_table
from .. import utils


def run(env, part):
    name = __name__.split(".")[-1]
    module_work_dir = opj(env["work_dir"], part, name)

    if os.path.exists(module_work_dir):
        return

    os.makedirs(module_work_dir)
    logger = json_line_logger.LoggerFile(opj(module_work_dir, "log.jsonl"))
    logger.info(__name__)

    with json_line_logger.TimeDelta(
        logger, "init trajectory reconstruction config"
    ):
        trajectory_config = {}
        trajectory_config["fuzzy_config"] = (
            gamrec.trajectory.v2020nov12fuzzy0.config.compile_user_config(
                user_config=env["config"]["reconstruction"]["trajectory"][
                    "fuzzy_method"
                ]
            )
        )
        trajectory_config["model_fit_config"] = (
            gamrec.trajectory.v2020dec04iron0b.config.compile_user_config(
                user_config=env["config"]["reconstruction"]["trajectory"][
                    "core_axis_fit"
                ]
            )
        )

    evttab = snt.SparseNumericTable(index_key="uid")
    evttab = event_table.add_levels_from_path(
        evttab=evttab,
        path=opj(
            env["work_dir"],
            "cls2rec",
            "extract_features_from_light_field",
            "event_table.snt.zip",
        ),
    )
    additional_level_keys = ["reconstructed_trajectory"]
    for key in additional_level_keys:
        evttab = event_table.add_empty_level(evttab, key)

    evttab = estimate_primary_trajectory(
        evttab=evttab,
        fuzzy_config=trajectory_config["fuzzy_config"],
        model_fit_config=trajectory_config["model_fit_config"],
        reconstructed_cherenkov_path=opj(
            env["work_dir"],
            "cer2cls",
            "simulate_instrument_and_reconstruct_cherenkov",
            "reconstructed_cherenkov.loph.tar",
        ),
        light_field_geometry=env["light_field_geometry"],
        logger=logger,
    )

    event_table.write_certain_levels_to_path(
        evttab=evttab,
        path=opj(module_work_dir, "event_table.snt.zip"),
        level_keys=additional_level_keys,
    )

    logger.info("done.")
    json_line_logger.shutdown(logger=logger)
    utils.gzip_file(opj(module_work_dir, "log.jsonl"))


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
        index_key="uid",
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
                rec["uid"] = uid

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

                evttab["reconstructed_trajectory"].append(rec)

    return evttab
