#!/usr/bin/python
import sys
import copy
import numpy as np
import plenoirf as irf
import atmospheric_cherenkov_response
import solid_angle_utils
import sparse_numeric_table as snt
import os
from os.path import join as opj
import json_utils

res = irf.summary.ScriptResources.from_argv(sys.argv)
res.start()

ONREGION_TYPES = res.analysis["on_off_measuremnent"]["onregion_types"]

passing_trigger = json_utils.tree.read(
    opj(res.paths["analysis_dir"], "0055_passing_trigger")
)
passing_quality = json_utils.tree.read(
    opj(res.paths["analysis_dir"], "0056_passing_basic_quality")
)
passing_trajectory_quality = json_utils.tree.read(
    opj(res.paths["analysis_dir"], "0059_passing_trajectory_quality")
)

MAX_SOURCE_ANGLE_DEG = res.analysis["gamma_ray_source_direction"][
    "max_angle_relative_to_pointing_deg"
]
MAX_SOURCE_ANGLE_RAD = np.deg2rad(MAX_SOURCE_ANGLE_DEG)
SOLID_ANGLE_TO_CONTAIN_SOURCE_SR = solid_angle_utils.cone.solid_angle(
    half_angle_rad=MAX_SOURCE_ANGLE_RAD
)
POSSIBLE_ONREGION_POLYGON = irf.reconstruction.onregion.make_circular_polygon(
    radius=MAX_SOURCE_ANGLE_RAD, num_steps=37
)
pointing_azimuth_deg = None
pointing_zenith_deg = None

energy_bin = res.energy_binning(key="trigger_acceptance_onregion")


def cut_candidates_for_detection(
    event_table,
    uid_trajectory_quality,
    uid_trigger,
    uid_quality,
):
    uid_self = event_table["primary"]["uid"]

    uid_candidates = snt.logic.intersection(
        [uid_self, uid_trigger, uid_quality, uid_trajectory_quality]
    )

    return snt.logic.cut_and_sort_table_on_indices(
        table=event_table,
        common_indices=uid_candidates,
    )


def make_wighted_mask_wrt_primary_table(
    primary_table, idx_dict_of_weights, default_weight=0.0
):
    num_primaries = primary_table["uid"].shape[0]
    mask = np.zeros(num_primaries)

    for ii in range(num_primaries):
        idx = primary_table["uid"][ii]
        if idx in idx_dict_of_weights:
            mask[ii] = idx_dict_of_weights[idx]
        else:
            mask[ii] = default_weight
    return mask


for ok in ONREGION_TYPES:
    for pk in res.PARTICLES:
        os.makedirs(opj(res.paths["out_dir"], ok, pk), exist_ok=True)


for pk in res.PARTICLES:

    with res.open_event_table(particle_key=pk) as arc:
        diffuse_thrown = arc.query(
            levels_and_columns={
                "primary": "__all__",
                "instrument_pointing": "__all__",
                "groundgrid_choice": "__all__",
                "reconstructed_trajectory": "__all__",
                "features": "__all__",
                "groundgrid": "__all__",
            }
        )

    uid_possible_onregion = (
        irf.analysis.cuts.cut_primary_direction_within_angle(
            event_table=diffuse_thrown,
            max_angle_between_primary_and_pointing_rad=MAX_SOURCE_ANGLE_RAD,
        )
    )

    # point source
    # -------------

    # thrown
    point_thrown = snt.logic.cut_table_on_indices(
        table=diffuse_thrown,
        common_indices=uid_possible_onregion,
    )

    # detected
    point_candidate = cut_candidates_for_detection(
        event_table=point_thrown,
        uid_trajectory_quality=passing_trajectory_quality[pk]["uid"],
        uid_trigger=passing_trigger[pk]["uid"],
        uid_quality=passing_quality[pk]["uid"],
    )

    poicanarr = irf.reconstruction.trajectory_quality.make_rectangular_table(
        event_table=point_candidate,
        instrument_pointing_model=res.config["pointing"]["model"],
    )

    for ok in ONREGION_TYPES:
        onregion_config = copy.deepcopy(ONREGION_TYPES[ok])
        idx_dict_source_in_onregion = {}
        for ii in range(poicanarr["uid"].shape[0]):
            _onregion = irf.reconstruction.onregion.estimate_onregion(
                reco_cx=poicanarr["reconstructed_trajectory/cx_rad"][ii],
                reco_cy=poicanarr["reconstructed_trajectory/cy_rad"][ii],
                reco_main_axis_azimuth=poicanarr[
                    "reconstructed_trajectory/fuzzy_main_axis_azimuth_rad"
                ][ii],
                reco_num_photons=poicanarr["features/num_photons"][ii],
                reco_core_radius=np.hypot(
                    poicanarr["reconstructed_trajectory/x_m"][ii],
                    poicanarr["reconstructed_trajectory/y_m"][ii],
                ),
                config=onregion_config,
            )

            hit = irf.reconstruction.onregion.is_direction_inside(
                cx=poicanarr["true_trajectory/cx_rad"][ii],
                cy=poicanarr["true_trajectory/cy_rad"][ii],
                onregion=_onregion,
            )

            idx_dict_source_in_onregion[poicanarr["uid"][ii]] = hit

        mask_detected = make_wighted_mask_wrt_primary_table(
            primary_table=point_thrown["primary"],
            idx_dict_of_weights=idx_dict_source_in_onregion,
        )

        (
            Qeff,
            Qeff_au,
        ) = atmospheric_cherenkov_response.analysis.effective_quantity_for_grid(
            energy_bin_edges_GeV=energy_bin["edges"],
            energy_GeV=point_thrown["primary"]["energy_GeV"],
            mask_detected=mask_detected,
            quantity_scatter=point_thrown["groundgrid"]["area_thrown_m2"],
            num_grid_cells_above_lose_threshold=point_thrown["groundgrid"][
                "num_bins_above_threshold"
            ],
            total_num_grid_cells=point_thrown["groundgrid"]["num_bins_thrown"],
        )

        json_utils.write(
            opj(res.paths["out_dir"], ok, pk, "point.json"),
            {
                "comment": (
                    "Effective area "
                    "for a point source, reconstructed in onregion. "
                    "VS energy"
                ),
                "unit": "m$^{2}$",
                "mean": Qeff,
                "absolute_uncertainty": Qeff_au,
            },
        )

    # diffuse source
    # --------------

    # thrown
    diffuse_thrown = diffuse_thrown

    # detected
    diffuse_candidate = cut_candidates_for_detection(
        event_table=diffuse_thrown,
        uid_trajectory_quality=passing_trajectory_quality[pk]["uid"],
        uid_trigger=passing_trigger[pk]["uid"],
        uid_quality=passing_quality[pk]["uid"],
    )

    difcanarr = irf.reconstruction.trajectory_quality.make_rectangular_table(
        event_table=diffuse_candidate,
        instrument_pointing_model=res.config["pointing"]["model"],
    )

    for ok in ONREGION_TYPES:
        onregion_config = copy.deepcopy(ONREGION_TYPES[ok])

        idx_dict_probability_for_source_in_onregion = {}
        for ii in range(difcanarr["uid"].shape[0]):
            _onregion = irf.reconstruction.onregion.estimate_onregion(
                reco_cx=difcanarr["reconstructed_trajectory/cx_rad"][ii],
                reco_cy=difcanarr["reconstructed_trajectory/cy_rad"][ii],
                reco_main_axis_azimuth=difcanarr[
                    "reconstructed_trajectory/fuzzy_main_axis_azimuth_rad"
                ][ii],
                reco_num_photons=difcanarr["features/num_photons"][ii],
                reco_core_radius=np.hypot(
                    difcanarr["reconstructed_trajectory/x_m"][ii],
                    difcanarr["reconstructed_trajectory/y_m"][ii],
                ),
                config=onregion_config,
            )

            onregion_polygon = irf.reconstruction.onregion.make_polygon(
                onregion=_onregion, num_steps=37
            )

            overlap_srad = (
                irf.reconstruction.onregion.intersecting_area_of_polygons(
                    a=onregion_polygon, b=POSSIBLE_ONREGION_POLYGON
                )
            )

            probability_to_contain_random_source = (
                overlap_srad / SOLID_ANGLE_TO_CONTAIN_SOURCE_SR
            )

            idx_dict_probability_for_source_in_onregion[
                difcanarr["uid"][ii]
            ] = probability_to_contain_random_source

        mask_probability_for_source_in_onregion = make_wighted_mask_wrt_primary_table(
            primary_table=diffuse_thrown["primary"],
            idx_dict_of_weights=idx_dict_probability_for_source_in_onregion,
        )

        (
            Qeff,
            Qeff_au,
        ) = atmospheric_cherenkov_response.analysis.effective_quantity_for_grid(
            energy_bin_edges_GeV=energy_bin["edges"],
            energy_GeV=diffuse_thrown["primary"]["energy_GeV"],
            mask_detected=mask_probability_for_source_in_onregion,
            quantity_scatter=(
                diffuse_thrown["groundgrid"]["area_thrown_m2"]
                * diffuse_thrown["primary"]["solid_angle_thrown_sr"]
            ),
            num_grid_cells_above_lose_threshold=diffuse_thrown["groundgrid"][
                "num_bins_above_threshold"
            ],
            total_num_grid_cells=diffuse_thrown["groundgrid"][
                "num_bins_thrown"
            ],
        )

        json_utils.write(
            opj(res.paths["out_dir"], ok, pk, "diffuse.json"),
            {
                "comment": (
                    "Effective acceptance (area x solid angle) "
                    "for a diffuse source, reconstructed in onregion. "
                    "VS energy"
                ),
                "unit": "m$^{2}$ sr",
                "mean": Qeff,
                "absolute_uncertainty": Qeff_au,
            },
        )

res.stop()
