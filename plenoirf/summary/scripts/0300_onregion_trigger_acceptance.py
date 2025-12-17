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

passing_trigger = res.read_passed_trigger(
    opj(res.paths["analysis_dir"], "0055_passing_trigger"),
    trigger_mode_key="far_accepting_focus_and_near_rejecting_focus",
)
passing_quality = json_utils.tree.Tree(
    opj(res.paths["analysis_dir"], "0056_passing_basic_quality")
)
passing_trajectory_quality = json_utils.tree.Tree(
    opj(res.paths["analysis_dir"], "0059_passing_trajectory_quality")
)

NUM_ONREGION_POLYGON_STEPS = 37
MAX_SOURCE_ANGLE_DEG = res.analysis["gamma_ray_source_direction"][
    "max_angle_relative_to_pointing_deg"
]
MAX_SOURCE_ANGLE_RAD = np.deg2rad(MAX_SOURCE_ANGLE_DEG)
SOLID_ANGLE_TO_CONTAIN_SOURCE_SR = solid_angle_utils.cone.solid_angle(
    half_angle_rad=MAX_SOURCE_ANGLE_RAD
)
POSSIBLE_ONREGION_POLYGON = irf.reconstruction.onregion.make_circular_polygon(
    radius=MAX_SOURCE_ANGLE_RAD, num_steps=NUM_ONREGION_POLYGON_STEPS
)

energy_bin = res.energy_binning(key="5_bins_per_decade")
zenith_bin = res.zenith_binning("3_bins_per_45deg")


def cut_candidates_for_detection(
    event_table,
    uid_trajectory_quality,
    uid_trigger,
    uid_quality,
):
    uid_self = event_table["primary"]["uid"]

    uid_candidates = snt.logic.intersection(
        uid_self,
        uid_trigger,
        uid_quality,
        uid_trajectory_quality,
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


# make out dirs
for zd in range(zenith_bin["num"]):
    zk = f"zd{zd:d}"
    for ok in ONREGION_TYPES:
        for pk in res.PARTICLES:
            os.makedirs(opj(res.paths["out_dir"], zk, ok, pk), exist_ok=True)


# point source
# ------------
for pk in res.PARTICLES:
    for zdbin in range(zenith_bin["num"]):
        zk = f"zd{zdbin:d}"

        effective_area = {}
        for ok in ONREGION_TYPES:
            effective_area[ok] = {
                "mean": np.zeros(energy_bin["num"]),
                "absolute_uncertainty": np.zeros(energy_bin["num"]),
            }

        for enbin in range(energy_bin["num"]):
            print(
                "point",
                pk,
                f"zd: {zdbin + 1:d}/{zenith_bin['num']:d}, "
                f"en: {enbin + 1:d}/{energy_bin['num']:d}",
            )

            diffuse_thrown = res.event_table(particle_key=pk).query(
                levels_and_columns={
                    "primary": (
                        "uid",
                        "energy_GeV",
                        "azimuth_rad",
                        "zenith_rad",
                        "solid_angle_thrown_sr",
                    ),
                    "instrument_pointing": (
                        "uid",
                        "azimuth_rad",
                        "zenith_rad",
                    ),
                    "reconstructed_trajectory": (
                        "uid",
                        "x_m",
                        "y_m",
                        "cx_rad",
                        "cy_rad",
                        "fuzzy_main_axis_azimuth_rad",
                    ),
                    "features": (
                        "uid",
                        "num_photons",
                        "image_half_depth_shift_cx",
                        "image_half_depth_shift_cy",
                    ),
                    "groundgrid": (
                        "uid",
                        "num_bins_thrown",
                        "num_bins_above_threshold",
                        "area_thrown_m2",
                    ),
                    "groundgrid_choice": ("uid", "core_x_m", "core_y_m"),
                },
                energy_start_GeV=energy_bin["edges"][enbin],
                energy_stop_GeV=energy_bin["edges"][enbin + 1],
                zenith_start_rad=zenith_bin["edges"][zdbin],
                zenith_stop_rad=zenith_bin["edges"][zdbin + 1],
            )

            uid_possible_onregion = irf.analysis.cuts.cut_primary_direction_within_angle(
                event_table=diffuse_thrown,
                max_angle_between_primary_and_pointing_rad=MAX_SOURCE_ANGLE_RAD,
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

            poicanarr = (
                irf.reconstruction.trajectory_quality.make_rectangular_table(
                    event_table=point_candidate,
                    instrument_pointing_model=res.config["pointing"]["model"],
                )
            )

            area_scatter_m2 = point_thrown["groundgrid"][
                "area_thrown_m2"
            ] * np.cos(point_thrown["primary"]["zenith_rad"])

            for ok in ONREGION_TYPES:
                onregion_config = copy.deepcopy(ONREGION_TYPES[ok])
                idx_dict_source_in_onregion = {}
                for ii in range(poicanarr["uid"].shape[0]):
                    _onregion = irf.reconstruction.onregion.estimate_onregion(
                        reco_cx=poicanarr["reconstructed_trajectory/cx_rad"][
                            ii
                        ],
                        reco_cy=poicanarr["reconstructed_trajectory/cy_rad"][
                            ii
                        ],
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
                    effective_area[ok]["mean"][enbin],
                    effective_area[ok]["absolute_uncertainty"][enbin],
                ) = atmospheric_cherenkov_response.analysis.effective_quantity_for_grid(
                    mask_detected=mask_detected,
                    quantity_scatter=area_scatter_m2,
                    num_grid_cells_above_lose_threshold=point_thrown[
                        "groundgrid"
                    ]["num_bins_above_threshold"],
                    total_num_grid_cells=point_thrown["groundgrid"][
                        "num_bins_thrown"
                    ],
                )

        for ok in ONREGION_TYPES:
            json_utils.write(
                opj(res.paths["out_dir"], zk, ok, pk, "point.json"),
                {
                    "comment": (
                        "Effective area "
                        "for a point source, reconstructed in onregion. "
                        "VS energy"
                    ),
                    "zenith_key": zk,
                    "particle_key": pk,
                    "onregion_key": ok,
                    "unit": "m$^{2}$",
                    "mean": effective_area[ok]["mean"],
                    "absolute_uncertainty": effective_area[ok][
                        "absolute_uncertainty"
                    ],
                },
            )


# diffuse source
# ---------------
for pk in res.PARTICLES:
    for zdbin in range(zenith_bin["num"]):
        zk = f"zd{zdbin:d}"

        effective_etendue = {}
        for ok in ONREGION_TYPES:
            effective_etendue[ok] = {
                "mean": np.zeros(energy_bin["num"]),
                "absolute_uncertainty": np.zeros(energy_bin["num"]),
            }

        for enbin in range(energy_bin["num"]):
            print(
                "diffuse",
                pk,
                f"zd: {zdbin + 1:d}/{zenith_bin['num']:d}, "
                f"en: {enbin + 1:d}/{energy_bin['num']:d}",
            )

            diffuse_thrown = res.event_table(particle_key=pk).query(
                levels_and_columns={
                    "primary": (
                        "uid",
                        "energy_GeV",
                        "azimuth_rad",
                        "zenith_rad",
                        "solid_angle_thrown_sr",
                    ),
                    "instrument_pointing": (
                        "uid",
                        "azimuth_rad",
                        "zenith_rad",
                    ),
                    "reconstructed_trajectory": (
                        "uid",
                        "x_m",
                        "y_m",
                        "cx_rad",
                        "cy_rad",
                        "fuzzy_main_axis_azimuth_rad",
                    ),
                    "features": (
                        "uid",
                        "num_photons",
                        "image_half_depth_shift_cx",
                        "image_half_depth_shift_cy",
                    ),
                    "groundgrid": (
                        "uid",
                        "num_bins_thrown",
                        "num_bins_above_threshold",
                        "area_thrown_m2",
                    ),
                    "groundgrid_choice": ("uid", "core_x_m", "core_y_m"),
                },
                energy_start_GeV=energy_bin["edges"][enbin],
                energy_stop_GeV=energy_bin["edges"][enbin + 1],
                zenith_start_rad=zenith_bin["edges"][zdbin],
                zenith_stop_rad=zenith_bin["edges"][zdbin + 1],
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

            difcanarr = (
                irf.reconstruction.trajectory_quality.make_rectangular_table(
                    event_table=diffuse_candidate,
                    instrument_pointing_model=res.config["pointing"]["model"],
                )
            )

            etendue_scatter_m2_sr = (
                diffuse_thrown["groundgrid"]["area_thrown_m2"]
                * diffuse_thrown["primary"]["solid_angle_thrown_sr"]
                * np.cos(diffuse_thrown["primary"]["zenith_rad"])
            )

            for ok in ONREGION_TYPES:
                print(f"{zk:s}, {pk:s}, {ok:s}, diffuse")

                onregion_config = copy.deepcopy(ONREGION_TYPES[ok])

                idx_dict_probability_for_source_in_onregion = {}
                for ii in range(difcanarr["uid"].shape[0]):
                    _onregion = irf.reconstruction.onregion.estimate_onregion(
                        reco_cx=difcanarr["reconstructed_trajectory/cx_rad"][
                            ii
                        ],
                        reco_cy=difcanarr["reconstructed_trajectory/cy_rad"][
                            ii
                        ],
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

                    onregion_polygon = (
                        irf.reconstruction.onregion.make_polygon(
                            onregion=_onregion,
                            num_steps=NUM_ONREGION_POLYGON_STEPS,
                        )
                    )

                    overlap_srad = irf.reconstruction.onregion.intersecting_area_of_polygons(
                        a=onregion_polygon, b=POSSIBLE_ONREGION_POLYGON
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
                    effective_etendue[ok]["mean"][enbin],
                    effective_etendue[ok]["absolute_uncertainty"][enbin],
                ) = atmospheric_cherenkov_response.analysis.effective_quantity_for_grid(
                    mask_detected=mask_probability_for_source_in_onregion,
                    quantity_scatter=etendue_scatter_m2_sr,
                    num_grid_cells_above_lose_threshold=diffuse_thrown[
                        "groundgrid"
                    ]["num_bins_above_threshold"],
                    total_num_grid_cells=diffuse_thrown["groundgrid"][
                        "num_bins_thrown"
                    ],
                )

        for ok in ONREGION_TYPES:
            json_utils.write(
                opj(res.paths["out_dir"], zk, ok, pk, "diffuse.json"),
                {
                    "comment": (
                        "Effective acceptance (area x solid angle) "
                        "for a diffuse source, reconstructed in onregion. "
                        "VS energy"
                    ),
                    "zenith_key": zk,
                    "particle_key": pk,
                    "onregion_key": ok,
                    "unit": "m$^{2}$ sr",
                    "mean": effective_etendue[ok]["mean"],
                    "absolute_uncertainty": effective_etendue[ok][
                        "absolute_uncertainty"
                    ],
                },
            )

res.stop()
