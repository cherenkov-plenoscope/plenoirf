#!/usr/bin/python
import sys
import numpy as np
import plenoirf as irf
import atmospheric_cherenkov_response
import sparse_numeric_table as snt
import os
from os.path import join as opj
import json_utils

res = irf.summary.ScriptResources.from_argv(sys.argv)
res.start()

MAX_SOURCE_ANGLE_RAD = np.deg2rad(
    res.analysis["gamma_ray_source_direction"][
        "max_angle_relative_to_pointing_deg"
    ]
)

energy_bin = res.energy_binning(key="10_bins_per_decade")
zenith_bin = res.zenith_binning(key="3_bins_per_45deg")

passing_trigger = res.read_passed_trigger(
    opj(res.paths["analysis_dir"], "0055_passing_trigger"),
    trigger_mode_key="far_accepting_focus_and_near_rejecting_focus",
)
trigger = res.trigger

SHAPE_THRESHOLDS_ENERGIES = (
    len(trigger["ratescan_thresholds_pe"]),
    energy_bin["num"],
)

# make out dirs
for zd in range(zenith_bin["num"]):
    zk = f"zd{zd:d}"
    for pk in res.PARTICLES:
        os.makedirs(opj(res.paths["out_dir"], zk, pk), exist_ok=True)


for pk in res.PARTICLES:
    for zdbin in range(zenith_bin["num"]):
        zk = f"zd{zdbin:d}"

        effective_area_point = {
            "Aeff": np.zeros(SHAPE_THRESHOLDS_ENERGIES),
            "Aeff_au": np.zeros(SHAPE_THRESHOLDS_ENERGIES),
        }

        effective_etendue_diffuse = {
            "Qeff": np.zeros(SHAPE_THRESHOLDS_ENERGIES),
            "Qeff_au": np.zeros(SHAPE_THRESHOLDS_ENERGIES),
        }

        for enbin in range(energy_bin["num"]):
            print(
                pk,
                f"zd: {zdbin:d}/{zenith_bin['num']:d}, en: {enbin:d}/{energy_bin['num']:d}",
            )
            _uid = res.event_table(particle_key=pk).query(
                levels_and_columns={
                    "primary": ("uid",),
                    "instrument_pointing": ("uid",),
                    "groundgrid": ("uid",),
                    "trigger": ("uid",),
                },
                energy_bin_indices=[enbin],
                zenith_bin_indices=[zdbin],
            )
            uid_common = snt.logic.intersection(
                _uid["primary"]["uid"],
                _uid["groundgrid"]["uid"],
                _uid["trigger"]["uid"],
            )
            del _uid

            diffuse_particle_table = res.event_table(particle_key=pk).query(
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
                    "groundgrid": (
                        "uid",
                        "area_thrown_m2",
                        "num_bins_thrown",
                        "num_bins_above_threshold",
                    ),
                    "trigger": ("uid",),
                },
                indices=uid_common,
                energy_bin_indices=[enbin],
                zenith_bin_indices=[zdbin],
                sort=True,
            )
            del uid_common

            # point source
            # ------------
            uid_possible_onregion = irf.analysis.cuts.cut_primary_direction_within_angle(
                event_table=diffuse_particle_table,
                max_angle_between_primary_and_pointing_rad=MAX_SOURCE_ANGLE_RAD,
            )

            point_particle_table = snt.logic.cut_table_on_indices(
                table=diffuse_particle_table,
                common_indices=uid_possible_onregion,
            )

            point_quantity_scatter = point_particle_table["groundgrid"][
                "area_thrown_m2"
            ] * np.cos(point_particle_table["primary"]["zenith_rad"])

            point_num_grid_cells_above_lose_threshold = point_particle_table[
                "groundgrid"
            ]["num_bins_above_threshold"]

            point_total_num_grid_cells = point_particle_table["groundgrid"][
                "num_bins_thrown"
            ]

            # diffuse source
            # --------------
            diffuse_quantity_scatter = (
                diffuse_particle_table["groundgrid"]["area_thrown_m2"]
                * np.cos(diffuse_particle_table["primary"]["zenith_rad"])
                * diffuse_particle_table["primary"]["solid_angle_thrown_sr"]
            )
            diffuse_num_grid_cells_above_lose_threshold = (
                diffuse_particle_table["groundgrid"][
                    "num_bins_above_threshold"
                ]
            )

            diffuse_total_num_grid_cells = diffuse_particle_table[
                "groundgrid"
            ]["num_bins_thrown"]

            for ithresh in range(SHAPE_THRESHOLDS_ENERGIES[0]):
                threshold_pe = trigger["ratescan_thresholds_pe"][ithresh]

                uid_detected = passing_trigger[pk].ratescan(
                    threshold_pe=threshold_pe,
                    zenith_bin_indices=[zdbin],
                    energy_bin_indices=[enbin],
                )

                # point
                # -----
                point_mask_detected = snt.logic.make_mask_of_right_in_left(
                    left_indices=point_particle_table["primary"]["uid"],
                    right_indices=uid_detected,
                )
                (
                    effective_area_point["Aeff"][ithresh][enbin],
                    effective_area_point["Aeff_au"][ithresh][enbin],
                ) = atmospheric_cherenkov_response.analysis.effective_quantity_for_grid(
                    mask_detected=point_mask_detected,
                    quantity_scatter=point_quantity_scatter,
                    num_grid_cells_above_lose_threshold=point_num_grid_cells_above_lose_threshold,
                    total_num_grid_cells=point_total_num_grid_cells,
                )

                # diffuse
                # -------
                diffuse_mask_detected = snt.logic.make_mask_of_right_in_left(
                    left_indices=diffuse_particle_table["primary"]["uid"],
                    right_indices=uid_detected,
                )
                (
                    effective_etendue_diffuse["Qeff"][ithresh][enbin],
                    effective_etendue_diffuse["Qeff_au"][ithresh][enbin],
                ) = atmospheric_cherenkov_response.analysis.effective_quantity_for_grid(
                    mask_detected=diffuse_mask_detected,
                    quantity_scatter=diffuse_quantity_scatter,
                    num_grid_cells_above_lose_threshold=diffuse_num_grid_cells_above_lose_threshold,
                    total_num_grid_cells=diffuse_total_num_grid_cells,
                )

        json_utils.write(
            opj(res.paths["out_dir"], zk, pk, "point.json"),
            {
                "comment": (
                    "Effective area for a point source. "
                    "VS trigger-ratescan-thresholds VS energy-bins"
                ),
                "zenith_key": zk,
                "energy_bin_edges_GeV": energy_bin["edges"],
                "unit": "m$^{2}$",
                "mean": effective_area_point["Aeff"],
                "absolute_uncertainty": effective_area_point["Aeff_au"],
            },
        )

        json_utils.write(
            opj(res.paths["out_dir"], zk, pk, "diffuse.json"),
            {
                "comment": (
                    "Effective acceptance (area x solid angle) "
                    "for a diffuse source. "
                    "VS trigger-ratescan-thresholds VS energy-bins"
                ),
                "zenith_key": zk,
                "unit": "m$^{2}$ sr",
                "mean": effective_etendue_diffuse["Qeff"],
                "absolute_uncertainty": effective_etendue_diffuse["Qeff_au"],
            },
        )

res.stop()
