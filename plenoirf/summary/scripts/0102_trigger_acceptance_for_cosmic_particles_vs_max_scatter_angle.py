#!/usr/bin/python
import sys
import numpy as np
import plenoirf as irf
import atmospheric_cherenkov_response
import sparse_numeric_table as snt
import os
from os.path import join as opj
import json_utils
import magnetic_deflection as mdfl
import spherical_coordinates
import solid_angle_utils

res = irf.summary.ScriptResources.from_argv(sys.argv)
res.start()

energy_bin = res.energy_binning(key="5_bins_per_decade")
zenith_bin = res.zenith_binning("3_bins_per_45deg")

passing_trigger = res.read_passed_trigger(
    opj(res.paths["analysis_dir"], "0055_passing_trigger"),
    trigger_mode_key="far_accepting_focus_and_near_rejecting_focus",
)

# make out dirs
for zd in range(zenith_bin["num"]):
    zk = f"zd{zd:d}"
    for pk in res.PARTICLES:
        os.makedirs(opj(res.paths["out_dir"], zk, pk), exist_ok=True)


for pk in res.PARTICLES:
    scatter_bin = res.scatter_binning(particle_key=pk)
    SHAPE_SOLIDANGLES_ENERGIES = (scatter_bin["num"], energy_bin["num"])

    for zdbin in range(zenith_bin["num"]):
        zk = f"zd{zdbin:d}"

        effective_etendue_diffuse = {
            "Qeff": np.zeros(SHAPE_SOLIDANGLES_ENERGIES),
            "Qeff_au": np.zeros(SHAPE_SOLIDANGLES_ENERGIES),
        }

        for enbin in range(energy_bin["num"]):
            print(
                pk,
                f"zd: {zdbin + 1:d}/{zenith_bin['num']:d}, "
                f"en: {enbin + 1:d}/{energy_bin['num']:d}",
            )

            _uid = res.event_table(particle_key=pk).query(
                levels_and_columns={
                    "primary": ("uid",),
                    "instrument_pointing": ("uid",),
                    "groundgrid": ("uid",),
                    "trigger": ("uid",),
                },
                energy_start_GeV=energy_bin["edges"][enbin],
                energy_stop_GeV=energy_bin["edges"][enbin + 1],
                zenith_start_rad=zenith_bin["edges"][zdbin],
                zenith_stop_rad=zenith_bin["edges"][zdbin + 1],
            )
            uid_common = snt.logic.intersection(
                _uid["primary"]["uid"],
                _uid["groundgrid"]["uid"],
                _uid["trigger"]["uid"],
            )
            del _uid

            shower_table = res.event_table(particle_key=pk).query(
                levels_and_columns={
                    "primary": [
                        "uid",
                        "energy_GeV",
                        "azimuth_rad",
                        "zenith_rad",
                    ],
                    "instrument_pointing": [
                        "uid",
                        "azimuth_rad",
                        "zenith_rad",
                    ],
                    "groundgrid": [
                        "uid",
                        "num_bins_thrown",
                        "area_thrown_m2",
                        "num_bins_above_threshold",
                    ],
                },
                indices=uid_common,
                energy_start_GeV=energy_bin["edges"][enbin],
                energy_stop_GeV=energy_bin["edges"][enbin + 1],
                zenith_start_rad=zenith_bin["edges"][zdbin],
                zenith_stop_rad=zenith_bin["edges"][zdbin + 1],
                sort=True,
            )
            del uid_common

            passing_trigger_uid = passing_trigger[pk].uid(
                energy_bin_indices=[enbin],
                zenith_bin_indices=[zdbin],
            )

            shower_table_scatter_angle_deg = np.rad2deg(
                spherical_coordinates.angle_between_az_zd(
                    azimuth1_rad=shower_table["primary"]["azimuth_rad"],
                    zenith1_rad=shower_table["primary"]["zenith_rad"],
                    azimuth2_rad=shower_table["instrument_pointing"][
                        "azimuth_rad"
                    ],
                    zenith2_rad=shower_table["instrument_pointing"][
                        "zenith_rad"
                    ],
                )
            )

            for iscatter in range(scatter_bin["num"]):
                scatter_cone_solid_angle_sr = scatter_bin["edges"][
                    iscatter + 1
                ]
                max_scatter_angle_rad = solid_angle_utils.cone.half_angle(
                    solid_angle_sr=scatter_cone_solid_angle_sr
                )
                max_scatter_angle_deg = np.rad2deg(max_scatter_angle_rad)

                print(
                    pk,
                    zk,
                    "max. scatter cone opening angle {:.3}deg".format(
                        max_scatter_angle_deg
                    ),
                )

                # cut subset of showers wich are within max scatter angle
                # -------------------------------------------------------
                mask_shower_within_max_scatter = (
                    shower_table_scatter_angle_deg <= max_scatter_angle_deg
                )
                uid_showers_within_max_scatter = shower_table["primary"][
                    "uid"
                ][mask_shower_within_max_scatter]

                S_shower_table = shower_table.query(
                    levels_and_columns={
                        "primary": ["uid", "energy_GeV"],
                        "groundgrid": "__all__",
                    },
                    indices=uid_showers_within_max_scatter,
                )

                S_shower_table = snt.logic.cut_and_sort_table_on_indices(
                    table=S_shower_table,
                    common_indices=uid_showers_within_max_scatter,
                    inplace=True,
                )

                S_mask_shower_detected = snt.logic.make_mask_of_right_in_left(
                    left_indices=S_shower_table["primary"]["uid"],
                    right_indices=passing_trigger_uid,
                )

                S_quantity_scatter = (
                    S_shower_table["groundgrid"]["area_thrown_m2"]
                    * scatter_cone_solid_angle_sr
                )

                S_num_grid_cells_above_lose_threshold = S_shower_table[
                    "groundgrid"
                ]["num_bins_above_threshold"]

                S_total_num_grid_cells = S_shower_table["groundgrid"][
                    "num_bins_thrown"
                ]

                (
                    effective_etendue_diffuse["Qeff"][iscatter][enbin],
                    effective_etendue_diffuse["Qeff_au"][iscatter][enbin],
                ) = atmospheric_cherenkov_response.analysis.effective_quantity_for_grid(
                    mask_detected=S_mask_shower_detected,
                    quantity_scatter=S_quantity_scatter,
                    num_grid_cells_above_lose_threshold=(
                        S_num_grid_cells_above_lose_threshold
                    ),
                    total_num_grid_cells=S_total_num_grid_cells,
                )

        json_utils.write(
            opj(res.paths["out_dir"], zk, pk, "diffuse.json"),
            {
                "comment": (
                    "Effective acceptance (area x solid angle) "
                    "for a diffuse source. "
                    "VS max. scatter-angle VS energy-bins"
                ),
                "unit": "m$^{2}$ sr",
                "mean": effective_etendue_diffuse["Qeff"],
                "absolute_uncertainty": effective_etendue_diffuse["Qeff_au"],
            },
        )

"""
for zd in range(zenith_bin["num"]):
    zk = f"zd{zd:d}"
    print(zk)

    zd_dir = opj(res.paths["out_dir"], zk)
    os.makedirs(zd_dir, exist_ok=True)

    for pk in res.PARTICLES:
        pk_dir = opj(zd_dir, pk)
        os.makedirs(pk_dir, exist_ok=True)

        scatter_bin = res.scatter_binning(particle_key=pk)

        with res.open_event_table(particle_key=pk) as arc:
            _shower_table = arc.query(
                levels_and_columns={
                    "primary": ["uid"],
                    "instrument_pointing": ["uid"],
                    "groundgrid": ["uid"],
                },
                indices=zenith_assignment[zk][pk],
            )

            uid_common = snt.logic.intersection(
                zenith_assignment[zk][pk],
                _shower_table["primary"]["uid"],
                _shower_table["instrument_pointing"]["uid"],
                _shower_table["groundgrid"]["uid"],
            )

            _shower_table = arc.query(
                levels_and_columns={
                    "primary": [
                        "uid",
                        "energy_GeV",
                        "azimuth_rad",
                        "zenith_rad",
                    ],
                    "instrument_pointing": [
                        "uid",
                        "azimuth_rad",
                        "zenith_rad",
                    ],
                    "groundgrid": [
                        "uid",
                        "num_bins_thrown",
                        "area_thrown_m2",
                        "num_bins_above_threshold",
                    ],
                },
                indices=uid_common,
            )

        shower_table = snt.logic.cut_and_sort_table_on_indices(
            table=_shower_table,
            common_indices=uid_common,
            inplace=True,
        )

        # diffuse source
        # --------------
        # num_grid_cells_above_lose_threshold = shower_table["grid"][
        #    "num_bins_above_threshold"
        #]
        #total_num_grid_cells = shower_table["grid"]["num_bins_thrown"]
        #idx_detected = passing_trigger[pk]["uid"]
        #
        #mask_shower_passed_trigger = snt.logic.make_mask_of_right_in_left(
        #    left_indices=shower_table["primary"]["uid"],
        #    right_indices=idx_detected,
        #)

        shower_table_scatter_angle_deg = np.rad2deg(
            spherical_coordinates.angle_between_az_zd(
                azimuth1_rad=shower_table["primary"]["azimuth_rad"],
                zenith1_rad=shower_table["primary"]["zenith_rad"],
                azimuth2_rad=shower_table["instrument_pointing"][
                    "azimuth_rad"
                ],
                zenith2_rad=shower_table["instrument_pointing"]["zenith_rad"],
            )
        )

        Q = []
        Q_au = []
        for ci in range(scatter_bin["num"]):
            scatter_cone_solid_angle_sr = scatter_bin["edges"][ci + 1]
            max_scatter_angle_rad = solid_angle_utils.cone.half_angle(
                solid_angle_sr=scatter_cone_solid_angle_sr
            )
            max_scatter_angle_deg = np.rad2deg(max_scatter_angle_rad)

            print(
                pk,
                zk,
                "max. scatter cone opening angle {:.3}deg".format(
                    max_scatter_angle_deg
                ),
            )

            # cut subset of showers wich are within max scatter angle
            # -------------------------------------------------------
            mask_shower_within_max_scatter = (
                shower_table_scatter_angle_deg <= max_scatter_angle_deg
            )
            uid_showers_within_max_scatter = shower_table["primary"]["uid"][
                mask_shower_within_max_scatter
            ]

            S_shower_table = shower_table.query(
                levels_and_columns={
                    "primary": ["uid", "energy_GeV"],
                    "groundgrid": "__all__",
                },
                indices=uid_showers_within_max_scatter,
            )

            S_shower_table = snt.logic.cut_and_sort_table_on_indices(
                table=S_shower_table,
                common_indices=uid_showers_within_max_scatter,
                inplace=True,
            )

            S_mask_shower_detected = snt.logic.make_mask_of_right_in_left(
                left_indices=S_shower_table["primary"]["uid"],
                right_indices=passing_trigger[pk]["uid"],
            )

            S_quantity_scatter = (
                S_shower_table["groundgrid"]["area_thrown_m2"]
                * scatter_cone_solid_angle_sr
            )

            S_num_grid_cells_above_lose_threshold = S_shower_table[
                "groundgrid"
            ]["num_bins_above_threshold"]

            S_total_num_grid_cells = S_shower_table["groundgrid"][
                "num_bins_thrown"
            ]

            (
                S_Q,
                S_Q_au,
            ) = atmospheric_cherenkov_response.analysis.effective_quantity_for_grid_vs_energy(
                energy_bin_edges_GeV=energy_bin["edges"],
                energy_GeV=S_shower_table["primary"]["energy_GeV"],
                mask_detected=S_mask_shower_detected,
                quantity_scatter=S_quantity_scatter,
                num_grid_cells_above_lose_threshold=(
                    S_num_grid_cells_above_lose_threshold
                ),
                total_num_grid_cells=S_total_num_grid_cells,
            )
            Q.append(S_Q)
            Q_au.append(S_Q_au)

        json_utils.write(
            opj(pk_dir, "diffuse.json"),
            {
                "comment": (
                    "Effective acceptance (area x solid angle) "
                    "for a diffuse source. "
                    "VS max. scatter-angle VS energy-bins"
                ),
                "unit": "m$^{2}$ sr",
                "mean": Q,
                "absolute_uncertainty": Q_au,
            },
        )
"""
res.stop()
