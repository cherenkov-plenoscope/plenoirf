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

MAX_SOURCE_ANGLE_DEG = res.analysis["gamma_ray_source_direction"][
    "max_angle_relative_to_pointing_deg"
]

energy_bin = res.energy_binning(key="trigger_acceptance_onregion")
zenith_bin = res.zenith_binning("once")

passing_trigger = json_utils.tree.read(
    opj(res.paths["analysis_dir"], "0055_passing_trigger")
)

for zd in range(zenith_bin["num"]):
    zk = f"zd{zd:d}"
    print(zk)

    zenith_start_rad = zenith_bin["edges"][zd]
    zenith_stop_rad = zenith_bin["edges"][zd + 1]

    zd_dir = opj(res.paths["out_dir"], zk)
    os.makedirs(zd_dir, exist_ok=True)

    for pk in res.PARTICLES:
        pk_dir = opj(zd_dir, pk)
        os.makedirs(pk_dir, exist_ok=True)

        scatter_bin = res.scatter_binning(particle_key=pk)

        with res.open_event_table(particle_key=pk) as arc:
            _shower_table = arc.query(
                levels_and_columns={
                    "primary": [
                        "uid",
                        "energy_GeV",
                        "azimuth_rad",
                        "zenith_rad",
                    ],
                    "instrument_pointing": "__all__",
                    "groundgrid": [
                        "uid",
                        "num_bins_thrown",
                        "area_thrown_m2",
                        "num_bins_above_threshold",
                    ],
                }
            )

        mask_zenith_bin = np.logical_and(
            _shower_table["instrument_pointing"]["zenith_rad"]
            >= zenith_start_rad,
            _shower_table["instrument_pointing"]["zenith_rad"]
            < zenith_stop_rad,
        )

        uid_common = snt.logic.intersection(
            [
                _shower_table["primary"]["uid"],
                _shower_table["instrument_pointing"]["uid"][mask_zenith_bin],
                _shower_table["groundgrid"]["uid"],
            ]
        )

        shower_table = snt.logic.cut_and_sort_table_on_indices(
            table=_shower_table,
            common_indices=uid_common,
        )

        # diffuse source
        # --------------
        """
        num_grid_cells_above_lose_threshold = shower_table["grid"][
            "num_bins_above_threshold"
        ]
        total_num_grid_cells = shower_table["grid"]["num_bins_thrown"]
        idx_detected = passing_trigger[pk]["uid"]

        mask_shower_passed_trigger = snt.logic.make_mask_of_right_in_left(
            left_indices=shower_table["primary"]["uid"],
            right_indices=idx_detected,
        )
        """

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
            idx_showers_within_max_scatter = shower_table["primary"]["uid"][
                mask_shower_within_max_scatter
            ]

            S_shower_table = snt.logic.cut_and_sort_table_on_indices(
                table=shower_table,
                common_indices=idx_showers_within_max_scatter,
                level_keys=["primary", "groundgrid"],
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
            ) = atmospheric_cherenkov_response.analysis.effective_quantity_for_grid(
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

res.stop()
