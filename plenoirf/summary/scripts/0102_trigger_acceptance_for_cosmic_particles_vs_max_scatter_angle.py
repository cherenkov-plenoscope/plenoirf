#!/usr/bin/python
import sys
import numpy as np
import plenoirf as irf
import atmospheric_cherenkov_response
import sparse_numeric_table as spt
import os
import json_utils
import magnetic_deflection as mdfl
import spherical_coordinates
import solid_angle_utils


argv = irf.summary.argv_since_py(sys.argv)
pa = irf.summary.paths_from_argv(argv)

irf_config = irf.summary.read_instrument_response_config(run_dir=pa["run_dir"])
sum_config = irf.summary.read_summary_config(summary_dir=pa["summary_dir"])

os.makedirs(pa["out_dir"], exist_ok=True)

PARTICLES = irf_config["config"]["particles"]
SITES = irf_config["config"]["sites"]
MAX_SOURCE_ANGLE_DEG = sum_config["gamma_ray_source_direction"][
    "max_angle_relative_to_pointing_deg"
]
pointing_azimuth_deg = irf_config["config"]["plenoscope_pointing"][
    "azimuth_deg"
]
pointing_zenith_deg = irf_config["config"]["plenoscope_pointing"]["zenith_deg"]

energy_bin = json_utils.read(
    os.path.join(pa["summary_dir"], "0005_common_binning", "energy.json")
)["trigger_acceptance_onregion"]

passing_trigger = json_utils.tree.read(
    os.path.join(pa["summary_dir"], "0055_passing_trigger")
)

scatter_bin = json_utils.read(
    os.path.join(pa["summary_dir"], "0005_common_binning", "scatter.json")
)

for sk in SITES:
    for pk in PARTICLES:
        sk_pk_dir = os.path.join(pa["out_dir"], sk, pk)

        os.makedirs(sk_pk_dir, exist_ok=True)

        shower_table = spt.read(
            path=os.path.join(
                pa["run_dir"],
                "event_table",
                sk,
                pk,
                "event_table.tar",
            ),
            structure=irf.table.STRUCTURE,
        )

        # diffuse source
        # --------------
        """
        num_grid_cells_above_lose_threshold = shower_table["grid"][
            "num_bins_above_threshold"
        ]
        total_num_grid_cells = shower_table["grid"]["num_bins_thrown"]
        idx_detected = passing_trigger[sk][pk]["idx"]

        mask_shower_passed_trigger = spt.make_mask_of_right_in_left(
            left_indices=shower_table["primary"][spt.IDX],
            right_indices=idx_detected,
        )
        """

        shower_table_scatter_angle_deg = np.rad2deg(
            spherical_coordinates.angle_between_az_zd(
                azimuth1_rad=shower_table["primary"]["azimuth_rad"],
                zenith1_rad=shower_table["primary"]["zenith_rad"],
                azimuth2_rad=shower_table["primary"]["magnet_azimuth_rad"],
                zenith2_rad=shower_table["primary"]["magnet_zenith_rad"],
            )
        )

        Q = []
        Q_au = []
        for ci in range(scatter_bin[pk]["num_bins"]):
            scatter_cone_solid_angle_sr = scatter_bin[pk]["edges"][ci + 1]
            max_scatter_angle_rad = solid_angle_utils.cone.half_angle(
                solid_angle_sr=scatter_cone_solid_angle_sr
            )
            max_scatter_angle_deg = np.rad2deg(max_scatter_angle_rad)

            print(
                sk,
                pk,
                "max. scatter cone opening angle {:.3}deg".format(
                    max_scatter_angle_deg
                ),
            )

            # cut subset of showers wich are within max scatter angle
            # -------------------------------------------------------
            mask_shower_within_max_scatter = (
                shower_table_scatter_angle_deg <= max_scatter_angle_deg
            )
            idx_showers_within_max_scatter = shower_table["primary"][spt.IDX][
                mask_shower_within_max_scatter
            ]

            S_shower_table = spt.cut_and_sort_table_on_indices(
                table=shower_table,
                common_indices=idx_showers_within_max_scatter,
                level_keys=["primary", "grid"],
            )

            S_mask_shower_detected = spt.make_mask_of_right_in_left(
                left_indices=S_shower_table["primary"][spt.IDX],
                right_indices=passing_trigger[sk][pk]["idx"],
            )

            S_quantity_scatter = (
                S_shower_table["grid"]["area_thrown_m2"]
                * scatter_cone_solid_angle_sr
            )

            S_num_grid_cells_above_lose_threshold = S_shower_table["grid"][
                "num_bins_above_threshold"
            ]

            S_total_num_grid_cells = S_shower_table["grid"]["num_bins_thrown"]

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
            os.path.join(sk_pk_dir, "diffuse.json"),
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
