#!/usr/bin/python
import sys
import numpy as np
import plenoirf as irf
import atmospheric_cherenkov_response
import sparse_numeric_table as spt
import os
import json_utils


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
)["trigger_acceptance"]


for sk in SITES:
    trigger_thresholds = sum_config["trigger"][sk]["ratescan_thresholds_pe"]
    trigger_modus = sum_config["trigger"][sk]["modus"]

    for pk in PARTICLES:
        site_particle_dir = os.path.join(pa["out_dir"], sk, pk)

        os.makedirs(site_particle_dir, exist_ok=True)

        diffuse_particle_table = spt.read(
            path=os.path.join(
                pa["run_dir"],
                "event_table",
                sk,
                pk,
                "event_table.tar",
            ),
            structure=irf.table.STRUCTURE,
        )

        # point source
        # ------------
        idx_possible_onregion = (
            irf.analysis.cuts.cut_primary_direction_within_angle(
                primary_table=diffuse_particle_table["primary"],
                radial_angle_deg=MAX_SOURCE_ANGLE_DEG,
                azimuth_deg=pointing_azimuth_deg,
                zenith_deg=pointing_zenith_deg,
            )
        )

        point_particle_table = spt.cut_table_on_indices(
            table=diffuse_particle_table,
            common_indices=idx_possible_onregion,
        )

        energy_GeV = point_particle_table["primary"]["energy_GeV"]
        quantity_scatter = point_particle_table["grid"]["area_thrown_m2"]
        num_grid_cells_above_lose_threshold = point_particle_table["grid"][
            "num_bins_above_threshold"
        ]
        total_num_grid_cells = point_particle_table["grid"]["num_bins_thrown"]

        value = []
        absolute_uncertainty = []
        for threshold in trigger_thresholds:
            idx_detected = irf.analysis.light_field_trigger_modi.make_indices(
                trigger_table=point_particle_table["trigger"],
                threshold=threshold,
                modus=trigger_modus,
            )
            mask_detected = spt.make_mask_of_right_in_left(
                left_indices=point_particle_table["primary"][spt.IDX],
                right_indices=idx_detected,
            )
            (
                _q_eff,
                _q_eff_au,
            ) = atmospheric_cherenkov_response.analysis.effective_quantity_for_grid(
                energy_bin_edges_GeV=energy_bin["edges"],
                energy_GeV=energy_GeV,
                mask_detected=mask_detected,
                quantity_scatter=quantity_scatter,
                num_grid_cells_above_lose_threshold=(
                    num_grid_cells_above_lose_threshold
                ),
                total_num_grid_cells=total_num_grid_cells,
            )
            value.append(_q_eff)
            absolute_uncertainty.append(_q_eff_au)

        json_utils.write(
            os.path.join(site_particle_dir, "point.json"),
            {
                "comment": (
                    "Effective area for a point source. "
                    "VS trigger-ratescan-thresholds VS energy-bins"
                ),
                "energy_bin_edges_GeV": energy_bin["edges"],
                "trigger": sum_config["trigger"],
                "unit": "m$^{2}$",
                "mean": value,
                "absolute_uncertainty": absolute_uncertainty,
            },
        )

        # diffuse source
        # --------------
        energy_GeV = diffuse_particle_table["primary"]["energy_GeV"]
        quantity_scatter = (
            diffuse_particle_table["grid"]["area_thrown_m2"]
            * diffuse_particle_table["primary"]["solid_angle_thrown_sr"]
        )
        num_grid_cells_above_lose_threshold = diffuse_particle_table["grid"][
            "num_bins_above_threshold"
        ]
        total_num_grid_cells = diffuse_particle_table["grid"][
            "num_bins_thrown"
        ]

        value = []
        absolute_uncertainty = []
        for threshold in trigger_thresholds:
            idx_detected = irf.analysis.light_field_trigger_modi.make_indices(
                trigger_table=diffuse_particle_table["trigger"],
                threshold=threshold,
                modus=trigger_modus,
            )
            mask_detected = spt.make_mask_of_right_in_left(
                left_indices=diffuse_particle_table["primary"][spt.IDX],
                right_indices=idx_detected,
            )
            (
                _q_eff,
                _q_eff_au,
            ) = atmospheric_cherenkov_response.analysis.effective_quantity_for_grid(
                energy_bin_edges_GeV=energy_bin["edges"],
                energy_GeV=energy_GeV,
                mask_detected=mask_detected,
                quantity_scatter=quantity_scatter,
                num_grid_cells_above_lose_threshold=(
                    num_grid_cells_above_lose_threshold
                ),
                total_num_grid_cells=total_num_grid_cells,
            )
            value.append(_q_eff)
            absolute_uncertainty.append(_q_eff_au)

        json_utils.write(
            os.path.join(site_particle_dir, "diffuse.json"),
            {
                "comment": (
                    "Effective acceptance (area x solid angle) "
                    "for a diffuse source. "
                    "VS trigger-ratescan-thresholds VS energy-bins"
                ),
                "unit": "m$^{2}$ sr",
                "mean": value,
                "absolute_uncertainty": absolute_uncertainty,
            },
        )
