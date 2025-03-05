#!/usr/bin/python
import sys
import numpy as np
import plenoirf as irf
import atmospheric_cherenkov_response
import sparse_numeric_table as snt
import os
import json_utils

paths = irf.summary.paths_from_argv(sys.argv)
res = irf.summary.Resources.from_argv(sys.argv)
os.makedirs(paths["out_dir"], exist_ok=True)

MAX_SOURCE_ANGLE_DEG = res.analysis["gamma_ray_source_direction"][
    "max_angle_relative_to_pointing_deg"
]

energy_bin = json_utils.read(
    os.path.join(paths["analysis_dir"], "0005_common_binning", "energy.json")
)["trigger_acceptance"]

trigger_thresholds = res.analysis["trigger"][res.site_key][
    "ratescan_thresholds_pe"
]
trigger_modus = res.analysis["trigger"][res.site_key]["modus"]

for pk in res.PARTICLES:
    pk_dir = os.path.join(paths["out_dir"], pk)
    os.makedirs(pk_dir, exist_ok=True)

    with res.open_event_table(particle_key=pk) as arc:
        diffuse_particle_table = arc.read_table(
            levels_and_columns={
                "primary": "__all__",
                "instrument_pointing": "__all__",
            }
        )

    _diff = snt.cut_on_common_indices(table=diffuse_particle_table)

    # point source
    # ------------
    idx_possible_onregion = (
        irf.analysis.cuts.cut_primary_direction_within_angle(
            primary_table=_diff["primary"],
            radial_angle_deg=MAX_SOURCE_ANGLE_DEG,
            azimuth_deg=np.rad2deg(
                _diff["instrument_pointing"]["azimuth_rad"]
            ),
            zenith_deg=np.rad2deg(_diff["instrument_pointing"]["zenith_rad"]),
        )
    )

    point_particle_table = snt.cut_table_on_indices(
        table=diffuse_particle_table,
        common_indices=idx_possible_onregion,
    )

    has_groundgrid_result = snt.make_mask_of_right_in_left(
        left_indices=point_particle_table["primary"][snt.IDX],
        right_indices=point_particle_table["groundgrid_result"][snt.IDX],
    )

    energy_GeV = point_particle_table["primary"]["energy_GeV"]
    quantity_scatter = point_particle_table["groundgrid"][
        "area_thrown_m2"
    ] * np.cos(point_particle_table["primary"]["zenith_rad"])
    num_grid_cells_above_lose_threshold = np.zeros(shape=len(energy_GeV))
    num_grid_cells_above_lose_threshold[has_groundgrid_result] = (
        point_particle_table["groundgrid_result"]["num_bins_above_threshold"]
    )
    total_num_grid_cells = point_particle_table["groundgrid"][
        "num_bins_thrown"
    ]

    value = []
    absolute_uncertainty = []
    for threshold in trigger_thresholds:
        print(pk, threshold)
        idx_detected = irf.analysis.light_field_trigger_modi.make_indices(
            trigger_table=point_particle_table["trigger"],
            threshold=threshold,
            modus=trigger_modus,
        )
        mask_detected = snt.make_mask_of_right_in_left(
            left_indices=point_particle_table["primary"][snt.IDX],
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
        os.path.join(pk_dir, "point.json"),
        {
            "comment": (
                "Effective area for a point source. "
                "VS trigger-ratescan-thresholds VS energy-bins"
            ),
            "energy_bin_edges_GeV": energy_bin["edges"],
            "unit": "m$^{2}$",
            "mean": value,
            "absolute_uncertainty": absolute_uncertainty,
        },
    )

    # diffuse source
    # --------------

    has_groundgrid_result = snt.make_mask_of_right_in_left(
        left_indices=diffuse_particle_table["primary"][snt.IDX],
        right_indices=diffuse_particle_table["groundgrid_result"][snt.IDX],
    )

    energy_GeV = diffuse_particle_table["primary"]["energy_GeV"]
    quantity_scatter = (
        diffuse_particle_table["groundgrid"]["area_thrown_m2"]
        * np.cos(diffuse_particle_table["primary"]["zenith_rad"])
        * diffuse_particle_table["primary"]["solid_angle_thrown_sr"]
    )
    num_grid_cells_above_lose_threshold = np.zeros(shape=len(energy_GeV))
    num_grid_cells_above_lose_threshold[has_groundgrid_result] = (
        diffuse_particle_table["groundgrid_result"]["num_bins_above_threshold"]
    )
    total_num_grid_cells = diffuse_particle_table["groundgrid"][
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
        mask_detected = snt.make_mask_of_right_in_left(
            left_indices=diffuse_particle_table["primary"][snt.IDX],
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
        os.path.join(pk_dir, "diffuse.json"),
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
