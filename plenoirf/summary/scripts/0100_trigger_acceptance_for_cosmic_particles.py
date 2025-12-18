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

TRIGGER_MODI = json_utils.read(
    opj(res.paths["analysis_dir"], "0055_passing_trigger", "trigger_modi.json")
)

passing_trigger = {}
for tk in TRIGGER_MODI:
    passing_trigger[tk] = res.read_passed_trigger(
        opj(res.paths["analysis_dir"], "0055_passing_trigger"),
        trigger_mode_key=tk,
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
        for tk in TRIGGER_MODI:
            os.makedirs(opj(res.paths["out_dir"], zk, pk, tk), exist_ok=True)


for pk in res.PARTICLES:
    for zdbin in range(zenith_bin["num"]):
        zk = f"zd{zdbin:d}"

        effective_area = {}
        effective_etendue = {}
        for tk in TRIGGER_MODI:
            effective_area[tk] = {
                "mean": np.zeros(SHAPE_THRESHOLDS_ENERGIES),
                "absolute_uncertainty": np.zeros(SHAPE_THRESHOLDS_ENERGIES),
            }
            effective_etendue[tk] = {
                "mean": np.zeros(SHAPE_THRESHOLDS_ENERGIES),
                "absolute_uncertainty": np.zeros(SHAPE_THRESHOLDS_ENERGIES),
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
                energy_start_GeV=energy_bin["edges"][enbin],
                energy_stop_GeV=energy_bin["edges"][enbin + 1],
                zenith_start_rad=zenith_bin["edges"][zdbin],
                zenith_stop_rad=zenith_bin["edges"][zdbin + 1],
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

            for tk in TRIGGER_MODI:
                for ithresh in range(SHAPE_THRESHOLDS_ENERGIES[0]):
                    threshold_pe = trigger["ratescan_thresholds_pe"][ithresh]

                    uid_detected = passing_trigger[tk][pk].ratescan(
                        threshold_pe=threshold_pe,
                        energy_start_GeV=energy_bin["edges"][enbin],
                        energy_stop_GeV=energy_bin["edges"][enbin + 1],
                        zenith_start_rad=zenith_bin["edges"][zdbin],
                        zenith_stop_rad=zenith_bin["edges"][zdbin + 1],
                    )

                    # point
                    # -----
                    point_mask_detected = snt.logic.make_mask_of_right_in_left(
                        left_indices=point_particle_table["primary"]["uid"],
                        right_indices=uid_detected,
                    )
                    (
                        effective_area[tk]["mean"][ithresh][enbin],
                        effective_area[tk]["absolute_uncertainty"][ithresh][
                            enbin
                        ],
                    ) = atmospheric_cherenkov_response.analysis.effective_quantity_for_grid(
                        mask_detected=point_mask_detected,
                        quantity_scatter=point_quantity_scatter,
                        num_grid_cells_above_lose_threshold=point_num_grid_cells_above_lose_threshold,
                        total_num_grid_cells=point_total_num_grid_cells,
                    )

                    # diffuse
                    # -------
                    diffuse_mask_detected = (
                        snt.logic.make_mask_of_right_in_left(
                            left_indices=diffuse_particle_table["primary"][
                                "uid"
                            ],
                            right_indices=uid_detected,
                        )
                    )
                    (
                        effective_etendue[tk]["mean"][ithresh][enbin],
                        effective_etendue[tk]["absolute_uncertainty"][ithresh][
                            enbin
                        ],
                    ) = atmospheric_cherenkov_response.analysis.effective_quantity_for_grid(
                        mask_detected=diffuse_mask_detected,
                        quantity_scatter=diffuse_quantity_scatter,
                        num_grid_cells_above_lose_threshold=diffuse_num_grid_cells_above_lose_threshold,
                        total_num_grid_cells=diffuse_total_num_grid_cells,
                    )

        for tk in TRIGGER_MODI:
            json_utils.write(
                opj(res.paths["out_dir"], zk, pk, tk, "point.json"),
                {
                    "comment": (
                        "Effective area for a point source. "
                        "VS trigger-ratescan-thresholds VS energy-bins"
                    ),
                    "zenith_key": zk,
                    "unit": "m$^{2}$",
                    "mean": effective_area[tk]["mean"],
                    "absolute_uncertainty": effective_area[tk][
                        "absolute_uncertainty"
                    ],
                },
            )

            json_utils.write(
                opj(res.paths["out_dir"], zk, pk, tk, "diffuse.json"),
                {
                    "comment": (
                        "Effective acceptance (area x solid angle) "
                        "for a diffuse source. "
                        "VS trigger-ratescan-thresholds VS energy-bins"
                    ),
                    "zenith_key": zk,
                    "unit": "m$^{2}$ sr",
                    "mean": effective_etendue[tk]["mean"],
                    "absolute_uncertainty": effective_etendue[tk][
                        "absolute_uncertainty"
                    ],
                },
            )

res.stop()
