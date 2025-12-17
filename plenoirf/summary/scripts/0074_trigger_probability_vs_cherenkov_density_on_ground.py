#!/usr/bin/python
import sys
import os
from os.path import join as opj
import numpy as np
import sparse_numeric_table as snt
import plenoirf as irf
import json_utils

res = irf.summary.ScriptResources.from_argv(sys.argv)
res.start()

zenith_bin = res.zenith_binning("3_bins_per_45deg")
energy_bin = res.energy_binning("10_bins_per_decade")

TRIGGER_MODI = [
    "far_accepting_focus_and_near_rejecting_focus",
    "far_accepting_focus",
]
passing_trigger_modi = {}
for tm in TRIGGER_MODI:
    passing_trigger_modi[tm] = {}
    for pk in res.PARTICLES:
        _passing_trigger = res.read_passed_trigger(
            opj(res.paths["analysis_dir"], "0055_passing_trigger"),
            trigger_mode_key=tm,
        )
        passing_trigger_modi[tm][pk] = {"uid": _passing_trigger[pk]["uid"]}


grid_bin_area_m2 = res.config["ground_grid"]["geometry"]["bin_width_m"] ** 2.0
density_bin_edges_per_m2 = np.geomspace(1e-3, 1e4, 7 * 5 + 1)

for zdbin in range(zenith_bin["num"]):
    zk = f"zd{zdbin:d}"

    for pk in res.PARTICLES:
        os.makedirs(opj(res.paths["out_dir"], zk, pk), exist_ok=True)

        event_table = res.event_table(particle_key=pk).query(
            levels_and_columns={
                "cherenkovsizepart": ["uid", "num_photons"],
                "instrument_pointing": "__all__",
                "trigger": ["uid"],
            },
            zenith_start_rad=zenith_bin["edges"][zdbin],
            zenith_stop_rad=zenith_bin["edges"][zdbin + 1],
        )
        event_table = snt.logic.cut_and_sort_table_on_indices(
            table=event_table,
            common_indices=event_table["trigger"]["uid"],
            inplace=True,
        )

        df = snt.logic.make_rectangular_DataFrame(table=event_table)

        for tm in passing_trigger_modi:
            mask_pasttrigger = snt.logic.make_mask_of_right_in_left(
                left_indices=df["uid"].values,
                right_indices=passing_trigger_modi[tm][pk]["uid"],
            ).astype(float)

            projected_area_m2 = (
                np.cos(df["instrument_pointing/zenith_rad"]) * grid_bin_area_m2
            )

            num_thrown = np.histogram(
                df["cherenkovsizepart/num_photons"] / projected_area_m2,
                bins=density_bin_edges_per_m2,
            )[0]

            num_pasttrigger = np.histogram(
                df["cherenkovsizepart/num_photons"] / projected_area_m2,
                bins=density_bin_edges_per_m2,
                weights=mask_pasttrigger,
            )[0]

            trigger_probability = irf.utils._divide_silent(
                numerator=num_pasttrigger,
                denominator=num_thrown,
                default=np.nan,
            )

            trigger_probability_unc = irf.utils._divide_silent(
                numerator=np.sqrt(num_pasttrigger),
                denominator=num_pasttrigger,
                default=np.nan,
            )

            json_utils.write(
                opj(res.paths["out_dir"], zk, pk, tm + ".json"),
                {
                    "Cherenkov_density_bin_edges_per_m2": density_bin_edges_per_m2,
                    "unit": "1",
                    "mean": trigger_probability,
                    "relative_uncertainty": trigger_probability_unc,
                },
            )

res.stop()
