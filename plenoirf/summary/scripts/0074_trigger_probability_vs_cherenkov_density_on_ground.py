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

passing_trigger = json_utils.tree.read(
    opj(res.paths["analysis_dir"], "0055_passing_trigger")
)

trigger_modi = {}
trigger_modi["passing_trigger"] = {}
trigger_modi["passing_trigger_if_only_accepting_not_rejecting"] = {}
for pk in res.PARTICLES:
    trigger_modi["passing_trigger"][pk] = {"uid": passing_trigger[pk]["uid"]}
    trigger_modi["passing_trigger_if_only_accepting_not_rejecting"][pk] = {"uid": passing_trigger[pk]["only_accepting_not_rejecting"]["uid"]}

grid_bin_area_m2 = res.config["ground_grid"]["geometry"]["bin_width_m"] ** 2.0
density_bin_edges_per_m2 = np.geomspace(1e-3, 1e4, 7 * 5 + 1)

for pk in res.PARTICLES:
    pk_dir = opj(res.paths["out_dir"], pk)
    os.makedirs(pk_dir, exist_ok=True)

    with res.open_event_table(particle_key=pk) as arc:
        event_table = arc.query(
            levels_and_columns={
                "cherenkovsizepart": "__all__",
                "instrument_pointing": "__all__",
                "trigger": "__all__",
            }
        )

    event_table = snt.logic.cut_and_sort_table_on_indices(
        table=event_table,
        common_indices=event_table["trigger"]["uid"],
    )
    df = snt.logic.make_rectangular_DataFrame(table=event_table)

    for tm in trigger_modi:
        mask_pasttrigger = snt.logic.make_mask_of_right_in_left(
            left_indices=df["uid"].values,
            right_indices=trigger_modi[tm][pk]["uid"],
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
            opj(pk_dir, tm + ".json"),
            {
                "Cherenkov_density_bin_edges_per_m2": density_bin_edges_per_m2,
                "unit": "1",
                "mean": trigger_probability,
                "relative_uncertainty": trigger_probability_unc,
            },
        )

res.stop()
