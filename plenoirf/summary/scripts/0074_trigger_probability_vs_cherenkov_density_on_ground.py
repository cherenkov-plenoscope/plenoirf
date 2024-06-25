#!/usr/bin/python
import sys
import os
from os.path import join as opj
import numpy as np
import sparse_numeric_table as snt
import plenoirf as irf
import json_utils

paths = irf.summary.paths_from_argv(sys.argv)
res = irf.summary.Resources.from_argv(sys.argv)
os.makedirs(paths["out_dir"], exist_ok=True)

trigger_modi = {}
trigger_modi["passing_trigger"] = json_utils.tree.read(
    os.path.join(paths["analysis_dir"], "0055_passing_trigger")
)
trigger_modi[
    "passing_trigger_if_only_accepting_not_rejecting"
] = json_utils.tree.read(
    os.path.join(
        paths["analysis_dir"],
        "0054_passing_trigger_if_only_accepting_not_rejecting",
    )
)

grid_bin_area_m2 = res.config["ground_grid"]["geometry"]["bin_width_m"] ** 2.0
density_bin_edges_per_m2 = np.geomspace(1e-3, 1e4, 7 * 5 + 1)

for pk in res.PARTICLES:
    pk_dir = opj(paths["out_dir"], pk)
    os.makedirs(pk_dir, exist_ok=True)

    event_table = res.read_event_table(particle_key=pk)
    event_table = snt.cut_and_sort_table_on_indices(
        table=event_table,
        common_indices=event_table["trigger"][snt.IDX],
        level_keys=["trigger", "cherenkovsizepart", "instrument_pointing"],
    )
    df = snt.make_rectangular_DataFrame(table=event_table)

    for tm in trigger_modi:
        mask_pasttrigger = snt.make_mask_of_right_in_left(
            left_indices=df[snt.IDX].values,
            right_indices=trigger_modi[tm][pk]["idx"],
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
            os.path.join(pk_dir, tm + ".json"),
            {
                "Cherenkov_density_bin_edges_per_m2": density_bin_edges_per_m2,
                "unit": "1",
                "mean": trigger_probability,
                "relative_uncertainty": trigger_probability_unc,
            },
        )
