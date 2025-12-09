#!/usr/bin/python
import sys
import plenoirf as irf
import os
import numpy as np
from os.path import join as opj
import sparse_numeric_table as snt
import json_utils

res = irf.summary.ScriptResources.from_argv(sys.argv)
res.start()

zenith_bin = res.zenith_binning(key="3_bins_per_45deg")
zenith_assignment = json_utils.tree.Tree(
    opj(res.paths["analysis_dir"], "0019_zenith_bin_assignment")
)

num_size_bins = 12
size_bin_edges = np.geomspace(1, 2**num_size_bins, (3 * num_size_bins) + 1)

passing_trigger = json_utils.tree.Tree(
    opj(res.paths["analysis_dir"], "0055_passing_trigger")
)

for zd in range(zenith_bin["num"]):
    zk = f"zd{zd:d}"

    for pk in res.PARTICLES:
        pk_dir = opj(res.paths["out_dir"], zk, pk)
        os.makedirs(pk_dir, exist_ok=True)

        with res.open_event_table(particle_key=pk) as arc:
            event_table = arc.query(
                levels_and_columns={"trigger": ("uid", "num_cherenkov_pe")}
            )
        event_table = snt.logic.cut_table_on_indices(
            table=event_table,
            common_indices=zenith_assignment[zk][pk],
        )

        key = "trigger_probability_vs_cherenkov_size"

        mask_pasttrigger = snt.logic.make_mask_of_right_in_left(
            left_indices=event_table["trigger"]["uid"],
            right_indices=passing_trigger[pk]["uid"],
        ).astype(float)

        num_thrown = np.histogram(
            event_table["trigger"]["num_cherenkov_pe"], bins=size_bin_edges
        )[0]

        num_pasttrigger = np.histogram(
            event_table["trigger"]["num_cherenkov_pe"],
            bins=size_bin_edges,
            weights=mask_pasttrigger,
        )[0]

        trigger_probability = irf.utils._divide_silent(
            numerator=num_pasttrigger, denominator=num_thrown, default=np.nan
        )

        trigger_probability_unc = irf.utils._divide_silent(
            numerator=np.sqrt(num_pasttrigger),
            denominator=num_pasttrigger,
            default=np.nan,
        )

        json_utils.write(
            opj(pk_dir, f"{key}.json"),
            {
                "true_Cherenkov_size_bin_edges_pe": size_bin_edges,
                "unit": "1",
                "mean": trigger_probability,
                "relative_uncertainty": trigger_probability_unc,
            },
        )

res.stop()
