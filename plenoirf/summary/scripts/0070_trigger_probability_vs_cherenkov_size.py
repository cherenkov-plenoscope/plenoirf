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

num_size_bins = 12
size_bin_edges = np.geomspace(1, 2**num_size_bins, (3 * num_size_bins) + 1)

passing_trigger = res.read_passed_trigger(
    opj(res.paths["analysis_dir"], "0055_passing_trigger"),
    trigger_mode_key="far_accepting_focus_and_near_rejecting_focus",
)

for zdbin in range(zenith_bin["num"]):
    zk = f"zd{zdbin:d}"

    for pk in res.PARTICLES:
        pk_dir = opj(res.paths["out_dir"], zk, pk)
        os.makedirs(pk_dir, exist_ok=True)

        event_table = res.event_table(particle_key=pk).query(
            levels_and_columns={"trigger": ("uid", "num_cherenkov_pe")},
            zenith_start_rad=zenith_bin["edges"][zdbin],
            zenith_stop_rad=zenith_bin["edges"][zdbin + 1],
        )

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
            opj(pk_dir, "trigger_probability_vs_cherenkov_size.json"),
            {
                "true_Cherenkov_size_bin_edges_pe": size_bin_edges,
                "unit": "1",
                "mean": trigger_probability,
                "relative_uncertainty": trigger_probability_unc,
            },
        )

res.stop()
