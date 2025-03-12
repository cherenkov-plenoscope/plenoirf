#!/usr/bin/python
import sys
import plenoirf as irf
import os
import numpy as np
from os.path import join as opj
import sparse_numeric_table as snt
import json_utils

paths = irf.summary.paths_from_argv(sys.argv)
res = irf.summary.Resources.from_argv(sys.argv)
os.makedirs(paths["out_dir"], exist_ok=True)

num_size_bins = 12
size_bin_edges = np.geomspace(1, 2**num_size_bins, (3 * num_size_bins) + 1)

passing_trigger = json_utils.tree.read(
    os.path.join(paths["analysis_dir"], "0055_passing_trigger")
)

for pk in res.PARTICLES:
    pk_dir = opj(paths["out_dir"], pk)
    os.makedirs(pk_dir, exist_ok=True)

    with res.open_event_table(particle_key=pk) as arc:
        event_table = arc.read_table(levels_and_columns={"trigger": "__all__"})

    key = "trigger_probability_vs_cherenkov_size"

    mask_pasttrigger = snt.make_mask_of_right_in_left(
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
        os.path.join(pk_dir, f"{key}.json"),
        {
            "true_Cherenkov_size_bin_edges_pe": size_bin_edges,
            "unit": "1",
            "mean": trigger_probability,
            "relative_uncertainty": trigger_probability_unc,
        },
    )
