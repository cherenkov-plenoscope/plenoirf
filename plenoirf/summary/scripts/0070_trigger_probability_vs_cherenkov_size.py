#!/usr/bin/python
import sys
import plenoirf as irf
import os
import numpy as np
from os.path import join as opj
import sparse_numeric_table as spt
import json_utils

argv = irf.summary.argv_since_py(sys.argv)
pa = irf.summary.paths_from_argv(argv)

irf_config = irf.summary.read_instrument_response_config(run_dir=pa["run_dir"])
sum_config = irf.summary.read_summary_config(summary_dir=pa["summary_dir"])

os.makedirs(pa["out_dir"], exist_ok=True)

num_size_bins = 12
size_bin_edges = np.geomspace(1, 2**num_size_bins, (3 * num_size_bins) + 1)

passing_trigger = json_utils.tree.read(
    os.path.join(pa["summary_dir"], "0055_passing_trigger")
)

for site_key in irf_config["config"]["sites"]:
    for particle_key in irf_config["config"]["particles"]:
        site_particle_dir = opj(pa["out_dir"], site_key, particle_key)
        os.makedirs(site_particle_dir, exist_ok=True)

        event_table = spt.read(
            path=os.path.join(
                pa["run_dir"],
                "event_table",
                site_key,
                particle_key,
                "event_table.tar",
            ),
            structure=irf.table.STRUCTURE,
        )

        key = "trigger_probability_vs_cherenkov_size"

        mask_pasttrigger = spt.make_mask_of_right_in_left(
            left_indices=event_table["trigger"][spt.IDX],
            right_indices=passing_trigger[site_key][particle_key]["idx"],
        )

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
            os.path.join(site_particle_dir, key + ".json"),
            {
                "true_Cherenkov_size_bin_edges_pe": size_bin_edges,
                "unit": "1",
                "mean": trigger_probability,
                "relative_uncertainty": trigger_probability_unc,
            },
        )
