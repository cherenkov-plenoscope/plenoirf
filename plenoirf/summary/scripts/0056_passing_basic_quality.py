#!/usr/bin/python
import sys
import plenoirf as irf
import sparse_numeric_table as spt
import os
import json_utils

argv = irf.summary.argv_since_py(sys.argv)
pa = irf.summary.paths_from_argv(argv)

irf_config = irf.summary.read_instrument_response_config(run_dir=pa["run_dir"])
sum_config = irf.summary.read_summary_config(summary_dir=pa["summary_dir"])

os.makedirs(pa["out_dir"], exist_ok=True)

max_relative_leakage = sum_config["quality"]["max_relative_leakage"]
min_reconstructed_photons = sum_config["quality"]["min_reconstructed_photons"]

for site_key in irf_config["config"]["sites"]:
    for particle_key in irf_config["config"]["particles"]:
        site_particle_dir = os.path.join(pa["out_dir"], site_key, particle_key)
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

        idx_pastquality = irf.analysis.cuts.cut_quality(
            feature_table=event_table["features"],
            max_relative_leakage=max_relative_leakage,
            min_reconstructed_photons=min_reconstructed_photons,
        )

        json_utils.write(
            path=os.path.join(site_particle_dir, "idx.json"),
            out_dict=idx_pastquality,
        )
