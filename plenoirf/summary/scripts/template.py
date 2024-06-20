#!/usr/bin/python
import sys
import copy
import plenoirf as irf
import sparse_numeric_table as snt
import os
import json_utils

argv = irf.summary.argv_since_py(sys.argv)
pa = irf.summary.paths_from_argv(argv)

irf_config = irf.summary.read_instrument_response_config(
    run_dir=paths["run_dir"]
)
sum_config = irf.summary.read_summary_config(summary_dir=paths["analysis_dir"])

os.makedirs(paths["out_dir"], exist_ok=True)

for site_key in irf_config["config"]["sites"]:
    for particle_key in irf_config["config"]["particles"]:
        event_table = snt.read(
            path=os.path.join(
                paths["run_dir"],
                "event_table",
                site_key,
                particle_key,
                "event_table.tar",
            ),
            structure=irf.table.STRUCTURE,
        )
