#!/usr/bin/python
import sys
import numpy as np
import plenoirf as irf
import os
import json_utils

argv = irf.summary.argv_since_py(sys.argv)
pa = irf.summary.paths_from_argv(argv)

irf_config = irf.summary.read_instrument_response_config(run_dir=pa["run_dir"])
sum_config = irf.summary.read_summary_config(summary_dir=pa["summary_dir"])

os.makedirs(pa["out_dir"], exist_ok=True)

observation_times = irf.utils.make_civil_times_points_in_quasi_logspace()

json_utils.write(
    os.path.join(pa["out_dir"], "observation_times.json"),
    {
        "observation_times": observation_times,
        "unit": "s",
        "comment": ("Typical civil times"),
    },
)
