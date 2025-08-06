#!/usr/bin/python
import sys
import numpy as np
import plenoirf as irf
import os
from os.path import join as opj
import json_utils

res = irf.summary.ScriptResources.from_argv(sys.argv)
res.start()

observation_times = irf.utils.make_civil_times_points_in_quasi_logspace()

json_utils.write(
    opj(res.paths["out_dir"], "observation_times.json"),
    {
        "observation_times": observation_times,
        "unit": "s",
        "comment": ("Typical civil times"),
    },
)

res.stop()
