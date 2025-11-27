#!/usr/bin/python
import sys
import numpy as np
import plenoirf as irf
import os
from os.path import join as opj
import json_utils

res = irf.summary.ScriptResources.from_argv(sys.argv)
res.start()

observation_times = np.geomspace(1e-3, 1e7, 161)

json_utils.write(
    opj(res.paths["out_dir"], "observation_times.json"),
    {
        "observation_times": observation_times,
        "unit": "s",
        "comment": ("Typical civil times"),
    },
)

res.stop()
