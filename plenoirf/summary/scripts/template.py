#!/usr/bin/python
import sys
import plenoirf as irf
import sparse_numeric_table as snt
import os
import json_utils
import numpy as np


paths = irf.summary.paths_from_argv(sys.argv)
res = irf.summary.Resources.from_argv(sys.argv)
os.makedirs(paths["out_dir"], exist_ok=True)

for pk in res.PARTICLES:
    with res.open_event_table(particle_key=pk) as arc:
        table = arc.query(
            levels_and_columns={"primary": ["uid", "energy_GeV"]}
        )
