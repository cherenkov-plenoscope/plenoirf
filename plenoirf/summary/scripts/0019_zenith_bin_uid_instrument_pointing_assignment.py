#!/usr/bin/python
import sys
import plenoirf as irf
import sparse_numeric_table as snt
import os
import json_utils
import numpy as np

res = irf.summary.ScriptResources.from_argv(sys.argv)
res.start()

zenith_bin = res.zenith_binning(key="once")

for zd in range(zenith_bin["num"]):
    zk = f"zd{zd:d}"
    zk_dir = os.path.join(res.paths["out_dir"], zk)
    os.makedirs(zk_dir)

    zenith_start_rad = zenith_bin["edges"][zd]
    zenith_stop_rad = zenith_bin["edges"][zd + 1]

    for pk in res.PARTICLES:

        with res.open_event_table(particle_key=pk) as arc:
            event_table = arc.query(
                levels_and_columns={
                    "instrument_pointing": "__all__",
                }
            )
        instrument_pointing = event_table["instrument_pointing"]

        mask = np.logical_and(
            instrument_pointing["zenith_rad"] >= zenith_start_rad,
            instrument_pointing["zenith_rad"] < zenith_stop_rad,
        )
        uid = instrument_pointing["uid"][mask]

        json_utils.write(os.path.join(zk_dir, f"{pk:s}.json"), uid)

res.stop()
