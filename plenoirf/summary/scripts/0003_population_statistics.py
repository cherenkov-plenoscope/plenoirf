#!/usr/bin/python
import sys
import plenoirf as irf
import os
from os.path import join as opj
import json_utils
import numpy as np
import sparse_numeric_table as snt
import sebastians_matplotlib_addons as sebplt

res = irf.summary.ScriptResources.from_argv(sys.argv)
res.start(sebplt)
energy_key = "10_bins_per_decade"
energy_bin = res.energy_binning(energy_key)
zenith_key = "3_bins_per_45deg"
zenith_bin = res.zenith_binning(key=zenith_key)

for pk in res.PARTICLES:

    thrown = np.zeros(
        shape=(energy_bin["num"], zenith_bin["num"]),
        dtype=int,
    )

    for zd in range(zenith_bin["num"]):
        zenith_start_rad = zenith_bin["edges"][zd]
        zenith_stop_rad = zenith_bin["edges"][zd + 1]

        for en in range(energy_bin["num"]):
            energy_start_GeV = energy_bin["edges"][en]
            energy_stop_GeV = energy_bin["edges"][en + 1]

            table = res.event_table(pk).query(
                energy_start_GeV=energy_start_GeV,
                energy_stop_GeV=energy_stop_GeV,
                zenith_start_rad=zenith_start_rad,
                zenith_stop_rad=zenith_stop_rad,
                levels_and_columns={"primary": ["uid"]},
            )

            thrown[en, zd] = table["primary"].shape[0]

    os.makedirs(opj(res.paths["out_dir"], pk), exist_ok=True)
    json_utils.write(
        opj(res.paths["out_dir"], pk, "num_thrown_energy_vs_zenith.json"),
        {
            "counts": thrown,
            "energy_key": energy_key,
            "zenith_key": zenith_key,
        },
    )

res.stop()
