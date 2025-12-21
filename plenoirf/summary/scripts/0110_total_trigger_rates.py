#!/usr/bin/python
import sys
import numpy as np
import plenoirf as irf
import os
import rename_after_writing as rnw
import propagate_uncertainties as pru
from os.path import join as opj
import json_utils

res = irf.summary.ScriptResources.from_argv(sys.argv)
res.start()

zenith_bin = res.zenith_binning("3_bins_per_45deg")
trigger_config = res.trigger

cosmic_rates = json_utils.tree.Tree(
    opj(res.paths["analysis_dir"], "0105_trigger_rates_for_cosmic_particles")
)

TRIGGER_MODI = json_utils.read(
    opj(res.paths["analysis_dir"], "0055_passing_trigger", "trigger_modi.json")
)
NUM_THRESHOLDS = len(trigger_config["ratescan_thresholds_pe"])

mean_key = "mean"
unc_key = "absolute_uncertainty"
intgr_key = "integral_rate"

for zd in range(zenith_bin["num"]):
    zk = f"zd{zd:d}"

    for tk in TRIGGER_MODI:

        R_total = np.zeros(NUM_THRESHOLDS)
        R_total_au = np.zeros(NUM_THRESHOLDS)

        for tt in range(NUM_THRESHOLDS):
            _R = []
            _R_au = []
            for pk in res.PARTICLES:
                _R.append(cosmic_rates[zk][pk][tk][intgr_key][mean_key][tt])
                _R_au.append(cosmic_rates[zk][pk][tk][intgr_key][unc_key][tt])
            R_total[tt], R_total_au[tt] = pru.sum(x=_R, x_au=_R_au)

        os.makedirs(opj(res.paths["out_dir"], zk), exist_ok=True)
        with rnw.open(
            opj(res.paths["out_dir"], zk, f"{tk:s}.json"), "wt"
        ) as f:
            f.write(
                json_utils.dumps(
                    {mean_key: R_total, unc_key: R_total_au}, indent=4
                )
            )

res.stop()
