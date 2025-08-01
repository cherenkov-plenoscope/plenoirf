#!/usr/bin/python
import sys
import numpy as np
import plenoirf as irf
import os
from os.path import join as opj
import json_utils

res = irf.summary.ScriptResources.from_argv(sys.argv)
res.start()

cosmic_rates = json_utils.tree.read(
    opj(res.paths["analysis_dir"], "0105_trigger_rates_for_cosmic_particles")
)
nsb_rates = json_utils.tree.read(
    opj(
        res.paths["analysis_dir"],
        "0120_trigger_rates_for_night_sky_background",
    )
)
zenith_bin = res.zenith_binning("once")
trigger = res.trigger

trigger_rates = {}

assert trigger["threshold_pe"] in trigger["ratescan_thresholds_pe"]
analysis_trigger_threshold_idx = (
    irf.utils.find_closest_index_in_array_for_value(
        arr=trigger["ratescan_thresholds_pe"], val=trigger["threshold_pe"]
    )
)

trigger_rates = {}
trigger_rates["night_sky_background"] = nsb_rates[
    "night_sky_background_rates"
]["mean"]

for zd in range(zenith_bin["num"]):
    zk = f"zd{zd:d}"
    trigger_rates[zk] = {}
    for pk in res.PARTICLES:
        trigger_rates[zk][pk] = np.array(
            cosmic_rates[zk][pk]["integral_rate"]["mean"]
        )

json_utils.write(
    opj(res.paths["out_dir"], "trigger_rates_by_origin.json"),
    {
        "comment": (
            "Trigger-rates by origin VS. zenith-bin VS. trigger-threshold. "
            "Including the analysis_trigger_threshold."
        ),
        "analysis_trigger_threshold_idx": analysis_trigger_threshold_idx,
        "unit": "s$^{-1}$",
        "origins": trigger_rates,
    },
)

res.stop()
