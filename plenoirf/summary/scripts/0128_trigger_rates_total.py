#!/usr/bin/python
import sys
import plenoirf as irf
import sparse_numeric_table as snt
import os
from os.path import join as opj
import numpy as np
import sebastians_matplotlib_addons as sebplt
import json_utils
import propagate_uncertainties as pru


res = irf.summary.ScriptResources.from_argv(sys.argv)
res.start(sebplt=sebplt)

trigger = res.trigger

cosmic_rates = json_utils.tree.Tree(
    opj(res.paths["analysis_dir"], "0105_trigger_rates_for_cosmic_particles")
)
nsb = json_utils.tree.Tree(
    opj(
        res.paths["analysis_dir"],
        "0120_trigger_rates_for_night_sky_background",
    )
)["night_sky_background_rates"]
zenith_bin = res.zenith_binning("once")

trigger_rates = {}

num_trigger_thresholds = len(trigger["ratescan_thresholds_pe"])

for zd in range(zenith_bin["num"]):
    zk = f"zd{zd:d}"
    trigger_rates[zk] = {}
    trigger_rates[zk]["night_sky_background"] = {}
    trigger_rates[zk]["night_sky_background"]["rate"] = nsb[zk]["rate"]
    trigger_rates[zk]["night_sky_background"]["rate_au"] = nsb[zk]["rate_au"]

    for pk in res.PARTICLES:
        trigger_rates[zk][pk] = {}
        trigger_rates[zk][pk]["rate"] = cosmic_rates[zk][pk]["integral_rate"][
            "mean"
        ]
        trigger_rates[zk][pk]["rate_au"] = cosmic_rates[zk][pk][
            "integral_rate"
        ]["absolute_uncertainty"]

json_utils.write(
    opj(res.paths["out_dir"], "trigger_rates_by_origin.json"),
    {
        "comment": (
            "Trigger rates by origin VS. zenith-bin VS. trigger threshold. "
        ),
        "unit": "s$^{-1}$",
        "origins": trigger_rates,
    },
)

res.stop()
