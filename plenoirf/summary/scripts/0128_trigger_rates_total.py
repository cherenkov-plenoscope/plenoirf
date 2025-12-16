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
TRIGGER_MODI = [
    "far_accepting_focus_and_near_rejecting_focus",
    "far_accepting_focus",
]

cosmic_rates = json_utils.tree.Tree(
    opj(res.paths["analysis_dir"], "0105_trigger_rates_for_cosmic_particles")
)
nsb = json_utils.tree.Tree(
    opj(
        res.paths["analysis_dir"],
        "0120_trigger_rates_for_night_sky_background",
    )
)
zenith_bin = res.zenith_binning("3_bins_per_45deg")

trigger_rates = {}

num_trigger_thresholds = len(trigger["ratescan_thresholds_pe"])

for zd in range(zenith_bin["num"]):
    zk = f"zd{zd:d}"

    trigger_modi = {
        "far_accepting_focus": "_far_accepting_focus",
        "far_accepting_focus_and_near_rejecting_focus": "",
    }
    trigger_rates[zk] = {}

    for trigger_modus in TRIGGER_MODI:
        nsb_key = f"night_sky_background{trigger_modi[trigger_modus]:s}"
        nsb_filename = f"night_sky_background_rates_{trigger_modus:s}"
        trigger_rates[zk][nsb_key] = {}
        trigger_rates[zk][nsb_key]["rate"] = nsb[nsb_filename][zk]["rate"]
        trigger_rates[zk][nsb_key]["rate_au"] = nsb[nsb_filename][zk][
            "rate_au"
        ]

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
