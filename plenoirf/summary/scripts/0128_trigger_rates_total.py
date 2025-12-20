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

trigger_config = res.trigger
TRIGGER_MODI = json_utils.read(
    opj(res.paths["analysis_dir"], "0055_passing_trigger", "trigger_modi.json")
)

cosmic_rates = json_utils.tree.Tree(
    opj(res.paths["analysis_dir"], "0105_trigger_rates_for_cosmic_particles")
)
nsb_rates = json_utils.tree.Tree(
    opj(
        res.paths["analysis_dir"],
        "0120_trigger_rates_for_night_sky_background",
    )
)
zenith_bin = res.zenith_binning("3_bins_per_45deg")


num_trigger_thresholds = len(trigger_config["ratescan_thresholds_pe"])

for zd in range(zenith_bin["num"]):
    zk = f"zd{zd:d}"

    os.makedirs(opj(res.paths["out_dir"], zk, "night_sky_background"))
    for tk in TRIGGER_MODI:
        rates = {}
        rates["rate"] = nsb_rates[tk][zk]["rate"]
        rates["rate_au"] = nsb_rates[tk][zk]["rate_au"]
        json_utils.write(
            opj(
                res.paths["out_dir"], zk, "night_sky_background", tk + ".json"
            ),
            rates,
        )

    for pk in res.PARTICLES:
        os.makedirs(opj(res.paths["out_dir"], zk, pk))

        for tk in TRIGGER_MODI:
            rates = {}
            rates["rate"] = cosmic_rates[zk][pk][tk]["integral_rate"]["mean"]
            rates["rate_au"] = cosmic_rates[zk][pk][tk]["integral_rate"][
                "absolute_uncertainty"
            ]
            json_utils.write(
                opj(res.paths["out_dir"], zk, pk, tk + ".json"),
                rates,
            )

res.stop()
