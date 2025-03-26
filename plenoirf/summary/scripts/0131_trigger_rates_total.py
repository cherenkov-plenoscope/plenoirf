#!/usr/bin/python
import sys
import numpy as np
import plenoirf as irf
import os
import json_utils

res = irf.summary.ScriptResources.from_argv(sys.argv)
res.start()

cosmic_rates = json_utils.tree.read(
    os.path.join(
        res.paths["analysis_dir"], "0105_trigger_rates_for_cosmic_particles"
    )
)
nsb_rates = json_utils.tree.read(
    os.path.join(
        res.paths["analysis_dir"],
        "0120_trigger_rates_for_night_sky_background",
    )
)

TRIGGER = res.analysis["trigger"][res.site_key]

trigger_rates = {}
trigger_thresholds = np.array(TRIGGER["ratescan_thresholds_pe"])
analysis_trigger_threshold = TRIGGER["threshold_pe"]

assert analysis_trigger_threshold in trigger_thresholds
analysis_trigger_threshold_idx = (
    irf.utils.find_closest_index_in_array_for_value(
        arr=trigger_thresholds, val=analysis_trigger_threshold
    )
)

trigger_rates = {}
trigger_rates["night_sky_background"] = nsb_rates[
    "night_sky_background_rates"
]["mean"]

for pk in res.PARTICLES:
    trigger_rates[pk] = cosmic_rates[pk]["integral_rate"]["mean"]

json_utils.write(
    os.path.join(res.paths["out_dir"], "trigger_rates_by_origin.json"),
    {
        "comment": (
            "Trigger-rates by origin VS. trigger-threshold. "
            "Including the analysis_trigger_threshold."
        ),
        "analysis_trigger_threshold_idx": analysis_trigger_threshold_idx,
        "unit": "s$^{-1}$",
        "origins": trigger_rates,
    },
)

res.stop()
