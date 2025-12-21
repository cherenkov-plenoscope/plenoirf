#!/usr/bin/python
import sys
import numpy as np
import plenoirf as irf
import os
import rename_after_writing as rnw
import propagate_uncertainties as pru
from os.path import join as opj
import binning_utils as bu
import json_utils

res = irf.summary.ScriptResources.from_argv(sys.argv)
res.start()

zenith_bin = res.zenith_binning("3_bins_per_45deg")

TRIGGER_MODI = json_utils.read(
    opj(res.paths["analysis_dir"], "0055_passing_trigger", "trigger_modi.json")
)
NUM_THRESHOLDS = len(res.trigger["ratescan_thresholds_pe"])

R_total = json_utils.tree.Tree(
    opj(res.paths["analysis_dir"], "0110_total_trigger_rates")
)

trej = irf.utils.find_closest_index_in_array_for_value(
    arr=res.trigger["ratescan_thresholds_pe"],
    val=res.analysis["trigger"]["threshold_pe"],
)


def find_closest_interolate(a, v):
    assert len(a) >= 2
    args = np.argsort(a)
    a_sorted = np.sort(a)
    assert bu.is_strictly_monotonic_increasing(a_sorted)
    i_upper = np.searchsorted(a=a_sorted, v=v)

    assert i_upper > 0, f"a={str(a_sorted):s}, v={v:f}"
    i_lower = i_upper - 1
    assert a_sorted[i_lower] <= v
    assert a_sorted[i_upper] > v
    xp = [a_sorted[i_lower], a_sorted[i_upper]]
    fp = [0, 1]
    w_upper = np.interp(x=v, xp=xp, fp=fp)
    w_lower = 1.0 - w_upper
    return {
        "lower": {"index": int(args[i_upper]), "weight": float(w_upper)},
        "upper": {"index": int(args[i_lower]), "weight": float(w_lower)},
    }


for zd in range(zenith_bin["num"]):
    zk = f"zd{zd:d}"

    R_total_at_analysis_threshold_rej = R_total[zk][
        "far_accepting_focus_and_near_rejecting_focus"
    ]["mean"][trej]

    match = find_closest_interolate(
        a=R_total[zk]["far_accepting_focus"]["mean"],
        v=R_total_at_analysis_threshold_rej,
    )

    with rnw.open(opj(res.paths["out_dir"], f"{zk:s}.json"), "wt") as f:
        f.write(json_utils.dumps(match, indent=4))

res.stop()
