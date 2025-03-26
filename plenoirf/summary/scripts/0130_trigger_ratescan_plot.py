#!/usr/bin/python
import sys
import plenoirf as irf
import sparse_numeric_table as snt
import os
from os.path import join as opj
import numpy as np
import sebastians_matplotlib_addons as sebplt
import json_utils


res = irf.summary.ScriptResources.from_argv(sys.argv)
res.start(sebplt=sebplt)

TRIGGER = res.analysis["trigger"][res.site_key]

cosmic_rates = json_utils.tree.read(
    opj(res.paths["analysis_dir"], "0105_trigger_rates_for_cosmic_particles")
)
nsb_rates = json_utils.tree.read(
    opj(
        res.paths["analysis_dir"],
        "0120_trigger_rates_for_night_sky_background",
    )
)

trigger_rates = {}
trigger_rates["night_sky_background"] = np.array(
    nsb_rates["night_sky_background_rates"]["mean"]
)
for pk in res.PARTICLES:
    trigger_rates[pk] = np.array(cosmic_rates[pk]["integral_rate"]["mean"])


trigger_thresholds = np.array(TRIGGER["ratescan_thresholds_pe"])
analysis_trigger_threshold = TRIGGER["threshold_pe"]

tr = trigger_rates

fig = sebplt.figure(irf.summary.figure.FIGURE_STYLE)
ax = sebplt.add_axes(fig=fig, span=irf.summary.figure.AX_SPAN)
ax.plot(
    trigger_thresholds,
    tr["night_sky_background"] + tr["electron"] + tr["proton"] + tr["helium"],
    "k",
    label="night-sky + cosmic-rays",
)
ax.plot(
    trigger_thresholds, tr["night_sky_background"], "k:", label="night-sky"
)

for ck in res.COSMIC_RAYS:
    ax.plot(
        trigger_thresholds,
        tr[ck],
        color=res.PARTICLE_COLORS[ck],
        label=ck,
    )

ax.semilogy()
ax.set_xlabel("trigger-threshold / photo-electrons")
ax.set_ylabel("trigger-rate / s$^{-1}$")
ax.legend(loc="best", fontsize=8)
ax.axvline(x=analysis_trigger_threshold, color="k", linestyle="-", alpha=0.25)
ax.set_ylim([1e0, 1e7])
fig.savefig(opj(res.paths["out_dir"], "ratescan.jpg"))
sebplt.close(fig)

res.stop()
