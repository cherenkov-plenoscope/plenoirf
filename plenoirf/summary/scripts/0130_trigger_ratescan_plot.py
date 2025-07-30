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

trigger = res.trigger

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

trigger_rates = {}
trigger_rates["night_sky_background"] = np.array(
    nsb_rates["night_sky_background_rates"]["mean"]
)
for zd in range(zenith_bin["num"]):
    zk = f"zd{zd:d}"
    trigger_rates[zk] = {}
    for pk in res.PARTICLES:
        trigger_rates[zk][pk] = np.array(
            cosmic_rates[zk][pk]["integral_rate"]["mean"]
        )

tr = trigger_rates

for zd in range(zenith_bin["num"]):
    zk = f"zd{zd:d}"

    fig = sebplt.figure(irf.summary.figure.FIGURE_STYLE)
    ax = sebplt.add_axes(fig=fig, span=irf.summary.figure.AX_SPAN)
    sebplt.add_axes_zenith_range_indicator(
        fig=fig,
        span=[0.0, 0.075, 0.175, 0.175],
        zenith_bin_edges_rad=zenith_bin["edges"],
        zenith_bin=zd,
        fontsize=6,
    )

    ax.plot(
        trigger["ratescan_thresholds_pe"],
        tr["night_sky_background"]
        + tr[zk]["electron"]
        + tr[zk]["proton"]
        + tr[zk]["helium"],
        "k",
        label="night-sky + cosmic-rays",
    )
    ax.plot(
        trigger["ratescan_thresholds_pe"],
        tr["night_sky_background"],
        "k:",
        label="night-sky",
    )

    for ck in res.COSMIC_RAYS:
        ax.plot(
            trigger["ratescan_thresholds_pe"],
            tr[zk][ck],
            color=res.PARTICLE_COLORS[ck],
            label=ck,
        )

    ax.semilogy()
    ax.set_xlabel("trigger threshold / photo electrons (p.e.)")
    ax.set_ylabel("trigger rate / s$^{-1}$")
    ax.legend(loc="best", fontsize=8)
    ax.axvline(x=trigger["threshold_pe"], color="k", linestyle="-", alpha=0.25)
    ax.set_ylim([1e0, 1e7])
    fig.savefig(opj(res.paths["out_dir"], f"zd{zd:d}_ratescan.jpg"))
    sebplt.close(fig)

res.stop()
