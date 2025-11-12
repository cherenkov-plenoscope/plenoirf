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
zenith_bin = res.zenith_binning("once")

trigger_rates = json_utils.tree.Tree(
    opj(res.paths["analysis_dir"], "0128_trigger_rates_total")
)["trigger_rates_by_origin"]["origins"]

num_trigger_thresholds = len(trigger["ratescan_thresholds_pe"])


def ax_plot_au(ax, x, y, y_au, alpha_ratio=0.25, **kwargs):
    if "alpha" in kwargs:
        line_alpha = kwargs.pop("alpha")
    else:
        line_alpha = 1.0

    fill_alpha = line_alpha * alpha_ratio

    if "label" in kwargs:
        line_label = kwargs.pop("label")
    else:
        line_label = None
    fill_label = None

    ax.fill_between(
        x=x,
        y1=y - y_au,
        y2=y + y_au,
        alpha=fill_alpha,
        label=fill_label,
        **kwargs,
    )
    ax.plot(x, y, alpha=line_alpha, label=line_label, **kwargs)


tr = trigger_rates

for zd in range(zenith_bin["num"]):
    zk = f"zd{zd:d}"

    sfig, sax = irf.summary.figure.style(key="4:3")
    fig = sebplt.figure(sfig)
    ax = sebplt.add_axes(fig=fig, span=sax)
    sebplt.add_axes_zenith_range_indicator(
        fig=fig,
        span=irf.summary.figure.AX_SPAN_ZENITH_INDICATOR,
        zenith_bin_edges_rad=zenith_bin["edges"],
        zenith_bin=zd,
        fontsize=6,
    )

    total_rate = np.zeros(num_trigger_thresholds)
    total_rate_au = np.zeros(num_trigger_thresholds)
    for tt in range(num_trigger_thresholds):
        _xsum = []
        _xsum_au = []
        for ck in tr[zk]:
            _xsum.append(tr[zk][ck]["rate"][tt])
            _xsum_au.append(tr[zk][ck]["rate_au"][tt])
        total_rate[tt], total_rate_au[tt] = pru.sum(x=_xsum, x_au=_xsum_au)

    ax_plot_au(
        ax=ax,
        x=trigger["ratescan_thresholds_pe"],
        y=total_rate,
        y_au=total_rate_au,
        color="black",
        label="night sky + cosmic rays",
    )

    ax_plot_au(
        ax=ax,
        x=trigger["ratescan_thresholds_pe"],
        y=tr[zk]["night_sky_background"]["rate"],
        y_au=tr[zk]["night_sky_background"]["rate_au"],
        color="black",
        linestyle=":",
        label="night sky",
    )

    for ck in res.COSMIC_RAYS:
        ax_plot_au(
            ax=ax,
            x=trigger["ratescan_thresholds_pe"],
            y=tr[zk][ck]["rate"],
            y_au=tr[zk][ck]["rate_au"],
            color=res.PARTICLE_COLORS[ck],
            label=ck,
        )

    ax.semilogy()
    ax.set_xlabel("trigger threshold / photo electrons (p.e.)")
    ax.set_ylabel("trigger rate / s$^{-1}$")
    ax.legend(loc="best", fontsize=8)

    zenith_corrected_threshold_pe = irf.light_field_trigger.get_trigger_threshold_corrected_for_pointing_zenith(
        pointing_zenith_rad=zenith_bin["centers"][zd],
        trigger=trigger,
        nominal_threshold_pe=trigger["threshold_pe"],
    )

    ax.axvline(
        x=zenith_corrected_threshold_pe, color="k", linestyle="--", alpha=0.25
    )
    ax.set_ylim([1e0, 1e6])
    fig.savefig(opj(res.paths["out_dir"], f"zd{zd:d}_ratescan.jpg"))
    sebplt.close(fig)

res.stop()
