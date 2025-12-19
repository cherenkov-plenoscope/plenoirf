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
zenith_bin = res.zenith_binning("3_bins_per_45deg")

TRIGGER_MODI = json_utils.read(
    opj(res.paths["analysis_dir"], "0055_passing_trigger", "trigger_modi.json")
)

trigger_rates = json_utils.tree.Tree(
    opj(res.paths["analysis_dir"], "0128_trigger_rates_total")
)

num_trigger_thresholds = len(trigger["ratescan_thresholds_pe"])

NSB_COLOR = "peru"
SUM_COLOR = "olive"
PLOT_LEGEND = False


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


RATE_CONTRIBUTIONS = [ck for ck in res.COSMIC_RAYS] + ["night_sky_background"]

trigger_modi_style = {
    "far_accepting_focus": {"linestyle": "-", "alpha": 0.2},
    "far_accepting_focus_and_near_rejecting_focus": {
        "linestyle": "-",
        "alpha": 1.0,
    },
}


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

    total_rate = {}
    total_rate_au = {}
    for tk in TRIGGER_MODI:
        total_rate[tk] = np.zeros(num_trigger_thresholds)
        total_rate_au[tk] = np.zeros(num_trigger_thresholds)

        for tt in range(num_trigger_thresholds):
            _xsum = []
            _xsum_au = []
            for ck in RATE_CONTRIBUTIONS:
                _xsum.append(trigger_rates[zk][ck][tk]["rate"][tt])
                _xsum_au.append(trigger_rates[zk][ck][tk]["rate_au"][tt])
            total_rate[tk][tt], total_rate_au[tk][tt] = pru.sum(
                x=_xsum, x_au=_xsum_au
            )

    for tk in TRIGGER_MODI:
        ax_plot_au(
            ax=ax,
            x=trigger["ratescan_thresholds_pe"],
            y=total_rate[tk],
            y_au=total_rate_au[tk],
            color=SUM_COLOR,
            label="night sky + cosmic rays",
            **trigger_modi_style[tk],
        )

        ax_plot_au(
            ax=ax,
            x=trigger["ratescan_thresholds_pe"],
            y=trigger_rates[zk]["night_sky_background"][tk]["rate"],
            y_au=trigger_rates[zk]["night_sky_background"][tk]["rate_au"],
            color=NSB_COLOR,
            label="night sky",
            **trigger_modi_style[tk],
        )

        for ck in res.COSMIC_RAYS:
            ax_plot_au(
                ax=ax,
                x=trigger["ratescan_thresholds_pe"],
                y=trigger_rates[zk][ck][tk]["rate"],
                y_au=trigger_rates[zk][ck][tk]["rate_au"],
                color=res.PARTICLE_COLORS[ck],
                label=ck,
                **trigger_modi_style[tk],
            )

    ax.semilogy()
    ax.set_xlabel("trigger threshold / photo electrons")
    ax.set_ylabel("trigger rate / s$^{-1}$")
    if PLOT_LEGEND:
        ax.legend(loc="best", fontsize=8)

    zenith_corrected_threshold_pe = irf.light_field_trigger.get_trigger_threshold_corrected_for_pointing_zenith(
        pointing_zenith_rad=zenith_bin["centers"][zd],
        trigger=trigger,
        nominal_threshold_pe=trigger["threshold_pe"],
    )

    ax.axvline(
        x=zenith_corrected_threshold_pe, color="k", linestyle="--", alpha=0.25
    )
    ax.set_ylim([1e0, 1e8])
    fig.savefig(opj(res.paths["out_dir"], f"zd{zd:d}_ratescan.jpg"))
    sebplt.close(fig)

res.stop()
