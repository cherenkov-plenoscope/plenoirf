#!/usr/bin/python
import sys
import plenoirf as irf
import os
import numpy as np
from os.path import join as opj
import sebastians_matplotlib_addons as sebplt
import json_utils

res = irf.summary.ScriptResources.from_argv(sys.argv)
res.start(sebplt=sebplt)

zenith_bin = res.zenith_binning(key="once")
trigger_vs_size = json_utils.tree.Tree(
    opj(
        res.paths["analysis_dir"], "0070_trigger_probability_vs_cherenkov_size"
    )
)

particle_colors = res.analysis["plot"]["particle_colors"]

# all particles together
# ----------------------
for zd in range(zenith_bin["num"]):
    zk = f"zd{zd:d}"

    fig = sebplt.figure(irf.summary.figure.FIGURE_STYLE)
    sebplt.add_axes_zenith_range_indicator(
        fig=fig,
        span=irf.summary.figure.AX_SPAN_ZENITH_INDICATOR,
        zenith_bin_edges_rad=zenith_bin["edges"],
        zenith_bin=zd,
        fontsize=6,
    )

    ax = sebplt.add_axes(fig=fig, span=irf.summary.figure.AX_SPAN)

    text_y = 0
    for pk in res.PARTICLES:
        size_bin_edges = np.array(
            trigger_vs_size[zk][pk]["trigger_probability_vs_cherenkov_size"][
                "true_Cherenkov_size_bin_edges_pe"
            ]
        )

        prob = np.array(
            trigger_vs_size[zk][pk]["trigger_probability_vs_cherenkov_size"][
                "mean"
            ]
        )
        prob_unc = np.array(
            trigger_vs_size[zk][pk]["trigger_probability_vs_cherenkov_size"][
                "relative_uncertainty"
            ]
        )

        sebplt.ax_add_histogram(
            ax=ax,
            bin_edges=size_bin_edges,
            bincounts=prob,
            linestyle="-",
            linecolor=particle_colors[pk],
            bincounts_upper=prob * (1 + prob_unc),
            bincounts_lower=prob * (1 - prob_unc),
            face_color=particle_colors[pk],
            face_alpha=0.25,
        )
        ax.text(
            0.85,
            0.1 + text_y,
            pk,
            color=particle_colors[pk],
            transform=ax.transAxes,
        )
        text_y += 0.06
    ax.semilogx()
    ax.semilogy()
    ax.set_xlim([1e1, np.max(size_bin_edges)])
    ax.set_ylim([1e-6, 1.5e-0])
    ax.set_xlabel("true Cherenkov size / photo electrons")
    ax.set_ylabel("trigger probability / 1")
    fig.savefig(
        opj(
            res.paths["out_dir"],
            f"trigger_probability_vs_cherenkov_size_{zk:s}.jpg",
        )
    )
    sebplt.close(fig)

res.stop()
