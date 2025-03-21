#!/usr/bin/python
import sys
import plenoirf as irf
import os
import numpy as np
from os.path import join as opj
import sebastians_matplotlib_addons as sebplt
import json_utils

paths = irf.summary.paths_from_argv(sys.argv)
res = irf.summary.Resources.from_argv(sys.argv)
os.makedirs(paths["out_dir"], exist_ok=True)
sebplt.matplotlib.rcParams.update(res.analysis["plot"]["matplotlib"])

trigger_vs_size = json_utils.tree.read(
    os.path.join(
        paths["analysis_dir"],
        "0074_trigger_probability_vs_cherenkov_density_on_ground",
    )
)

trigger_modi = {
    "passing_trigger": "trigger probability",
    "passing_trigger_if_only_accepting_not_rejecting": "trigger probability\nif only accepting not rejecting",
}

particle_colors = res.analysis["plot"]["particle_colors"]
key = "trigger_probability_vs_cherenkov_size_in_grid_bin"

for tm in trigger_modi:
    # all particles together
    # ----------------------
    fig = sebplt.figure(irf.summary.figure.FIGURE_STYLE)
    ax = sebplt.add_axes(fig=fig, span=irf.summary.figure.AX_SPAN)

    text_y = 0
    for pk in res.PARTICLES:
        density_bin_edges = trigger_vs_size[pk][tm][
            "Cherenkov_density_bin_edges_per_m2"
        ]

        prob = trigger_vs_size[pk][tm]["mean"]
        prob_unc = trigger_vs_size[pk][tm]["relative_uncertainty"]

        sebplt.ax_add_histogram(
            ax=ax,
            bin_edges=density_bin_edges,
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
    ax.set_xlim([np.min(density_bin_edges), np.max(density_bin_edges)])
    ax.set_ylim([1e-6, 1.5e-0])
    ax.set_xlabel("density of Cherenkov photons on mirror / m$^{-2}$")
    ax.set_ylabel("{:s} / 1".format(trigger_modi[tm]))
    fig.savefig(opj(paths["out_dir"], f"{tm}.jpg"))
    sebplt.close(fig)
