#!/usr/bin/python
import sys
import plenoirf as irf
import os
import numpy as np
from os.path import join as opj
import sebastians_matplotlib_addons as seb
import json_utils

argv = irf.summary.argv_since_py(sys.argv)
pa = irf.summary.paths_from_argv(argv)

irf_config = irf.summary.read_instrument_response_config(run_dir=pa["run_dir"])
sum_config = irf.summary.read_summary_config(summary_dir=pa["summary_dir"])
seb.matplotlib.rcParams.update(sum_config["plot"]["matplotlib"])

os.makedirs(pa["out_dir"], exist_ok=True)

trigger_vs_size = json_utils.tree.read(
    os.path.join(
        pa["summary_dir"], "0070_trigger_probability_vs_cherenkov_size"
    )
)

particle_colors = sum_config["plot"]["particle_colors"]

for site_key in irf_config["config"]["sites"]:
    # all particles together
    # ----------------------
    fig = seb.figure(irf.summary.figure.FIGURE_STYLE)
    ax = seb.add_axes(fig=fig, span=irf.summary.figure.AX_SPAN)

    text_y = 0
    for particle_key in irf_config["config"]["particles"]:
        size_bin_edges = np.array(
            trigger_vs_size[site_key][particle_key][
                "trigger_probability_vs_cherenkov_size"
            ]["true_Cherenkov_size_bin_edges_pe"]
        )

        prob = np.array(
            trigger_vs_size[site_key][particle_key][
                "trigger_probability_vs_cherenkov_size"
            ]["mean"]
        )
        prob_unc = np.array(
            trigger_vs_size[site_key][particle_key][
                "trigger_probability_vs_cherenkov_size"
            ]["relative_uncertainty"]
        )

        seb.ax_add_histogram(
            ax=ax,
            bin_edges=size_bin_edges,
            bincounts=prob,
            linestyle="-",
            linecolor=particle_colors[particle_key],
            bincounts_upper=prob * (1 + prob_unc),
            bincounts_lower=prob * (1 - prob_unc),
            face_color=particle_colors[particle_key],
            face_alpha=0.25,
        )
        ax.text(
            0.85,
            0.1 + text_y,
            particle_key,
            color=particle_colors[particle_key],
            transform=ax.transAxes,
        )
        text_y += 0.06
    ax.semilogx()
    ax.semilogy()
    ax.set_xlim([1e1, np.max(size_bin_edges)])
    ax.set_ylim([1e-6, 1.5e-0])
    ax.set_xlabel("true Cherenkov-size / p.e.")
    ax.set_ylabel("trigger-probability / 1")
    fig.savefig(
        opj(
            pa["out_dir"],
            site_key + "_trigger_probability_vs_cherenkov_size.jpg",
        )
    )
    seb.close(fig)
