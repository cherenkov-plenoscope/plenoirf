#!/usr/bin/python
import sys
import numpy as np
import plenoirf as irf
import os
import sebastians_matplotlib_addons as seb
import json_utils

argv = irf.summary.argv_since_py(sys.argv)
pa = irf.summary.paths_from_argv(argv)

irf_config = irf.summary.read_instrument_response_config(run_dir=pa["run_dir"])
sum_config = irf.summary.read_summary_config(summary_dir=pa["summary_dir"])
seb.matplotlib.rcParams.update(sum_config["plot"]["matplotlib"])

os.makedirs(pa["out_dir"], exist_ok=True)

SITES = irf_config["config"]["sites"]
PARTICLES = irf_config["config"]["particles"]
TRIGGER = sum_config["trigger"]
cosmic_rates = json_utils.tree.read(
    os.path.join(pa["summary_dir"], "0105_trigger_rates_for_cosmic_particles")
)
nsb_rates = json_utils.tree.read(
    os.path.join(
        pa["summary_dir"], "0120_trigger_rates_for_night_sky_background"
    )
)

particle_colors = sum_config["plot"]["particle_colors"]

trigger_rates = {}
for sk in SITES:
    trigger_rates[sk] = {}
    trigger_rates[sk]["night_sky_background"] = np.array(
        nsb_rates[sk]["night_sky_background_rates"]["mean"]
    )
    for pk in PARTICLES:
        trigger_rates[sk][pk] = np.array(
            cosmic_rates[sk][pk]["integral_rate"]["mean"]
        )

COSMIC_RAYS = list(PARTICLES.keys())
COSMIC_RAYS.remove("gamma")

for sk in SITES:
    trigger_thresholds = np.array(TRIGGER[sk]["ratescan_thresholds_pe"])
    analysis_trigger_threshold = TRIGGER[sk]["threshold_pe"]

    tr = trigger_rates[sk]

    fig = seb.figure(irf.summary.figure.FIGURE_STYLE)
    ax = seb.add_axes(fig=fig, span=irf.summary.figure.AX_SPAN)
    ax.plot(
        trigger_thresholds,
        tr["night_sky_background"]
        + tr["electron"]
        + tr["proton"]
        + tr["helium"],
        "k",
        label="night-sky + cosmic-rays",
    )
    ax.plot(
        trigger_thresholds, tr["night_sky_background"], "k:", label="night-sky"
    )

    for ck in COSMIC_RAYS:
        ax.plot(
            trigger_thresholds,
            tr[ck],
            color=particle_colors[ck],
            label=ck,
        )

    ax.semilogy()
    ax.set_xlabel("trigger-threshold / photo-electrons")
    ax.set_ylabel("trigger-rate / s$^{-1}$")
    ax.legend(loc="best", fontsize=8)
    ax.axvline(
        x=analysis_trigger_threshold, color="k", linestyle="-", alpha=0.25
    )
    ax.set_ylim([1e0, 1e7])
    fig.savefig(os.path.join(pa["out_dir"], "{:s}_ratescan.jpg".format(sk)))
    seb.close(fig)
