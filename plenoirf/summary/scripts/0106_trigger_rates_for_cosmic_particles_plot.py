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

cosmic_rates = json_utils.tree.read(
    os.path.join(pa["summary_dir"], "0105_trigger_rates_for_cosmic_particles")
)

fine_energy_bin = json_utils.read(
    os.path.join(pa["summary_dir"], "0005_common_binning", "energy.json")
)["interpolation"]

particle_colors = sum_config["plot"]["particle_colors"]

mean_key = "mean"
unc_key = "absolute_uncertainty"

for sk in irf_config["config"]["sites"]:
    trigger_thresholds = np.array(
        sum_config["trigger"][sk]["ratescan_thresholds_pe"]
    )
    analysis_trigger_threshold = sum_config["trigger"][sk]["threshold_pe"]

    tt = 0
    for tt, trigger_threshold in enumerate(trigger_thresholds):
        if trigger_threshold == analysis_trigger_threshold:
            break

    fig = seb.figure(irf.summary.figure.FIGURE_STYLE)
    ax = seb.add_axes(fig=fig, span=irf.summary.figure.AX_SPAN)

    text_y = 0.7
    for pk in irf_config["config"]["particles"]:
        dRdE = cosmic_rates[sk][pk]["differential_rate"][mean_key]
        dRdE_au = cosmic_rates[sk][pk]["differential_rate"][unc_key]

        ax.plot(
            fine_energy_bin["centers"],
            dRdE[tt, :],
            color=particle_colors[pk],
            label=pk,
        )
        ax.fill_between(
            x=fine_energy_bin["centers"],
            y1=dRdE[tt, :] - dRdE_au[tt, :],
            y2=dRdE[tt, :] + dRdE_au[tt, :],
            facecolor=particle_colors[pk],
            alpha=0.2,
            linewidth=0.0,
        )
        ax.text(
            0.5,
            0.1 + text_y,
            pk,
            color=particle_colors[pk],
            transform=ax.transAxes,
        )
        ir = cosmic_rates[sk][pk]["integral_rate"][mean_key][tt]
        ir_abs_unc = cosmic_rates[sk][pk]["integral_rate"][unc_key][tt]
        ax.text(
            0.6,
            0.1 + text_y,
            r"{: 8.1f} $\pm${: 6.1f} s$^{{-1}}$".format(
                ir, np.ceil(ir_abs_unc)
            ),
            color="k",
            family="monospace",
            transform=ax.transAxes,
        )
        text_y += 0.06

    ax.set_xlabel("energy / GeV")
    ax.set_ylabel("differential trigger-rate /\ns$^{-1}$ (GeV)$^{-1}$")
    ax.loglog()
    ax.set_xlim(fine_energy_bin["limits"])
    ax.set_ylim([1e-3, 1e5])
    fig.savefig(
        os.path.join(
            pa["out_dir"],
            "{:s}_differential_trigger_rate.jpg".format(sk),
        )
    )
    seb.close(fig)
