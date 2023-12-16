#!/usr/bin/python
import sys
import numpy as np
import plenoirf as irf
import os
import sebastians_matplotlib_addons as seb
import lima1983analysis
import json_utils

argv = irf.summary.argv_since_py(sys.argv)
pa = irf.summary.paths_from_argv(argv)

irf_config = irf.summary.read_instrument_response_config(run_dir=pa["run_dir"])
sum_config = irf.summary.read_summary_config(summary_dir=pa["summary_dir"])
seb.matplotlib.rcParams.update(sum_config["plot"]["matplotlib"])

os.makedirs(pa["out_dir"], exist_ok=True)

SITES = irf_config["config"]["sites"]
PARTICLES = irf_config["config"]["particles"]
ONREGION_TYPES = sum_config["on_off_measuremnent"]["onregion_types"]

onregion_rates = json_utils.tree.read(
    os.path.join(
        pa["summary_dir"], "0320_onregion_trigger_rates_for_cosmic_rays"
    )
)

fine_energy_bin = json_utils.read(
    os.path.join(pa["summary_dir"], "0005_common_binning", "energy.json")
)["interpolation"]

particle_colors = sum_config["plot"]["particle_colors"]

mean_key = "mean"
unc_key = "absolute_uncertainty"

for sk in SITES:
    for ok in ONREGION_TYPES:
        fig = seb.figure(irf.summary.figure.FIGURE_STYLE)
        ax = seb.add_axes(fig=fig, span=irf.summary.figure.AX_SPAN)

        text_y = 0.7

        for pk in PARTICLES:
            dRdE = onregion_rates[sk][ok][pk]["differential_rate"][mean_key]
            dRdE_au = onregion_rates[sk][ok][pk]["differential_rate"][unc_key]
            ax.plot(
                fine_energy_bin["centers"],
                dRdE,
                color=sum_config["plot"]["particle_colors"][pk],
            )
            ax.fill_between(
                x=fine_energy_bin["centers"],
                y1=dRdE - dRdE_au,
                y2=dRdE + dRdE_au,
                facecolor=sum_config["plot"]["particle_colors"][pk],
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
            ir = onregion_rates[sk][ok][pk]["integral_rate"][mean_key]
            ir_abs_unc = onregion_rates[sk][ok][pk]["integral_rate"][unc_key]
            ax.text(
                0.6,
                0.1 + text_y,
                r"{: 8.1f} $\pm${: 6.1f} s$^{{-1}}$".format(ir, ir_abs_unc),
                color="k",
                family="monospace",
                transform=ax.transAxes,
            )
            text_y += 0.06

        ax.set_xlim(fine_energy_bin["limits"])
        ax.set_ylim([1e-5, 1e3])
        ax.loglog()
        ax.set_xlabel("Energy / GeV")
        ax.set_ylabel("differential rate /\ns$^{-1}$ (GeV)$^{-1}$")
        fig.savefig(
            os.path.join(
                pa["out_dir"],
                "{:s}_{:s}_differential_event_rates.jpg".format(sk, ok),
            )
        )
        seb.close(fig)
