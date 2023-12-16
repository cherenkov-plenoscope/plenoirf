#!/usr/bin/python
import sys
import numpy as np
import plenoirf as irf
import os
import json_utils
import sebastians_matplotlib_addons as seb


argv = irf.summary.argv_since_py(sys.argv)
pa = irf.summary.paths_from_argv(argv)

os.makedirs(pa["out_dir"], exist_ok=True)

irf_config = irf.summary.read_instrument_response_config(run_dir=pa["run_dir"])
sum_config = irf.summary.read_summary_config(summary_dir=pa["summary_dir"])
seb.matplotlib.rcParams.update(sum_config["plot"]["matplotlib"])

airshower_fluxes = json_utils.tree.read(
    os.path.join(pa["summary_dir"], "0015_flux_of_airshowers")
)

energy_bin = json_utils.read(
    os.path.join(pa["summary_dir"], "0005_common_binning", "energy.json")
)["interpolation"]

particle_colors = sum_config["plot"]["particle_colors"]

for sk in irf_config["config"]["sites"]:
    fig = seb.figure(irf.summary.figure.FIGURE_STYLE)
    ax = seb.add_axes(fig=fig, span=irf.summary.figure.AX_SPAN)
    for pk in airshower_fluxes[sk]:
        dFdE = airshower_fluxes[sk][pk]["differential_flux"]["values"]
        dFdE_au = airshower_fluxes[sk][pk]["differential_flux"][
            "absolute_uncertainty"
        ]

        ax.plot(
            energy_bin["centers"],
            dFdE,
            label=pk,
            color=particle_colors[pk],
        )
        ax.fill_between(
            x=energy_bin["centers"],
            y1=dFdE - dFdE_au,
            y2=dFdE + dFdE_au,
            facecolor=particle_colors[pk],
            alpha=0.2,
            linewidth=0.0,
        )

    ax.set_xlabel("energy / GeV")
    ax.set_ylabel(
        "differential flux of airshowers /\n"
        + "m$^{-2}$ s$^{-1}$ sr$^{-1}$ (GeV)$^{-1}$"
    )
    ax.loglog()
    ax.set_xlim(energy_bin["limits"])
    ax.legend()
    fig.savefig(
        os.path.join(
            pa["out_dir"],
            "{:s}_airshower_differential_flux.jpg".format(sk),
        )
    )
    seb.close(fig)
