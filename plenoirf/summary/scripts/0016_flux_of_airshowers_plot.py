#!/usr/bin/python
import sys
import numpy as np
import plenoirf as irf
import os
import json_utils
import sebastians_matplotlib_addons as seb


paths = irf.summary.paths_from_argv(sys.argv)
res = irf.summary.Resources.from_argv(sys.argv)
os.makedirs(paths["out_dir"], exist_ok=True)

seb.matplotlib.rcParams.update(res.analysis["plot"]["matplotlib"])

airshower_fluxes = json_utils.tree.read(
    os.path.join(paths["analysis_dir"], "0015_flux_of_airshowers")
)

energy_bin = json_utils.read(
    os.path.join(paths["analysis_dir"], "0005_common_binning", "energy.json")
)["interpolation"]

particle_colors = res.analysis["plot"]["particle_colors"]


fig = seb.figure(irf.summary.figure.FIGURE_STYLE)
ax = seb.add_axes(fig=fig, span=irf.summary.figure.AX_SPAN)
for pk in airshower_fluxes:
    dFdE = airshower_fluxes[pk]["differential_flux"]["values"]
    dFdE_au = airshower_fluxes[pk]["differential_flux"]["absolute_uncertainty"]

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
ax.text(
    0.1,
    0.1,
    f"site: {res.SITE['name']}",
    horizontalalignment="center",
    # verticalalignment="center",
    transform=ax.transAxes,
)
ax.loglog()
ax.set_xlim(energy_bin["limits"])
ax.legend()
fig.savefig(
    os.path.join(
        paths["out_dir"],
        "airshower_differential_flux.jpg",
    )
)
seb.close(fig)
