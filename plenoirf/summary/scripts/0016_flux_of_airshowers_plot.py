#!/usr/bin/python
import sys
import numpy as np
import plenoirf as irf
import os
from os.path import join as opj
import json_utils
import sebastians_matplotlib_addons as sebplt


res = irf.summary.ScriptResources.from_argv(sys.argv)
res.start(sebplt=sebplt)

airshower_fluxes = json_utils.tree.Tree(
    opj(res.paths["analysis_dir"], "0015_flux_of_airshowers")
)

energy_bin = res.energy_binning(key="interpolation")

fig = sebplt.figure(irf.summary.figure.FIGURE_STYLE)
ax = sebplt.add_axes(fig=fig, span=irf.summary.figure.AX_SPAN)
for pk in airshower_fluxes:
    dFdE = airshower_fluxes[pk]["differential_flux"]["values"]
    dFdE_au = airshower_fluxes[pk]["differential_flux"]["absolute_uncertainty"]

    ax.plot(
        energy_bin["centers"],
        dFdE,
        label=pk,
        color=res.PARTICLE_COLORS[pk],
    )
    ax.fill_between(
        x=energy_bin["centers"],
        y1=dFdE - dFdE_au,
        y2=dFdE + dFdE_au,
        facecolor=res.PARTICLE_COLORS[pk],
        alpha=0.2,
        linewidth=0.0,
    )

ax.set_xlabel("energy / GeV")
ax.set_ylabel(
    "differential flux of airshowers /\n"
    + "m$^{-2}$ s$^{-1}$ sr$^{-1}$ (GeV)$^{-1}$"
)
"""
ax.text(
    0.1,
    0.1,
    f"site: {res.SITE['name']}",
    horizontalalignment="center",
    # verticalalignment="center",
    transform=ax.transAxes,
)
"""
ax.loglog()
ax.set_xlim(energy_bin["limits"])
ax.legend()
fig.savefig(
    opj(
        res.paths["out_dir"],
        "airshower_differential_flux.jpg",
    )
)
sebplt.close(fig)

res.stop()
