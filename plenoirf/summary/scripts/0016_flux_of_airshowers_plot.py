#!/usr/bin/python
import sys
import numpy as np
import plenoirf as irf
import os
from os.path import join as opj
import json_utils
import sebastians_matplotlib_addons as sebplt
import binning_utils


res = irf.summary.ScriptResources.from_argv(sys.argv)
res.start(sebplt=sebplt)

airshower_fluxes = json_utils.tree.Tree(
    opj(res.paths["analysis_dir"], "0015_flux_of_airshowers")
)

energy_bin = res.energy_binning(key="interpolation")


def _ge(x, v):
    return x >= v


def _lt(x, v):
    return x < v


parts = {
    "is_simulated": {"alpha": 1.0, "maskmaker": _ge, "label": False},
    "not_simulated": {"alpha": 0.25, "maskmaker": _lt, "label": False},
}

fstyle, aspan = irf.summary.figure.style(key="4:3")
fig = sebplt.figure(fstyle)
ax = sebplt.add_axes(fig=fig, span=aspan)

for pk in res.COSMIC_RAYS:

    E_start = binning_utils.power10.lower_bin_edge(
        **res.config["particles_simulated_energy_distribution"][pk][
            "energy_start_GeV_power10"
        ]
    )

    dFdE = airshower_fluxes[pk]["differential_flux"]["values"]
    dFdE_au = airshower_fluxes[pk]["differential_flux"]["absolute_uncertainty"]

    for part in parts:
        alpha = parts[part]["alpha"]
        _maskmaker = parts[part]["maskmaker"]
        mask = _maskmaker(x=energy_bin["centers"], v=E_start)
        label = parts[part]["label"]

        ax.plot(
            energy_bin["centers"][mask],
            dFdE[mask],
            label=pk if label else None,
            color=res.PARTICLE_COLORS[pk],
            alpha=alpha,
        )
        ax.fill_between(
            x=energy_bin["centers"][mask],
            y1=dFdE[mask] - dFdE_au[mask],
            y2=dFdE[mask] + dFdE_au[mask],
            facecolor=res.PARTICLE_COLORS[pk],
            alpha=0.2 * alpha,
            linewidth=0.0,
        )

ax.set_xlabel("energy / GeV")
ax.set_ylabel(
    "differential flux of airshowers /\n"
    + "m$^{-2}$ s$^{-1}$ sr$^{-1}$ (GeV)$^{-1}$"
)
ax.loglog()
ax.set_xlim(energy_bin["limits"])
# ax.legend()
fig.savefig(
    opj(
        res.paths["out_dir"],
        "airshower_differential_flux.jpg",
    )
)
sebplt.close(fig)

res.stop()
