#!/usr/bin/python
import sys
import numpy as np
import plenoirf as irf
import os
from os.path import join as opj
import sebastians_matplotlib_addons as sebplt
import json_utils

"""
Rebin the diff. flux of cosmic-rays dFdE into the energy-binning used
for the diff. sensitivity.
"""

res = irf.summary.ScriptResources.from_argv(sys.argv)
res.start(sebplt=sebplt)

# load
# ----
airshower_fluxes = json_utils.tree.read(
    opj(res.paths["analysis_dir"], "0015_flux_of_airshowers")
)

# prepare
# -------
energy_binning_key = "trigger_acceptance_onregion"
energy_bin = res.energy_binning(key=energy_binning_key)
fine_energy_bin = res.energy_binning(key="interpolation")

fine_energy_bin_matches = []
for E in energy_bin["edges"]:
    match = np.argmin(np.abs(fine_energy_bin["edges"] - E))
    fine_energy_bin_matches.append(match)

# work
# ----
diff_flux = {}
diff_flux_au = {}

for pk in res.COSMIC_RAYS:
    fine_dFdE = airshower_fluxes[pk]["differential_flux"]["values"]
    fine_dFdE_au = airshower_fluxes[pk]["differential_flux"][
        "absolute_uncertainty"
    ]
    dFdE = np.zeros(energy_bin["num"])
    dFdE_au = np.zeros(energy_bin["num"])

    for ebin in range(energy_bin["num"]):
        fe_start = fine_energy_bin_matches[ebin]
        fe_stop = fine_energy_bin_matches[ebin + 1]
        dFdE[ebin] = np.mean(fine_dFdE[fe_start:fe_stop])
        dFdE_au[ebin] = np.mean(fine_dFdE_au[fe_start:fe_stop])

    diff_flux[pk] = dFdE
    diff_flux_au[pk] = dFdE_au

    json_utils.write(
        opj(res.paths["out_dir"], pk + ".json"),
        {
            "energy_binning_key": energy_binning_key,
            "differential_flux": dFdE,
            "absolute_uncertainty": dFdE_au,
            "unit": "m$^{-2}$ sr$^{-1}$ s$^{-1}$ (GeV)$^{-1}$",
        },
    )

# plot
# ----

fig = sebplt.figure(irf.summary.figure.FIGURE_STYLE)
ax = sebplt.add_axes(fig=fig, span=irf.summary.figure.AX_SPAN)
for pk in res.COSMIC_RAYS:
    ax.plot(
        fine_energy_bin["centers"],
        airshower_fluxes[pk]["differential_flux"]["values"],
        color=res.analysis["plot"]["particle_colors"][pk],
    )
    sebplt.ax_add_histogram(
        ax=ax,
        bin_edges=energy_bin["edges"],
        bincounts=diff_flux[pk],
        bincounts_upper=diff_flux[pk] + diff_flux_au[pk],
        bincounts_lower=diff_flux[pk] - diff_flux_au[pk],
        linecolor=res.analysis["plot"]["particle_colors"][pk],
        face_color=res.analysis["plot"]["particle_colors"][pk],
        face_alpha=0.2,
    )
ax.set_ylabel("differential flux /\nm$^{-2}$ sr$^{-1}$ s$^{-1}$ (GeV)$^{-1}$")
ax.set_xlabel("energy / GeV")
ax.set_ylim([1e-6, 1e2])
ax.loglog()
ax.text(
    0.1,
    0.1,
    f"site: {res.SITE['name']}",
    horizontalalignment="center",
    # verticalalignment="center",
    transform=ax.transAxes,
)
fig.savefig(
    opj(res.paths["out_dir"], "airshower_differential_flux_rebinned.jpg")
)
sebplt.close(fig)

res.stop()
