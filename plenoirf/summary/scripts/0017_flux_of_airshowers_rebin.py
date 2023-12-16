#!/usr/bin/python
import sys
import numpy as np
import plenoirf as irf
import os
import sebastians_matplotlib_addons as seb
import json_utils

"""
Rebin the diff. flux of cosmic-rays dFdE into the energy-binning used
for the diff. sensitivity.
"""

argv = irf.summary.argv_since_py(sys.argv)
pa = irf.summary.paths_from_argv(argv)

irf_config = irf.summary.read_instrument_response_config(run_dir=pa["run_dir"])
sum_config = irf.summary.read_summary_config(summary_dir=pa["summary_dir"])
seb.matplotlib.rcParams.update(sum_config["plot"]["matplotlib"])

os.makedirs(pa["out_dir"], exist_ok=True)

SITES = irf_config["config"]["sites"]
PARTICLES = irf_config["config"]["particles"]
COSMIC_RAYS = irf.utils.filter_particles_with_electric_charge(PARTICLES)

# load
# ----
airshower_fluxes = json_utils.tree.read(
    os.path.join(pa["summary_dir"], "0015_flux_of_airshowers")
)

energy_binning = json_utils.read(
    os.path.join(pa["summary_dir"], "0005_common_binning", "energy.json")
)

# prepare
# -------
energy_bin = energy_binning["trigger_acceptance_onregion"]
fine_energy_bin = energy_binning["interpolation"]
fine_energy_bin_matches = []
for E in energy_bin["edges"]:
    match = np.argmin(np.abs(fine_energy_bin["edges"] - E))
    fine_energy_bin_matches.append(match)

# work
# ----
diff_flux = {}
diff_flux_au = {}
for sk in SITES:
    diff_flux[sk] = {}
    diff_flux_au[sk] = {}
    sk_dir = os.path.join(pa["out_dir"], sk)
    os.makedirs(sk_dir, exist_ok=True)
    for pk in COSMIC_RAYS:
        fine_dFdE = airshower_fluxes[sk][pk]["differential_flux"]["values"]
        fine_dFdE_au = airshower_fluxes[sk][pk]["differential_flux"][
            "absolute_uncertainty"
        ]
        dFdE = np.zeros(energy_bin["num_bins"])
        dFdE_au = np.zeros(energy_bin["num_bins"])

        for ebin in range(energy_bin["num_bins"]):
            fe_start = fine_energy_bin_matches[ebin]
            fe_stop = fine_energy_bin_matches[ebin + 1]
            dFdE[ebin] = np.mean(fine_dFdE[fe_start:fe_stop])
            dFdE_au[ebin] = np.mean(fine_dFdE_au[fe_start:fe_stop])

        diff_flux[sk][pk] = dFdE
        diff_flux_au[sk][pk] = dFdE_au

        json_utils.write(
            os.path.join(pa["out_dir"], sk, pk + ".json"),
            {
                "energy_binning_key": energy_bin["key"],
                "differential_flux": dFdE,
                "absolute_uncertainty": dFdE_au,
                "unit": "m$^{-2}$ sr$^{-1}$ s$^{-1}$ (GeV)$^{-1}$",
            },
        )

# plot
# ----
for sk in SITES:
    fig = seb.figure(irf.summary.figure.FIGURE_STYLE)
    ax = seb.add_axes(fig=fig, span=irf.summary.figure.AX_SPAN)
    for pk in COSMIC_RAYS:
        ax.plot(
            fine_energy_bin["centers"],
            airshower_fluxes[sk][pk]["differential_flux"]["values"],
            color=sum_config["plot"]["particle_colors"][pk],
        )
        seb.ax_add_histogram(
            ax=ax,
            bin_edges=energy_bin["edges"],
            bincounts=diff_flux[sk][pk],
            bincounts_upper=diff_flux[sk][pk] + diff_flux_au[sk][pk],
            bincounts_lower=diff_flux[sk][pk] - diff_flux_au[sk][pk],
            linecolor=sum_config["plot"]["particle_colors"][pk],
            face_color=sum_config["plot"]["particle_colors"][pk],
            face_alpha=0.2,
        )
    ax.set_ylabel(
        "differential flux /\nm$^{-2}$ sr$^{-1}$ s$^{-1}$ (GeV)$^{-1}$"
    )
    ax.set_xlabel("energy / GeV")
    ax.set_ylim([1e-6, 1e2])
    ax.loglog()
    fig.savefig(os.path.join(pa["out_dir"], sk + ".jpg"))
    seb.close(fig)
