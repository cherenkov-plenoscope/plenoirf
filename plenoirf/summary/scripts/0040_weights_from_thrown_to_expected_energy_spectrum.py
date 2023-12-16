#!/usr/bin/python
import sys
import plenoirf as irf
import sparse_numeric_table as spt
import os
import numpy as np
import sebastians_matplotlib_addons as seb
import json_utils

argv = irf.summary.argv_since_py(sys.argv)
pa = irf.summary.paths_from_argv(argv)

irf_config = irf.summary.read_instrument_response_config(run_dir=pa["run_dir"])
sum_config = irf.summary.read_summary_config(summary_dir=pa["summary_dir"])
seb.matplotlib.rcParams.update(sum_config["plot"]["matplotlib"])

os.makedirs(pa["out_dir"], exist_ok=True)

energy_binning = json_utils.read(
    os.path.join(pa["summary_dir"], "0005_common_binning", "energy.json")
)
energy_bin = energy_binning["trigger_acceptance"]
fine_energy_bin = energy_binning["interpolation"]

PARTICLES = irf_config["config"]["particles"]
SITES = irf_config["config"]["sites"]

particle_colors = sum_config["plot"]["particle_colors"]

# AIRSHOWER RATES
# ===============

airshower_rates = {}
airshower_rates["energy_bin_centers"] = fine_energy_bin["centers"]

# cosmic-ray-flux
# ----------------
_airshower_differential_fluxes = json_utils.tree.read(
    os.path.join(pa["summary_dir"], "0015_flux_of_airshowers")
)

# gamma-ray-flux of reference source
# ----------------------------------
gamma_reference_source = json_utils.read(
    os.path.join(
        pa["summary_dir"], "0009_flux_of_gamma_rays", "reference_source.json"
    )
)

for sk in SITES:
    _airshower_differential_fluxes[sk]["gamma"] = {}
    _airshower_differential_fluxes[sk]["gamma"][
        "differential_flux"
    ] = gamma_reference_source["differential_flux"]


airshower_rates["rates"] = {}
for sk in SITES:
    airshower_rates["rates"][sk] = {}
    for pk in PARTICLES:
        airshower_rates["rates"][sk][pk] = (
            airshower_rates["energy_bin_centers"]
            * _airshower_differential_fluxes[sk][pk]["differential_flux"][
                "values"
            ]
        )

# Read features
# =============

tables = {}

thrown_spectrum = {}
thrown_spectrum["energy_bin_edges"] = energy_bin["edges"]
thrown_spectrum["energy_bin_centers"] = energy_bin["centers"]
thrown_spectrum["rates"] = {}

energy_ranges = {}

for sk in SITES:
    tables[sk] = {}
    thrown_spectrum["rates"][sk] = {}
    energy_ranges[sk] = {}
    for pk in PARTICLES:
        thrown_spectrum["rates"][sk][pk] = {}
        energy_ranges[sk][pk] = {}

        _table = spt.read(
            path=os.path.join(
                pa["run_dir"],
                "event_table",
                sk,
                pk,
                "event_table.tar",
            ),
            structure=irf.table.STRUCTURE,
        )

        thrown_spectrum["rates"][sk][pk] = np.histogram(
            _table["primary"]["energy_GeV"],
            bins=thrown_spectrum["energy_bin_edges"],
        )[0]
        energy_ranges[sk][pk]["min"] = np.min(_table["primary"]["energy_GeV"])
        energy_ranges[sk][pk]["max"] = np.max(_table["primary"]["energy_GeV"])

for sk in SITES:
    for pk in PARTICLES:
        site_particle_dir = os.path.join(pa["out_dir"], sk, pk)
        os.makedirs(site_particle_dir, exist_ok=True)

        w_energy = np.geomspace(
            energy_ranges[sk][pk]["min"],
            energy_ranges[sk][pk]["max"],
            fine_energy_bin["num_bins"],
        )
        w_weight = irf.analysis.reweight.reweight(
            initial_energies=thrown_spectrum["energy_bin_centers"],
            initial_rates=thrown_spectrum["rates"][sk][pk],
            target_energies=airshower_rates["energy_bin_centers"],
            target_rates=airshower_rates["rates"][sk][pk],
            event_energies=w_energy,
        )

        json_utils.write(
            os.path.join(site_particle_dir, "weights_vs_energy.json"),
            {
                "comment": (
                    "Weights vs. energy to transform from thrown "
                    "energy-spectrum to expected energy-spectrum of "
                    "air-showers. In contrast to the energy-spectrum of "
                    "cosmic-rays, this already includes the "
                    "geomagnetic-cutoff."
                ),
                "energy_GeV": w_energy,
                "unit": "1",
                "mean": w_weight,
            },
        )

weights = json_utils.tree.read(pa["out_dir"])

for sk in SITES:
    fig = seb.figure(seb.FIGURE_16_9)
    ax = seb.add_axes(fig=fig, span=(0.1, 0.1, 0.8, 0.8))
    for pk in PARTICLES:
        ax.plot(
            weights[sk][pk]["weights_vs_energy"]["energy_GeV"],
            weights[sk][pk]["weights_vs_energy"]["mean"],
            color=particle_colors[pk],
        )
    ax.loglog()
    ax.set_xlabel("energy / GeV")
    ax.set_ylabel("relative re-weights/ 1")
    ax.set_xlim([1e-1, 1e4])
    ax.set_ylim([1e-6, 1.0])
    fig.savefig(os.path.join(pa["out_dir"], "{:s}_weights.jpg".format(sk)))
    seb.close(fig)
