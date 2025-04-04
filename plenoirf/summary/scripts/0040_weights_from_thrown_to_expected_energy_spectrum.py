#!/usr/bin/python
import sys
import plenoirf as irf
import os
from os.path import join as opj
import numpy as np
import sebastians_matplotlib_addons as sebplt
import json_utils


res = irf.summary.ScriptResources.from_argv(sys.argv)
res.start(sebplt=sebplt)

energy_bin = res.energy_binning(key="trigger_acceptance")
fine_energy_bin = res.energy_binning(key="interpolation")

# AIRSHOWER RATES
# ===============
airshower_rates = {}
airshower_rates["energy_bin_centers"] = fine_energy_bin["centers"]

# cosmic-ray-flux
# ----------------
_airshower_differential_fluxes = json_utils.tree.read(
    opj(res.paths["analysis_dir"], "0015_flux_of_airshowers")
)

# gamma-ray-flux of reference source
# ----------------------------------
gamma_reference_source = json_utils.read(
    opj(
        res.paths["analysis_dir"],
        "0009_flux_of_gamma_rays",
        "reference_source.json",
    )
)

_airshower_differential_fluxes["gamma"] = {}
_airshower_differential_fluxes["gamma"]["differential_flux"] = (
    gamma_reference_source["differential_flux"]
)

airshower_rates["rates"] = {}

airshower_rates["rates"] = {}
for pk in res.PARTICLES:
    airshower_rates["rates"][pk] = (
        airshower_rates["energy_bin_centers"]
        * _airshower_differential_fluxes[pk]["differential_flux"]["values"]
    )

# Read features
# =============

tables = {}

thrown_spectrum = {}
thrown_spectrum["energy_bin_edges"] = energy_bin["edges"]
thrown_spectrum["energy_bin_centers"] = energy_bin["centers"]
thrown_spectrum["rates"] = {}

energy_ranges = {}

tables = {}
thrown_spectrum["rates"] = {}
energy_ranges = {}
for pk in res.PARTICLES:
    thrown_spectrum["rates"][pk] = {}
    energy_ranges[pk] = {}

    with res.open_event_table(particle_key=pk) as arc:
        _table = arc.query(
            levels_and_columns={"primary": ["uid", "energy_GeV"]}
        )

    thrown_spectrum["rates"][pk] = np.histogram(
        _table["primary"]["energy_GeV"],
        bins=thrown_spectrum["energy_bin_edges"],
    )[0]
    energy_ranges[pk]["min"] = np.min(_table["primary"]["energy_GeV"])
    energy_ranges[pk]["max"] = np.max(_table["primary"]["energy_GeV"])

for pk in res.PARTICLES:
    particle_dir = opj(res.paths["out_dir"], pk)
    os.makedirs(particle_dir, exist_ok=True)

    w_energy = np.geomspace(
        energy_ranges[pk]["min"],
        energy_ranges[pk]["max"],
        fine_energy_bin["num"],
    )
    w_weight = irf.analysis.reweight.reweight(
        initial_energies=thrown_spectrum["energy_bin_centers"],
        initial_rates=thrown_spectrum["rates"][pk],
        target_energies=airshower_rates["energy_bin_centers"],
        target_rates=airshower_rates["rates"][pk],
        event_energies=w_energy,
    )

    json_utils.write(
        opj(particle_dir, "weights_vs_energy.json"),
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

weights = json_utils.tree.read(res.paths["out_dir"])

fig = sebplt.figure(irf.summary.figure.FIGURE_STYLE)
ax = sebplt.add_axes(fig=fig, span=irf.summary.figure.AX_SPAN)
for pk in res.PARTICLES:
    ax.plot(
        weights[pk]["weights_vs_energy"]["energy_GeV"],
        weights[pk]["weights_vs_energy"]["mean"],
        color=res.PARTICLE_COLORS[pk],
    )
ax.loglog()
ax.set_xlabel("energy / GeV")
ax.set_ylabel("relative re-weights / 1")
ax.set_xlim([1e-1, 1e4])
ax.set_ylim([1e-6, 1.0])
ax.text(
    0.1,
    0.1,
    f"site: {res.SITE['name']}",
    horizontalalignment="center",
    # verticalalignment="center",
    transform=ax.transAxes,
)
fig.savefig(opj(res.paths["out_dir"], "weights.jpg"))
sebplt.close(fig)

res.stop()
