#!/usr/bin/python
import sys
import numpy as np
import plenoirf as irf
import flux_sensitivity
import spectral_energy_distribution_units as sed
from plenoirf.analysis import spectral_energy_distribution as sed_styles
import cosmic_fluxes
import os
from os.path import join as opj
import sebastians_matplotlib_addons as sebplt
import json_utils

res = irf.summary.ScriptResources.from_argv(sys.argv)
res.start(sebplt=sebplt)

fermi = irf.other_instruments.fermi_lat
cta = irf.other_instruments.cherenkov_telescope_array_south
portal = irf.other_instruments.portal


ONREGION_TYPES = res.analysis["on_off_measuremnent"]["onregion_types"]

# load
# ----
zenith_bin = res.zenith_binning("once")
ZENITH_ZD_ZK = [(zd, f"zd{zd:d}") for zd in range(zenith_bin["num"])]

dS = json_utils.tree.Tree(
    opj(res.paths["analysis_dir"], "0540_diffsens_estimate")
)

diff_sens_scenario = res.analysis["differential_sensitivity"][
    "gamma_ray_effective_area_scenario"
]
pivot_energies = {"cta": 25.0, "portal": 2.5}

PLOT_FERMI_LAT_ESTIMATE_BY_HINTON_AND_FUNK = False

systematic_uncertainties = res.analysis["on_off_measuremnent"][
    "systematic_uncertainties"
]
num_systematic_uncertainties = len(systematic_uncertainties)

for pe in pivot_energies:
    fls = (
        irf.other_instruments.fermi_lat.flux_sensitivity_vs_observation_time_vs_energy()
    )
    assert fls["dnde"]["unit"] == "cm-2 MeV-1 ph s-1"
    odnde = {"dnde": {}}
    odnde["dnde"]["value"] = fls["dnde"]["value"] * 1e4 * 1e3
    odnde["dnde"]["unit"] = "m-2 GeV ph s-1"
    odnde["energy_bin_edges"] = {}
    odnde["energy_bin_edges"]["value"] = (
        fls["energy_bin_edges"]["value"] * 1e-3
    )
    odnde["energy_bin_edges"]["unit"] = "GeV"
    odnde["observation_times"] = fls["observation_times"]
    lo_ebin = (
        np.digitize(
            pivot_energies[pe], bins=odnde["energy_bin_edges"]["value"]
        )
        - 1
    )

    energy_bin = res.energy_binning(key="trigger_acceptance_onregion")

    # gamma-ray-flux of crab-nebula
    # -----------------------------
    crab_flux = cosmic_fluxes.read_crab_nebula_flux_from_resources()

    internal_sed_style = sed_styles.PLENOIRF_SED_STYLE

    enidx = irf.utils.find_closest_index_in_array_for_value(
        arr=energy_bin["edges"],
        val=pivot_energies[pe],
        max_rel_error=0.25,
    )

    x_lim_s = np.array([1e0, 1e7])
    e_lim_GeV = np.array([1e-1, 1e4])
    y_lim_per_m2_per_s_per_GeV = np.array(
        [1e0, 1e-6]
    )  # np.array([1e3, 1e-16])

    # work
    # ----
    for zd, zk in ZENITH_ZD_ZK:
        for ok in ONREGION_TYPES:
            for dk in flux_sensitivity.differential.SCENARIOS:
                os.makedirs(
                    opj(res.paths["out_dir"], zk, ok, dk), exist_ok=True
                )

                observation_times = dS[zk][ok][dk]["observation_times"]

                components = []

                # Crab reference fluxes
                # ---------------------
                for i in range(4):
                    scale_factor = np.power(10.0, (-1) * i)
                    _flux = scale_factor * np.interp(
                        x=pivot_energies[pe],
                        xp=np.array(crab_flux["energy"]["values"]),
                        fp=np.array(crab_flux["differential_flux"]["values"]),
                    )
                    com = {}
                    com["observation_time"] = observation_times
                    com["energy"] = pivot_energies[pe] * np.ones(
                        len(observation_times)
                    )
                    com["differential_flux"] = _flux * np.ones(
                        len(observation_times)
                    )
                    com["label"] = (
                        "{:1.1e} Crab".format(scale_factor) if i == 0 else None
                    )
                    com["color"] = "black"
                    com["alpha"] = 0.25 / (1.0 + i)
                    com["linestyle"] = "--"
                    components.append(com.copy())

                # Fermi-LAT
                # ---------
                try:
                    if PLOT_FERMI_LAT_ESTIMATE_BY_HINTON_AND_FUNK:
                        fermi_s_vs_t = fermi.sensitivity_vs_observation_time(
                            energy_GeV=pivot_energies[pe]
                        )
                        com = {}
                        com["energy"] = pivot_energies[pe] * np.ones(
                            len(fermi_s_vs_t["observation_time"]["values"])
                        )
                        com["observation_time"] = np.array(
                            fermi_s_vs_t["observation_time"]["values"]
                        )
                        com["differential_flux"] = np.array(
                            fermi_s_vs_t["differential_flux"]["values"]
                        )
                        com["label"] = fermi.LABEL
                        com["color"] = fermi.COLOR
                        com["alpha"] = 1.0
                        com["linestyle"] = "-"
                        components.append(com)
                except AssertionError as asserr:
                    print("Fermi-LAT official", pe, pivot_energies[pe], asserr)

                com = {}
                com["energy"] = pivot_energies[pe] * np.ones(
                    len(odnde["observation_times"]["value"])
                )
                com["observation_time"] = odnde["observation_times"]["value"]
                com["differential_flux"] = odnde["dnde"]["value"][:, lo_ebin]
                com["label"] = fermi.LABEL + "sebplt."
                com["color"] = fermi.COLOR
                com["alpha"] = 1.0
                com["linestyle"] = "-"
                components.append(com)

                # CTA South
                # ---------
                try:
                    cta_s_vs_t = irf.other_instruments.cherenkov_telescope_array_south.sensitivity_vs_observation_time(
                        energy_GeV=pivot_energies[pe]
                    )
                    com = {}
                    com["energy"] = pivot_energies[pe] * np.ones(
                        len(cta_s_vs_t["observation_time"]["values"])
                    )
                    com["observation_time"] = np.array(
                        cta_s_vs_t["observation_time"]["values"]
                    )
                    com["differential_flux"] = np.array(
                        cta_s_vs_t["differential_flux"]["values"]
                    )
                    com["label"] = cta.LABEL
                    com["color"] = cta.COLOR
                    com["alpha"] = 1.0
                    com["linestyle"] = "-"
                    components.append(com)
                except AssertionError as asserr:
                    print("CTA official", pe, pivot_energies[pe], asserr)

                # Plenoscope
                # ----------

                for sysuncix in range(num_systematic_uncertainties):
                    portal_dFdE = dS[zk][ok][dk]["differential_flux"][
                        :, :, sysuncix
                    ]
                    if sysuncix == 0:
                        _alpha = 0.5
                        _linestyle = ":"
                    else:
                        _alpha = 1.0
                        _linestyle = "-"

                    com = {}
                    com["observation_time"] = observation_times
                    com["energy"] = pivot_energies[pe] * np.ones(
                        len(observation_times)
                    )
                    com["differential_flux"] = portal_dFdE[enidx, :]
                    com["label"] = (
                        f"{portal.LABEL:s} sys.: {systematic_uncertainties[sysuncix]:.1e}"
                    )
                    com["color"] = portal.COLOR
                    com["alpha"] = _alpha
                    com["linestyle"] = _linestyle
                    components.append(com)

                # figure
                # ------

                sfig, sax = irf.summary.figure.style("4:3")
                fig = sebplt.figure(sfig)
                ax = sebplt.add_axes(fig=fig, span=sax)

                sebplt.add_axes_zenith_range_indicator(
                    fig=fig,
                    span=irf.summary.figure.AX_SPAN_ZENITH_INDICATOR,
                    zenith_bin_edges_rad=zenith_bin["edges"],
                    zenith_bin=zd,
                    fontsize=6,
                )

                for com in components:
                    _energy, _dFdE = sed.convert_units_with_style(
                        x=com["energy"],
                        y=com["differential_flux"],
                        input_style=internal_sed_style,
                        target_style=sed_styles.PLENOIRF_SED_STYLE,
                    )
                    _ = _energy
                    ax.plot(
                        com["observation_time"],
                        _dFdE,
                        label=com["label"],
                        color=com["color"],
                        alpha=com["alpha"],
                        linestyle=com["linestyle"],
                    )

                _E_lim, _dFdE_lim = sed.convert_units_with_style(
                    x=e_lim_GeV,
                    y=y_lim_per_m2_per_s_per_GeV,
                    input_style=internal_sed_style,
                    target_style=sed_styles.PLENOIRF_SED_STYLE,
                )
                _ = _E_lim

                ax.set_xlim(x_lim_s)
                ax.set_ylim(np.sort(_dFdE_lim))
                ax.loglog()
                # ax.legend(loc="best", fontsize=10)
                ax.set_xlabel("observation time / s")
                ax.set_ylabel(
                    sed_styles.PLENOIRF_SED_STYLE["y_label"]
                    + " /\n "
                    + sed_styles.PLENOIRF_SED_STYLE["y_unit"]
                )

                fig.savefig(
                    opj(
                        res.paths["out_dir"],
                        zk,
                        ok,
                        dk,
                        "sensitivity_vs_obseravtion_time_{:d}MeV.jpg".format(
                            int(pivot_energies[pe] * 1e3)
                        ),
                    )
                )
                sebplt.close(fig)

res.stop()
