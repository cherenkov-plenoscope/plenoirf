#!/usr/bin/python
import sys
import numpy as np
import plenoirf as irf
import flux_sensitivity
import spectral_energy_distribution_units as sed
from plenoirf.analysis import spectral_energy_distribution as sed_styles
import cosmic_fluxes
import os
import sebastians_matplotlib_addons as seb
import json_utils

argv = irf.summary.argv_since_py(sys.argv)
pa = irf.summary.paths_from_argv(argv)

irf_config = irf.summary.read_instrument_response_config(run_dir=pa["run_dir"])
sum_config = irf.summary.read_summary_config(summary_dir=pa["summary_dir"])
seb.matplotlib.rcParams.update(sum_config["plot"]["matplotlib"])

os.makedirs(pa["out_dir"], exist_ok=True)

SITES = irf_config["config"]["sites"]
PARTICLES = irf_config["config"]["particles"]
COSMIC_RAYS = irf.utils.filter_particles_with_electric_charge(PARTICLES)
ONREGION_TYPES = sum_config["on_off_measuremnent"]["onregion_types"]

dS = json_utils.tree.read(
    os.path.join(pa["summary_dir"], "0540_diffsens_estimate")
)

energy_bin = json_utils.read(
    os.path.join(pa["summary_dir"], "0005_common_binning", "energy.json")
)["trigger_acceptance_onregion"]

fermi = irf.other_instruments.fermi_lat
cta = irf.other_instruments.cherenkov_telescope_array_south


# gamma-ray-flux of crab-nebula
# -----------------------------
crab_flux = cosmic_fluxes.read_crab_nebula_flux_from_resources()

internal_sed_style = sed_styles.PLENOIRF_SED_STYLE

SED_STYLES = {
    "portal": sed_styles.PLENOIRF_SED_STYLE,
    "science": sed_styles.SCIENCE_SED_STYLE,
    "fermi": sed_styles.FERMI_SED_STYLE,
    "cta": sed_styles.CHERENKOV_TELESCOPE_ARRAY_SED_STYLE,
}


def find_observation_time_index(
    observation_times, observation_time, max_rel_error=0.1
):
    return irf.utils.find_closest_index_in_array_for_value(
        arr=observation_times,
        val=observation_time,
        max_rel_error=max_rel_error,
    )


def com_add_diff_flux(
    com, energy_bin_edges, dVdE, dVdE_au, obstidx, sysuncix=None
):
    com["energy"] = []
    com["differential_flux"] = []
    com["differential_flux_au"] = []
    num_energy_bins = len(energy_bin_edges) - 1

    for ebin in range(num_energy_bins):
        com["energy"].append(
            [
                energy_bin_edges[ebin],
                energy_bin_edges[ebin + 1],
            ]
        )
        if sysuncix is not None:
            com["differential_flux"].append(
                [
                    dVdE[ebin, obstidx, sysuncix],
                    dVdE[ebin, obstidx, sysuncix],
                ]
            )
            com["differential_flux_au"].append(
                [
                    dVdE_au[ebin, obstidx, sysuncix],
                    dVdE_au[ebin, obstidx, sysuncix],
                ]
            )
        else:
            com["differential_flux"].append(
                [
                    dVdE[ebin, obstidx],
                    dVdE[ebin, obstidx],
                ]
            )
            com["differential_flux_au"].append(
                [
                    dVdE_au[ebin, obstidx],
                    dVdE_au[ebin, obstidx],
                ]
            )
    return com


cta_diffsens = json_utils.tree.read(
    os.path.join(pa["summary_dir"], "0545_diffsens_estimate_cta_south")
)

fermi_diffsens = json_utils.tree.read(
    os.path.join(pa["summary_dir"], "0544_diffsens_estimate_fermi_lat")
)["flux_sensitivity"]


observation_times = {"1800s": 1800, "0180s": 180}
sysuncix = 0

for obsk in observation_times:
    observation_time = observation_times[obsk]
    observation_time_str = irf.utils.make_civil_time_str(
        time_s=observation_time, format_seconds="{:.0f}"
    )

    sys_unc = sum_config["on_off_measuremnent"]["systematic_uncertainties"][
        sysuncix
    ]

    x_lim_GeV = np.array([1e-1, 1e4])
    y_lim_per_m2_per_s_per_GeV = np.array([1e3, 1e-16])

    for sk in SITES:
        for ok in ONREGION_TYPES:
            for sedk in SED_STYLES:
                os.makedirs(
                    os.path.join(pa["out_dir"], sk, ok, sedk), exist_ok=True
                )

    for sk in SITES:
        for ok in ONREGION_TYPES:
            for dk in flux_sensitivity.differential.SCENARIOS:
                components = []

                # Crab reference fluxes
                # ---------------------
                for i in range(4):
                    com = {}
                    scale_factor = np.power(10.0, (-1) * i)
                    com["energy"] = [np.array(crab_flux["energy"]["values"])]
                    com["differential_flux"] = [
                        scale_factor
                        * np.array(crab_flux["differential_flux"]["values"])
                    ]
                    com[
                        "label"
                    ] = None  # "{:1.1e} Crab".format(scale_factor) if i == 0 else None
                    com["color"] = "k"
                    com["alpha"] = 0.25 / (1.0 + i)
                    com["linestyle"] = "--"
                    components.append(com.copy())

                # Fermi-LAT diff
                # --------------
                fermi_obstidx = find_observation_time_index(
                    observation_times=fermi_diffsens["observation_times_s"],
                    observation_time=observation_time,
                )
                com = {}
                com = com_add_diff_flux(
                    com=com,
                    energy_bin_edges=fermi_diffsens["energy_bin_edges_GeV"],
                    dVdE=fermi_diffsens["dVdE_per_m2_per_GeV_per_s"],
                    dVdE_au=fermi_diffsens["dVdE_per_m2_per_GeV_per_s_au"],
                    obstidx=fermi_obstidx,
                )
                com["label"] = fermi.LABEL + ", " + observation_time_str
                com["color"] = fermi.COLOR
                com["alpha"] = 1.0
                com["linestyle"] = "-"
                components.append(com)

                # CTA-South, my own estimate
                # --------------------------
                cta_obstidx = find_observation_time_index(
                    observation_times=cta_diffsens["config"][
                        "observation_times"
                    ],
                    observation_time=observation_time,
                )
                com = {}
                com = com_add_diff_flux(
                    com=com,
                    energy_bin_edges=cta_diffsens[
                        "instrument_response_function"
                    ]["energy_bin_edges_GeV"],
                    dVdE=cta_diffsens[dk]["flux_sensitivity"][
                        "dVdE_per_m2_per_GeV_per_s"
                    ],
                    dVdE_au=cta_diffsens[dk]["flux_sensitivity"][
                        "dVdE_per_m2_per_GeV_per_s_au"
                    ],
                    obstidx=cta_obstidx,
                )
                com["label"] = cta.LABEL + ", " + observation_time_str
                com["color"] = cta.COLOR
                com["alpha"] = 1.0
                com["linestyle"] = "-"
                components.append(com)

                # CTA-South, Jim Hinton and Stefan Funk, 30min
                # --------------------------------------------
                try:
                    cta_diff = cta.differential_sensitivity(
                        observation_time=observation_time
                    )
                    com = {}
                    com["energy"] = [np.array(cta_diff["energy"]["values"])]
                    com["differential_flux"] = [
                        np.array(cta_diff["differential_flux"]["values"])
                    ]
                    com["label"] = (
                        cta.LABEL + " (official)" + ", " + observation_time_str
                    )
                    com["color"] = cta.COLOR
                    com["alpha"] = 1.0
                    com["linestyle"] = ":"
                    components.append(com)
                except AssertionError as asserr:
                    print("CTA official", asserr)

                # Plenoscope diff
                # ---------------
                obstidx = find_observation_time_index(
                    observation_times=dS[sk][ok][dk]["observation_times"],
                    observation_time=observation_time,
                )
                com = {}
                com = com_add_diff_flux(
                    com=com,
                    energy_bin_edges=energy_bin["edges"],
                    dVdE=dS[sk][ok][dk]["differential_flux"],
                    dVdE_au=dS[sk][ok][dk]["differential_flux_au"],
                    obstidx=obstidx,
                    sysuncix=sysuncix,
                )
                com["label"] = "Portal, " + observation_time_str
                com["color"] = "black"
                com["alpha"] = 1.0
                com["linestyle"] = "-"
                components.append(com)

                for sedk in SED_STYLES:
                    sed_style = SED_STYLES[sedk]

                    fig = seb.figure(irf.summary.figure.FIGURE_STYLE)
                    ax = seb.add_axes(fig=fig, span=irf.summary.figure.AX_SPAN)

                    for com in components:
                        for ii in range(len(com["energy"])):
                            _energy, _dFdE = sed.convert_units_with_style(
                                x=com["energy"][ii],
                                y=com["differential_flux"][ii],
                                input_style=internal_sed_style,
                                target_style=sed_style,
                            )

                            ax.plot(
                                _energy,
                                _dFdE,
                                label=com["label"] if ii == 0 else None,
                                color=com["color"],
                                alpha=com["alpha"],
                                linestyle=com["linestyle"],
                            )

                            if "differential_flux_au" in com:
                                _, _dFdE_au = sed.convert_units_with_style(
                                    x=com["energy"][ii],
                                    y=com["differential_flux_au"][ii],
                                    input_style=internal_sed_style,
                                    target_style=sed_style,
                                )
                                _d_ru = _dFdE_au / _dFdE
                                _dFdE_lu = _dFdE - _dFdE_au
                                _dFdE_uu = _dFdE + _dFdE_au
                                ax.fill_between(
                                    x=_energy,
                                    y1=_dFdE_lu,
                                    y2=_dFdE_uu,
                                    label=None,
                                    color=com["color"],
                                    alpha=com["alpha"] * 0.15,
                                    linewidth=0.0,
                                )

                    _x_lim, _y_lim = sed.convert_units_with_style(
                        x=x_lim_GeV,
                        y=y_lim_per_m2_per_s_per_GeV,
                        input_style=internal_sed_style,
                        target_style=sed_style,
                    )

                    ax.set_xlim(np.sort(_x_lim))
                    ax.set_ylim(np.sort(_y_lim))
                    ax.loglog()
                    ax.legend(loc="best", fontsize=10)
                    etype = flux_sensitivity.differential.SCENARIOS[dk][
                        "energy_axes_label"
                    ]
                    ax.set_xlabel(
                        etype
                        + " "
                        + sed_style["x_label"]
                        + " / "
                        + sed_style["x_unit"]
                    )
                    ax.set_ylabel(
                        sed_style["y_label"] + " /\n " + sed_style["y_unit"]
                    )
                    fig.savefig(
                        os.path.join(
                            pa["out_dir"],
                            sk,
                            ok,
                            sedk,
                            "{:s}_{:s}_{:s}_differential_sensitivity_sed_style_{:s}_{:s}.jpg".format(
                                sk, ok, dk, sedk, obsk
                            ),
                        )
                    )
                    seb.close(fig)
