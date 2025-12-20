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
import scipy
import sebastians_matplotlib_addons as sebplt
import json_utils

res = irf.summary.ScriptResources.from_argv(sys.argv)
res.start(sebplt=sebplt)

trigger_mode = "far_accepting_focus"
trigger_config = res.trigger

energy_bin = res.energy_binning(key="10_bins_per_decade")

"""
In Sebastian's Ph.D. thesis, the trigger rate for gamma rays was estiamted by
taking the overall acceptance for gamma rays from a point source and
multiplying it with the PSF containment factor of 0.68.

Cosmic ray background only included protons and leptons, but nu Helium.
The trigger rate for cosmic rays was based on the acceptance over the entire
field-of-view and multiplied by the ratio of the entire field-of-view's
solid angle over the onregions's solid angle.

The psf enclosure radius was only 0.31 deg as there was no magnetic deflection.
"""
ZK = "zd0"
PHD_PSF_RADIUS_DEG = 0.31
PHD_PSF_CONTAINMENT_FACTOR = 0.68
PHD_COSMIC_RAY_KEYS = ["electron", "proton"]

PHD_DETECTION_THRESHOLD_STD = 5.0
PHD_ON_OVER_OFF_RATIO = 1.0 / 5.0
PHD_OBSERVATION_TIME_S = 50 * 3600

all_fov_acceptance = json_utils.tree.Tree(
    opj(
        res.paths["analysis_dir"],
        "0100_trigger_acceptance_for_cosmic_particles",
    )
)

all_fov_rates = json_utils.tree.Tree(
    opj(res.paths["analysis_dir"], "0105_trigger_rates_for_cosmic_particles")
)

fermi = irf.other_instruments.fermi_lat

num_isez_energy_supports = 137

# gamma-ray-flux of crab-nebula
# -----------------------------
crab_flux = cosmic_fluxes.read_crab_nebula_flux_from_resources()

internal_sed_style = sed_styles.PLENOIRF_SED_STYLE

output_sed_styles = {
    "plenoirf": sed_styles.PLENOIRF_SED_STYLE,
    "science": sed_styles.SCIENCE_SED_STYLE,
    "fermi": sed_styles.FERMI_SED_STYLE,
}

# background rates
# ----------------
FOV_RADIUS_DEG = (
    0.5 * res.instrument["light_field_sensor_geometry"]["max_FoV_diameter_deg"]
)
onregion_over_all_fov_ratio = PHD_PSF_RADIUS_DEG**2 / FOV_RADIUS_DEG**2


trigger_thresholds = np.array(trigger_config["ratescan_thresholds_pe"])
analysis_trigger_threshold = trigger_config["threshold_pe"]
trigger_threshold_key = np.where(
    trigger_thresholds == analysis_trigger_threshold
)[0][0]

electron_rate_onregion = (
    all_fov_rates[ZK]["electron"][trigger_mode]["integral_rate"]["mean"][
        trigger_threshold_key
    ]
    * onregion_over_all_fov_ratio
)

cosmic_ray_rate_onregion = 0
for ck in PHD_COSMIC_RAY_KEYS:
    cosmic_ray_rate_onregion += (
        all_fov_rates[ZK][ck][trigger_mode]["integral_rate"]["mean"][
            trigger_threshold_key
        ]
        * onregion_over_all_fov_ratio
    )

x_lim_GeV = np.array([1e-1, 1e4])
y_lim_per_m2_per_s_per_GeV = np.array([1e-0, 1e-16])


components = []

# Crab reference fluxes
# ---------------------
for i in range(4):
    com = {}
    scale_factor = np.power(10.0, (-1) * i)
    com["energy"] = [np.array(crab_flux["energy"]["values"])]
    com["differential_flux"] = [
        scale_factor * np.array(crab_flux["differential_flux"]["values"])
    ]
    com["label"] = "{:.3f} Crab".format(scale_factor)
    com["color"] = "k"
    com["alpha"] = 1.0 / (1.0 + i)
    com["linestyle"] = "--"
    components.append(com.copy())

# Fermi-LAT diff
# --------------
fermi_diff = fermi.differential_sensitivity(l=0, b=90)
com = {}
com["energy"] = [np.array(fermi_diff["energy"]["values"])]
com["differential_flux"] = [
    np.array(fermi_diff["differential_flux"]["values"])
]
com["label"] = fermi.LABEL + ", 10y, (l=0, b=90), diff."
com["color"] = fermi.COLOR
com["alpha"] = 1.0
com["linestyle"] = "-"
components.append(com)

# plenoscope
# ----------
all_fov_gamma_effective_area_m2 = (
    np.array(
        all_fov_acceptance[ZK]["gamma"][trigger_mode]["point"]["mean"][
            trigger_threshold_key
        ]
    )
    * PHD_PSF_CONTAINMENT_FACTOR
)

assert (
    energy_bin["num"]
    == all_fov_acceptance["zd0"]["gamma"][trigger_mode]["point"]["mean"].shape[
        1
    ]
)

(
    critical_signal_rate_per_s,
    critical_signal_rate_per_s_au,
) = flux_sensitivity.critical_rate.estimate_critical_signal_rate(
    background_rate_onregion_in_scenario_per_s=cosmic_ray_rate_onregion,
    background_rate_onregion_in_scenario_per_s_au=np.zeros(
        shape=cosmic_ray_rate_onregion.shape
    ),
    onregion_over_offregion_ratio=PHD_ON_OVER_OFF_RATIO,
    observation_time_s=PHD_OBSERVATION_TIME_S,
    instrument_systematic_uncertainty_relative=0.0,
    detection_threshold_std=PHD_DETECTION_THRESHOLD_STD,
    estimator_statistics="LiMaEq17",
)

(
    isez_energy_GeV,
    isez_differential_flux_per_GeV_per_m2_per_s,
) = flux_sensitivity.integral.estimate_spectral_exclusion_zone(
    signal_area_m2=all_fov_gamma_effective_area_m2,
    energy_bin_edges_GeV=energy_bin["edges"],
    critical_signal_rate_per_s=critical_signal_rate_per_s,
    power_law_spectral_indices=np.linspace(start=-5, stop=-0.5, num=137),
    power_law_pivot_energy_GeV=1.0,
)
com = {}
com["energy"] = [isez_energy_GeV]
com["differential_flux"] = [isez_differential_flux_per_GeV_per_m2_per_s]
com["label"] = "Portal {:2.0f} h, trigger".format(
    PHD_OBSERVATION_TIME_S / 3600.0
)
com["color"] = "red"
com["alpha"] = 1.0
com["linestyle"] = "-"
components.append(com)

for sed_style_key in output_sed_styles:
    sed_style = output_sed_styles[sed_style_key]

    fig = sebplt.figure()
    ax = sebplt.add_axes(fig=fig, span=[0.15, 0.15, 0.75, 0.75])
    ax.set_title(
        "Analysis as in Sebastian's Ph.D.\n"
        "Only p and e, fix psf-radius "
        "{:.2f}$^\\circ$, fix psf-containment {:.2f}.".format(
            PHD_PSF_RADIUS_DEG, PHD_PSF_CONTAINMENT_FACTOR
        )
    )

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
    ax.set_xlabel(sed_style["x_label"] + " / " + sed_style["x_unit"])
    ax.set_ylabel(sed_style["y_label"] + " /\n" + sed_style["y_unit"])
    fig.savefig(
        opj(
            res.paths["out_dir"],
            f"integral_spectral_exclusion_zone_style_{sed_style_key:s}.jpg",
        )
    )
    sebplt.close(fig)

res.stop()
