#!/usr/bin/python
import sys
import copy
import numpy as np
import plenoirf as irf
import flux_sensitivity
import propagate_uncertainties as pru
import os
import sebastians_matplotlib_addons as seb
import lima1983analysis
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

# load
# ----
energy_binning = json_utils.read(
    os.path.join(pa["summary_dir"], "0005_common_binning", "energy.json")
)
energy_bin = energy_binning["trigger_acceptance_onregion"]
energy_bin_width_au = np.zeros(energy_bin["num_bins"])

S = json_utils.tree.read(
    os.path.join(
        pa["summary_dir"],
        "0534_diffsens_signal_area_and_background_rates_for_multiple_scenarios",
    )
)

detection_threshold_std = sum_config["on_off_measuremnent"][
    "detection_threshold_std"
]

systematic_uncertainties = sum_config["on_off_measuremnent"][
    "systematic_uncertainties"
]
num_systematic_uncertainties = len(systematic_uncertainties)

observation_times = json_utils.read(
    os.path.join(
        pa["summary_dir"],
        "0539_diffsens_observation_times",
        "observation_times.json",
    )
)["observation_times"]

num_observation_times = len(observation_times)

estimator_statistics = sum_config["on_off_measuremnent"][
    "estimator_for_critical_signal_rate"
]

# prepare
# -------
for sk in SITES:
    for ok in ONREGION_TYPES:
        os.makedirs(os.path.join(pa["out_dir"], sk, ok), exist_ok=True)

# work
# ----
for sk in SITES:
    for ok in ONREGION_TYPES:
        on_over_off_ratio = ONREGION_TYPES[ok]["on_over_off_ratio"]
        for dk in flux_sensitivity.differential.SCENARIOS:
            print(sk, ok, dk)

            A_gamma_scenario = S[sk][ok][dk]["gamma"]["area"]["mean"]
            A_gamma_scenario_au = S[sk][ok][dk]["gamma"]["area"][
                "absolute_uncertainty"
            ]

            # Sum up components of background rate in scenario
            # ------------------------------------------------
            R_background_components = []
            R_background_components_au = []
            for ck in COSMIC_RAYS:
                R_background_components.append(
                    S[sk][ok][dk][ck]["rate"]["mean"][:]
                )
                R_background_components_au.append(
                    S[sk][ok][dk][ck]["rate"]["absolute_uncertainty"][:]
                )

            R_background_scenario, R_background_scenario_au = pru.sum_axis0(
                x=R_background_components,
                x_au=R_background_components_au,
            )

            critical_dVdE = np.nan * np.ones(
                shape=(
                    energy_bin["num_bins"],
                    num_observation_times,
                    num_systematic_uncertainties,
                )
            )
            critical_dVdE_au = np.nan * np.ones(critical_dVdE.shape)

            for obstix in range(num_observation_times):
                for sysuncix in range(num_systematic_uncertainties):
                    (
                        R_gamma_scenario,
                        R_gamma_scenario_au,
                    ) = flux_sensitivity.differential.estimate_critical_signal_rate_vs_energy(
                        background_rate_onregion_in_scenario_per_s=R_background_scenario,
                        background_rate_onregion_in_scenario_per_s_au=R_background_scenario_au,
                        onregion_over_offregion_ratio=on_over_off_ratio,
                        observation_time_s=observation_times[obstix],
                        instrument_systematic_uncertainty_relative=systematic_uncertainties[
                            sysuncix
                        ],
                        detection_threshold_std=detection_threshold_std,
                        estimator_statistics=estimator_statistics,
                    )

                    (
                        dVdE,
                        dVdE_au,
                    ) = flux_sensitivity.differential.estimate_differential_sensitivity(
                        energy_bin_edges_GeV=energy_bin["edges"],
                        signal_area_in_scenario_m2=A_gamma_scenario,
                        signal_area_in_scenario_m2_au=A_gamma_scenario_au,
                        critical_signal_rate_in_scenario_per_s=R_gamma_scenario,
                        critical_signal_rate_in_scenario_per_s_au=R_gamma_scenario_au,
                    )

                    critical_dVdE[:, obstix, sysuncix] = dVdE
                    critical_dVdE_au[:, obstix, sysuncix] = dVdE_au

            json_utils.write(
                os.path.join(pa["out_dir"], sk, ok, dk + ".json"),
                {
                    "energy_binning_key": energy_bin["key"],
                    "observation_times": observation_times,
                    "systematic_uncertainties": systematic_uncertainties,
                    "differential_flux": critical_dVdE,
                    "differential_flux_au": critical_dVdE_au,
                    "comment": (
                        "Differential flux-sensitivity "
                        "VS energy VS observation-time "
                        "VS systematic uncertainties."
                    ),
                },
            )
