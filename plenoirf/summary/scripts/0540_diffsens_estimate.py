#!/usr/bin/python
import sys
import copy
import numpy as np
import plenoirf as irf
import flux_sensitivity
import propagate_uncertainties as pru
import os
from os.path import join as opj
import sebastians_matplotlib_addons as sebplt
import lima1983analysis
import json_utils

res = irf.summary.ScriptResources.from_argv(sys.argv)
res.start(sebplt=sebplt)

ONREGION_TYPES = res.analysis["on_off_measuremnent"]["onregion_types"]

# load
# ----
energy_bin = res.energy_binning(key="trigger_acceptance_onregion")
zenith_bin = res.zenith_binning("once")

energy_bin_width_au = np.zeros(energy_bin["num"])

S = json_utils.tree.Tree(
    opj(
        res.paths["analysis_dir"],
        "0534_diffsens_signal_area_and_background_rates_for_multiple_scenarios",
    )
)

detection_threshold_std = res.analysis["on_off_measuremnent"][
    "detection_threshold_std"
]

systematic_uncertainties = res.analysis["on_off_measuremnent"][
    "systematic_uncertainties"
]
num_systematic_uncertainties = len(systematic_uncertainties)

observation_times = json_utils.read(
    opj(
        res.paths["analysis_dir"],
        "0539_diffsens_observation_times",
        "observation_times.json",
    )
)["observation_times"]

num_observation_times = len(observation_times)

estimator_statistics = res.analysis["on_off_measuremnent"][
    "estimator_for_critical_signal_rate"
]

# prepare
# -------
for zd in range(zenith_bin["num"]):
    zk = f"zd{zd:d}"
    for ok in ONREGION_TYPES:
        os.makedirs(opj(res.paths["out_dir"], zk, ok), exist_ok=True)

# work
# ----
for zd in range(zenith_bin["num"]):
    zk = f"zd{zd:d}"
    for ok in ONREGION_TYPES:
        on_over_off_ratio = ONREGION_TYPES[ok]["on_over_off_ratio"]
        for dk in flux_sensitivity.differential.SCENARIOS:
            print(zk, ok, dk)

            A_gamma_scenario = S[zk][ok][dk]["gamma"]["area"]["mean"]
            A_gamma_scenario_au = S[zk][ok][dk]["gamma"]["area"][
                "absolute_uncertainty"
            ]

            # Sum up components of background rate in scenario
            # ------------------------------------------------
            R_background_components = []
            R_background_components_au = []
            for ck in res.COSMIC_RAYS:
                R_background_components.append(
                    S[zk][ok][dk][ck]["rate"]["mean"][:]
                )
                R_background_components_au.append(
                    S[zk][ok][dk][ck]["rate"]["absolute_uncertainty"][:]
                )

            R_background_scenario, R_background_scenario_au = pru.sum_axis0(
                x=R_background_components,
                x_au=R_background_components_au,
            )

            critical_dVdE = np.nan * np.ones(
                shape=(
                    energy_bin["num"],
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
                opj(res.paths["out_dir"], zk, ok, dk + ".json"),
                {
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

res.stop()
