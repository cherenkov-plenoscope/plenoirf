#!/usr/bin/python
import sys
import flux_sensitivity
import numpy as np
import json_utils
import plenoirf as irf
import pkg_resources
import binning_utils
import os


argv = irf.summary.argv_since_py(sys.argv)
pa = irf.summary.paths_from_argv(argv)

irf_config = irf.summary.read_instrument_response_config(run_dir=pa["run_dir"])
sum_config = irf.summary.read_summary_config(summary_dir=pa["summary_dir"])

os.makedirs(pa["out_dir"], exist_ok=True)

observation_times = json_utils.read(
    os.path.join(
        pa["summary_dir"],
        "0539_diffsens_observation_times",
        "observation_times.json",
    )
)["observation_times"]
num_observation_times = len(observation_times)

# config for diff-sens
# --------------------
CTA_IRF_CONFIG = {}
CTA_IRF_CONFIG["name"] = "Prod5-South-20deg-AverageAz-14MSTs37SSTs.1800s-v0.1"
CTA_IRF_CONFIG["num_bins_per_decade"] = sum_config["energy_binning"][
    "num_bins_per_decade"
]
CTA_IRF_CONFIG["roi_opening_deg"] = 3.25
CTA_IRF_CONFIG["detection_threshold_std"] = sum_config["on_off_measuremnent"][
    "detection_threshold_std"
]
CTA_IRF_CONFIG["estimator_statistics"] = sum_config["on_off_measuremnent"][
    "estimator_for_critical_signal_rate"
]
CTA_IRF_CONFIG["on_over_off_ratio"] = 0.2  # CTA-specific
CTA_IRF_CONFIG["systematic_uncertainty_relative"] = 1e-2  # CTA-specific
CTA_IRF_CONFIG["observation_times"] = observation_times

json_utils.write(
    os.path.join(pa["out_dir"], "config.json"),
    CTA_IRF_CONFIG,
)


# instrument-response-function
# ----------------------------
cta_irf_path = pkg_resources.resource_filename(
    "flux_sensitivity",
    os.path.join(
        "tests", "resources", "cta", CTA_IRF_CONFIG["name"] + ".fits.gz"
    ),
)

cta_irf = (
    flux_sensitivity.io.gamma_astro_data.read_instrument_response_function(
        path=cta_irf_path
    )
)

cta_irf = flux_sensitivity.io.gamma_astro_data.average_instrument_response_over_field_of_view(
    irf=cta_irf,
    roi_opening_deg=CTA_IRF_CONFIG["roi_opening_deg"],
)

energy_bin_edges_GeV = (
    flux_sensitivity.io.gamma_astro_data.find_common_energy_bin_edges(
        components=cta_irf,
        num_bins_per_decade=CTA_IRF_CONFIG["num_bins_per_decade"],
    )
)
num_energy_bins = len(energy_bin_edges_GeV) - 1

probability_reco_given_true = flux_sensitivity.io.gamma_astro_data.integrate_dPdMu_to_get_probability_reco_given_true(
    dPdMu=cta_irf["energy_dispersion"]["dPdMu"],
    dPdMu_energy_bin_edges=cta_irf["energy_dispersion"][
        "energy_bin_edges_GeV"
    ],
    dPdMu_Mu_bin_edges=cta_irf["energy_dispersion"]["Mu_bin_edges"],
    energy_bin_edges=energy_bin_edges_GeV,
)
probability_reco_given_true_au = np.zeros(probability_reco_given_true.shape)

signal_area_m2 = np.interp(
    x=binning_utils.centers(energy_bin_edges_GeV),
    xp=binning_utils.centers(
        cta_irf["effective_area"]["energy_bin_edges_GeV"]
    ),
    fp=cta_irf["effective_area"]["area_m2"],
)
signal_area_m2_au = np.zeros(signal_area_m2.shape)

background_per_s_per_sr_per_GeV = np.interp(
    x=binning_utils.centers(energy_bin_edges_GeV),
    xp=binning_utils.centers(cta_irf["background"]["energy_bin_edges_GeV"]),
    fp=cta_irf["background"]["background_per_s_per_sr_per_GeV"],
)

point_spread_function_sigma_deg = np.interp(
    x=binning_utils.centers(energy_bin_edges_GeV),
    xp=binning_utils.centers(
        cta_irf["point_spread_function"]["energy_bin_edges_GeV"]
    ),
    fp=cta_irf["point_spread_function"]["sigma_deg"],
)

background_rate_onregion_per_s = (
    flux_sensitivity.io.gamma_astro_data.integrate_background_rate_in_onregion(
        background_per_s_per_sr_per_GeV=background_per_s_per_sr_per_GeV,
        point_spread_function_sigma_deg=point_spread_function_sigma_deg,
        energy_bin_edges_GeV=energy_bin_edges_GeV,
    )
)
background_rate_onregion_per_s_au = np.zeros(
    background_rate_onregion_per_s.shape
)

blk = {}
blk["energy_bin_edges_GeV"] = energy_bin_edges_GeV
blk["probability_reco_given_true"] = probability_reco_given_true
blk["probability_reco_given_true_au"] = probability_reco_given_true_au
blk["signal_area_m2"] = signal_area_m2
blk["signal_area_m2_au"] = signal_area_m2_au
blk["background_rate_onregion_per_s"] = background_rate_onregion_per_s
blk["background_rate_onregion_per_s_au"] = background_rate_onregion_per_s_au
json_utils.write(
    os.path.join(pa["out_dir"], "instrument_response_function.json"), blk
)


# scenarios
# ---------
for dk in flux_sensitivity.differential.SCENARIOS:
    scenario_dir = os.path.join(pa["out_dir"], dk)
    os.makedirs(scenario_dir, exist_ok=True)

    scenario = flux_sensitivity.differential.init_scenario_matrices_for_signal_and_background(
        probability_reco_given_true=blk["probability_reco_given_true"],
        probability_reco_given_true_au=blk["probability_reco_given_true_au"],
        scenario_key=dk,
    )

    (
        signal_area_in_scenario_m2,
        signal_area_in_scenario_m2_au,
    ) = flux_sensitivity.differential.apply_scenario_to_signal_effective_area(
        signal_area_m2=blk["signal_area_m2"],
        signal_area_m2_au=blk["signal_area_m2_au"],
        scenario_G_matrix=scenario["G_matrix"],
        scenario_G_matrix_au=scenario["G_matrix_au"],
    )

    (
        background_rate_onregion_in_scenario_per_s,
        background_rate_onregion_in_scenario_per_s_au,
    ) = flux_sensitivity.differential.apply_scenario_to_background_rate(
        rate_in_reco_energy_per_s=blk["background_rate_onregion_per_s"],
        rate_in_reco_energy_per_s_au=blk["background_rate_onregion_per_s_au"],
        scenario_B_matrix=scenario["B_matrix"],
        scenario_B_matrix_au=scenario["B_matrix_au"],
    )
    scenario["signal_area_in_scenario_m2"] = signal_area_in_scenario_m2
    scenario["signal_area_in_scenario_m2_au"] = signal_area_in_scenario_m2_au
    scenario[
        "background_rate_onregion_in_scenario_per_s"
    ] = background_rate_onregion_in_scenario_per_s
    scenario[
        "background_rate_onregion_in_scenario_per_s_au"
    ] = background_rate_onregion_in_scenario_per_s_au
    json_utils.write(os.path.join(scenario_dir, "scenario.json"), scenario)

    critrate = np.zeros(shape=(num_energy_bins, num_observation_times))
    critrate_au = np.zeros(shape=(num_energy_bins, num_observation_times))

    dVdE = np.zeros(shape=(num_energy_bins, num_observation_times))
    dVdE_au = np.zeros(shape=(num_energy_bins, num_observation_times))

    # observation_times
    # -----------------
    for obstix in range(num_observation_times):
        (
            critrate[:, obstix],
            critrate_au[:, obstix],
        ) = flux_sensitivity.differential.estimate_critical_signal_rate_vs_energy(
            background_rate_onregion_in_scenario_per_s=scenario[
                "background_rate_onregion_in_scenario_per_s"
            ],
            background_rate_onregion_in_scenario_per_s_au=scenario[
                "background_rate_onregion_in_scenario_per_s_au"
            ],
            onregion_over_offregion_ratio=CTA_IRF_CONFIG["on_over_off_ratio"],
            observation_time_s=observation_times[obstix],
            instrument_systematic_uncertainty_relative=CTA_IRF_CONFIG[
                "systematic_uncertainty_relative"
            ],
            detection_threshold_std=CTA_IRF_CONFIG["detection_threshold_std"],
            estimator_statistics=CTA_IRF_CONFIG["estimator_statistics"],
        )

        (
            dVdE[:, obstix],
            dVdE_au[:, obstix],
        ) = flux_sensitivity.differential.estimate_differential_sensitivity(
            energy_bin_edges_GeV=blk["energy_bin_edges_GeV"],
            signal_area_in_scenario_m2=scenario["signal_area_in_scenario_m2"],
            signal_area_in_scenario_m2_au=scenario[
                "signal_area_in_scenario_m2_au"
            ],
            critical_signal_rate_in_scenario_per_s=critrate[:, obstix],
            critical_signal_rate_in_scenario_per_s_au=critrate_au[:, obstix],
        )

    piy = {}
    piy["critical_signal_rate_in_scenario_per_s"] = dVdE
    piy["critical_signal_rate_in_scenario_per_s_au"] = dVdE_au
    json_utils.write(
        os.path.join(scenario_dir, "critical_signal_rate.json"), piy
    )

    out = {}
    out["dVdE_per_m2_per_GeV_per_s"] = dVdE
    out["dVdE_per_m2_per_GeV_per_s_au"] = dVdE_au
    json_utils.write(os.path.join(scenario_dir, "flux_sensitivity.json"), out)
