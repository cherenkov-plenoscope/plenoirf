#!/usr/bin/python
import sys
import copy
import numpy as np
import plenoirf as irf
import flux_sensitivity
import os
from os.path import join as opj
import json_utils

res = irf.summary.ScriptResources.from_argv(sys.argv)
res.start()

ONREGION_TYPES = res.analysis["on_off_measuremnent"]["onregion_types"]

# load
# ----
energy_bin = res.energy_binning(key="5_bins_per_decade")
zenith_bin = res.zenith_binning("3_bins_per_45deg")

Q = json_utils.tree.Tree(
    opj(res.paths["analysis_dir"], "0300_onregion_trigger_acceptance")
)

M = json_utils.tree.Tree(
    opj(res.paths["analysis_dir"], "0066_energy_estimate_quality")
)

R = json_utils.tree.Tree(
    opj(res.paths["analysis_dir"], "0530_diffsens_background_diff_rates")
)

# prepare
# -------
for zd in range(zenith_bin["num"]):
    zk = f"zd{zd:d}"
    for ok in ONREGION_TYPES:
        for dk in flux_sensitivity.differential.SCENARIOS:
            for pk in res.PARTICLES:
                os.makedirs(
                    opj(res.paths["out_dir"], zk, ok, dk, pk),
                    exist_ok=True,
                )

for zd in range(zenith_bin["num"]):
    zk = f"zd{zd:d}"
    M_gamma = M["gamma"]

    for ok in ONREGION_TYPES:
        for dk in flux_sensitivity.differential.SCENARIOS:
            print(zk, ok, dk)

            scenario = flux_sensitivity.differential.init_scenario_matrices_for_signal_and_background(
                probability_reco_given_true=M_gamma["reco_given_true"],
                probability_reco_given_true_au=M_gamma[
                    "reco_given_true_abs_unc"
                ],
                scenario_key=dk,
            )

            json_utils.write(
                opj(
                    res.paths["out_dir"], zk, ok, dk, "gamma", "scenario.json"
                ),
                scenario,
            )

            (
                A_gamma_scenario,
                A_gamma_scenario_au,
            ) = flux_sensitivity.differential.apply_scenario_to_signal_effective_area(
                signal_area_m2=Q[zk][ok]["gamma"]["point"]["mean"],
                signal_area_m2_au=Q[zk][ok]["gamma"]["point"][
                    "absolute_uncertainty"
                ],
                scenario_G_matrix=scenario["G_matrix"],
                scenario_G_matrix_au=scenario["G_matrix_au"],
            )

            json_utils.write(
                opj(res.paths["out_dir"], zk, ok, dk, "gamma", "area.json"),
                {
                    "mean": A_gamma_scenario,
                    "absolute_uncertainty": A_gamma_scenario_au,
                },
            )

            # background rates
            # ----------------
            for ck in res.COSMIC_RAYS:
                (
                    R_cosmic_ray_scenario,
                    R_cosmic_ray_scenario_au,
                ) = flux_sensitivity.differential.apply_scenario_to_background_rate(
                    rate_in_reco_energy_per_s=R[zk][ok][ck]["reco"]["mean"],
                    rate_in_reco_energy_per_s_au=R[zk][ok][ck]["reco"][
                        "absolute_uncertainty"
                    ],
                    scenario_B_matrix=scenario["B_matrix"],
                    scenario_B_matrix_au=scenario["B_matrix_au"],
                )

                json_utils.write(
                    opj(res.paths["out_dir"], zk, ok, dk, ck, "rate.json"),
                    {
                        "mean": R_cosmic_ray_scenario,
                        "absolute_uncertainty": R_cosmic_ray_scenario_au,
                    },
                )

res.stop()
