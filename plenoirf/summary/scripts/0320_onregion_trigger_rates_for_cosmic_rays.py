#!/usr/bin/python
import sys
import numpy as np
import propagate_uncertainties as pru
import plenoirf as irf
import cosmic_fluxes
import os
import json_utils

argv = irf.summary.argv_since_py(sys.argv)
pa = irf.summary.paths_from_argv(argv)

irf_config = irf.summary.read_instrument_response_config(run_dir=pa["run_dir"])
sum_config = irf.summary.read_summary_config(summary_dir=pa["summary_dir"])

os.makedirs(pa["out_dir"], exist_ok=True)

SITES = irf_config["config"]["sites"]
PARTICLES = irf_config["config"]["particles"]
ONREGION_TYPES = sum_config["on_off_measuremnent"]["onregion_types"]

onregion_acceptance = json_utils.tree.read(
    os.path.join(pa["summary_dir"], "0300_onregion_trigger_acceptance")
)

energy_binning = json_utils.read(
    os.path.join(pa["summary_dir"], "0005_common_binning", "energy.json")
)
energy_bin = energy_binning["trigger_acceptance_onregion"]
fenergy_bin = energy_binning["interpolation"]

# cosmic-ray-flux
# ----------------
airshower_fluxes = json_utils.tree.read(
    os.path.join(pa["summary_dir"], "0015_flux_of_airshowers")
)

# gamma-ray-flux of reference source
# ----------------------------------
gamma_source = json_utils.read(
    os.path.join(
        pa["summary_dir"], "0009_flux_of_gamma_rays", "reference_source.json"
    )
)
gamma_dKdE = gamma_source["differential_flux"]["values"]
gamma_dKdE_au = np.zeros(shape=gamma_dKdE.shape)

comment_differential = "Differential trigger-rate, reconstructed in onregion."
comment_integral = "Integral trigger-rate, reconstructed in onregion."


"""
A / m^{2}
Q / m^{2} sr

R / s^{-1}
dRdE / s^{-1} (GeV)^{-1}

F / s^{-1} m^{-2} (sr)^{-1}
dFdE / s^{-1} m^{-2} (sr)^{-1} (GeV)^{-1}

K / s^{-1} m^{-2}
dKdE / s^{-1} m^{-2} (GeV)^{-1}
"""

for sk in SITES:
    for ok in ONREGION_TYPES:
        for pk in PARTICLES:
            os.makedirs(os.path.join(pa["out_dir"], sk, ok, pk), exist_ok=True)

for sk in SITES:
    # gamma-ray
    # ---------
    for ok in ONREGION_TYPES:
        _A = onregion_acceptance[sk][ok]["gamma"]["point"]["mean"]
        _A_au = onregion_acceptance[sk][ok]["gamma"]["point"][
            "absolute_uncertainty"
        ]

        A = np.interp(
            x=fenergy_bin["centers"], xp=energy_bin["centers"], fp=_A
        )
        A_au = np.interp(
            x=fenergy_bin["centers"], xp=energy_bin["centers"], fp=_A_au
        )

        dRdE, dRdE_au = pru.multiply(
            x=gamma_dKdE,
            x_au=gamma_dKdE_au,
            y=A,
            y_au=A_au,
        )

        R, R_au = irf.utils.integrate_rate_where_known(
            dRdE=dRdE,
            dRdE_au=dRdE_au,
            E_edges=fenergy_bin["edges"],
        )

        json_utils.write(
            os.path.join(
                pa["out_dir"], sk, ok, "gamma", "differential_rate.json"
            ),
            {
                "comment": comment_differential
                + ", "
                + gamma_source["name"]
                + " VS onregion-radius",
                "unit": "s$^{-1} (GeV)$^{-1}$",
                "mean": dRdE,
                "absolute_uncertainty": dRdE_au,
            },
        )
        json_utils.write(
            os.path.join(pa["out_dir"], sk, ok, "gamma", "integral_rate.json"),
            {
                "comment": comment_integral
                + ", "
                + gamma_source["name"]
                + " VS onregion-radius",
                "unit": "s$^{-1}$",
                "mean": R,
                "absolute_uncertainty": R_au,
            },
        )

        # cosmic-rays
        # -----------
        for ck in airshower_fluxes[sk]:
            cosmic_dFdE = airshower_fluxes[sk][ck]["differential_flux"][
                "values"
            ]
            cosmic_dFdE_au = airshower_fluxes[sk][ck]["differential_flux"][
                "absolute_uncertainty"
            ]

            _Q = onregion_acceptance[sk][ok][ck]["diffuse"]["mean"]
            _Q_au = onregion_acceptance[sk][ok][ck]["diffuse"][
                "absolute_uncertainty"
            ]

            Q = np.interp(
                x=fenergy_bin["centers"],
                xp=energy_bin["centers"],
                fp=_Q,
            )
            Q_au = np.interp(
                x=fenergy_bin["centers"],
                xp=energy_bin["centers"],
                fp=_Q_au,
            )

            dRdE, dRdE_au = pru.multiply(
                x=cosmic_dFdE,
                x_au=cosmic_dFdE_au,
                y=Q,
                y_au=Q_au,
            )

            R, R_au = irf.utils.integrate_rate_where_known(
                dRdE=dRdE,
                dRdE_au=dRdE_au,
                E_edges=fenergy_bin["edges"],
            )

            json_utils.write(
                os.path.join(
                    pa["out_dir"], sk, ok, ck, "differential_rate.json"
                ),
                {
                    "comment": comment_differential + " VS onregion-radius",
                    "unit": "s$^{-1} (GeV)$^{-1}$",
                    "mean": dRdE,
                    "absolute_uncertainty": dRdE_au,
                },
            )
            json_utils.write(
                os.path.join(pa["out_dir"], sk, ok, ck, "integral_rate.json"),
                {
                    "comment": comment_integral + " VS onregion-radius",
                    "unit": "s$^{-1}$",
                    "mean": R,
                    "absolute_uncertainty": R_au,
                },
            )
