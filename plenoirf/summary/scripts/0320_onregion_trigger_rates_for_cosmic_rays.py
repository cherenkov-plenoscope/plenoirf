#!/usr/bin/python
import sys
import numpy as np
import propagate_uncertainties as pru
import plenoirf as irf
import cosmic_fluxes
import os
from os.path import join as opj
import json_utils

res = irf.summary.ScriptResources.from_argv(sys.argv)
res.start()

onregion_acceptance = json_utils.tree.Tree(
    opj(res.paths["analysis_dir"], "0300_onregion_trigger_acceptance")
)

energy_bin = res.energy_binning(key="5_bins_per_decade")
fenergy_bin = res.energy_binning(key="60_bins_per_decade")

zenith_bin = res.zenith_binning("3_bins_per_45deg")

ONREGION_TYPES = res.analysis["on_off_measuremnent"]["onregion_types"]


# cosmic-ray-flux
# ----------------
airshower_fluxes = json_utils.tree.Tree(
    opj(res.paths["analysis_dir"], "0015_flux_of_airshowers")
)

# gamma-ray-flux of reference source
# ----------------------------------
gamma_source = json_utils.read(
    opj(
        res.paths["analysis_dir"],
        "0009_flux_of_gamma_rays",
        "reference_source.json",
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

for zd in range(zenith_bin["num"]):
    zk = f"zd{zd:d}"
    for ok in ONREGION_TYPES:
        for pk in res.PARTICLES:
            os.makedirs(opj(res.paths["out_dir"], zk, ok, pk), exist_ok=True)


for zd in range(zenith_bin["num"]):
    zk = f"zd{zd:d}"

    # gamma-ray
    # ---------
    for ok in ONREGION_TYPES:
        _A = onregion_acceptance[zk][ok]["gamma"]["point"]["mean"]
        _A_au = onregion_acceptance[zk][ok]["gamma"]["point"][
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
            opj(
                res.paths["out_dir"], zk, ok, "gamma", "differential_rate.json"
            ),
            {
                "comment": comment_differential
                + ", "
                + gamma_source["name"]
                + " VS onregion-radius",
                "zenith_key": zk,
                "particle_key": "gamma",
                "onregion_key": ok,
                "unit": "s$^{-1} (GeV)$^{-1}$",
                "mean": dRdE,
                "absolute_uncertainty": dRdE_au,
            },
        )
        json_utils.write(
            opj(res.paths["out_dir"], zk, ok, "gamma", "integral_rate.json"),
            {
                "comment": comment_integral
                + ", "
                + gamma_source["name"]
                + " VS onregion-radius",
                "zenith_key": zk,
                "particle_key": "gamma",
                "onregion_key": ok,
                "unit": "s$^{-1}$",
                "mean": R,
                "absolute_uncertainty": R_au,
            },
        )

        # cosmic-rays
        # -----------
        for ck in res.COSMIC_RAYS:
            cosmic_dFdE = airshower_fluxes[ck]["differential_flux"]["values"]
            cosmic_dFdE_au = airshower_fluxes[ck]["differential_flux"][
                "absolute_uncertainty"
            ]

            _Q = onregion_acceptance[zk][ok][ck]["diffuse"]["mean"]
            _Q_au = onregion_acceptance[zk][ok][ck]["diffuse"][
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
                opj(
                    res.paths["out_dir"], zk, ok, ck, "differential_rate.json"
                ),
                {
                    "comment": comment_differential + " VS onregion-radius",
                    "zenith_key": zk,
                    "particle_key": ck,
                    "onregion_key": ok,
                    "unit": "s$^{-1} (GeV)$^{-1}$",
                    "mean": dRdE,
                    "absolute_uncertainty": dRdE_au,
                },
            )
            json_utils.write(
                opj(res.paths["out_dir"], zk, ok, ck, "integral_rate.json"),
                {
                    "comment": comment_integral + " VS onregion-radius",
                    "zenith_key": zk,
                    "particle_key": ck,
                    "onregion_key": ok,
                    "unit": "s$^{-1}$",
                    "mean": R,
                    "absolute_uncertainty": R_au,
                },
            )

res.stop()
